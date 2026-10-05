import copy
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os, sys, argparse
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import get_device
from data_processing.dataset_parser import (load_combined_dataset,
                                             create_dead_reckoning_dataset,
                                             INPUT_SIZE)
from models.dead_reckoning_model import DeadReckoningLSTM


def train_model(sources=('nav',), num_epochs=10, normalize=True,
                 early_stopping_patience=5, min_epochs=3, batch_size=64,
                 tag_suffix='', override_df=None, held_out_val_df=None,
                 feature_cols=None):
    """Trains DeadReckoningLSTM with early stopping (patience-based, keyed on
    val loss) + ReduceLROnPlateau, adapted from the pattern in
    D:\\ΠΜΣ\\Βαθιά Μάθηση\\Εργασίες εξαμήνου\\2η Άσκηση\\25118\\25118.py, see
    AI_RECOVERY_EXECUTION_PLAN.md §8/§9. Needed because the honest (post-
    leakage-fix) val curve bottoms out around epoch 3 and rises after,
    without this, "more epochs" makes the saved model worse, not better.

    batch_size default is 64, deliberately kept at the original value
    despite a real, measured tradeoff: on this machine's AMD GPU (DirectML,
    see device_utils.py), batch=256 trains faster and reaches better
    windowed validation loss (0.632 vs. 0.72-0.78 over 8 epochs) than
    batch=64, but a direct chained-trajectory comparison
    (evaluate_trajectory.py, the metric that actually matters for this
    project, not the windowed proxy) showed batch=256 training reaching
    substantially worse real-world accuracy (37.18m vs. 26.34m final error
    on the same held-out segment) despite its better windowed number, a
    likely large-batch-SGD generalization effect (see models/ANALYSIS.md's
    "Batch size vs. chained accuracy" note for the full finding). batch=64
    stays the default because it is the config with actual chained-accuracy
    evidence behind it; batch_size remains a parameter for anyone who wants
    to explore the tradeoff further, not because larger is assumed better.

    tag_suffix appends to the auto-computed save tag (e.g. 'imu_norm' ->
    'imu_normv2') so a retraining run for the ablation matrix (Track A,
    Phase A5) never silently overwrites dr_lstm_imu_norm.pth, the file
    recovery_orchestrator.py loads for the live, already-demoed system.

    override_df, if given, is used as the entire training-pool dataframe
    instead of load_combined_dataset(base_dir, sources=sources) -- e.g. real
    px4 rows plus synthetic-augmented ones from
    data_processing/px4_synthetic_augment.py concatenated by the caller.
    Deliberately a full replacement, not an additive concat of the
    sources-loaded df: sources=('px4',) would load ALL real px4 rows,
    including whatever the caller means to hold out for validation (see
    held_out_val_df below) -- concatenating augmented rows on top of that
    would leak held-out rows into the train pool. The caller is expected to
    build override_df from only the real rows it intends to train on.

    held_out_val_df, if given, is passed straight through to
    create_dead_reckoning_dataset() -- see that function's own docstring.
    Pairs with override_df: without a fixed, real-only held_out_val_df, a
    plain row-sequential split on override_df would put whatever rows
    happen to be concatenated last (e.g. synthetic ones) in the validation
    tail instead of real held-out data, silently changing what "val loss"
    measures between an augmented run and the real-only baseline it's meant
    to be compared against.

    feature_cols, if given, overrides FEATURE_COLS/INPUT_SIZE for this run
    (e.g. data_processing.dataset_parser.MOTOR_FEATURE_COLS for a px4-only
    motor-informed arm) -- the model is constructed with
    input_size=len(feature_cols) instead of the module-level INPUT_SIZE.
    Defaults to None, unchanged behavior (and unchanged INPUT_SIZE) for
    every existing call site.

    Returns (model, scaler_X, scaler_y, history) where history contains the
    full per-epoch train/val loss lists (for plotting) plus which epoch was
    kept.
    """
    print(f"\nStarting Training  |  sources={list(sources)}  "
          f"max_epochs={num_epochs}  normalize={normalize}  "
          f"early_stopping_patience={early_stopping_patience}  batch_size={batch_size}"
          + (f"  override_df_rows={len(override_df)}" if override_df is not None else "")
          + (f"  held_out_val_rows={len(held_out_val_df)}" if held_out_val_df is not None else "")
          + (f"  feature_cols={len(feature_cols)}" if feature_cols is not None else ""))

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

    df = override_df if override_df is not None else load_combined_dataset(base_dir, sources=sources)
    input_size = len(feature_cols) if feature_cols is not None else INPUT_SIZE
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=10, normalize=normalize, held_out_val_df=held_out_val_df,
        feature_cols=feature_cols)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    device = get_device()

    model     = DeadReckoningLSTM(input_size=input_size, hidden_size=64,
                                  num_layers=2, output_size=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        t_loss = sum(
            _step(model, bx.to(device), by.to(device), optimizer, criterion)
            for bx, by in train_dl
        ) / len(train_dl)

        model.eval()
        with torch.no_grad():
            v_loss = sum(
                criterion(model(bx.to(device)), by.to(device)).item()
                for bx, by in val_dl
            ) / len(val_dl)

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:>2}/{num_epochs}]  Train: {t_loss:.4f}  "
              f"Val: {v_loss:.4f}  LR: {lr_now:.2e}"
              f"{'  <- best' if improved else ''}")

        if epoch >= min_epochs and epochs_since_best >= early_stopping_patience:
            print(f"Early stopping: no val improvement for {early_stopping_patience} "
                  f"epochs (best was epoch {best_epoch}, val={best_val:.4f}).")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training done in {elapsed:.1f}s. Restored weights from epoch "
          f"{best_epoch} (best val={best_val:.4f}).")

    tag = "_".join(sources) + ("_norm" if normalize else "") + tag_suffix
    out_dir  = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'dr_lstm_{tag}.pth')
    # .cpu() before saving: a DirectML-device state_dict (this machine's GPU
    # backend) pickles in a form torch>=2.6's default weights_only=True
    # loader rejects elsewhere; plain CPU tensors are the portable format.
    torch.save(model.cpu().state_dict(), out_path)
    print(f"Model saved → {out_path}")

    if normalize and scaler_X is not None:
        sx_path = os.path.join(out_dir, f'scaler_X_{tag}.pkl')
        sy_path = os.path.join(out_dir, f'scaler_y_{tag}.pkl')
        joblib.dump(scaler_X, sx_path)
        joblib.dump(scaler_y, sy_path)
        print(f"Scalers saved → {sx_path}")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val': best_val,
        'elapsed_s': elapsed,
        'tag': tag,
    }
    hist_path = os.path.join(out_dir, f'history_{tag}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {hist_path}")

    return model, scaler_X, scaler_y, history


def _step(model, bx, by, opt, crit):
    opt.zero_grad()
    loss = crit(model(bx), by)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'],
                    choices=['nav', 'imu', 'px4'],
                    help='Dataset sources to use (space-separated). '
                         "'nav' is deprecated (not a real trajectory, see "
                         "AI_RECOVERY_EXECUTION_PLAN.md); 'imu' is the "
                         "recommended default; 'px4' is real PX4 flight-log data.")
    ap.add_argument('--epochs',    type=int,  default=30,
                    help='Max epochs -- early stopping will likely stop sooner.')
    ap.add_argument('--patience',  type=int,  default=5,
                    help='Early-stopping patience (epochs with no val improvement).')
    ap.add_argument('--no-norm',   action='store_true',
                    help='Disable StandardScaler normalization')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='Training batch size (default 256, tuned for this '
                         'machine\'s AMD/DirectML GPU; pass 64 on CPU-only '
                         'hardware if that regime measures better there).')
    args = ap.parse_args()

    train_model(sources=tuple(args.sources),
                num_epochs=args.epochs,
                normalize=not args.no_norm,
                early_stopping_patience=args.patience,
                batch_size=args.batch_size)
