"""Continues training the LSTM checkpoint from ablation_matrix.py's seed 4
(Track A Phase A5's 5-seed extension, run label cpu_5seeds) past its
original 30-epoch ceiling.

Why this specific checkpoint, and why this is a deliberate, documented
exception rather than a change to the formal ablation study: history_imu_
norm_ablation_seed4.json shows this one run never triggered early stopping
-- it was still improving at epoch 30 (best_epoch=30, patience=5 never saw
5 consecutive non-improving epochs) -- and it already had both the best
windowed validation loss (0.6846) AND the best chained-trajectory error
(24.78m) of all 5 seeds trained in that run (see runs/ablation/cpu_5seeds/
ANALYSIS.md's training-dynamics section). That is two independent metrics
agreeing this run was still converging, not a single lucky chained-error
roll -- the basis for treating it as worth finishing, not cherry-picking.
The ablation matrix's own 5-seed comparison already ran to completion and
its conclusions (Transformer is the best architecture, statistically) stand
exactly as reported; this script does not re-run or revise that comparison,
it only lets this one already-best run actually finish.

Reuses the exact same deterministic data/scaler pipeline seed 4's original
training used (load_combined_dataset + create_dead_reckoning_dataset with
identical arguments -- no shuffling before the split, a plain StandardScaler
fit, so this reproduces numerically identical scalers without needing to
load the old scaler files separately) so continued training is consistent
with the weights already learned under that normalization.

Only promotes the result to the production dr_lstm_imu_norm.pth if it
actually beats the current production checkpoint on the real chained-
trajectory metric -- printed either way, never assumed.

Usage:
  python ai_backend/continue_training_best_lstm.py
"""
import copy
import json
import os
import sys
import time

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import get_device
from data_processing.dataset_parser import (load_combined_dataset,
                                             create_dead_reckoning_dataset,
                                             INPUT_SIZE)
from models.dead_reckoning_model import DeadReckoningLSTM
from evaluate_trajectory import load_val_segment, reconstruct_lstm, ground_truth, euclidean_error

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
SEED4_TAG = 'imu_norm_ablation_seed4'
SEED4_BEST_VAL = 0.6846   # from history_imu_norm_ablation_seed4.json -- the known starting point


def _step(model, bx, by, opt, crit):
    opt.zero_grad()
    loss = crit(model(bx), by)
    loss.backward()
    opt.step()
    return loss.item()


def continue_training(max_additional_epochs: int = 100, patience: int = 5, batch_size: int = 64):
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=('imu',))
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=10, normalize=True)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                           batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                         batch_size=batch_size, shuffle=False)

    device = get_device()
    model = DeadReckoningLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=3)
    ckpt_path = os.path.join(MODELS_DIR, f'dr_lstm_{SEED4_TAG}.pth')
    # weights_only=False: trusted, self-generated checkpoint (see device_utils.py's docstring).
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
    model = model.to(device)
    print(f"Resumed from {ckpt_path}")
    print(f"Starting point: epoch 30 of the original run, best_val={SEED4_BEST_VAL:.4f} "
          f"(still improving when that run's fixed epoch budget ended).")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_val = SEED4_BEST_VAL
    best_state = copy.deepcopy(model.state_dict())   # the loaded seed-4 weights are the baseline to beat
    best_epoch = 0   # 0 means "the original seed-4 checkpoint itself was never beaten"
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, max_additional_epochs + 1):
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
        print(f"Epoch [{epoch:>3}] (continuing past original epoch 30)  Train: {t_loss:.4f}  "
              f"Val: {v_loss:.4f}  LR: {lr_now:.2e}{'  <- best' if improved else ''}")

        if epochs_since_best >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs "
                  f"(best val={best_val:.4f}).")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    print(f"\nContinued training done in {elapsed:.1f}s.")
    if best_epoch == 0:
        print("No improvement found beyond the original seed-4 checkpoint -- "
              "it had, in fact, already converged; keeping it unchanged.")
    else:
        print(f"Improved: best_val {SEED4_BEST_VAL:.4f} -> {best_val:.4f} "
              f"at continued epoch {best_epoch}.")

    history = {
        'started_from': SEED4_TAG,
        'started_from_best_val': SEED4_BEST_VAL,
        'continued_train_losses': train_losses,
        'continued_val_losses': val_losses,
        'continued_best_epoch': best_epoch,
        'best_val': best_val,
        'elapsed_s': elapsed,
    }
    return model, scaler_X, scaler_y, history


def evaluate_chained(model, scaler_X, scaler_y) -> dict:
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    seg = load_val_segment(base_dir, n_samples=3000)
    gt = ground_truth(seg)
    pos = reconstruct_lstm(seg, model, scaler_X, scaler_y)
    err = euclidean_error(pos, gt)
    return {'final': float(err[-1]), 'mean': float(err.mean()), 'max': float(err.max())}


if __name__ == '__main__':
    model, scaler_X, scaler_y, history = continue_training()

    print("\n=== Evaluating continued model on the real chained-trajectory segment ===")
    continued_result = evaluate_chained(model, scaler_X, scaler_y)
    print(f"Continued model:  final={continued_result['final']:.2f}m  "
          f"mean={continued_result['mean']:.2f}m  max={continued_result['max']:.2f}m")

    # Always save under its own tag first -- never overwrites production
    # unconditionally, only after the comparison below says it earned it.
    continued_tag = 'imu_norm_lstm_continued'
    out_path = os.path.join(MODELS_DIR, f'dr_lstm_{continued_tag}.pth')
    torch.save(model.cpu().state_dict(), out_path)
    joblib.dump(scaler_X, os.path.join(MODELS_DIR, f'scaler_X_{continued_tag}.pkl'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, f'scaler_y_{continued_tag}.pkl'))
    history['chained_eval'] = continued_result
    with open(os.path.join(MODELS_DIR, f'history_{continued_tag}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved continued model -> {out_path}")

    # Compare against whatever is CURRENTLY the production checkpoint, fresh
    # (not a cached/possibly-stale cited number), so the promotion decision
    # is based on a real apples-to-apples comparison at the moment this runs.
    prod_path = os.path.join(MODELS_DIR, 'dr_lstm_imu_norm.pth')
    prod_sx_path = os.path.join(MODELS_DIR, 'scaler_X_imu_norm.pkl')
    prod_sy_path = os.path.join(MODELS_DIR, 'scaler_y_imu_norm.pkl')
    print("\n=== Evaluating the CURRENT production model, fresh, for comparison ===")
    prod_model = DeadReckoningLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=3)
    prod_model.load_state_dict(torch.load(prod_path, map_location='cpu', weights_only=False))
    prod_model.eval()
    prod_sx = joblib.load(prod_sx_path)
    prod_sy = joblib.load(prod_sy_path)
    prod_result = evaluate_chained(prod_model, prod_sx, prod_sy)
    print(f"Current production dr_lstm_imu_norm.pth:  final={prod_result['final']:.2f}m  "
          f"mean={prod_result['mean']:.2f}m  max={prod_result['max']:.2f}m")

    print(f"\n{'Model':<32}{'Final [m]':>12}{'Mean [m]':>12}{'Max [m]':>12}")
    print(f"{'Current production':<32}{prod_result['final']:>12.2f}{prod_result['mean']:>12.2f}"
          f"{prod_result['max']:>12.2f}")
    print(f"{'Continued (from seed4)':<32}{continued_result['final']:>12.2f}"
          f"{continued_result['mean']:>12.2f}{continued_result['max']:>12.2f}")

    if continued_result['final'] < prod_result['final']:
        print(f"\nPROMOTING: continued model beats current production "
              f"({continued_result['final']:.2f}m < {prod_result['final']:.2f}m) on the real "
              f"chained-trajectory metric. Overwriting dr_lstm_imu_norm.pth.")
        import shutil
        # Back up the outgoing production checkpoint under a dated tag first
        # -- overwriting it is a one-way door otherwise, and this is a
        # production artifact other tools (recovery_orchestrator.py, live_
        # mission_demo.py, evaluate_trajectory.py's default tag) load by
        # this exact filename.
        backup_tag = 'imu_norm_pre_seed4_continuation'
        shutil.copy(prod_path, os.path.join(MODELS_DIR, f'dr_lstm_{backup_tag}.pth'))
        shutil.copy(prod_sx_path, os.path.join(MODELS_DIR, f'scaler_X_{backup_tag}.pkl'))
        shutil.copy(prod_sy_path, os.path.join(MODELS_DIR, f'scaler_y_{backup_tag}.pkl'))
        print(f"Backed up outgoing production checkpoint with tag '{backup_tag}'.")

        shutil.copy(out_path, prod_path)
        shutil.copy(os.path.join(MODELS_DIR, f'scaler_X_{continued_tag}.pkl'), prod_sx_path)
        shutil.copy(os.path.join(MODELS_DIR, f'scaler_y_{continued_tag}.pkl'), prod_sy_path)
        with open(os.path.join(MODELS_DIR, 'history_imu_norm.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print("Promoted. dr_lstm_imu_norm.pth is now the continued-from-seed4 model.")
    else:
        print(f"\nNOT promoting: continued model ({continued_result['final']:.2f}m) did not beat "
              f"current production ({prod_result['final']:.2f}m) on the chained metric. "
              f"Production checkpoint left unchanged; the continued model remains saved "
              f"under tag '{continued_tag}' for reference.")
