"""Trains DeadReckoningTransformer from scratch, mirroring train.py's
structure exactly (early stopping + ReduceLROnPlateau) so its results are
directly comparable to the from-scratch LSTM baseline. The from-scratch arm
counterpart to ssl_pretext_transformer.py's SSL-pretrained arm (Track A,
Phase A3 of the coursework-grounded extension roadmap).

Usage:
  python ai_backend\\train_transformer.py --sources imu --epochs 30
"""
import argparse
import copy
import json
import os
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import get_device
from data_processing.dataset_parser import (load_combined_dataset,
                                             create_dead_reckoning_dataset,
                                             INPUT_SIZE)
from models.dr_transformer import DeadReckoningTransformer, augment_batch, WINDOW_SIZE

# Fixed (not per-window-random) permutation of the 10 timestep slots, used
# only when shuffle_window_order=True: diagnostic for whether the
# Transformer's real advantage over GBT (order-blind) depends on genuine
# chronological adjacency, or just on having a consistent, learnable
# structure at all. A single fixed permutation preserves the latter (the
# model can still learn "slot i always holds original timestep perm[i]")
# while destroying the former (slot i is no longer i steps before the
# prediction), applied identically to train and val so the model is
# trained and evaluated under the same (consistent, just non-chronological)
# convention -- see ai_backend/models/ANALYSIS.md's Transformer section for
# the attention-interpretability finding this follows up on.
_SHUFFLE_PERM = np.random.RandomState(42).permutation(WINDOW_SIZE)


def train_model(sources=('imu',), num_epochs=30, normalize=True,
                 early_stopping_patience=5, min_epochs=3, use_augmentation=True,
                 batch_size=64, tag_suffix='', shuffle_window_order=False):
    """Same shape as train.py::train_model. use_augmentation applies
    dr_transformer.augment_batch() to training batches only (never val),
    the mitigation for the data-diversity risk flagged in the roadmap's
    Data sufficiency check. Kept as a flag so the ablation matrix (A5)
    can run both settings and report whether augmentation actually helped,
    rather than assuming it did.

    batch_size default is 64, kept at the original value; see
    train.py::train_model's docstring for why (a larger batch measured
    better windowed loss but worse real chained-trajectory accuracy, and
    for this Transformer specifically the effect was even larger: 29.60m
    vs. 18.33m final chained error at batch=256 vs. batch=64).

    shuffle_window_order=True applies the fixed _SHUFFLE_PERM to every
    window's timestep axis (both train and val) right after dataset
    creation, before any training happens -- see the module-level
    _SHUFFLE_PERM comment for why a single fixed permutation, not a
    per-window random one.

    tag_suffix appends to the auto-computed save tag, same convention as
    train.py::train_model, so a retraining run (e.g. the ablation matrix's
    per-seed runs) never overwrites dr_transformer_imu_norm.pth."""
    print(f"\nStarting Transformer Training  |  sources={list(sources)}  "
          f"max_epochs={num_epochs}  normalize={normalize}  "
          f"early_stopping_patience={early_stopping_patience}  "
          f"augmentation={use_augmentation}  batch_size={batch_size}  "
          f"shuffle_window_order={shuffle_window_order}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=normalize)

    if shuffle_window_order:
        print(f"Applying fixed timestep-order permutation {_SHUFFLE_PERM.tolist()} "
              f"to every window (train and val alike)...")
        X_train = X_train[:, _SHUFFLE_PERM, :]
        X_val = X_val[:, _SHUFFLE_PERM, :]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = get_device()

    model = DeadReckoningTransformer(input_size=INPUT_SIZE, window_size=WINDOW_SIZE).to(device)
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
        t_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            if use_augmentation:
                bx = augment_batch(bx)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

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
    out_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'dr_transformer_{tag}.pth')
    # .cpu() before saving: see train.py::train_model's save-path comment.
    torch.save(model.cpu().state_dict(), out_path)
    print(f"Model saved -> {out_path}")

    if normalize and scaler_X is not None:
        sx_path = os.path.join(out_dir, f'scaler_X_transformer_{tag}.pkl')
        sy_path = os.path.join(out_dir, f'scaler_y_transformer_{tag}.pkl')
        joblib.dump(scaler_X, sx_path)
        joblib.dump(scaler_y, sy_path)
        print(f"Scalers saved -> {sx_path}")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val': best_val,
        'elapsed_s': elapsed,
        'tag': tag,
        'augmentation': use_augmentation,
    }
    hist_path = os.path.join(out_dir, f'history_transformer_{tag}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {hist_path}")

    return model, scaler_X, scaler_y, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'],
                     choices=['nav', 'imu', 'px4'])
    ap.add_argument('--epochs', type=int, default=30,
                     help='Max epochs, early stopping will likely stop sooner.')
    ap.add_argument('--patience', type=int, default=5,
                     help='Early-stopping patience (epochs with no val improvement).')
    ap.add_argument('--no-norm', action='store_true',
                     help='Disable StandardScaler normalization')
    ap.add_argument('--no-augment', action='store_true',
                     help='Disable time-series data augmentation during training.')
    ap.add_argument('--batch-size', type=int, default=64,
                     help='Training batch size (default 64, see train.py for why '
                          'a larger batch is not used despite being faster on GPU).')
    ap.add_argument('--shuffle-order', action='store_true',
                     help='Diagnostic: apply a fixed timestep-order permutation to every '
                          'window (train and val), testing whether the model\'s advantage '
                          'over an order-blind model depends on genuine chronological order. '
                          'Auto-appends "_shuffled" to the saved tag.')
    args = ap.parse_args()

    # Non-default CLI settings get an explicit tag suffix so a diagnostic
    # run (e.g. --no-augment, --shuffle-order) never silently overwrites
    # the validated-default dr_transformer_{tag}.pth checkpoint every other
    # evaluation in this project loads by that exact filename.
    suffix_parts = []
    if args.no_augment:
        suffix_parts.append('_noaugment')
    if args.shuffle_order:
        suffix_parts.append('_shuffled')

    train_model(sources=tuple(args.sources),
                num_epochs=args.epochs,
                normalize=not args.no_norm,
                early_stopping_patience=args.patience,
                use_augmentation=not args.no_augment,
                batch_size=args.batch_size,
                tag_suffix=''.join(suffix_parts),
                shuffle_window_order=args.shuffle_order)
