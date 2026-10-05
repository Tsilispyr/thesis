"""Trains DeadReckoningLSTMUncertainty with the same early-stopping +
ReduceLROnPlateau structure as train.py, but with gaussian_nll_loss instead
of MSELoss. Track A, Phase A2 of the coursework-grounded extension roadmap.

Saved under dr_lstm_uncertainty_{tag}.pth -- deliberately not overwriting
dr_lstm_{tag}.pth, since recovery_orchestrator.py loads the plain
DeadReckoningLSTM by that exact filename and this is an additive, offline
arm, not a replacement for the live production model.

Usage:
  python ai_backend\\train_uncertainty.py --sources imu --epochs 30
"""
import argparse
import copy
import json
import os
import sys
import time

import joblib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import get_device
from data_processing.dataset_parser import (load_combined_dataset,
                                             create_dead_reckoning_dataset,
                                             INPUT_SIZE)
from models.dead_reckoning_model_uncertainty import (DeadReckoningLSTMUncertainty,
                                                       gaussian_nll_loss)


def train_model(sources=('imu',), num_epochs=30, normalize=True,
                 early_stopping_patience=5, min_epochs=3, batch_size=64):
    """Same shape as train.py::train_model, criterion swapped for
    gaussian_nll_loss. Returns (model, scaler_X, scaler_y, history); history
    additionally reports val MSE-on-mu (mu vs. target, ignoring log_var) so
    it's directly comparable to the plain LSTM's val loss in STATUS.md.

    batch_size default is 64, kept at the original value; see
    train.py::train_model's docstring for why (a larger batch measured
    better windowed loss but worse real chained-trajectory accuracy).
    """
    print(f"\nStarting Uncertainty Training  |  sources={list(sources)}  "
          f"max_epochs={num_epochs}  normalize={normalize}  "
          f"early_stopping_patience={early_stopping_patience}  batch_size={batch_size}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')

    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=10, normalize=normalize)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = get_device()

    model = DeadReckoningLSTMUncertainty(input_size=INPUT_SIZE, hidden_size=64,
                                          num_layers=2, output_size=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=2)

    train_losses, val_losses, val_mse_mu = [], [], []
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
            optimizer.zero_grad()
            mu, log_var = model(bx)
            loss = gaussian_nll_loss(mu, log_var, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

        model.eval()
        v_loss, v_mse = 0.0, 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                mu, log_var = model(bx)
                v_loss += gaussian_nll_loss(mu, log_var, by).item()
                v_mse += ((mu - by) ** 2).mean().item()
        v_loss /= len(val_dl)
        v_mse /= len(val_dl)

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_mse_mu.append(v_mse)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:>2}/{num_epochs}]  Train NLL: {t_loss:.4f}  "
              f"Val NLL: {v_loss:.4f}  Val MSE(mu): {v_mse:.4f}  LR: {lr_now:.2e}"
              f"{'  <- best' if improved else ''}")

        if epoch >= min_epochs and epochs_since_best >= early_stopping_patience:
            print(f"Early stopping: no val NLL improvement for {early_stopping_patience} "
                  f"epochs (best was epoch {best_epoch}, val NLL={best_val:.4f}).")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training done in {elapsed:.1f}s. Restored weights from epoch "
          f"{best_epoch} (best val NLL={best_val:.4f}).")

    tag = "_".join(sources) + ("_norm" if normalize else "")
    out_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'dr_lstm_uncertainty_{tag}.pth')
    # .cpu() before saving: see train.py::train_model's save-path comment.
    torch.save(model.cpu().state_dict(), out_path)
    print(f"Model saved -> {out_path}")

    if normalize and scaler_X is not None:
        sx_path = os.path.join(out_dir, f'scaler_X_uncertainty_{tag}.pkl')
        sy_path = os.path.join(out_dir, f'scaler_y_uncertainty_{tag}.pkl')
        joblib.dump(scaler_X, sx_path)
        joblib.dump(scaler_y, sy_path)
        print(f"Scalers saved -> {sx_path}")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mse_mu': val_mse_mu,
        'best_epoch': best_epoch,
        'best_val': best_val,
        'elapsed_s': elapsed,
        'tag': tag,
    }
    hist_path = os.path.join(out_dir, f'history_uncertainty_{tag}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {hist_path}")

    return model, scaler_X, scaler_y, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'],
                    choices=['nav', 'imu', 'px4'],
                    help='Dataset sources to use (space-separated).')
    ap.add_argument('--epochs', type=int, default=30,
                    help='Max epochs, early stopping will likely stop sooner.')
    ap.add_argument('--patience', type=int, default=5,
                    help='Early-stopping patience (epochs with no val NLL improvement).')
    ap.add_argument('--no-norm', action='store_true',
                    help='Disable StandardScaler normalization')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='Training batch size (default 64, see train.py for why '
                         'a larger batch is not used despite being faster on GPU).')
    args = ap.parse_args()

    train_model(sources=tuple(args.sources),
                num_epochs=args.epochs,
                normalize=not args.no_norm,
                early_stopping_patience=args.patience,
                batch_size=args.batch_size)
