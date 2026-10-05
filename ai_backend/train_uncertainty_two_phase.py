"""Two-phase variant of train_uncertainty.py, testing the concrete fix
ai_backend/models/ANALYSIS.md proposed for the uncertainty head's weak
calibration (r=0.203): "separating the mean and variance training into two
phases (train mu to convergence first, then freeze it and fit log_var alone)
to avoid the overconfidence-race dynamic" documented there -- validation NLL
peaks at epoch 1 while validation MSE-on-mu keeps improving through epoch 6,
so the production checkpoint (restored from epoch 1) uses an under-converged
mu.

DeadReckoningLSTMUncertainty's mu and log_var share one nn.Linear(32, 6) head
(models/dead_reckoning_model_uncertainty.py:52-56, first 3 outputs = mu, last
3 = log_var), not two separate heads -- so "freezing mu" can't be done with a
plain requires_grad=False on a whole layer. Phase 2 instead freezes the
shared trunk (lstm + fc[0]) via requires_grad=False, and additionally
zero-masks the mu rows (0:3) of fc[2].weight/bias's gradient every step
before the optimizer applies it, so only the log_var rows (3:6) of that
shared layer actually move -- mathematically equivalent to two independent
heads with mu's frozen, with zero architecture change (the checkpoint stays
load-compatible with the existing DeadReckoningLSTMUncertainty class and
evaluate_trajectory.py::reconstruct_lstm_uncertainty()).

Phase 1 trains on plain MSE(mu, target) only (like the plain LSTM), to a
proper convergence (early-stopped on val MSE) -- not the NLL-driven epoch-1
stop the current production uncertainty checkpoint uses. Phase 2 then trains
NLL with the above freeze/mask, early-stopped on val NLL, monitoring both
val NLL and the calibration correlation directly.

Saved under a separate _twophase tag (not overwriting the existing
dr_lstm_uncertainty_imu_norm.pth), consistent with this project's practice
of testing an alternative before deciding whether to promote it.

Usage:
  python ai_backend/train_uncertainty_two_phase.py --sources imu
"""
import argparse
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
from models.dead_reckoning_model_uncertainty import (DeadReckoningLSTMUncertainty,
                                                       gaussian_nll_loss)


def train_two_phase(sources=('imu',), phase1_epochs=30, phase2_epochs=30,
                     normalize=True, phase1_patience=5, phase2_patience=5,
                     min_epochs=3, batch_size=64):
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

    # ---- Phase 1: train mu to convergence on plain MSE, log_var untouched ----
    print(f"\n=== Phase 1: train mu on plain MSE (max {phase1_epochs} epochs) ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    mse = nn.MSELoss()

    p1_train_losses, p1_val_losses = [], []
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        t_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            mu, _ = model(bx)
            loss = mse(mu, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                mu, _ = model(bx)
                v_loss += mse(mu, by).item()
        v_loss /= len(val_dl)

        scheduler.step(v_loss)
        p1_train_losses.append(t_loss)
        p1_val_losses.append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(f"[Phase 1] Epoch [{epoch:>2}/{phase1_epochs}]  Train MSE: {t_loss:.4f}  "
              f"Val MSE: {v_loss:.4f}{'  <- best' if improved else ''}")

        if epoch >= min_epochs and epochs_since_best >= phase1_patience:
            print(f"[Phase 1] Early stopping (best epoch {best_epoch}, val MSE={best_val:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    phase1_elapsed = time.time() - t0
    phase1_best_epoch = best_epoch
    phase1_best_val = best_val
    print(f"Phase 1 done in {phase1_elapsed:.1f}s, mu restored from epoch "
          f"{phase1_best_epoch} (val MSE={phase1_best_val:.4f}).")

    # ---- Phase 2: freeze trunk + mu rows, train only log_var rows on NLL ----
    print(f"\n=== Phase 2: freeze mu, fit log_var alone on NLL (max {phase2_epochs} epochs) ===")
    for p in model.lstm.parameters():
        p.requires_grad = False
    for p in model.fc[0].parameters():   # Linear(hidden_size, 32) -- shared trunk
        p.requires_grad = False
    final_layer = model.fc[2]            # Linear(32, output_size*2): rows 0:3=mu, 3:6=log_var
    output_size = model.output_size

    # weight_decay=0 here deliberately: Adam applies weight decay as
    # grad += weight_decay * param INSIDE step(), using the .grad tensor
    # AFTER mask_mu_grad() zeroes it -- a nonzero weight_decay would silently
    # re-introduce a small but real gradient on the "frozen" mu rows (found
    # by comparing phase 1's final val MSE against phase 2 epoch 1's
    # supposedly-unchanged mu MSE: 0.7080 vs. 1.0314, not equal as they must
    # be if mu is truly frozen -- root-caused to exactly this).
    optimizer2 = optim.Adam([final_layer.weight, final_layer.bias], lr=0.001, weight_decay=0.0)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=2)

    def mask_mu_grad():
        # Zero the mu rows' gradient every step so only log_var rows (output_size:) move,
        # even though both live in the same nn.Linear weight/bias tensor.
        if final_layer.weight.grad is not None:
            final_layer.weight.grad[:output_size].zero_()
        if final_layer.bias.grad is not None:
            final_layer.bias.grad[:output_size].zero_()

    p2_train_losses, p2_val_losses, p2_val_mse_mu = [], [], []
    best_val2 = float('inf')
    best_state2 = None
    best_epoch2 = 0
    epochs_since_best2 = 0
    t1 = time.time()

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        t_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer2.zero_grad()
            mu, log_var = model(bx)
            loss = gaussian_nll_loss(mu, log_var, by)
            loss.backward()
            mask_mu_grad()
            optimizer2.step()
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

        scheduler2.step(v_loss)
        p2_train_losses.append(t_loss)
        p2_val_losses.append(v_loss)
        p2_val_mse_mu.append(v_mse)

        improved = v_loss < best_val2
        if improved:
            best_val2 = v_loss
            best_state2 = copy.deepcopy(model.state_dict())
            best_epoch2 = epoch
            epochs_since_best2 = 0
        else:
            epochs_since_best2 += 1

        print(f"[Phase 2] Epoch [{epoch:>2}/{phase2_epochs}]  Train NLL: {t_loss:.4f}  "
              f"Val NLL: {v_loss:.4f}  Val MSE(mu, should be frozen~{phase1_best_val:.4f}): "
              f"{v_mse:.4f}{'  <- best' if improved else ''}")

        if epoch >= min_epochs and epochs_since_best2 >= phase2_patience:
            print(f"[Phase 2] Early stopping (best epoch {best_epoch2}, val NLL={best_val2:.4f}).")
            break

    if best_state2 is not None:
        model.load_state_dict(best_state2)
    phase2_elapsed = time.time() - t1
    print(f"Phase 2 done in {phase2_elapsed:.1f}s, restored from epoch "
          f"{best_epoch2} (val NLL={best_val2:.4f}).")

    for p in model.parameters():
        p.requires_grad = True   # restore for any downstream fine-tuning / re-saving

    tag = "_".join(sources) + ("_norm" if normalize else "") + "_twophase"
    out_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f'dr_lstm_uncertainty_{tag}.pth')
    torch.save(model.cpu().state_dict(), out_path)
    print(f"Model saved -> {out_path}")

    if normalize and scaler_X is not None:
        joblib.dump(scaler_X, os.path.join(out_dir, f'scaler_X_uncertainty_{tag}.pkl'))
        joblib.dump(scaler_y, os.path.join(out_dir, f'scaler_y_uncertainty_{tag}.pkl'))

    history = {
        'phase1_train_losses': p1_train_losses, 'phase1_val_losses': p1_val_losses,
        'phase1_best_epoch': phase1_best_epoch, 'phase1_best_val_mse': phase1_best_val,
        'phase1_elapsed_s': phase1_elapsed,
        'phase2_train_losses': p2_train_losses, 'phase2_val_losses': p2_val_losses,
        'phase2_val_mse_mu': p2_val_mse_mu,
        'phase2_best_epoch': best_epoch2, 'phase2_best_val_nll': best_val2,
        'phase2_elapsed_s': phase2_elapsed,
        'tag': tag,
    }
    hist_path = os.path.join(out_dir, f'history_uncertainty_{tag}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {hist_path}")

    return model, scaler_X, scaler_y, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    ap.add_argument('--phase1-epochs', type=int, default=30)
    ap.add_argument('--phase2-epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    train_two_phase(sources=tuple(args.sources),
                     phase1_epochs=args.phase1_epochs,
                     phase2_epochs=args.phase2_epochs,
                     batch_size=args.batch_size)
