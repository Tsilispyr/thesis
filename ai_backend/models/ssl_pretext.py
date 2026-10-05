"""Self-supervised masked-timestep reconstruction pretext task (Goal 1 item 2,
LSTM primary arm - AI_RECOVERY_EXECUTION_PLAN.md §12).

Pretrain an LSTM encoder to reconstruct randomly-masked timesteps in a
14-feature IMU window (no labels needed - self-supervised), then transfer
the encoder's weights into DeadReckoningLSTM and fine-tune on the existing
supervised Δposition regression task. Run as an ablation against the
from-scratch supervised baseline (train.py) - does pretraining actually
help, and by how much (Goal 2's 5-model benchmark matrix).

CLI:
  python ai_backend/models/ssl_pretext.py --sources imu --ssl-epochs 10 --finetune-epochs 10
"""
import copy
import json
import os
import sys
import time
import argparse

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_utils import get_device
from models.dead_reckoning_model import DeadReckoningLSTM, INPUT_SIZE
from data_processing.dataset_parser import load_combined_dataset, create_dead_reckoning_dataset

HIDDEN_SIZE = 64
NUM_LAYERS = 2


class MaskedLSTMAutoencoder(nn.Module):
    """Encoder is architecturally identical to DeadReckoningLSTM.lstm (same
    input_size/hidden_size/num_layers/dropout) so its weights transfer
    directly via transfer_encoder_weights() below - no key-name remapping
    needed. A per-timestep linear decoder head reconstructs the full
    INPUT_SIZE-dim feature vector at every timestep from the encoder's
    hidden output; loss is computed only over masked positions.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x_masked.size(0), self.hidden_size, device=x_masked.device)
        c0 = torch.zeros(self.num_layers, x_masked.size(0), self.hidden_size, device=x_masked.device)
        out, _ = self.encoder(x_masked, (h0, c0))
        return self.decoder(out)   # (batch, seq_len, input_size) reconstruction


def random_mask(x: torch.Tensor, mask_ratio: float = 0.3):
    """Zeros a random subset of timesteps per sample (whole-timestep masking,
    not per-feature - matches "mask random windows of the 14-feature
    sequence" from the plan). Returns (x_masked, mask); mask is a
    (batch, seq_len) bool tensor, True where masked.

    mask_ratio is a documented ablation axis (Goal 2, §12) - sweep this
    alongside window/hidden size when comparing configurations.

    Vectorized (random per-position scores + topk) rather than a per-sample
    Python loop with torch.randperm: on this machine's DirectML GPU at
    batch=256, the loop version measured 156ms/call versus 0.9ms for this
    one, found while training ssl_pretext_transformer.py's SSL arm (see
    that file's own random_mask docstring for the full story) and ported
    back here since Phase A5's multi-seed matrix runs this same function
    repeatedly for the SSL-LSTM arm.
    """
    batch, seq_len, _ = x.shape
    n_mask = max(1, int(round(seq_len * mask_ratio)))
    scores = torch.rand(batch, seq_len, device=x.device)
    _, mask_idx = torch.topk(scores, n_mask, dim=1, largest=False)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)
    mask.scatter_(1, mask_idx, True)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def masked_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged over feature dim, then restricted to masked timesteps
    only (standard masked-reconstruction / MAE-style objective - loss on
    the imputed positions, not positions the model could already see)."""
    diff = ((recon - target) ** 2).mean(dim=-1)   # (batch, seq_len)
    masked_diff = diff[mask]
    if masked_diff.numel() == 0:
        return diff.sum() * 0.0
    return masked_diff.mean()


def transfer_encoder_weights(ssl_model: MaskedLSTMAutoencoder, dr_model: DeadReckoningLSTM) -> None:
    """Copies pretrained encoder weights into a DeadReckoningLSTM's `lstm`
    submodule for fine-tuning. Both are plain nn.LSTM(14, 64, 2, ...)
    instances, so state_dict keys match exactly (weight_ih_l0, weight_hh_l0,
    bias_ih_l0, bias_hh_l0, ..._l1) - no remapping required."""
    dr_model.lstm.load_state_dict(ssl_model.encoder.state_dict())


def _run_early_stopped(model, train_dl, val_dl, opt, scheduler, epoch_fn,
                        epochs, patience, min_epochs, log_prefix):
    """Shared early-stopping loop for both SSL stages: epoch_fn(model, dl, opt_or_None)
    -> avg_loss (opt=None means eval-only/no backward pass). Returns
    (best_state, history) - caller restores best_state into the model."""
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tl = epoch_fn(model, train_dl, opt)
        model.eval()
        with torch.no_grad():
            vl = epoch_fn(model, val_dl, None)

        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        improved = vl < best_val
        if improved:
            best_val, best_epoch, epochs_since_best = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_best += 1

        lr_now = opt.param_groups[0]['lr']
        print(f"{log_prefix} Epoch [{epoch:>2}/{epochs}]  Train: {tl:.4f}  "
              f"Val: {vl:.4f}  LR: {lr_now:.2e}{'  <- best' if improved else ''}")

        if epoch >= min_epochs and epochs_since_best >= patience:
            print(f"{log_prefix} Early stopping: no val improvement for {patience} "
                  f"epochs (best epoch {best_epoch}, val={best_val:.4f}).")
            break

    elapsed = time.time() - t0
    history = {'train_losses': train_losses, 'val_losses': val_losses,
               'best_epoch': best_epoch, 'best_val': best_val, 'elapsed_s': elapsed}
    return best_state, history


def pretrain_ssl(sources=('imu',), epochs=15, mask_ratio=0.3, batch_size=64, lr=1e-3,
                  patience=4, min_epochs=3):
    """Stage 1: self-supervised masked-reconstruction pretraining. Reuses
    dataset_parser's full leakage-safe pipeline (source-boundary window
    skipping, split-then-scale) for X; the delta-position targets it also
    returns are simply unused here - SSL needs no labels."""
    print(f"\n[SSL] Pretraining  |  sources={list(sources)}  max_epochs={epochs}  "
          f"mask_ratio={mask_ratio}  patience={patience}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, _y_train, X_val, _y_val, scaler_X, _scaler_y = create_dead_reckoning_dataset(
        df, window_size=10, normalize=True)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=batch_size, shuffle=False)

    device = get_device()
    model = MaskedLSTMAutoencoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    def _epoch(m, dl, optimizer):
        total = 0.0
        for (bx,) in dl:
            bx = bx.to(device)
            x_masked, mask = random_mask(bx, mask_ratio)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = masked_reconstruction_loss(m(x_masked), bx, mask)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()
        return total / len(dl)

    best_state, history = _run_early_stopped(
        model, train_dl, val_dl, opt, scheduler, _epoch, epochs, patience, min_epochs, "[SSL]")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[SSL] Pretraining done in {history['elapsed_s']:.1f}s, "
          f"restored epoch {history['best_epoch']} (val={history['best_val']:.4f}).")

    return model, scaler_X, history


def finetune_from_ssl(ssl_model: MaskedLSTMAutoencoder, sources=('imu',), epochs=30, batch_size=64, lr=1e-3,
                       patience=5, min_epochs=3):
    """Stage 2: transfer the pretrained encoder into DeadReckoningLSTM and
    fine-tune on the supervised Δposition regression task - same objective,
    data pipeline, and early-stopping discipline as train.py, so results are
    directly comparable to the from-scratch supervised baseline (Goal 2's
    ablation matrix)."""
    print(f"\n[SSL] Fine-tuning for Δposition regression  |  sources={list(sources)}  "
          f"max_epochs={epochs}  patience={patience}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=10, normalize=True)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                           batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                         batch_size=batch_size, shuffle=False)

    device = get_device()
    model = DeadReckoningLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                               num_layers=NUM_LAYERS, output_size=3).to(device)
    transfer_encoder_weights(ssl_model, model)

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    def _epoch(m, dl, optimizer):
        total = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = criterion(m(bx), by)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()
        return total / len(dl)

    best_state, history = _run_early_stopped(
        model, train_dl, val_dl, opt, scheduler, _epoch, epochs, patience, min_epochs, "[SSL-finetune]")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[SSL-finetune] Done in {history['elapsed_s']:.1f}s, "
          f"restored epoch {history['best_epoch']} (val={history['best_val']:.4f}).")

    return model, scaler_X, scaler_y, history


def save_ssl_lstm_artifacts(ssl_model: MaskedLSTMAutoencoder, ft_model: DeadReckoningLSTM,
                             scaler_X, scaler_y, ssl_history: dict, ft_history: dict,
                             sources, tag_suffix: str = '') -> str:
    """Persists both stages' checkpoints/scalers/histories to models/, tagged
    by sources (+ optional tag_suffix, same convention as
    train.py::train_model, so per-seed ablation-matrix runs don't overwrite
    each other or the live-system files). Returns the tag used.

    Extracted from this module's own __main__ block so ablation_matrix.py
    can call it too -- previously ablation_matrix.py trained SSL-LSTM via
    pretrain_ssl()/finetune_from_ssl() directly and discarded their returned
    history, so no checkpoint or history ever reached disk for that arm."""
    tag = "_".join(sources) + "_ssl_norm" + tag_suffix
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    # .cpu() before saving: see train.py::train_model's save-path comment.
    torch.save(ssl_model.cpu().state_dict(), os.path.join(out_dir, f'ssl_pretrain_{tag}.pth'))
    torch.save(ft_model.cpu().state_dict(), os.path.join(out_dir, f'dr_lstm_{tag}.pth'))
    joblib.dump(scaler_X, os.path.join(out_dir, f'scaler_X_{tag}.pkl'))
    joblib.dump(scaler_y, os.path.join(out_dir, f'scaler_y_{tag}.pkl'))
    with open(os.path.join(out_dir, f'history_ssl_pretrain_{tag}.json'), 'w') as f:
        json.dump(ssl_history, f, indent=2)
    with open(os.path.join(out_dir, f'history_{tag}.json'), 'w') as f:
        json.dump(ft_history, f, indent=2)
    print(f"Saved SSL-pretrained encoder + fine-tuned model with tag '{tag}' -> {out_dir}")
    return tag


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    ap.add_argument('--ssl-epochs', type=int, default=15,
                    help='Max SSL pretraining epochs -- early stopping will likely stop sooner.')
    ap.add_argument('--finetune-epochs', type=int, default=30,
                    help='Max fine-tuning epochs -- early stopping will likely stop sooner.')
    ap.add_argument('--mask-ratio', type=float, default=0.3)
    ap.add_argument('--ssl-patience', type=int, default=4)
    ap.add_argument('--finetune-patience', type=int, default=5)
    args = ap.parse_args()

    ssl_model, _, ssl_history = pretrain_ssl(
        sources=tuple(args.sources), epochs=args.ssl_epochs,
        mask_ratio=args.mask_ratio, patience=args.ssl_patience)
    ft_model, scaler_X, scaler_y, ft_history = finetune_from_ssl(
        ssl_model, sources=tuple(args.sources), epochs=args.finetune_epochs,
        patience=args.finetune_patience)

    save_ssl_lstm_artifacts(ssl_model, ft_model, scaler_X, scaler_y,
                             ssl_history, ft_history, args.sources)
