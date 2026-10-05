"""Self-supervised masked-timestep reconstruction for the Transformer arm,
mirroring ssl_pretext.py's two-stage structure exactly (Track A, Phase A3
of the coursework-grounded extension roadmap).

Pretrain a TransformerTrunk encoder to reconstruct randomly-masked
timesteps in a 14-feature IMU window, then transfer its weights into
DeadReckoningTransformer and fine-tune on the supervised delta-position
regression task, run as an ablation against both the from-scratch
Transformer (dr_transformer.py) and the existing LSTM/SSL-LSTM results.

Given the LSTM's own SSL fine-tuning result was an honest negative (see
models/ANALYSIS.md), this is a genuine hypothesis test, not a promised win,
and is reported the same way either way it comes out.

CLI:
  python ai_backend/models/ssl_pretext_transformer.py --sources imu --ssl-epochs 15 --finetune-epochs 30
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_utils import get_device
from data_processing.dataset_parser import load_combined_dataset, create_dead_reckoning_dataset
from models.dr_transformer import (DeadReckoningTransformer, TransformerTrunk, augment_batch,
                                    INPUT_SIZE, WINDOW_SIZE)

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1


class MaskedTransformerAutoencoder(nn.Module):
    """encoder (a TransformerTrunk) is architecturally identical to
    DeadReckoningTransformer.trunk, so its weights transfer directly via
    transfer_encoder_weights() below, the same relationship
    MaskedLSTMAutoencoder.encoder has to DeadReckoningLSTM.lstm. A per-
    timestep linear decoder reconstructs the full INPUT_SIZE-dim feature
    vector at every non-cls token position; loss is computed only over
    masked positions (the cls token itself is never a reconstruction
    target, it has no corresponding input timestep).
    """

    def __init__(self, input_size=INPUT_SIZE, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, window_size=WINDOW_SIZE):
        super().__init__()
        self.encoder = TransformerTrunk(input_size, d_model, nhead, num_layers,
                                         dim_feedforward, dropout, window_size)
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x_masked)          # (batch, window+1, d_model), cls at index 0
        timestep_tokens = tokens[:, 1:, :]        # drop cls, keep one token per input timestep
        return self.decoder(timestep_tokens)      # (batch, window, input_size) reconstruction


def random_mask(x: torch.Tensor, mask_ratio: float = 0.3):
    """Identical semantics to ssl_pretext.py::random_mask (whole-timestep
    masking, not per-feature). Kept as a separate copy rather than a shared
    import so this file has no coupling to the LSTM SSL module beyond the
    dataset pipeline, matching how little the two model families otherwise
    share.

    Vectorized (random per-position scores + topk, instead of a per-sample
    Python loop with torch.randperm): measured 156ms/call on this machine's
    DirectML GPU at batch=256 for the original loop version, versus 0.9ms
    for this one, a real bottleneck found and fixed while training the SSL
    arm for the first time (that first run's 15-epoch pretraining stage
    took ~7467s, disproportionate to fine-tuning's ~1400s for more epochs
    of a comparable-cost model, this masking call was why)."""
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
    diff = ((recon - target) ** 2).mean(dim=-1)   # (batch, seq_len)
    masked_diff = diff[mask]
    if masked_diff.numel() == 0:
        return diff.sum() * 0.0
    return masked_diff.mean()


def transfer_encoder_weights(ssl_model: MaskedTransformerAutoencoder,
                              dr_model: DeadReckoningTransformer) -> None:
    """Copies the pretrained trunk into a DeadReckoningTransformer's `trunk`
    submodule. Both are TransformerTrunk(14, 64, 4, 4, 256, 0.1, 10)
    instances built with identical constructor arguments, so state_dict
    keys match exactly, no remapping required."""
    dr_model.trunk.load_state_dict(ssl_model.encoder.state_dict())


def _run_early_stopped(model, train_dl, val_dl, opt, scheduler, epoch_fn,
                        epochs, patience, min_epochs, log_prefix):
    """Same shared early-stopping loop shape as ssl_pretext.py's
    _run_early_stopped (kept as its own copy rather than a shared import,
    same reasoning as random_mask above)."""
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


def pretrain_ssl_transformer(sources=('imu',), epochs=15, mask_ratio=0.3, batch_size=64,
                              lr=1e-3, patience=4, min_epochs=3):
    """Stage 1: self-supervised masked-reconstruction pretraining, same data
    pipeline discipline as ssl_pretext.py::pretrain_ssl (source-boundary
    window skipping, split-then-scale)."""
    print(f"\n[SSL-Transformer] Pretraining  |  sources={list(sources)}  max_epochs={epochs}  "
          f"mask_ratio={mask_ratio}  patience={patience}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, _y_train, X_val, _y_val, scaler_X, _scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=True)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=batch_size, shuffle=False)

    device = get_device()
    model = MaskedTransformerAutoencoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    def _epoch(m, dl, optimizer):
        total = 0.0
        for (bx,) in dl:
            bx = bx.to(device)
            if optimizer is not None:
                bx = augment_batch(bx)   # train-only time-series augmentation, see dr_transformer.py
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
        model, train_dl, val_dl, opt, scheduler, _epoch, epochs, patience, min_epochs,
        "[SSL-Transformer]")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[SSL-Transformer] Pretraining done in {history['elapsed_s']:.1f}s, "
          f"restored epoch {history['best_epoch']} (val={history['best_val']:.4f}).")

    return model, scaler_X, history


def finetune_from_ssl_transformer(ssl_model: MaskedTransformerAutoencoder, sources=('imu',),
                                   epochs=30, batch_size=64, lr=1e-3, patience=5, min_epochs=3):
    """Stage 2: transfer the pretrained trunk into DeadReckoningTransformer
    and fine-tune on the supervised delta-position regression task, same
    objective/pipeline/early-stopping discipline as train.py and
    ssl_pretext.py::finetune_from_ssl, so results are directly comparable."""
    print(f"\n[SSL-Transformer] Fine-tuning for position-delta regression  |  sources={list(sources)}  "
          f"max_epochs={epochs}  patience={patience}")

    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    df = load_combined_dataset(base_dir, sources=sources)
    X_train, y_train, X_val, y_val, scaler_X, scaler_y = create_dead_reckoning_dataset(
        df, window_size=WINDOW_SIZE, normalize=True)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                           batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                         batch_size=batch_size, shuffle=False)

    device = get_device()
    model = DeadReckoningTransformer(input_size=INPUT_SIZE, d_model=D_MODEL, nhead=NHEAD,
                                      num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                                      dropout=DROPOUT, output_size=3,
                                      window_size=WINDOW_SIZE).to(device)
    transfer_encoder_weights(ssl_model, model)

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    def _epoch(m, dl, optimizer):
        total = 0.0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            if optimizer is not None:
                bx = augment_batch(bx)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = criterion(m(bx), by)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()
        return total / len(dl)

    best_state, history = _run_early_stopped(
        model, train_dl, val_dl, opt, scheduler, _epoch, epochs, patience, min_epochs,
        "[SSL-Transformer-finetune]")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[SSL-Transformer-finetune] Done in {history['elapsed_s']:.1f}s, "
          f"restored epoch {history['best_epoch']} (val={history['best_val']:.4f}).")

    return model, scaler_X, scaler_y, history


def save_ssl_transformer_artifacts(ssl_model: MaskedTransformerAutoencoder,
                                    ft_model: DeadReckoningTransformer,
                                    scaler_X, scaler_y, ssl_history: dict, ft_history: dict,
                                    sources, tag_suffix: str = '') -> str:
    """Persists both stages' checkpoints/scalers/histories to models/, same
    convention as ssl_pretext.py::save_ssl_lstm_artifacts. Extracted from
    this module's own __main__ so ablation_matrix.py can call it too -- it
    previously trained this arm via pretrain_ssl_transformer()/
    finetune_from_ssl_transformer() directly and discarded their returned
    history, so no checkpoint or history ever reached disk for this arm."""
    tag = "_".join(sources) + "_ssl_norm" + tag_suffix
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    # .cpu() before saving: see train.py::train_model's save-path comment.
    torch.save(ssl_model.cpu().state_dict(), os.path.join(out_dir, f'ssl_pretrain_transformer_{tag}.pth'))
    torch.save(ft_model.cpu().state_dict(), os.path.join(out_dir, f'dr_transformer_{tag}.pth'))
    joblib.dump(scaler_X, os.path.join(out_dir, f'scaler_X_transformer_{tag}.pkl'))
    joblib.dump(scaler_y, os.path.join(out_dir, f'scaler_y_transformer_{tag}.pkl'))
    with open(os.path.join(out_dir, f'history_ssl_pretrain_transformer_{tag}.json'), 'w') as f:
        json.dump(ssl_history, f, indent=2)
    with open(os.path.join(out_dir, f'history_transformer_{tag}.json'), 'w') as f:
        json.dump(ft_history, f, indent=2)
    print(f"Saved SSL-pretrained Transformer trunk + fine-tuned model with tag '{tag}' -> {out_dir}")
    return tag


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    ap.add_argument('--ssl-epochs', type=int, default=15,
                     help='Max SSL pretraining epochs, early stopping will likely stop sooner.')
    ap.add_argument('--finetune-epochs', type=int, default=30,
                     help='Max fine-tuning epochs, early stopping will likely stop sooner.')
    ap.add_argument('--mask-ratio', type=float, default=0.3)
    ap.add_argument('--ssl-patience', type=int, default=4)
    ap.add_argument('--finetune-patience', type=int, default=5)
    args = ap.parse_args()

    ssl_model, _, ssl_history = pretrain_ssl_transformer(
        sources=tuple(args.sources), epochs=args.ssl_epochs,
        mask_ratio=args.mask_ratio, patience=args.ssl_patience)
    ft_model, scaler_X, scaler_y, ft_history = finetune_from_ssl_transformer(
        ssl_model, sources=tuple(args.sources), epochs=args.finetune_epochs,
        patience=args.finetune_patience)

    save_ssl_transformer_artifacts(ssl_model, ft_model, scaler_X, scaler_y,
                                    ssl_history, ft_history, args.sources)
