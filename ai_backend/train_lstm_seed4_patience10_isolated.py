"""Isolation check for train_lstm_patience_experiment.py's seed 4 result:
that run trained seeds 0-4 sequentially in one Python process, and seed 4
stopped at epoch 3 (val 0.7200) -- a sharp departure from the original
cpu_5seeds seed 4, which never triggered early stopping and improved to
epoch 30 (val 0.6846). Since set_global_seed(4) is called immediately
before this seed's training either way, a real behavioral difference here
would mean CPU/PyTorch execution isn't as deterministic as set_global_seed()
implies once 4 prior training runs have already executed in the same
process (thread-pool/allocator state, not the RNG seed itself). Runs ONLY
seed 4, patience=10, in a fresh process, nothing else executed first.

Usage:
  python ai_backend/train_lstm_seed4_patience10_isolated.py
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_matrix import set_global_seed
from train import train_model
from evaluate_trajectory import load_val_segment, ground_truth, euclidean_error, reconstruct_lstm

SEED = 4
PATIENCE = 10


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    set_global_seed(SEED)
    tag_suffix = f'_ablation_seed{SEED}_patience10_isolated'
    model, scaler_X, scaler_y, history = train_model(
        sources=('imu',), num_epochs=30, early_stopping_patience=PATIENCE,
        batch_size=64, tag_suffix=tag_suffix)

    pos = reconstruct_lstm(seg, model, scaler_X, scaler_y)
    err = euclidean_error(pos, gt_pos)

    result = {
        'seed': SEED, 'patience': PATIENCE, 'isolated_process': True,
        'best_epoch': history['best_epoch'],
        'best_val_loss': history['best_val'],
        'epochs_run': len(history['val_losses']),
        'final_error_m': float(err[-1]),
        'mean_error_m': float(err.mean()),
    }
    print(f"\nSeed 4 (isolated process), patience=10: best_epoch={history['best_epoch']}/"
          f"{len(history['val_losses'])} run, best_val={history['best_val']:.4f}, "
          f"final={err[-1]:.2f} m, mean={err.mean():.2f} m")
    print("Compare: original cpu_5seeds seed 4 (isolated process, patience=5): "
          "best_epoch=30/30, best_val=0.6846, final=24.78 m")
    print("Compare: this experiment's seed 4 (sequential same-process, patience=10): "
          "best_epoch=3/13, best_val=0.7200, final=27.04 m")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                             'lstm_seed4_patience10_isolated_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
