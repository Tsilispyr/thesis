"""Tests cpu_5seeds/ANALYSIS.md's own proposed diagnostic directly: does
raising the LSTM's early-stopping patience from 5 to 10 systematically close
the gap seed 4 revealed (it never triggered early stopping at patience=5,
kept improving to epoch 30, and independently had both the best windowed
loss and the best chained error of all 5 seeds) -- or was seed 4 simply a
favorable initialization that happened to keep improving?

Retrains the same 5 seeds (0-4) used in runs/ablation/cpu_5seeds, with
set_global_seed() called identically to ablation_matrix.py::train_one_seed()
so each seed's initial weights match, only early_stopping_patience differs
(10 instead of 5). Saved under a distinct _patience10 tag_suffix, not
overwriting the cpu_5seeds checkpoints.

Usage:
  python ai_backend/train_lstm_patience_experiment.py
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_matrix import set_global_seed
from train import train_model
from evaluate_trajectory import load_val_segment, ground_truth, euclidean_error, reconstruct_lstm

SEEDS = [0, 1, 2, 3, 4]
PATIENCE = 10
SOURCES = ('imu',)
EPOCHS = 30


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    results = {}
    for seed in SEEDS:
        print(f"\n{'=' * 70}\nSeed {seed}, patience={PATIENCE}\n{'=' * 70}")
        set_global_seed(seed)
        tag_suffix = f'_ablation_seed{seed}_patience10'
        model, scaler_X, scaler_y, history = train_model(
            sources=SOURCES, num_epochs=EPOCHS, early_stopping_patience=PATIENCE,
            batch_size=64, tag_suffix=tag_suffix)

        pos = reconstruct_lstm(seg, model, scaler_X, scaler_y)
        err = euclidean_error(pos, gt_pos)

        results[seed] = {
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val'],
            'early_stopped': len(history['val_losses']) < EPOCHS,
            'epochs_run': len(history['val_losses']),
            'final_error_m': float(err[-1]),
            'mean_error_m': float(err.mean()),
        }
        print(f"Seed {seed}: best_epoch={history['best_epoch']}/{len(history['val_losses'])} run, "
              f"best_val={history['best_val']:.4f}, final_error={err[-1]:.2f} m, "
              f"mean_error={err.mean():.2f} m")

    print(f"\n{'=' * 70}\nSummary: patience={PATIENCE} vs. cpu_5seeds' patience=5\n{'=' * 70}")
    print(f"{'seed':>5} {'best_epoch':>11} {'epochs_run':>11} {'best_val':>10} "
          f"{'final_m':>9} {'mean_m':>8}")
    for seed, r in results.items():
        print(f"{seed:>5} {r['best_epoch']:>11} {r['epochs_run']:>11} "
              f"{r['best_val_loss']:>10.4f} {r['final_error_m']:>9.2f} {r['mean_error_m']:>8.2f}")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                             'lstm_patience10_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
