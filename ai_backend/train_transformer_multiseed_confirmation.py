"""Multi-seed confirmation for the two single-run Transformer findings that
were explicitly flagged as inconclusive on n=1: the order-shuffle result
(9.78m vs. 12.46m) and the no-augment result (6.47m vs. 12.46m). Both were
suggestive but not distinguishable from the Transformer's own well-documented
71% seed-to-seed CV without the same multi-seed treatment every other arm in
the ablation matrix already got -- this script gives them that treatment.

Trains 5 seeds (0-4, matching cpu_5seeds exactly) for each variant,
set_global_seed(seed) called identically to ablation_matrix.py so initial
weights match the existing per-seed data already on disk. Every checkpoint
saved under a distinct tag_suffix (_ablation_seed{N}_shuffled /
_ablation_seed{N}_noaugment) -- cannot collide with cpu_5seeds' own
_ablation_seed{N} checkpoints or with dr_transformer_imu_norm.pth, verified
by inspection of train_transformer.py's tag-building logic before this was
run (the exact bug class caught and fixed in the single-run version of this
experiment).

Runs shuffle-order first (faster per seed, ~45min based on the single-run
precedent) then no-augment (slower, ~1h45min per seed based on the same),
saving results incrementally after every seed so a partial run is never lost.

Usage:
  python ai_backend/train_transformer_multiseed_confirmation.py
"""
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_matrix import set_global_seed, confidence_interval
from train_transformer import train_model, _SHUFFLE_PERM
from evaluate_transformer_shuffled import reconstruct_transformer_shuffled
from evaluate_trajectory import load_val_segment, ground_truth, euclidean_error, reconstruct_transformer

SEEDS = [0, 1, 2, 3, 4]
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs',
                         'transformer_multiseed_confirmation.json')


def load_results():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {'shuffled': {}, 'noaugment': {}}


def save_results(results):
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def summarize(per_seed_finals):
    mean, lo, hi = confidence_interval(per_seed_finals)
    return {'mean': mean, 'ci_low': lo, 'ci_high': hi,
            'per_seed_final': per_seed_finals}


def run_variant(variant: str, seg, gt_pos, results: dict):
    print(f"\n{'#' * 70}\n# Variant: {variant}\n{'#' * 70}")
    for seed in SEEDS:
        if str(seed) in results[variant]:
            print(f"[{variant}] seed {seed}: already done, skipping "
                  f"(final={results[variant][str(seed)]['final_error_m']:.2f} m)")
            continue

        print(f"\n{'=' * 70}\n[{variant}] seed {seed}\n{'=' * 70}")
        t0 = time.time()
        set_global_seed(seed)

        if variant == 'shuffled':
            model, scaler_X, scaler_y, history = train_model(
                sources=('imu',), num_epochs=30, early_stopping_patience=5,
                batch_size=64, tag_suffix=f'_ablation_seed{seed}_shuffled',
                shuffle_window_order=True)
            pos = reconstruct_transformer_shuffled(seg, model, scaler_X, scaler_y, _SHUFFLE_PERM)
        else:
            model, scaler_X, scaler_y, history = train_model(
                sources=('imu',), num_epochs=30, early_stopping_patience=5,
                batch_size=64, tag_suffix=f'_ablation_seed{seed}_noaugment',
                use_augmentation=False)
            pos = reconstruct_transformer(seg, model, scaler_X, scaler_y)

        err = euclidean_error(pos, gt_pos)
        elapsed = time.time() - t0

        results[variant][str(seed)] = {
            'best_epoch': history['best_epoch'],
            'best_val_loss': history['best_val'],
            'epochs_run': len(history['val_losses']),
            'final_error_m': float(err[-1]),
            'mean_error_m': float(err.mean()),
            'train_elapsed_s': elapsed,
        }
        save_results(results)
        print(f"[{variant}] seed {seed} done in {elapsed:.1f}s: "
              f"final={err[-1]:.2f} m, mean={err.mean():.2f} m "
              f"(best_epoch={history['best_epoch']}/{len(history['val_losses'])} run)")


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    print("Loading held-out segment...")
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    results = load_results()
    if 'shuffled' not in results:
        results['shuffled'] = {}
    if 'noaugment' not in results:
        results['noaugment'] = {}

    run_variant('shuffled', seg, gt_pos, results)
    run_variant('noaugment', seg, gt_pos, results)

    print(f"\n{'#' * 70}\n# Final summary\n{'#' * 70}")
    for variant in ['shuffled', 'noaugment']:
        finals = [results[variant][str(s)]['final_error_m'] for s in SEEDS]
        summary = summarize(finals)
        results[f'{variant}_summary'] = summary
        print(f"\n{variant}: per-seed final = {[round(f, 2) for f in finals]}")
        print(f"  mean = {summary['mean']:.2f} m, 95% CI = "
              f"[{summary['ci_low']:.2f}, {summary['ci_high']:.2f}] m")

    print("\nComparison points (single-run, from the earlier round of testing):")
    print("  Real order-preserving Transformer (cpu_5seeds, 5-seed): "
          "9.10 m mean, 95% CI [2.61, 15.60] m")
    print("  Shuffled (single run): 9.78 m final")
    print("  No-augment (single run): 6.47 m final")

    save_results(results)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == '__main__':
    main()
