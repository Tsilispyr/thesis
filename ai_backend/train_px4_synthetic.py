"""Multi-seed comparison: real-only px4 baseline vs. real px4 + physically-
motivated synthetic diversity (rotation + splice, see
data_processing/px4_synthetic_augment.py), completing the AI_RECOVERY plan's
Part B.2b -- the one item left over from the PX4 data-diversity thread after
the Mission-filtered re-download (SOURCE.md's "Batch 2") already confirmed
the earlier val loss ~1.5 result was a data-quantity/diversity problem
(fixed: 0.2768 on the larger real-only corpus).

Both arms train against the SAME fixed, real-only held-out validation rows
(the last 20% of the real px4 corpus, row-sequential, matching
create_dead_reckoning_dataset()'s own default split convention) so "val
loss" means the same thing in both arms -- augmentation only ever touches
the training pool, never validation. Multiple seeds, not one run: this
project's own established standard after two single-run "improvements"
(Transformer order-shuffle, no-augment) both turned out to be seed luck
under confirmation, not real effects -- see TRAINING_ANALYSIS.md.

Usage:
  python ai_backend/train_px4_synthetic.py --n-seeds 3
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from data_processing.dataset_parser import load_combined_dataset
from data_processing.px4_synthetic_augment import rotate_augment, splice_augment
from plot_style import (style_axes, COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE,
                         COLOR_GROUND_TRUTH, COLOR_EKF)
from train import train_model

_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Dedicated output folder, separate from ai_backend/models/ (checkpoints/
# scalers only, per that folder's existing convention) and from every other
# runs/ subfolder already in use (rl_seek_comparison.png, ablation/*) --
# this experiment's own summary/plot/logs live here so nothing it produces
# is at risk of colliding with or overwriting anything made before it.
_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'px4_synthetic_diversity')
_VAL_FRACTION = 0.2


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confidence_interval(values: list, confidence: float = 0.95):
    """Point estimate + t-distribution CI, same convention as
    ablation_matrix.py's own (duplicated here rather than imported -- a
    3-line utility, not worth coupling to that file's larger orchestrator)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * sem
    return mean, max(0.0, mean - margin), mean + margin


def _split_real(real_df: pd.DataFrame, val_fraction: float = _VAL_FRACTION):
    """Row-sequential split matching create_dead_reckoning_dataset()'s own
    default convention -- the val portion is fixed and never touched by
    augmentation, shared by every arm/seed below."""
    cut = int(len(real_df) * (1.0 - val_fraction))
    return real_df.iloc[:cut].reset_index(drop=True), real_df.iloc[cut:].reset_index(drop=True)


def run_arm(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int, tag_suffix: str,
            epochs: int, patience: int):
    set_global_seed(seed)
    _, _, _, history = train_model(
        sources=('px4',), num_epochs=epochs, normalize=True,
        early_stopping_patience=patience, override_df=train_df,
        held_out_val_df=val_df, tag_suffix=tag_suffix)
    # ai_backend/models/ keeps every checkpoint/scaler/history this project
    # has ever produced, per its established convention (RL checkpoints,
    # ablation seeds, etc.) -- untouched. A copy of just this run's history
    # also goes to _OUT_DIR so this experiment's full logs are readable in
    # one dedicated place without hunting through that shared folder.
    os.makedirs(_OUT_DIR, exist_ok=True)
    hist_name = f"history_px4_norm{tag_suffix}.json"
    shutil.copy2(os.path.join(_MODELS_DIR, hist_name), os.path.join(_OUT_DIR, hist_name))
    return history


def _plot_comparison(baseline_results, augmented_results, out_path,
                      labels=('Baseline\n(real only)', 'Augmented\n(real+synthetic)'),
                      title='PX4 synthetic diversity: val loss by seed'):
    """labels/title default to this script's own synthetic-diversity framing
    -- overridden by other callers (e.g. train_motor_informed.py) reusing
    this function for a differently-labeled two-arm comparison, so this
    stays genuinely reusable rather than hardcoded to one experiment."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    positions = [0, 1]
    data = [baseline_results, augmented_results]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                     medianprops=dict(color='black'))
    for patch, color in zip(bp['boxes'], [COLOR_ACCENT_ORANGE, COLOR_ACCENT_BLUE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, vals in enumerate(data):
        ax.scatter([positions[i]] * len(vals), vals, color='black', zorder=3, s=20)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{labels[0]}\nn={len(baseline_results)}',
                         f'{labels[1]}\nn={len(augmented_results)}'])
    style_axes(ax, title=title,
               xlabel='', ylabel='Best val loss (MSE, normalized)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_training_curves_multiseed(histories: list, title: str, out_path: str):
    """Train/val loss per epoch, overlaid across seeds -- same convention as
    visualize_ablation.py's plot_training_curve_multiseed (train solid black,
    val dotted slate gray, seed identity carried by decreasing alpha),
    duplicated here rather than imported to avoid pulling in that module's
    much larger evaluate_trajectory.py/ablation_matrix.py dependency chain
    for one small plotting function."""
    alphas = np.linspace(0.9, 0.35, len(histories)) if len(histories) > 1 else [0.9]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(histories):
        epochs = list(range(1, len(history['train_losses']) + 1))
        a = alphas[i]
        ax.plot(epochs, history['train_losses'], color=COLOR_GROUND_TRUTH, linewidth=1.5,
                linestyle='-', alpha=a, label=f'Train Loss (seed {i})')
        ax.plot(epochs, history['val_losses'], color=COLOR_EKF, linewidth=1.3,
                linestyle=':', alpha=a, label=f'Val Loss (seed {i})')
    style_axes(ax, title=title, xlabel='Epoch', ylabel='MSE Loss (normalized)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=8,
              frameon=False, ncol=min(len(histories), 3))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_val_curve_overlay(baseline_histories: list, augmented_histories: list, out_path: str,
                             labels=('Baseline (real only)', 'Augmented (real+synthetic)'),
                             title='Validation loss: baseline vs. synthetic-augmented (mean +-1 std)'):
    """Baseline vs. augmented val-loss curves on one axis (mean across
    seeds, +-1 std band) -- the direct visual answer to 'does augmented
    data change the SHAPE of convergence, not just the final number.'
    labels/title default to this script's own synthetic-diversity framing --
    overridden by other callers reusing this function for a differently-
    labeled two-arm comparison."""
    def _mean_std(histories):
        max_len = max(len(h['val_losses']) for h in histories)
        arr = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            arr[i, :len(h['val_losses'])] = h['val_losses']
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    base_mean, base_std = _mean_std(baseline_histories)
    aug_mean, aug_std = _mean_std(augmented_histories)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mean, std, color, label in [
        (base_mean, base_std, COLOR_ACCENT_ORANGE, labels[0]),
        (aug_mean, aug_std, COLOR_ACCENT_BLUE, labels[1]),
    ]:
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    style_axes(ax, title=title,
               xlabel='Epoch', ylabel='MSE Loss (normalized)')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--n-rotations', type=int, default=2,
                     help='Rotated copies of the real training rows per seed.')
    ap.add_argument('--n-splices', type=int, default=20,
                     help='Synthetic spliced flights attempted per seed.')
    ap.add_argument('--tag', type=str, default='',
                     help='Extra tag inserted into every saved checkpoint/history name '
                          '(e.g. "_dryrun") so a quick test run never collides with a '
                          'real run\'s files for the same seed.')
    args = ap.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    real_df = load_combined_dataset(base_dir, sources=('px4',))
    real_train_df, real_val_df = _split_real(real_df)
    print(f"\nReal px4: {len(real_df)} rows -> {len(real_train_df)} train-pool / "
          f"{len(real_val_df)} held-out val rows (fixed across every arm/seed below).")

    baseline_histories, augmented_histories = [], []
    for seed in range(args.n_seeds):
        print(f"\n{'='*70}\nSeed {seed} -- baseline (real px4 only)\n{'='*70}")
        baseline_histories.append(run_arm(
            real_train_df, real_val_df, seed,
            tag_suffix=f'_mission_v2_synthetic_baseline{args.tag}_seed{seed}',
            epochs=args.epochs, patience=args.patience))

        print(f"\n{'='*70}\nSeed {seed} -- augmented (real + rotation + splice)\n{'='*70}")
        rotated = rotate_augment(real_train_df, n_copies=args.n_rotations, seed=seed)
        spliced = splice_augment(real_train_df, n_splices=args.n_splices, seed=seed)
        print(f"  +{len(rotated)} rotated rows ({rotated['_src_id'].nunique()} synthetic flights), "
              f"+{len(spliced)} spliced rows ({spliced['_src_id'].nunique()} synthetic flights)")
        augmented_train_df = pd.concat([real_train_df, rotated, spliced], ignore_index=True)
        augmented_histories.append(run_arm(
            augmented_train_df, real_val_df, seed,
            tag_suffix=f'_mission_v2_synthetic_augmented{args.tag}_seed{seed}',
            epochs=args.epochs, patience=args.patience))

    baseline_results = [h['best_val'] for h in baseline_histories]
    augmented_results = [h['best_val'] for h in augmented_histories]
    base_mean, base_lo, base_hi = confidence_interval(baseline_results)
    aug_mean, aug_lo, aug_hi = confidence_interval(augmented_results)

    print(f"\n{'='*70}\nResult ({args.n_seeds} seeds, 95% CI)\n{'='*70}")
    print(f"  Baseline (real only):        val_loss = {base_mean:.4f}  "
          f"[{base_lo:.4f}, {base_hi:.4f}]  {baseline_results}")
    print(f"  Augmented (real+synthetic):  val_loss = {aug_mean:.4f}  "
          f"[{aug_lo:.4f}, {aug_hi:.4f}]  {augmented_results}")
    overlap = not (aug_hi < base_lo or base_hi < aug_lo)
    print(f"  95% CIs {'OVERLAP -- not distinguishable from noise' if overlap else 'DO NOT overlap -- a real difference'}")

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f'comparison{args.tag}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_seeds': args.n_seeds,
            'n_rotations': args.n_rotations, 'n_splices': args.n_splices,
            'epochs': args.epochs, 'patience': args.patience,
            'baseline_val_losses': baseline_results,
            'augmented_val_losses': augmented_results,
            'baseline_mean_ci': [base_mean, base_lo, base_hi],
            'augmented_mean_ci': [aug_mean, aug_lo, aug_hi],
            'cis_overlap': overlap,
        }, f, indent=2)
    print(f"Saved -> {out_path}")

    # This pipeline only ever splits real px4 rows into train/val (see
    # create_dead_reckoning_dataset()) -- there is no separate held-out
    # "test" set anywhere in this project's LSTM training, on px4 or any
    # other source. The held-out val set used here *is* that honest,
    # never-trained-on measurement (real rows only, fixed across both arms,
    # see _split_real()) -- these figures cover train and val because
    # that's the whole pipeline, not because a test-set figure was skipped.
    fig_dir = os.path.join(_OUT_DIR, 'figures')
    _plot_training_curves_multiseed(
        baseline_histories, f'PX4 baseline (real only, {args.n_seeds} seeds)',
        os.path.join(fig_dir, f'train_curves_baseline{args.tag}.png'))
    _plot_training_curves_multiseed(
        augmented_histories, f'PX4 augmented (real+synthetic, {args.n_seeds} seeds)',
        os.path.join(fig_dir, f'train_curves_augmented{args.tag}.png'))
    _plot_val_curve_overlay(baseline_histories, augmented_histories,
                             os.path.join(fig_dir, f'val_curve_overlay{args.tag}.png'))

    if args.n_seeds >= 2:
        _plot_comparison(baseline_results, augmented_results,
                          os.path.join(fig_dir, f'comparison{args.tag}.png'))
    else:
        print("(Skipping box-plot comparison -- n_seeds < 2, a single point has no meaningful spread to show.)")
