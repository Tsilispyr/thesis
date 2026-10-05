"""Ablation-matrix figures (Track A, Phase A5) -- consumes the completed
3-seed x 6-model + 2-baseline results from runs/ablation/cpu_run1/results.json
and the per-seed checkpoints/histories under ai_backend/models/ to produce the
full figure set for the multi-seed ablation study. Fixed color palette /
line-style conventions, see plot_style.py; checkpoint loading is reused
directly from ablation_matrix.py's _try_load_cached_* helpers rather than
reimplemented here, and the mean +/- 95% CI convention is reused directly
from ablation_matrix.py's confidence_interval() (t-distribution, not z,
lower bound clipped at 0).

Every figure that can only show one trajectory per model (3D/XY/XZ, error-
over-time) uses that model's REPRESENTATIVE seed -- the one whose 'final'
chained error is the MEDIAN of its 3 seeds, not the best and not the worst
(see median_seed_index() below, and figures/ANALYSIS.md for why). Training-
curve figures instead overlay all 3 seeds via alpha, since a single-seed
curve would hide exactly the seed-to-seed spread runs/ablation/cpu_run1/
ANALYSIS.md's coefficient-of-variation finding is about.

Usage:
  python ai_backend/visualize_ablation.py --run-label cpu_run1
  python ai_backend/visualize_ablation.py --run-label cpu_5seeds

Not hardcoded to one run: --run-label picks which runs/ablation/<label>/
results.json to read, and the number of seeds is read from that file
(results.json's top-level 'seeds' field), not assumed to be 3 -- so a
5-seed (or any-N-seed) run gets the exact same figure treatment as the
original 3-seed cpu_run1 run, for direct comparison across runs.
"""
import argparse
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import ABLATION_MODEL_COLORS, COLOR_GROUND_TRUTH, COLOR_EKF, GRID_KW, style_axes
from evaluate_trajectory import (load_val_segment, reconstruct_lstm, reconstruct_transformer,
                                  reconstruct_classical, reconstruct_filter, ground_truth,
                                  euclidean_error, WINDOW_SIZE)
from ablation_matrix import (_try_load_cached_lstm, _try_load_cached_transformer,
                              _try_load_cached_classical, confidence_interval)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')

# Set by main() from --run-label, before any other function in this module
# runs -- every function below reads these as module globals rather than
# taking a run_dir parameter through the whole call chain, since this
# script is always a single, one-shot process invocation (never imports
# main() and calls it twice with different labels in the same process).
RUN_LABEL = None
RUN_DIR = None
FIG_DIR = None
RESULTS_JSON = None
SEEDS = None

STOCHASTIC_MODELS = ['LSTM', 'SSL-LSTM', 'Transformer', 'SSL-Transformer', 'RandomForest', 'GBT']

# One (display label, results-JSON key, line style) triple per series --
# solid for the six learned models (primary/predicted role), dotted for the
# two classical baselines (reference role), matching this project's
# established solid=primary/dotted=baseline convention (plot_style.py,
# memory feedback_ai_recovery_graph_style.md). Keys follow evaluate_
# trajectory.py's own snake_case convention ('random_forest', 'pure_inertial',
# etc.) so trajectory_eval_ablation.json reads like a natural extension of
# that file's schema, not a parallel one.
ALL_SERIES = [
    ('LSTM',             'lstm',            '-'),
    ('SSL-LSTM',         'ssl_lstm',        '-'),
    ('Transformer',      'transformer',     '-'),
    ('SSL-Transformer',  'ssl_transformer', '-'),
    ('RandomForest',     'random_forest',   '-'),
    ('GBT',              'gbt',             '-'),
    ('EKF',              'ekf',             ':'),
    ('Pure Inertial',    'pure_inertial',   ':'),
]
KEY_MAP = {label: key for label, key, _ in ALL_SERIES}

# Per runs/ablation/cpu_run1/ANALYSIS.md's ranking (best point-estimate
# chained error, Transformer 8.18m / GBT 18.73m / SSL-Transformer 14.72m),
# used only for the reduced "top performers" figures -- the full 8-series
# figures are always produced too, never replaced by these.
TOP_PERFORMERS = ('Transformer', 'GBT', 'SSL-Transformer', 'EKF', 'Pure Inertial')


# ---------------------------------------------------------------------------
# Representative (median) seed selection
# ---------------------------------------------------------------------------

def median_seed_index(per_seed: list) -> int:
    """Index (0/1/2) of the seed whose 'final' chained error is the MEDIAN
    of the 3 seeds -- not the best, not the worst. This is the project's
    chosen rule for picking one representative trajectory per stochastic
    model wherever a figure can only show a single path (3D/XY/XZ, error-
    over-time): showing the best-of-3 seed would silently cherry-pick a
    lucky run, exactly the "an isolated metric looking better is not
    evidence of real improvement" trap this project has flagged repeatedly
    elsewhere (ai_backend/models/ANALYSIS.md's batch-size finding, the
    EKF's "physically correct but 132m-drifting" result)."""
    finals = [r['final'] for r in per_seed]
    order = np.argsort(finals)
    median_pos = int(order[len(order) // 2])
    return median_pos


def configure(run_label: str) -> None:
    """Sets the module-level RUN_LABEL/RUN_DIR/FIG_DIR/RESULTS_JSON globals
    for the given run label. Must be called before load_ablation_results()
    (which also sets SEEDS, from the loaded file's own seed count -- so a
    5-seed run correctly overlays 5 curves/alphas, not the original 3)."""
    global RUN_LABEL, RUN_DIR, FIG_DIR, RESULTS_JSON
    RUN_LABEL = run_label
    RUN_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'ablation', run_label)
    FIG_DIR = os.path.join(RUN_DIR, 'figures')
    RESULTS_JSON = os.path.join(RUN_DIR, 'results.json')


def load_ablation_results() -> dict:
    global SEEDS
    with open(RESULTS_JSON) as f:
        full = json.load(f)
    SEEDS = tuple(range(full['seeds']))
    return full['results']


def compute_representative_seeds(results: dict) -> dict:
    rep = {}
    for name in STOCHASTIC_MODELS:
        per_seed = results[name]['per_seed']
        idx = median_seed_index(per_seed)
        rep[name] = {'seed': idx, 'final': per_seed[idx]['final'],
                      'all_finals': [r['final'] for r in per_seed]}
    return rep


# ---------------------------------------------------------------------------
# Training curves -- 3 seeds overlaid via alpha (never line style, that role
# is already taken by train-vs-val here, and by primary-vs-baseline project-
# wide -- reusing it for seed identity would collide with both meanings).
# ---------------------------------------------------------------------------

def load_history(filename: str) -> dict:
    with open(os.path.join(MODELS_DIR, filename)) as f:
        return json.load(f)


def plot_training_curve_multiseed(histories: list, title: str, out_path: str) -> None:
    """Adapts visualize_training.py's plot_training_curve to N overlaid
    seeds (N = len(histories), not assumed to be 3): same colors/linestyles
    per role (train loss solid black, val loss dotted slate gray), seed
    identity carried by decreasing alpha (seed 0 most opaque, last seed
    most transparent) instead of line style."""
    alphas = np.linspace(0.9, 0.35, len(histories)) if len(histories) > 1 else [0.9]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, history in enumerate(histories):
        epochs = list(range(1, len(history['train_losses']) + 1))
        a = alphas[i]
        ax.plot(epochs, history['train_losses'], color=COLOR_GROUND_TRUTH, linewidth=1.5,
                linestyle='-', alpha=a, label=f'Train Loss (seed {i})')
        ax.plot(epochs, history['val_losses'], color=COLOR_EKF, linewidth=1.3,
                linestyle=':', alpha=a, label=f'Val Loss (seed {i})')
    style_axes(ax, title=title, xlabel='Epoch', ylabel='MSE Loss')
    # Val loss (dotted) can sit in a flat, full-width band near the top of
    # the axes for some arms (e.g. LSTM/SSL-LSTM finetune), which collides
    # with an in-axes corner legend regardless of which corner is picked --
    # train loss (solid) instead spans top-left to bottom-right, so no
    # single in-axes corner is guaranteed clear across all 6 callers here.
    # Placed below the axes instead: always non-overlapping, still minimal/
    # frameless per the project's legend convention.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=8,
              frameon=False, ncol=3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run_training_curves() -> None:
    n = len(SEEDS)
    jobs = [
        ('train_lstm.png', f'LSTM Training (imu, {n} seeds)',
         [f'history_imu_norm_ablation_seed{s}.json' for s in SEEDS]),
        ('train_ssl_lstm_pretrain.png',
         f'SSL-LSTM Pretraining -- Masked Reconstruction ({n} seeds)',
         [f'history_ssl_pretrain_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]),
        ('train_ssl_lstm_finetune.png', f'SSL-LSTM Fine-Tuning (imu, {n} seeds)',
         [f'history_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]),
        ('train_transformer.png', f'Transformer Training (imu, {n} seeds)',
         [f'history_transformer_imu_norm_ablation_seed{s}.json' for s in SEEDS]),
        ('train_ssl_transformer_pretrain.png',
         f'SSL-Transformer Pretraining -- Masked Reconstruction ({n} seeds)',
         [f'history_ssl_pretrain_transformer_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]),
        ('train_ssl_transformer_finetune.png', f'SSL-Transformer Fine-Tuning (imu, {n} seeds)',
         [f'history_transformer_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]),
    ]
    for fname, title, hist_files in jobs:
        histories = [load_history(h) for h in hist_files]
        plot_training_curve_multiseed(histories, title, os.path.join(FIG_DIR, fname))

    print("(RandomForest and GBT are single-shot GridSearchCV fits, not "
          "epoch-trained -- no per-epoch curve exists for either, so no "
          "train_randomforest.png / train_gbt.png is produced. See "
          "figures/ANALYSIS.md for this limitation stated plainly.)")


# ---------------------------------------------------------------------------
# Windowed validation-loss comparison (comparison_val_loss.png)
# ---------------------------------------------------------------------------

def get_val_losses(model_name: str) -> list:
    """3 per-seed WINDOWED validation losses for model_name's final/fine-
    tuned arm -- never the SSL pretrain-stage loss (different target/
    objective, masked reconstruction vs delta-position regression, not
    comparable to the other five bars)."""
    if model_name == 'LSTM':
        files = [f'history_imu_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'best_val'
    elif model_name == 'SSL-LSTM':
        files = [f'history_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'best_val'
    elif model_name == 'Transformer':
        files = [f'history_transformer_imu_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'best_val'
    elif model_name == 'SSL-Transformer':
        files = [f'history_transformer_imu_ssl_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'best_val'
    elif model_name == 'RandomForest':
        files = [f'history_rf_imu_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'val_mse'
    elif model_name == 'GBT':
        files = [f'history_gbt_imu_norm_ablation_seed{s}.json' for s in SEEDS]
        key = 'val_mse'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return [load_history(f)[key] for f in files]


def plot_val_loss_comparison(out_path: str) -> None:
    """Same bar style / color mapping / CI-error-bar style as ablation_
    matrix.py's own ablation_matrix.png (which shows the CHAINED metric) --
    deliberately so the two are visually comparable side by side. This one
    shows the WINDOWED metric instead, to check whether the two metrics
    agree on ranking (they do not, for the Transformer arms -- see
    figures/ANALYSIS.md)."""
    names = STOCHASTIC_MODELS
    means, lo_err, hi_err = [], [], []
    for name in names:
        vals = get_val_losses(name)
        m, lo, hi = confidence_interval(vals)
        means.append(m)
        lo_err.append(m - lo)
        hi_err.append(hi - m)

    colors = [ABLATION_MODEL_COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, means, color=colors, edgecolor='white',
           yerr=[lo_err, hi_err], capsize=4, ecolor='#333333')
    ax.set_ylabel('Windowed validation MSE loss (normalized)')
    ax.set_title(f'Ablation Matrix: windowed validation loss by model '
                  f'({len(SEEDS)}-seed 95% CI)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', **GRID_KW)
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Trajectory reconstruction on the shared held-out segment
# ---------------------------------------------------------------------------

def reconstruct_all_models(rep_seeds: dict) -> dict:
    print(f"Loading val segment (3000 samples, held-out tail of imu_data.csv, "
          f"same segment ablation_matrix.py used)...")
    seg = load_val_segment(BASE_DIR, n_samples=3000)
    gt = ground_truth(seg)
    t = np.cumsum(seg['dt'].values[WINDOW_SIZE:])

    results = {'t': t.tolist(), 'ground_truth': gt.tolist()}

    sequence_loaders = {
        'LSTM':            ('imu_norm_ablation_seed{seed}',
                             _try_load_cached_lstm, reconstruct_lstm),
        'SSL-LSTM':        ('imu_ssl_norm_ablation_seed{seed}',
                             _try_load_cached_lstm, reconstruct_lstm),
        'Transformer':     ('imu_norm_ablation_seed{seed}',
                             _try_load_cached_transformer, reconstruct_transformer),
        'SSL-Transformer': ('imu_ssl_norm_ablation_seed{seed}',
                             _try_load_cached_transformer, reconstruct_transformer),
    }
    for name, (tag_fmt, loader, reconstructor) in sequence_loaders.items():
        seed = rep_seeds[name]['seed']
        tag = tag_fmt.format(seed=seed)
        loaded = loader(MODELS_DIR, tag)
        if loaded is None:
            raise RuntimeError(f"No cached checkpoint found for {name} tag '{tag}'")
        model, sx, sy = loaded
        pos = reconstructor(seg, model, sx, sy)
        key = KEY_MAP[name]
        results[key] = pos.tolist()
        results[f'error_{key}'] = euclidean_error(pos, gt).tolist()
        print(f"{name:<16} (seed {seed}, representative/median): "
              f"final error = {results[f'error_{key}'][-1]:.2f} m")

    for name, prefix in [('RandomForest', 'rf'), ('GBT', 'gbt')]:
        seed = rep_seeds[name]['seed']
        tag = f'imu_norm_ablation_seed{seed}'
        loaded = _try_load_cached_classical(MODELS_DIR, prefix, tag)
        if loaded is None:
            raise RuntimeError(f"No cached checkpoint found for {name} tag '{tag}'")
        model, sx, sy = loaded
        pos = reconstruct_classical(seg, model, sx, sy)
        key = KEY_MAP[name]
        results[key] = pos.tolist()
        results[f'error_{key}'] = euclidean_error(pos, gt).tolist()
        print(f"{name:<16} (seed {seed}, representative/median): "
              f"final error = {results[f'error_{key}'][-1]:.2f} m")

    print("Reconstructing EKF (deterministic baseline)...")
    ekf_pos = reconstruct_filter(seg, use_baro=True)
    results['ekf'] = ekf_pos.tolist()
    results['error_ekf'] = euclidean_error(ekf_pos, gt).tolist()
    print(f"{'EKF':<16} (deterministic): final error = {results['error_ekf'][-1]:.2f} m")

    print("Reconstructing Pure Inertial (deterministic baseline)...")
    pi_pos = reconstruct_filter(seg, use_baro=False)
    results['pure_inertial'] = pi_pos.tolist()
    results['error_pure_inertial'] = euclidean_error(pi_pos, gt).tolist()
    print(f"{'Pure Inertial':<16} (deterministic): "
          f"final error = {results['error_pure_inertial'][-1]:.2f} m")

    return results


# ---------------------------------------------------------------------------
# Trajectory figures -- adapts visualize_trajectory.py's plot_3d_trajectory /
# plot_projection / plot_error_over_time from 3-4 series to a parameterized
# list of (label, key, linestyle) series, 8 in the full versions.
# ---------------------------------------------------------------------------

def plot_3d_trajectory_multi(results: dict, series: list, title: str, out_path: str,
                              legend_fontsize: int = 8) -> None:
    gt = np.array(results['ground_truth'])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=COLOR_GROUND_TRUTH, linewidth=1.5,
            linestyle='-', label='Ground Truth (Executed)')
    for label, key, ls in series:
        pos = np.array(results[key])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=ABLATION_MODEL_COLORS[label],
                linewidth=1.2, linestyle=ls, label=label)
    ax.scatter([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], color=COLOR_GROUND_TRUTH, s=40, marker='o')
    ax.zaxis.labelpad = 10
    style_axes(ax, title=title, xlabel='Position X [m]', ylabel='Position Y [m]',
               zlabel='Altitude Z [m]')
    ax.legend(loc='upper left', fontsize=legend_fontsize, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_projection_multi(results: dict, series: list, axis_pair: tuple, labels: tuple,
                           title: str, out_path: str, legend_fontsize: int = 9) -> None:
    gt = np.array(results['ground_truth'])
    i, j = axis_pair
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt[:, i], gt[:, j], color=COLOR_GROUND_TRUTH, linewidth=1.5, label='Ground Truth')
    for label, key, ls in series:
        pos = np.array(results[key])
        ax.plot(pos[:, i], pos[:, j], color=ABLATION_MODEL_COLORS[label],
                linewidth=1.2, linestyle=ls, label=label)
    ax.scatter([gt[0, i]], [gt[0, j]], color=COLOR_GROUND_TRUTH, s=50, zorder=5)
    ax.set_aspect('equal', adjustable='datalim')
    style_axes(ax, title=title, xlabel=labels[0], ylabel=labels[1])
    ax.legend(loc='upper left', fontsize=legend_fontsize, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_error_over_time_multi(results: dict, series: list, out_path: str) -> None:
    t = np.array(results['t'])
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, key, ls in series:
        err = results[f'error_{key}']
        ax.plot(t, err, color=ABLATION_MODEL_COLORS[label], linewidth=1.3,
                linestyle=ls, label=label)
    style_axes(ax, title='Position Error Growth over Time -- All 8 Methods',
               xlabel='Time [s]', ylabel='Euclidean Error [m]')
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run_trajectory_figures(rep_seeds: dict) -> dict:
    results = reconstruct_all_models(rep_seeds)

    json_out = os.path.join(FIG_DIR, 'trajectory_eval_ablation.json')
    with open(json_out, 'w') as f:
        json.dump(results, f)
    print(f"Saved {json_out}")

    full_series = list(ALL_SERIES)
    top_series = [s for s in ALL_SERIES if s[0] in TOP_PERFORMERS]

    # Full 8-series versions -- always produced, never replaced by the
    # reduced versions below. Legibility risk (8 overlapping paths) is
    # highest for 3D/XY, so those get a smaller legend font (7pt) than the
    # established 8-9pt convention; XZ's altitude band stays visually
    # narrow across all methods (per figures_single_drone/ANALYSIS.md) so
    # keeps the standard size.
    plot_3d_trajectory_multi(results, full_series,
                              '3D Trajectory Reconstruction -- All 8 Methods',
                              os.path.join(FIG_DIR, 'trajectory_3d.png'), legend_fontsize=7)
    plot_projection_multi(results, full_series, (0, 1),
                           ('Position X [m]', 'Position Y [m]'),
                           'Top-Down XY Path View -- All 8 Methods',
                           os.path.join(FIG_DIR, 'trajectory_xy.png'), legend_fontsize=7)
    plot_projection_multi(results, full_series, (0, 2),
                           ('Position X [m]', 'Altitude Z [m]'),
                           'Elevation XZ Path View -- All 8 Methods',
                           os.path.join(FIG_DIR, 'trajectory_xz.png'), legend_fontsize=8)
    plot_error_over_time_multi(results, full_series, os.path.join(FIG_DIR, 'error_over_time.png'))

    # Reduced "top performers" versions (Transformer, GBT, SSL-Transformer,
    # both baselines, ground truth) -- for the two plots where 8 overlapping
    # paths risk real visual clutter (a spiral/looping real flight path).
    plot_3d_trajectory_multi(results, top_series,
                              '3D Trajectory Reconstruction -- Top Performers',
                              os.path.join(FIG_DIR, 'trajectory_3d_top_performers.png'),
                              legend_fontsize=9)
    plot_projection_multi(results, top_series, (0, 1),
                           ('Position X [m]', 'Position Y [m]'),
                           'Top-Down XY Path View -- Top Performers',
                           os.path.join(FIG_DIR, 'trajectory_xy_top_performers.png'),
                           legend_fontsize=9)

    return results


# ---------------------------------------------------------------------------

def main(run_label: str) -> None:
    configure(run_label)
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_ablation_results()
    rep_seeds = compute_representative_seeds(results)

    print(f"Run label: {run_label}  ({len(SEEDS)} seeds)")
    print("Representative (median) seed per stochastic model:")
    for name in STOCHASTIC_MODELS:
        info = rep_seeds[name]
        print(f"  {name:<16} seed {info['seed']}  final error = {info['final']:.2f} m  "
              f"(all {len(SEEDS)} seeds: {[f'{v:.2f}' for v in info['all_finals']]})")

    print("\n=== Training curves ===")
    run_training_curves()

    print("\n=== Windowed validation-loss comparison ===")
    plot_val_loss_comparison(os.path.join(FIG_DIR, 'comparison_val_loss.png'))

    print("\n=== Trajectory reconstruction + comparison figures ===")
    run_trajectory_figures(rep_seeds)

    print(f"\nAll ablation figures written to {FIG_DIR}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-label', type=str, default='cpu_run1',
                     help='Which runs/ablation/<label>/ to build figures for. The number of '
                          'seeds is read from that run\'s own results.json, not assumed.')
    args = ap.parse_args()
    main(args.run_label)
