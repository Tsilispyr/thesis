"""Formal multi-seed ablation matrix (Track A, Phase A5 of the coursework-
grounded extension roadmap): trains every stochastic model arm from
Phases A2-A4 across multiple random seeds, evaluates each on the same
real, held-out chained-trajectory segment `evaluate_trajectory.py` already
uses for every model in this project, and reports mean +/- confidence
interval per model rather than a single, possibly-lucky run.

Pattern ported: the run_experiment -> results dict -> comparison table ->
plot -> results.json pattern is confirmed twice independently in the DL
coursework (D:\\ΠΜΣ\\Υπολογιστική όραση\\Εργασία\\project-2\\ex3_cifar.py,
ex1_sbd.py) and is already the house style inside this repo's own
ml_pipeline.py (STEP 6-12). The point-estimate + CI table pattern is
ported from D:\\ΠΜΣ\\Στατιστική\\final στατιστική\\...\\stat-fin.R's
`or_table <- cbind(OR=ors, ci)`.

Deterministic models (EKF, pure inertial) get a single point value; their
CI columns read "n/a (deterministic)" rather than a fabricated interval.
Stochastic models (LSTM, SSL-LSTM, Transformer, SSL-Transformer,
RandomForest, GBT) each retrain from scratch per seed with the global
random seed set beforehand, then get a t-distribution confidence interval
(not z=1.96, correct at small n, e.g. n=3 -> t~4.30, n=5 -> t~2.78).

Given the batch-size finding documented in models/ANALYSIS.md's
"Infrastructure note" (batch=256/GPU trains faster but measurably WORSE
chained-trajectory accuracy for both LSTM and Transformer), every model
here trains at the validated batch_size=64 default, deliberately not the
faster-but-worse GPU-friendly configuration.

Usage:
  python ai_backend/ablation_matrix.py --seeds 3 --sources imu
  python ai_backend/ablation_matrix.py --quick   # 2 seeds, few epochs, fast sanity check
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import get_device
from plot_style import ABLATION_MODEL_COLORS, GRID_KW
from evaluate_trajectory import (load_val_segment, reconstruct_lstm, reconstruct_transformer,
                                  reconstruct_classical, reconstruct_filter, ground_truth,
                                  euclidean_error, WINDOW_SIZE)
from data_processing.dataset_parser import INPUT_SIZE
from models.dead_reckoning_model import DeadReckoningLSTM
from models.dr_transformer import DeadReckoningTransformer

STOCHASTIC_MODELS = ['LSTM', 'SSL-LSTM', 'Transformer', 'SSL-Transformer', 'RandomForest', 'GBT']
DETERMINISTIC_MODELS = ['EKF', 'Pure Inertial']

BATCH_SIZE = 64   # deliberately the validated config, see module docstring


class _Tee:
    """Duplicates writes to multiple streams. Used below to mirror stdout
    and stderr into a log file under runs/ablation/ so the full multi-hour,
    every-epoch, every-seed console record is durable in the repo rather
    than living only in whatever ephemeral process/terminal launched this
    script. Every training function this module calls runs in-process (no
    subprocess), so redirecting this process's own stdout/stderr captures
    everything printed by every model's training loop without touching any
    of those scripts' own print() calls."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confidence_interval(values: list, confidence: float = 0.95):
    """Point estimate + t-distribution CI (correct at small n, unlike a
    fixed z=1.96). n=1 has no meaningful interval, returns (mean, mean, mean).
    Lower bound clipped at 0: this measures a Euclidean distance error, which
    cannot be negative, so a t-interval that dips below 0 (a real risk at
    very small n, e.g. n=2) is clipped rather than reported as a physically
    impossible negative error."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * sem
    return mean, max(0.0, mean - margin), mean + margin


def _try_load_cached_lstm(models_dir: str, tag: str):
    """Returns (model, scaler_X, scaler_y) if a complete checkpoint for this
    tag already exists on disk, else None. Lets a resumed run reuse a seed's
    work from an earlier interrupted run instead of re-paying hours of
    training -- every stochastic model now saves a complete per-seed
    checkpoint (the persistence fix), so this is safe: a seed only shows up
    as cached once every one of its files landed on disk. Shared by both
    the from-scratch LSTM and the SSL-LSTM arms -- the fine-tuned model is
    the same DeadReckoningLSTM class either way, only the tag differs."""
    pth = os.path.join(models_dir, f'dr_lstm_{tag}.pth')
    sx_p = os.path.join(models_dir, f'scaler_X_{tag}.pkl')
    sy_p = os.path.join(models_dir, f'scaler_y_{tag}.pkl')
    if not (os.path.exists(pth) and os.path.exists(sx_p) and os.path.exists(sy_p)):
        return None
    model = DeadReckoningLSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=3)
    model.load_state_dict(torch.load(pth, map_location='cpu', weights_only=False))
    model.eval()
    return model, joblib.load(sx_p), joblib.load(sy_p)


def _try_load_cached_transformer(models_dir: str, tag: str):
    """Same as _try_load_cached_lstm, for the Transformer/SSL-Transformer arms."""
    pth = os.path.join(models_dir, f'dr_transformer_{tag}.pth')
    sx_p = os.path.join(models_dir, f'scaler_X_transformer_{tag}.pkl')
    sy_p = os.path.join(models_dir, f'scaler_y_transformer_{tag}.pkl')
    if not (os.path.exists(pth) and os.path.exists(sx_p) and os.path.exists(sy_p)):
        return None
    model = DeadReckoningTransformer(input_size=INPUT_SIZE, window_size=WINDOW_SIZE)
    model.load_state_dict(torch.load(pth, map_location='cpu', weights_only=False))
    model.eval()
    return model, joblib.load(sx_p), joblib.load(sy_p)


def _try_load_cached_classical(models_dir: str, prefix: str, tag: str):
    """Same idea for RandomForest ('rf') / GBT ('gbt'), plain joblib objects,
    no torch state_dict involved."""
    model_p = os.path.join(models_dir, f'{prefix}_{tag}.pkl')
    sx_p = os.path.join(models_dir, f'scaler_X_{prefix}_{tag}.pkl')
    sy_p = os.path.join(models_dir, f'scaler_y_{prefix}_{tag}.pkl')
    if not (os.path.exists(model_p) and os.path.exists(sx_p) and os.path.exists(sy_p)):
        return None
    return joblib.load(model_p), joblib.load(sx_p), joblib.load(sy_p)


def train_one_seed(model_name: str, seed: int, sources: tuple, epochs: int,
                    grid_search_once: dict, n_jobs: int = -1) -> dict:
    """Trains model_name for one seed, returns its chained-trajectory error
    dict (final/mean/max) on the shared held-out segment. grid_search_once
    is a mutable cache dict so RandomForest/GBT hyperparameters are tuned
    once (seed 0) and reused with only random_state varying afterward,
    standard ablation practice: tune once, measure seed variance separately,
    rather than re-running GridSearchCV per seed for no methodological
    benefit at large extra cost.

    n_jobs is passed straight through to RandomForest/GBT (see
    classical_baselines.py); defaults to -1 (all cores) but is a real
    parameter so two seeds can each be trained in a separate concurrent
    process without both claiming every core -- see the module's
    --n-jobs/--seed-list CLI flags."""
    set_global_seed(seed)
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    seg = load_val_segment(base_dir, n_samples=3000)
    gt = ground_truth(seg)
    tag_suffix = f'_ablation_seed{seed}'

    if model_name == 'LSTM':
        tag = "_".join(sources) + "_norm" + tag_suffix
        cached = _try_load_cached_lstm(models_dir, tag)
        if cached is not None:
            print(f"LSTM seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
        else:
            from train import train_model
            model, sx, sy, _ = train_model(sources=sources, num_epochs=epochs,
                                            early_stopping_patience=5, batch_size=BATCH_SIZE,
                                            tag_suffix=tag_suffix)
        pos = reconstruct_lstm(seg, model, sx, sy)

    elif model_name == 'SSL-LSTM':
        tag = "_".join(sources) + "_ssl_norm" + tag_suffix
        cached = _try_load_cached_lstm(models_dir, tag)
        if cached is not None:
            print(f"SSL-LSTM seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
        else:
            from models.ssl_pretext import pretrain_ssl, finetune_from_ssl, save_ssl_lstm_artifacts
            ssl_model, _, ssl_history = pretrain_ssl(sources=sources, epochs=epochs, patience=4,
                                                       batch_size=BATCH_SIZE)
            model, sx, sy, ft_history = finetune_from_ssl(ssl_model, sources=sources, epochs=epochs,
                                                            patience=5, batch_size=BATCH_SIZE)
            save_ssl_lstm_artifacts(ssl_model, model, sx, sy, ssl_history, ft_history,
                                     sources, tag_suffix)
        pos = reconstruct_lstm(seg, model, sx, sy)

    elif model_name == 'Transformer':
        tag = "_".join(sources) + "_norm" + tag_suffix
        cached = _try_load_cached_transformer(models_dir, tag)
        if cached is not None:
            print(f"Transformer seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
        else:
            from train_transformer import train_model
            model, sx, sy, _ = train_model(sources=sources, num_epochs=epochs,
                                            early_stopping_patience=5, batch_size=BATCH_SIZE,
                                            tag_suffix=tag_suffix)
        pos = reconstruct_transformer(seg, model, sx, sy)

    elif model_name == 'SSL-Transformer':
        tag = "_".join(sources) + "_ssl_norm" + tag_suffix
        cached = _try_load_cached_transformer(models_dir, tag)
        if cached is not None:
            print(f"SSL-Transformer seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
        else:
            from models.ssl_pretext_transformer import (pretrain_ssl_transformer,
                                                          finetune_from_ssl_transformer,
                                                          save_ssl_transformer_artifacts)
            ssl_model, _, ssl_history = pretrain_ssl_transformer(sources=sources, epochs=epochs,
                                                                   patience=4, batch_size=BATCH_SIZE)
            model, sx, sy, ft_history = finetune_from_ssl_transformer(
                ssl_model, sources=sources, epochs=epochs, patience=5, batch_size=BATCH_SIZE)
            save_ssl_transformer_artifacts(ssl_model, model, sx, sy, ssl_history, ft_history,
                                            sources, tag_suffix)
        pos = reconstruct_transformer(seg, model, sx, sy)

    elif model_name == 'RandomForest':
        tag = "_".join(sources) + "_norm" + tag_suffix
        cached = _try_load_cached_classical(models_dir, 'rf', tag)
        if cached is not None:
            print(f"RandomForest seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
            if 'rf_params' not in grid_search_once:
                hist_p = os.path.join(models_dir, f'history_rf_{tag}.json')
                if os.path.exists(hist_p):
                    with open(hist_p) as f:
                        grid_search_once['rf_params'] = json.load(f)['best_params']
        else:
            from models.classical_baselines import train_random_forest, save_random_forest_artifacts
            if 'rf_params' not in grid_search_once:
                # First seed only: the grid_search=True call already fits and
                # returns a fully-trained model on the tuned params, no need
                # to train a second time for this seed.
                model, sx, sy, info = train_random_forest(sources=sources, grid_search=True,
                                                            random_state=seed, n_jobs=n_jobs)
                grid_search_once['rf_params'] = info['best_params']
            else:
                model, sx, sy, info = train_random_forest(
                    sources=sources, grid_search=False, random_state=seed, n_jobs=n_jobs,
                    **grid_search_once['rf_params'])
            save_random_forest_artifacts(model, sx, sy, info, sources, tag_suffix)
        pos = reconstruct_classical(seg, model, sx, sy)

    elif model_name == 'GBT':
        tag = "_".join(sources) + "_norm" + tag_suffix
        cached = _try_load_cached_classical(models_dir, 'gbt', tag)
        if cached is not None:
            print(f"GBT seed {seed}: reusing cached checkpoint (tag '{tag}')")
            model, sx, sy = cached
            if 'gbt_params' not in grid_search_once:
                hist_p = os.path.join(models_dir, f'history_gbt_{tag}.json')
                if os.path.exists(hist_p):
                    with open(hist_p) as f:
                        grid_search_once['gbt_params'] = json.load(f)['best_params']
        else:
            from models.classical_baselines import train_gbt, save_gbt_artifacts
            if 'gbt_params' not in grid_search_once:
                model, sx, sy, info = train_gbt(sources=sources, grid_search=True,
                                                 random_state=seed, n_jobs=n_jobs)
                grid_search_once['gbt_params'] = info['best_params']
            else:
                model, sx, sy, info = train_gbt(
                    sources=sources, grid_search=False, random_state=seed, n_jobs=n_jobs,
                    **grid_search_once['gbt_params'])
            save_gbt_artifacts(model, sx, sy, info, sources, tag_suffix)
        pos = reconstruct_classical(seg, model, sx, sy)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    err = euclidean_error(pos, gt)
    return {'final': float(err[-1]), 'mean': float(err.mean()), 'max': float(err.max())}


def _run_readme(run_label: str, seeds: int, epochs: int, sources: tuple, quick: bool,
                 models: list, results: dict) -> str:
    """Factual run metadata only (config, timestamp, final table) -- not
    interpretation. Written automatically so every run-labeled folder is
    self-describing even before anyone writes the deeper ANALYSIS.md by
    hand (same split this project already uses elsewhere: history_*.json
    is facts, ANALYSIS.md is interpretation). Lets multiple ablation runs
    sit side by side under runs/ablation/ and stay individually readable
    and comparable, rather than each new run overwriting the last one."""
    lines = [
        f"# Ablation Matrix Run: {run_label}",
        "",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Config: seeds={seeds} epochs={epochs} sources={list(sources)} quick={quick}",
        f"Models: {models or STOCHASTIC_MODELS}",
        "",
        "Full per-epoch console output: `full_console_log.txt`",
        "Raw per-seed numbers: `results.json`",
        "Chart: `ablation_matrix.png`",
        "",
        "## Final chained-trajectory error",
        "",
        "| Model | Final err [m] | 95% CI | Mean err [m] | 95% CI |",
        "|---|---|---|---|---|",
    ]
    order = DETERMINISTIC_MODELS + [m for m in (models or STOCHASTIC_MODELS)]
    for name in order:
        r = results.get(name)
        if r is None:
            continue
        if r['deterministic']:
            ci_f = ci_m = "n/a (deterministic)"
        else:
            ci_f = f"[{r['final_ci_low']:.2f}, {r['final_ci_high']:.2f}]"
            ci_m = f"[{r['mean_ci_low']:.2f}, {r['mean_ci_high']:.2f}]"
        lines.append(f"| {name} | {r['final_mean']:.2f} | {ci_f} | {r['mean_mean']:.2f} | {ci_m} |")
    lines.append("")
    lines.append("See `ANALYSIS.md` in this folder for interpretation, if present.")
    return "\n".join(lines) + "\n"


def run(seeds: int = 3, sources: tuple = ('imu',), epochs: int = 30, quick: bool = False,
        models: list = None, out_dir: str = None, run_label: str = 'default',
        seed_list: list = None, n_jobs: int = -1) -> dict:
    """seed_list, if given, overrides seeds/range(seeds) with an explicit list
    of seed indices to train (e.g. [3] or [4]) -- for running two specific
    seeds in two separate concurrent terminal processes without both
    claiming seed 0 (the default range(seeds) always starts at 0, which
    would make two concurrent full runs collide on the same checkpoint
    files). When seed_list is given, this is understood to be a partial,
    cache-populating invocation: it trains and persists just those seeds'
    checkpoints and prints their per-seed results, but skips the final
    table/plot/results.json/README -- those need every seed present to be
    meaningful, and are produced by a later, ordinary (no --seed-list)
    invocation once all seeds are cached (near-instant, since it just
    reuses every cached checkpoint)."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', 'ablation', run_label)
    if quick:
        epochs = min(epochs, 3)
        seeds = min(seeds, 2)
    models = models or STOCHASTIC_MODELS
    seeds_to_run = seed_list if seed_list is not None else list(range(seeds))

    print(f"\n=== Ablation Matrix: seeds={seeds_to_run} epochs={epochs} sources={sources} "
          f"models={models} quick={quick} partial={seed_list is not None} ===\n")

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    seg = load_val_segment(base_dir, n_samples=3000)
    gt = ground_truth(seg)

    results = {}

    # Deterministic models: single point value, no seed loop. Skipped in
    # partial (--seed-list) mode -- cheap either way, but pointless clutter
    # when this invocation's only job is training one specific seed.
    if seed_list is None:
        for name, use_baro in [('EKF', True), ('Pure Inertial', False)]:
            pos = reconstruct_filter(seg, use_baro=use_baro)
            err = euclidean_error(pos, gt)
            results[name] = {
                'seeds': None,
                'per_seed': None,
                'final_mean': float(err[-1]), 'final_ci_low': None, 'final_ci_high': None,
                'mean_mean': float(err.mean()), 'mean_ci_low': None, 'mean_ci_high': None,
                'deterministic': True,
            }
            print(f"{name:<16} (deterministic)  final={err[-1]:.2f}m  mean={err.mean():.2f}m")

    # Stochastic models: seed loop.
    grid_search_once = {}
    for model_name in models:
        per_seed = []
        t0 = time.time()
        for seed in seeds_to_run:
            print(f"\n--- {model_name}, seed {seed} ---")
            r = train_one_seed(model_name, seed, sources, epochs, grid_search_once, n_jobs=n_jobs)
            per_seed.append(r)
            print(f"{model_name} seed {seed}: final={r['final']:.2f}m mean={r['mean']:.2f}m")

        finals = [r['final'] for r in per_seed]
        means = [r['mean'] for r in per_seed]
        f_mean, f_lo, f_hi = confidence_interval(finals)
        m_mean, m_lo, m_hi = confidence_interval(means)

        results[model_name] = {
            'seeds': len(seeds_to_run),
            'per_seed': per_seed,
            'final_mean': f_mean, 'final_ci_low': f_lo, 'final_ci_high': f_hi,
            'mean_mean': m_mean, 'mean_ci_low': m_lo, 'mean_ci_high': m_hi,
            'deterministic': False,
        }
        elapsed = time.time() - t0
        print(f"\n{model_name}: final={f_mean:.2f}m [{f_lo:.2f}, {f_hi:.2f}]  "
              f"mean={m_mean:.2f}m [{m_lo:.2f}, {m_hi:.2f}]  "
              f"({elapsed:.1f}s for {len(seeds_to_run)} seed(s))")

    if seed_list is not None:
        next_total = max(seed_list) + 1
        print(f"\nPartial run complete for seeds {seed_list}. Checkpoints are cached under "
              f"ai_backend/models/. No table/plot/results.json/README written -- those need "
              f"every seed present. Once every seed 0..{next_total - 1} has been trained "
              f"(across however many partial runs), aggregate them with:\n"
              f"  python ai_backend/ablation_matrix.py --seeds {next_total} --run-label <label>\n"
              f"which will find every seed already cached and complete almost instantly.")
        return results

    # --- Table ---
    print(f"\n{'Model':<18}{'Final err [m]':>18}{'95% CI':>22}{'Mean err [m]':>16}{'95% CI':>22}")
    order = DETERMINISTIC_MODELS + [m for m in models]
    for name in order:
        r = results[name]
        if r['deterministic']:
            ci_f = "n/a (deterministic)"
            ci_m = "n/a (deterministic)"
        else:
            ci_f = f"[{r['final_ci_low']:.2f}, {r['final_ci_high']:.2f}]"
            ci_m = f"[{r['mean_ci_low']:.2f}, {r['mean_ci_high']:.2f}]"
        print(f"{name:<18}{r['final_mean']:>18.2f}{ci_f:>22}{r['mean_mean']:>16.2f}{ci_m:>22}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = order
    vals = [results[n]['final_mean'] for n in names]
    errs = [
        [results[n]['final_mean'] - (results[n]['final_ci_low'] or results[n]['final_mean'])
         for n in names],
        [(results[n]['final_ci_high'] or results[n]['final_mean']) - results[n]['final_mean']
         for n in names],
    ]
    colors = [ABLATION_MODEL_COLORS.get(n, '#888888') for n in names]
    bars = ax.bar(names, vals, color=colors, edgecolor='white',
                   yerr=errs, capsize=4, ecolor='#333333')
    ax.set_ylabel('Chained-trajectory final error (m)')
    ax.set_title(f'Ablation Matrix: final chained-trajectory error by model '
                 f'({seeds}-seed 95% CI where stochastic)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', **GRID_KW)
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'ablation_matrix.png')
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {fig_path}")

    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump({'seeds': seeds, 'epochs': epochs, 'sources': list(sources),
                    'quick': quick, 'run_label': run_label, 'results': results}, f, indent=2)
    print(f"Saved results -> {json_path}")

    readme_path = os.path.join(out_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(_run_readme(run_label, seeds, epochs, sources, quick, models, results))
    print(f"Saved README -> {readme_path}")

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3,
                     help='Number of random seeds per stochastic model (default 3; the '
                          'roadmap suggested 3-5, 3 chosen given real per-model training '
                          'time observed this session, see models/ANALYSIS.md).')
    ap.add_argument('--sources', nargs='+', default=['imu'], choices=['nav', 'imu', 'px4'])
    ap.add_argument('--epochs', type=int, default=30,
                     help='Max epochs per training run (early stopping usually stops sooner).')
    ap.add_argument('--quick', action='store_true',
                     help='Fast sanity-check mode: 2 seeds, <=3 epochs, to verify the '
                          'full pipeline runs end to end before committing to a long run.')
    ap.add_argument('--models', nargs='+', default=None, choices=STOCHASTIC_MODELS,
                     help='Subset of stochastic models to run (default: all six).')
    ap.add_argument('--run-label', type=str, default=None,
                     help='Name for this run\'s output folder under runs/ablation/ (e.g. '
                          '"cpu_run1"), so multiple runs stay side by side and comparable '
                          'instead of overwriting each other. Defaults to a timestamp.')
    ap.add_argument('--seed-list', type=int, nargs='+', default=None,
                     help='Train exactly these seed indices (e.g. "--seed-list 3") instead of '
                          '0..seeds-1. For running specific seeds in separate concurrent '
                          'terminal processes without both colliding on seed 0 -- skips the '
                          'final table/plot/results.json/README (those need every seed '
                          'present); run again without --seed-list once all seeds are cached '
                          'to produce the real aggregate output.')
    ap.add_argument('--n-jobs', type=int, default=-1,
                     help='sklearn n_jobs for RandomForest/GBT (default -1, all cores). Lower '
                          'this (e.g. 4) when running multiple seeds concurrently in separate '
                          'terminals on the same machine, so they do not both claim every core.')
    ap.add_argument('--threads', type=int, default=None,
                     help='Caps torch.set_num_threads() for this process (PyTorch defaults to '
                          'one thread per physical core). Same reasoning as --n-jobs: set this '
                          'when running two concurrent seeds so each leaves room for the other.')
    args = ap.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    run_label = args.run_label or datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', 'ablation', run_label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'full_console_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"Run label: {run_label}")
    if args.threads is not None:
        print(f"torch threads capped at {args.threads}")
    print(f"Logging full console output -> {log_path}")

    run(seeds=args.seeds, sources=tuple(args.sources), epochs=args.epochs,
        quick=args.quick, models=args.models, out_dir=out_dir, run_label=run_label,
        seed_list=args.seed_list, n_jobs=args.n_jobs)
