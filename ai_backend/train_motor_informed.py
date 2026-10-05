"""Multi-seed comparison: real px4 IMU-only baseline vs. the same real px4
corpus with motor/actuator output added as input features
(MOTOR_FEATURE_COLS, data_processing/dataset_parser.py) -- does knowing what
the motors were actually commanded to do help reconstruct position, on top
of what the IMU alone already provides?

px4 is the only source this can run on: imu_data.csv (the production
training source) has no motor/actuator column at all, confirmed by reading
its header directly. Motor data is genuine signal on the real corpus, not a
placeholder -- confirmed directly (motor_0..3 mean~0.3, std~0.5 across the
46-log corpus), not assumed.

This is train + evaluate only, deliberately not wired into
recovery_orchestrator.py in this pass (per explicit scope decision): answer
whether motor data actually helps before deciding whether/how to add a new
fallback layer to the live hierarchy. If it does, the natural follow-up is a
manually-selectable option alongside the existing Auto/LSTM/EKF/RULE_BASED
force-layer picker -- there's no automatic trigger condition (no IMU-health
signal exists anywhere in the codebase) worth building around yet.

Reuses train_px4_synthetic.py's utilities directly (set_global_seed,
confidence_interval, _split_real, and all three plotting functions) rather
than re-implementing them -- same held-out-real-val-rows discipline, same
multi-seed rigor this session established for every PX4 experiment.

Usage:
  python ai_backend/train_motor_informed.py --n-seeds 3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing.dataset_parser import load_combined_dataset, FEATURE_COLS, MOTOR_FEATURE_COLS
from train import train_model
from train_px4_synthetic import (set_global_seed, confidence_interval, _split_real,
                                  _plot_comparison, _plot_training_curves_multiseed,
                                  _plot_val_curve_overlay)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Dedicated output folder, same discipline as px4_synthetic_diversity/ -- never
# ai_backend/models/ for anything but raw checkpoints/scalers/per-seed history
# (that folder's own existing convention), never any other runs/ subfolder.
_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'px4_motor_informed')


def run_arm(train_df, val_df, seed, tag_suffix, epochs, patience, feature_cols):
    set_global_seed(seed)
    _, _, _, history = train_model(
        sources=('px4',), num_epochs=epochs, normalize=True,
        early_stopping_patience=patience, override_df=train_df,
        held_out_val_df=val_df, tag_suffix=tag_suffix, feature_cols=feature_cols)
    os.makedirs(_OUT_DIR, exist_ok=True)
    hist_name = f"history_px4_norm{tag_suffix}.json"
    import shutil
    shutil.copy2(os.path.join(_MODELS_DIR, hist_name), os.path.join(_OUT_DIR, hist_name))
    return history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
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
    # Sanity check before trusting anything downstream: confirm motor columns
    # are genuinely present and non-constant on the actual rows this run will
    # use, not silently zero-filled/dropped somewhere upstream.
    motor_cols = ['motor_0', 'motor_1', 'motor_2', 'motor_3']
    stds = real_train_df[motor_cols].std()
    print(f"  Motor column std on train-pool rows: {dict(stds.round(3))}")
    assert (stds > 1e-6).all(), (
        f"Motor columns are constant on the train pool -- would make this comparison "
        f"meaningless. stds: {dict(stds)}")

    baseline_histories, motor_histories = [], []
    for seed in range(args.n_seeds):
        print(f"\n{'='*70}\nSeed {seed} -- baseline (IMU-only, {len(FEATURE_COLS)} features)\n{'='*70}")
        baseline_histories.append(run_arm(
            real_train_df, real_val_df, seed,
            tag_suffix=f'_mission_v2_motor_baseline{args.tag}_seed{seed}',
            epochs=args.epochs, patience=args.patience, feature_cols=FEATURE_COLS))

        print(f"\n{'='*70}\nSeed {seed} -- motor-informed (IMU+motor, {len(MOTOR_FEATURE_COLS)} features)\n{'='*70}")
        motor_histories.append(run_arm(
            real_train_df, real_val_df, seed,
            tag_suffix=f'_mission_v2_motor_informed{args.tag}_seed{seed}',
            epochs=args.epochs, patience=args.patience, feature_cols=MOTOR_FEATURE_COLS))

    baseline_results = [h['best_val'] for h in baseline_histories]
    motor_results = [h['best_val'] for h in motor_histories]
    base_mean, base_lo, base_hi = confidence_interval(baseline_results)
    motor_mean, motor_lo, motor_hi = confidence_interval(motor_results)

    print(f"\n{'='*70}\nResult ({args.n_seeds} seeds, 95% CI)\n{'='*70}")
    print(f"  Baseline (IMU-only):      val_loss = {base_mean:.4f}  "
          f"[{base_lo:.4f}, {base_hi:.4f}]  {baseline_results}")
    print(f"  Motor-informed (IMU+motor): val_loss = {motor_mean:.4f}  "
          f"[{motor_lo:.4f}, {motor_hi:.4f}]  {motor_results}")
    overlap = not (motor_hi < base_lo or base_hi < motor_lo)
    print(f"  95% CIs {'OVERLAP -- not distinguishable from noise' if overlap else 'DO NOT overlap -- a real difference'}")

    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f'comparison{args.tag}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_seeds': args.n_seeds, 'epochs': args.epochs, 'patience': args.patience,
            'baseline_n_features': len(FEATURE_COLS), 'motor_n_features': len(MOTOR_FEATURE_COLS),
            'baseline_val_losses': baseline_results,
            'motor_informed_val_losses': motor_results,
            'baseline_mean_ci': [base_mean, base_lo, base_hi],
            'motor_informed_mean_ci': [motor_mean, motor_lo, motor_hi],
            'cis_overlap': overlap,
        }, f, indent=2)
    print(f"Saved -> {out_path}")

    # Same convention as px4_synthetic_diversity/: no separate "test" plot --
    # this pipeline only ever splits real px4 rows into train/val, the val
    # curves below ARE the honest, never-trained-on measurement.
    fig_dir = os.path.join(_OUT_DIR, 'figures')
    _plot_training_curves_multiseed(
        baseline_histories, f'PX4 baseline (IMU-only, {args.n_seeds} seeds)',
        os.path.join(fig_dir, f'train_curves_baseline{args.tag}.png'))
    _plot_training_curves_multiseed(
        motor_histories, f'PX4 motor-informed (IMU+motor, {args.n_seeds} seeds)',
        os.path.join(fig_dir, f'train_curves_motor{args.tag}.png'))
    _plot_val_curve_overlay(
        baseline_histories, motor_histories,
        os.path.join(fig_dir, f'val_curve_overlay{args.tag}.png'),
        labels=('Baseline (IMU-only)', 'Motor-informed (IMU+motor)'),
        title='Validation loss: IMU-only vs. motor-informed (mean +-1 std)')

    if args.n_seeds >= 2:
        _plot_comparison(
            baseline_results, motor_results,
            os.path.join(fig_dir, f'comparison{args.tag}.png'),
            labels=('Baseline\n(IMU-only)', 'Motor-informed\n(IMU+motor)'),
            title='PX4 motor-informed features: val loss by seed')
    else:
        print("(Skipping box-plot comparison -- n_seeds < 2, a single point has no meaningful spread to show.)")
