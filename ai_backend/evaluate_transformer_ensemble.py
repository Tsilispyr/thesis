"""Does ensembling the Transformer's 5 already-trained ablation-seed
checkpoints reduce its known seed-to-seed unreliability (71% coefficient of
variation in runs/ablation/cpu_5seeds/ANALYSIS.md), without training anything
new?

At each chained-trajectory step, all 5 seeds' per-window delta predictions
(in real units, after each seed's own scaler_y.inverse_transform) are
averaged BEFORE being added to the running position -- an ensemble of the
prediction each model actually makes at that step, not a post-hoc average of
5 independently-drifted trajectories. Evaluated on the exact same held-out
segment every other model in this project is evaluated on
(evaluate_trajectory.load_val_segment), so the ensemble's number is directly
comparable to the per-seed numbers already reported in
runs/ablation/cpu_5seeds/results.json.

Usage:
  python ai_backend/evaluate_transformer_ensemble.py
"""
import os
import sys
import json

import numpy as np
import torch
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dead_reckoning_model import INPUT_SIZE
from models.dr_transformer import DeadReckoningTransformer
from evaluate_trajectory import (
    load_val_segment, ground_truth, euclidean_error, reconstruct_transformer,
    FEATURE_COLS, POS_COLS, WINDOW_SIZE,
)

SEEDS = [0, 1, 2, 3, 4]
TAG = 'imu_norm_ablation_seed{seed}'


def load_seed_model(models_dir: str, seed: int):
    tag = TAG.format(seed=seed)
    model = DeadReckoningTransformer(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f'dr_transformer_{tag}.pth'), map_location='cpu',
        weights_only=False))
    model.eval()
    scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_transformer_{tag}.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_transformer_{tag}.pkl'))
    return model, scaler_X, scaler_y


def reconstruct_transformer_ensemble(seg, models_and_scalers) -> np.ndarray:
    """Same chaining convention as evaluate_trajectory.reconstruct_transformer,
    but the delta added at each step is the mean of all N seeds' predicted
    delta at that step (each seed's own scaler_X for the window, own
    scaler_y to invert its prediction back to real metres, before
    averaging)."""
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()

    for m, _, _ in models_and_scalers:
        m.eval()

    with torch.no_grad():
        for i in range(n):
            deltas = []
            for model, scaler_X, scaler_y in models_and_scalers:
                window_scaled = scaler_X.transform(feat[i:i + WINDOW_SIZE])
                x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)
                pred_scaled = model(x).numpy()
                deltas.append(scaler_y.inverse_transform(pred_scaled)[0])
            mean_delta = np.mean(deltas, axis=0)
            pos = pos + mean_delta
            positions[i] = pos
    return positions


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    print("Loading held-out segment (same one every other evaluation in this project uses)...")
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    print(f"Loading {len(SEEDS)} Transformer ablation-seed checkpoints...")
    models_and_scalers = [load_seed_model(models_dir, s) for s in SEEDS]

    per_seed = {}
    for seed, (model, scaler_X, scaler_y) in zip(SEEDS, models_and_scalers):
        pos = reconstruct_transformer(seg, model, scaler_X, scaler_y)
        err = euclidean_error(pos, gt_pos)
        per_seed[seed] = {'final': float(err[-1]), 'mean': float(err.mean())}
        print(f"  seed {seed}: final={err[-1]:.2f} m, mean={err.mean():.2f} m "
              f"(cross-check against results.json per_seed[{seed}])")

    print("\nReconstructing ensemble trajectory (mean of 5 seeds' per-step delta, "
          "averaged before chaining)...")
    ens_pos = reconstruct_transformer_ensemble(seg, models_and_scalers)
    ens_err = euclidean_error(ens_pos, gt_pos)

    seed_finals = np.array([v['final'] for v in per_seed.values()])
    seed_means = np.array([v['mean'] for v in per_seed.values()])

    result = {
        'per_seed': per_seed,
        'seed_final_mean': float(seed_finals.mean()),
        'seed_final_std': float(seed_finals.std()),
        'seed_final_cv_pct': float(100 * seed_finals.std() / seed_finals.mean()),
        'ensemble_final': float(ens_err[-1]),
        'ensemble_mean': float(ens_err.mean()),
    }

    print(f"\n5 individual seeds -- final error: mean={seed_finals.mean():.2f} m, "
          f"std={seed_finals.std():.2f} m, CV={result['seed_final_cv_pct']:.0f}%, "
          f"range=[{seed_finals.min():.2f}, {seed_finals.max():.2f}] m")
    print(f"Ensemble (5 seeds, averaged per-step)  -- final={ens_err[-1]:.2f} m, "
          f"mean={ens_err.mean():.2f} m")
    print(f"\nEnsemble vs. best individual seed (final): "
          f"{ens_err[-1]:.2f} m vs. {seed_finals.min():.2f} m")
    print(f"Ensemble vs. worst individual seed (final): "
          f"{ens_err[-1]:.2f} m vs. {seed_finals.max():.2f} m")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                             'transformer_ensemble_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
