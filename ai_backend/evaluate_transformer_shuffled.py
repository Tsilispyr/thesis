"""Evaluates the order-shuffled Transformer (train_transformer.py --shuffle-order)
on the same held-out chained-trajectory segment and same chaining convention
every other model in this project is evaluated on, applying the identical
fixed permutation (imported from train_transformer._SHUFFLE_PERM, not
redefined here) to each window before prediction, matching what the model
was trained on.

Answers the open question directly: does the real Transformer's advantage
over GBT (order-blind) depend on genuine chronological order, or does a
fixed-but-non-chronological slot assignment work almost as well?

Usage:
  python ai_backend/evaluate_transformer_shuffled.py
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
from train_transformer import _SHUFFLE_PERM
from evaluate_trajectory import (
    load_val_segment, ground_truth, euclidean_error,
    FEATURE_COLS, POS_COLS, WINDOW_SIZE,
)

TAG = 'imu_norm_shuffled'


def reconstruct_transformer_shuffled(seg, model, scaler_X, scaler_y, perm) -> np.ndarray:
    """Same chaining convention as evaluate_trajectory.reconstruct_transformer,
    with perm applied to each window's timestep axis before prediction --
    must match how the model was trained (train_transformer.py's
    shuffle_window_order=True path)."""
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()

    model.eval()
    with torch.no_grad():
        for i in range(n):
            window = feat[i:i + WINDOW_SIZE][perm]
            window_scaled = scaler_X.transform(window)
            x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)
            pred_scaled = model(x).numpy()
            delta = scaler_y.inverse_transform(pred_scaled)[0]
            pos = pos + delta
            positions[i] = pos
    return positions


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    print("Loading held-out segment...")
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    print(f"Loading order-shuffled Transformer (tag={TAG}), permutation={_SHUFFLE_PERM.tolist()}...")
    model = DeadReckoningTransformer(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f'dr_transformer_{TAG}.pth'), map_location='cpu',
        weights_only=False))
    scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_transformer_{TAG}.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_transformer_{TAG}.pkl'))

    print("Reconstructing chained trajectory (with the same fixed permutation applied "
          "to every window)...")
    pos = reconstruct_transformer_shuffled(seg, model, scaler_X, scaler_y, _SHUFFLE_PERM)
    err = euclidean_error(pos, gt_pos)

    result = {
        'tag': TAG,
        'permutation': _SHUFFLE_PERM.tolist(),
        'final_error_m': float(err[-1]),
        'mean_error_m': float(err.mean()),
    }

    print(f"\nOrder-shuffled Transformer -- final error: {err[-1]:.2f} m, "
          f"mean error: {err.mean():.2f} m")
    print("Comparison point (real, order-preserving Transformer, from "
          "ai_backend/models/ANALYSIS.md): 12.46 m final / 9.75 m mean")
    print("Comparison point (GBT, order-blind, from the same doc): 18.06 m final / 11.66 m mean")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                             'transformer_shuffled_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
