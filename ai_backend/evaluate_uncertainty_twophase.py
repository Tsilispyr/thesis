"""Evaluates the two-phase uncertainty checkpoint (train_uncertainty_two_phase.py)
on the exact same held-out chained-trajectory segment and calibration check
evaluate_trajectory.py uses for the original single-phase uncertainty model,
so the two are directly comparable. Not folded into evaluate_trajectory.py's
run() since that function couples the uncertainty tag to a plain-LSTM tag
that doesn't exist for this twophase-only checkpoint.

Usage:
  python ai_backend/evaluate_uncertainty_twophase.py
"""
import os
import sys
import json

import torch
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dead_reckoning_model import INPUT_SIZE
from models.dead_reckoning_model_uncertainty import DeadReckoningLSTMUncertainty
from evaluate_trajectory import (
    load_val_segment, ground_truth, euclidean_error, reconstruct_lstm_uncertainty,
    uncertainty_calibration,
)

TAG = 'imu_norm_twophase'


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    print("Loading held-out segment...")
    seg = load_val_segment(base_dir, n_samples=3000)
    gt_pos = ground_truth(seg)

    print(f"Loading two-phase uncertainty checkpoint (tag={TAG})...")
    model = DeadReckoningLSTMUncertainty(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f'dr_lstm_uncertainty_{TAG}.pth'), map_location='cpu',
        weights_only=False))
    scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_uncertainty_{TAG}.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_uncertainty_{TAG}.pkl'))

    print("Reconstructing chained trajectory...")
    pos, pred_std = reconstruct_lstm_uncertainty(seg, model, scaler_X, scaler_y)
    err = euclidean_error(pos, gt_pos)
    calib = uncertainty_calibration(pred_std, err)

    result = {
        'tag': TAG,
        'final_error_m': float(err[-1]),
        'mean_error_m': float(err.mean()),
        'calibration_pearson_r': calib['pearson_r'],
        'calibration_bins': calib['bins'],
    }

    print(f"\nTwo-phase uncertainty model -- final error: {err[-1]:.2f} m, "
          f"mean error: {err.mean():.2f} m")
    print(f"Calibration (predicted uncertainty vs. actual error): r = {calib['pearson_r']:.3f}")
    print("Comparison point (original single-phase model, from ai_backend/models/ANALYSIS.md): "
          "29.44 m final, r = 0.203 (range 0.119-0.212 across 3 retrains)")
    print("\nBinned calibration table (5 bins by predicted uncertainty, low to high):")
    for i, b in enumerate(calib['bins']):
        print(f"  bin {i}: n={b['n']:>5}  mean_pred_std={b['mean_pred_std']:.2f}  "
              f"mean_actual_error={b['mean_actual_error']:.2f}")

    out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                             'uncertainty_twophase_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
