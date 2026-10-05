"""Single-drone trajectory reconstruction + comparison harness.

Takes a trained LSTM model and reconstructs a real, continuous trajectory by
CHAINING its per-window delta predictions -- exactly how it would be used in
deployment (cumulative drift included), not just isolated per-window MSE --
alongside two non-learned baselines built from the SAME real sensor data:
the classical EKF (models/ekf_baseline.py) and pure inertial integration
(the same EKF class, never given a barometer correction). Ground truth is
the dataset's own real position.

Produces a results JSON consumed by visualize_trajectory.py -- computation
and plotting are deliberately separate so the (slower) reconstruction
doesn't have to rerun every time a plot's styling changes.

Usage:
  python ai_backend/evaluate_trajectory.py --model-tag imu_norm --n-samples 3000
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.dataset_parser import FEATURE_COLS, load_and_clean_imu_data
from models.dead_reckoning_model import DeadReckoningLSTM, INPUT_SIZE
from models.dead_reckoning_model_uncertainty import DeadReckoningLSTMUncertainty
from models.dr_transformer import DeadReckoningTransformer
from models.ekf_baseline import DeadReckoningEKF, rotation_from_euler_deg

WINDOW_SIZE = 10
POS_COLS = ['latitude', 'longitude', 'altitude']   # already relative metres for the imu source


def load_val_segment(base_dir: str, n_samples: int = 3000, val_fraction: float = 0.2):
    """A real, contiguous, held-out segment -- the tail val_fraction of
    imu_data.csv, the same split boundary used everywhere else in the
    project (create_dead_reckoning_dataset's sequential 80/20 split)."""
    imu_path = os.path.join(base_dir, 'imu_data.csv')
    df = load_and_clean_imu_data(imu_path)
    split = int(len(df) * (1.0 - val_fraction))
    val_df = df.iloc[split:].reset_index(drop=True)
    seg = val_df.iloc[:n_samples + WINDOW_SIZE].reset_index(drop=True)
    return seg


def reconstruct_lstm(seg, model: DeadReckoningLSTM, scaler_X, scaler_y) -> np.ndarray:
    """Chains the model's per-window delta predictions into a running
    position estimate -- the honest way to evaluate dead reckoning (drift
    compounds exactly as it would in real operation), not an isolated
    per-window MSE."""
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()   # anchor: true position under the first window

    model.eval()
    with torch.no_grad():
        for i in range(n):
            window_scaled = scaler_X.transform(feat[i:i + WINDOW_SIZE])
            x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)
            pred_scaled = model(x).numpy()
            delta = scaler_y.inverse_transform(pred_scaled)[0]
            pos = pos + delta
            positions[i] = pos
    return positions


def reconstruct_lstm_uncertainty(seg, model: DeadReckoningLSTMUncertainty, scaler_X, scaler_y):
    """Same chaining as reconstruct_lstm(), for the uncertainty model. Also
    returns the model's predicted per-axis 1-sigma std at every step,
    converted from scaled-space log-variance back to real metres: since
    scaler_y is a linear StandardScaler (y_scaled = (y - mean_) / scale_),
    std in original units = std_scaled * scaler_y.scale_ (per axis).

    Returns (positions, pred_std), positions shape (n, 3) as in
    reconstruct_lstm(), pred_std shape (n, 3) in metres, one std per axis
    per step (not yet reduced to a single scalar, run() combines it into
    a per-step magnitude via the same L2 convention as euclidean_error()).
    """
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pred_std = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()

    model.eval()
    with torch.no_grad():
        for i in range(n):
            window_scaled = scaler_X.transform(feat[i:i + WINDOW_SIZE])
            x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)
            mu_scaled, log_var_scaled = model(x)
            mu_scaled = mu_scaled.numpy()
            std_scaled = np.exp(0.5 * log_var_scaled.numpy())

            delta = scaler_y.inverse_transform(mu_scaled)[0]
            # scaler_y.inverse_transform is affine (x*scale_+mean_), so its
            # linear part alone (scale_) converts a *std*, no mean_ shift.
            std_real = (std_scaled[0] * scaler_y.scale_)

            pos = pos + delta
            positions[i] = pos
            pred_std[i] = std_real
    return positions, pred_std


def reconstruct_transformer(seg, model: DeadReckoningTransformer, scaler_X, scaler_y,
                             collect_attention: bool = False):
    """Same chaining convention as reconstruct_lstm(). collect_attention=True
    additionally records each step's attention weights (from
    DeadReckoningTransformer's need_weights=True path), returned as a list
    of length n, each entry a list of num_layers arrays shaped
    (nhead, window+1, window+1), for the analysis deliverable's attention
    visualization. Off by default since it roughly doubles memory for a
    quantity only the analysis, not the chained-error numbers, needs."""
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()
    attn_log = [] if collect_attention else None

    model.eval()
    with torch.no_grad():
        for i in range(n):
            window_scaled = scaler_X.transform(feat[i:i + WINDOW_SIZE])
            x = torch.tensor(window_scaled[np.newaxis], dtype=torch.float32)
            if collect_attention:
                pred_scaled, attn_weights = model(x, need_weights=True)
                attn_log.append([w[0].numpy() for w in attn_weights])   # drop batch dim (size 1)
            else:
                pred_scaled = model(x)
            delta = scaler_y.inverse_transform(pred_scaled.numpy())[0]
            pos = pos + delta
            positions[i] = pos

    if collect_attention:
        return positions, attn_log
    return positions


def reconstruct_classical(seg, model, scaler_X, scaler_y) -> np.ndarray:
    """Same chaining convention as reconstruct_lstm(), for a classical
    (non-sequential) regressor from classical_baselines.py, RandomForest
    or MultiOutputRegressor(HistGradientBoostingRegressor). model.predict()
    takes a flattened (n_samples, window*features) array rather than a
    (n_samples, window, features) tensor, everything else about the chained
    reconstruction (anchor at the true position under the first window,
    accumulate predicted deltas) is identical."""
    feat = seg[FEATURE_COLS].values.astype(np.float32)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    positions = np.zeros((n, 3))
    pos = true_pos[WINDOW_SIZE - 1].copy()

    for i in range(n):
        window_scaled = scaler_X.transform(feat[i:i + WINDOW_SIZE])
        x_flat = window_scaled.reshape(1, -1)
        pred_scaled = model.predict(x_flat)
        delta = scaler_y.inverse_transform(pred_scaled)[0]
        pos = pos + delta
        positions[i] = pos
    return positions


def reconstruct_filter(seg, use_baro: bool = True, baro_noise_std: float = 0.3,
                        baro_every: int = 24, seed: int = 0,
                        return_std: bool = False):
    """Runs DeadReckoningEKF over the same real accel/attitude/dt sequence.
    use_baro=False gives "pure inertial integration" for free -- same class,
    same process model, just never calling update_baro().

    return_std=True additionally returns the EKF's own per-axis
    position_uncertainty at every step (its native covariance-derived
    1-sigma estimate), for comparing against the learned NLL uncertainty
    head (dead_reckoning_model_uncertainty.py) on the same segment."""
    accel = seg[['imu_acc_x', 'imu_acc_y', 'imu_acc_z']].values.astype(np.float64)
    rpy = seg[['roll', 'pitch', 'yaw']].values.astype(np.float64)
    dt_arr = seg['dt'].values.astype(np.float64)
    true_pos = seg[POS_COLS].values.astype(np.float64)
    n = len(seg) - WINDOW_SIZE

    ekf = DeadReckoningEKF()
    ekf.reset(position=true_pos[WINDOW_SIZE - 1])
    rng = np.random.default_rng(seed)

    positions = np.zeros((n, 3))
    stds = np.zeros((n, 3)) if return_std else None
    for i in range(n):
        idx = WINDOW_SIZE + i
        R = rotation_from_euler_deg(*rpy[idx])
        ekf.predict(accel[idx], R, dt_arr[idx])
        if use_baro and i % baro_every == 0:
            noisy_alt = true_pos[idx, 2] + rng.normal(0, baro_noise_std)
            ekf.update_baro(noisy_alt)
        positions[i] = ekf.position
        if return_std:
            stds[i] = ekf.position_uncertainty
    if return_std:
        return positions, stds
    return positions


def ground_truth(seg) -> np.ndarray:
    return seg[POS_COLS].values.astype(np.float64)[WINDOW_SIZE:]


def euclidean_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=1)


def uncertainty_calibration(pred_std: np.ndarray, actual_error: np.ndarray) -> dict:
    """Does the model's predicted uncertainty actually track its real error?
    pred_std is per-axis (n, 3); reduced to a single per-step magnitude via
    the same L2 convention as euclidean_error() so it's directly comparable
    to actual_error (n,). Returns Pearson correlation plus a coarse binned
    calibration table (5 bins by predicted-uncertainty quantile: mean
    predicted std vs. mean actual error per bin, a well-calibrated model
    should show those two columns tracking each other bin-for-bin)."""
    pred_mag = np.linalg.norm(pred_std, axis=1)
    corr = float(np.corrcoef(pred_mag, actual_error)[0, 1])

    n_bins = 5
    order = np.argsort(pred_mag)
    bin_edges = np.array_split(order, n_bins)
    bins = []
    for b in bin_edges:
        bins.append({
            'n': int(len(b)),
            'mean_pred_std': float(pred_mag[b].mean()),
            'mean_actual_error': float(actual_error[b].mean()),
        })
    return {'pearson_r': corr, 'bins': bins}


def run(model_tag: str = 'imu_norm', n_samples: int = 3000, out_path: str = None,
        include_uncertainty: bool = True, include_transformer: bool = True,
        include_classical: bool = True) -> dict:
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    print(f"Loading val segment ({n_samples} samples, held-out tail of imu_data.csv)...")
    seg = load_val_segment(base_dir, n_samples=n_samples)

    print(f"Loading model tag '{model_tag}'...")
    model = DeadReckoningLSTM(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f'dr_lstm_{model_tag}.pth'), map_location='cpu',
        weights_only=False))   # trusted, self-generated checkpoint (see device_utils.py)
    scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_{model_tag}.pkl'))
    scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_{model_tag}.pkl'))

    print("Reconstructing LSTM trajectory (chained predictions)...")
    lstm_pos = reconstruct_lstm(seg, model, scaler_X, scaler_y)

    print("Reconstructing EKF trajectory (real accel + periodic simulated-baro correction)...")
    ekf_pos, ekf_std = reconstruct_filter(seg, use_baro=True, return_std=True)

    print("Reconstructing pure-inertial trajectory (no correction)...")
    inertial_pos = reconstruct_filter(seg, use_baro=False)

    gt_pos = ground_truth(seg)
    t = np.cumsum(seg['dt'].values[WINDOW_SIZE:])

    results = {
        'model_tag': model_tag,
        'n_samples': int(len(gt_pos)),
        't': t.tolist(),
        'ground_truth': gt_pos.tolist(),
        'lstm': lstm_pos.tolist(),
        'ekf': ekf_pos.tolist(),
        'pure_inertial': inertial_pos.tolist(),
        'error_lstm': euclidean_error(lstm_pos, gt_pos).tolist(),
        'error_ekf': euclidean_error(ekf_pos, gt_pos).tolist(),
        'error_pure_inertial': euclidean_error(inertial_pos, gt_pos).tolist(),
        'ekf_pred_std': ekf_std.tolist(),
    }

    uncertainty_tag = f'uncertainty_{model_tag}'
    uncertainty_model_path = os.path.join(models_dir, f'dr_lstm_{uncertainty_tag}.pth')
    if include_uncertainty and os.path.exists(uncertainty_model_path):
        print("Reconstructing uncertainty-LSTM trajectory (mean + learned std)...")
        u_model = DeadReckoningLSTMUncertainty(input_size=INPUT_SIZE)
        u_model.load_state_dict(torch.load(uncertainty_model_path, map_location='cpu',
                                            weights_only=False))
        u_scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_{uncertainty_tag}.pkl'))
        u_scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_{uncertainty_tag}.pkl'))

        u_pos, u_std = reconstruct_lstm_uncertainty(seg, u_model, u_scaler_X, u_scaler_y)
        u_error = euclidean_error(u_pos, gt_pos)
        calib = uncertainty_calibration(u_std, u_error)

        # Same calibration check applied to the EKF's native covariance, so
        # the two uncertainty sources are compared on identical footing.
        ekf_error = np.array(results['error_ekf'])
        ekf_calib = uncertainty_calibration(ekf_std, ekf_error)

        results['lstm_uncertainty'] = u_pos.tolist()
        results['error_lstm_uncertainty'] = u_error.tolist()
        results['lstm_uncertainty_pred_std'] = u_std.tolist()
        results['lstm_uncertainty_calibration'] = calib
        results['ekf_calibration'] = ekf_calib

        print(f"\nUncertainty calibration (Pearson r between predicted std magnitude "
              f"and actual error):")
        print(f"  LSTM-uncertainty head: r = {calib['pearson_r']:.3f}")
        print(f"  EKF native covariance: r = {ekf_calib['pearson_r']:.3f}")
    else:
        print(f"\n(Skipping uncertainty comparison: no trained model found at "
              f"{uncertainty_model_path}, run train_uncertainty.py first.)")

    transformer_model_path = os.path.join(models_dir, f'dr_transformer_{model_tag}.pth')
    if include_transformer and os.path.exists(transformer_model_path):
        print("Reconstructing Transformer trajectory (chained predictions)...")
        t_model = DeadReckoningTransformer(input_size=INPUT_SIZE)
        t_model.load_state_dict(torch.load(transformer_model_path, map_location='cpu',
                                            weights_only=False))
        t_scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_transformer_{model_tag}.pkl'))
        t_scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_transformer_{model_tag}.pkl'))

        t_pos = reconstruct_transformer(seg, t_model, t_scaler_X, t_scaler_y)
        results['transformer'] = t_pos.tolist()
        results['error_transformer'] = euclidean_error(t_pos, gt_pos).tolist()
    else:
        print(f"\n(Skipping Transformer comparison: no trained model found at "
              f"{transformer_model_path}, run train_transformer.py first.)")

    classical_tag = f'{model_tag.rsplit("_norm", 1)[0]}_norm'
    rf_path = os.path.join(models_dir, f'rf_{classical_tag}.pkl')
    gbt_path = os.path.join(models_dir, f'gbt_{classical_tag}.pkl')
    if include_classical and os.path.exists(rf_path) and os.path.exists(gbt_path):
        print("Reconstructing RandomForest and GBT trajectories (chained predictions)...")
        rf_model = joblib.load(rf_path)
        rf_scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_rf_{classical_tag}.pkl'))
        rf_scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_rf_{classical_tag}.pkl'))
        gbt_model = joblib.load(gbt_path)
        gbt_scaler_X = joblib.load(os.path.join(models_dir, f'scaler_X_gbt_{classical_tag}.pkl'))
        gbt_scaler_y = joblib.load(os.path.join(models_dir, f'scaler_y_gbt_{classical_tag}.pkl'))

        rf_pos = reconstruct_classical(seg, rf_model, rf_scaler_X, rf_scaler_y)
        gbt_pos = reconstruct_classical(seg, gbt_model, gbt_scaler_X, gbt_scaler_y)
        results['random_forest'] = rf_pos.tolist()
        results['error_random_forest'] = euclidean_error(rf_pos, gt_pos).tolist()
        results['gbt'] = gbt_pos.tolist()
        results['error_gbt'] = euclidean_error(gbt_pos, gt_pos).tolist()
    else:
        print(f"\n(Skipping classical-ML comparison: no trained models found at "
              f"{rf_path} / {gbt_path}, run classical_baselines.py first.)")

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'runs',
                                 f'trajectory_eval_{model_tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved trajectory evaluation -> {out_path}")

    print(f"\n{'Method':<20}{'Final err [m]':>15}{'Mean err [m]':>15}{'Max err [m]':>13}")
    methods = [('LSTM', 'error_lstm'), ('EKF', 'error_ekf'),
               ('Pure Inertial', 'error_pure_inertial')]
    if 'error_lstm_uncertainty' in results:
        methods.append(('LSTM-uncertainty', 'error_lstm_uncertainty'))
    if 'error_transformer' in results:
        methods.append(('Transformer', 'error_transformer'))
    if 'error_random_forest' in results:
        methods.append(('RandomForest', 'error_random_forest'))
    if 'error_gbt' in results:
        methods.append(('GBT', 'error_gbt'))
    for name, key in methods:
        err = np.array(results[key])
        print(f"{name:<20}{err[-1]:>15.2f}{err.mean():>15.2f}{err.max():>13.2f}")

    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', default='imu_norm',
                    help="Model tag under ai_backend/models/ (dr_lstm_{tag}.pth + scalers).")
    ap.add_argument('--n-samples', type=int, default=3000,
                    help="Segment length in samples (240Hz for the imu source, so 3000 ~= 12.5s).")
    ap.add_argument('--no-uncertainty', action='store_true',
                    help="Skip the uncertainty-model comparison even if a trained one exists.")
    ap.add_argument('--no-transformer', action='store_true',
                    help="Skip the Transformer comparison even if a trained one exists.")
    ap.add_argument('--no-classical', action='store_true',
                    help="Skip the RandomForest/GBT comparison even if trained models exist.")
    args = ap.parse_args()
    run(model_tag=args.model_tag, n_samples=args.n_samples,
        include_uncertainty=not args.no_uncertainty,
        include_transformer=not args.no_transformer,
        include_classical=not args.no_classical)
