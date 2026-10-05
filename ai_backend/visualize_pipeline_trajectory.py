"""Adds a trajectory-reconstruction figure to runs/figures/ (the 12-step
ml_pipeline.py dashboard), continuing its numbered sequence as figures
13-15. That dashboard's own Run C (imu, normalized) is the only trustworthy
run in it (see runs/figures/ANALYSIS.md), but its checkpoint
(runs/models/C_imu_norm.pth) was saved without its scalers -- ml_pipeline.py's
_train_run() discards create_dead_reckoning_dataset()'s returned scaler_X/
scaler_y -- so its raw predictions can no longer be converted back into real
metres. Rather than retrain a redundant near-duplicate model just to recover
scalers, this reuses the already-trained, already-validated imu_norm model
(the same one runs/figures_single_drone/ uses, and the same data source/
normalization Run C represents) via its already-computed
runs/trajectory_eval_imu_norm.json. Reuses visualize_trajectory.py's own
plot_3d_trajectory/plot_projection functions directly rather than
duplicating their plotting logic.

Usage:
  python ai_backend/visualize_pipeline_trajectory.py
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualize_trajectory import plot_3d_trajectory, plot_projection


def run(results_path: str, out_dir: str) -> None:
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    plot_3d_trajectory(results, os.path.join(out_dir, '13_trajectory_reconstruction_3d.png'))
    plot_projection(results, (0, 1), ('Position X [m]', 'Position Y [m]'),
                     'Run C (imu_norm) Trajectory -- Top-Down XY View',
                     os.path.join(out_dir, '14_trajectory_reconstruction_xy.png'))
    plot_projection(results, (0, 2), ('Position X [m]', 'Altitude Z [m]'),
                     'Run C (imu_norm) Trajectory -- Elevation XZ View',
                     os.path.join(out_dir, '15_trajectory_reconstruction_xz.png'))


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..')
    results_path = os.path.join(root, 'runs', 'trajectory_eval_imu_norm.json')
    out_dir = os.path.join(root, 'runs', 'figures')
    run(results_path, out_dir)
