"""Renders the trajectory-evaluation results from evaluate_trajectory.py as
Category A (3D trajectory + XY/XZ projections) and Category B (Euclidean
error-over-time) figures, per AI_RECOVERY_EXECUTION_PLAN.md §12 Goal 4.
Fixed color palette / line-style conventions -- see plot_style.py.

Usage:
  python ai_backend/visualize_trajectory.py
  (defaults to runs/trajectory_eval_imu_norm.json + the SSL-tagged one if present)
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import (COLOR_REFERENCE, COLOR_PREDICTED, COLOR_GROUND_TRUTH,
                         COLOR_EKF, COLOR_PURE_INERTIAL, style_axes)


def plot_3d_trajectory(results: dict, out_path: str) -> None:
    gt = np.array(results['ground_truth'])
    pred = np.array(results['lstm'])
    ekf = np.array(results['ekf'])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=COLOR_GROUND_TRUTH, linewidth=1.5,
            linestyle='-', label='Ground Truth (Executed)')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=COLOR_PREDICTED, linewidth=1.3,
            linestyle='-', label='LSTM Reconstruction (Predicted)')
    ax.plot(ekf[:, 0], ekf[:, 1], ekf[:, 2], color=COLOR_EKF, linewidth=1.2,
            linestyle=':', label='Classical EKF (Baseline)')
    ax.scatter([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], color=COLOR_GROUND_TRUTH, s=40, marker='o')
    ax.zaxis.labelpad = 10
    style_axes(ax, title='3D Trajectory Reconstruction',
               xlabel='Position X [m]', ylabel='Position Y [m]', zlabel='Altitude Z [m]')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_projection(results: dict, axis_pair: tuple, labels: tuple, title: str, out_path: str) -> None:
    gt = np.array(results['ground_truth'])
    pred = np.array(results['lstm'])
    ekf = np.array(results['ekf'])
    i, j = axis_pair

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt[:, i], gt[:, j], color=COLOR_GROUND_TRUTH, linewidth=1.5, label='Ground Truth')
    ax.plot(pred[:, i], pred[:, j], color=COLOR_PREDICTED, linewidth=1.3, label='LSTM Prediction')
    ax.plot(ekf[:, i], ekf[:, j], color=COLOR_EKF, linewidth=1.2, linestyle=':', label='EKF')
    ax.scatter([gt[0, i]], [gt[0, j]], color=COLOR_GROUND_TRUTH, s=50, zorder=5)
    ax.set_aspect('equal', adjustable='datalim')
    style_axes(ax, title=title, xlabel=labels[0], ylabel=labels[1])
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_error_over_time(results: dict, out_path: str, ssl_results: dict = None) -> None:
    t = np.array(results['t'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, results['error_pure_inertial'], color=COLOR_PURE_INERTIAL, linewidth=1.2,
            linestyle=':', label='Pure Inertial Integration')
    ax.plot(t, results['error_ekf'], color=COLOR_EKF, linewidth=1.2,
            linestyle=':', label='Classical EKF')
    ax.plot(t, results['error_lstm'], color=COLOR_PREDICTED, linewidth=1.4,
            linestyle='-', label='Supervised LSTM')
    if ssl_results is not None:
        ax.plot(t, ssl_results['error_lstm'], color=COLOR_REFERENCE, linewidth=1.4,
                linestyle='-', label='SSL-Pretrained LSTM')
    style_axes(ax, title='Position Error Growth over Time',
               xlabel='Time [s]', ylabel='Euclidean Error [m]')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run(results_path: str, out_dir: str, ssl_results_path: str = None) -> None:
    with open(results_path) as f:
        results = json.load(f)
    ssl_results = None
    if ssl_results_path and os.path.exists(ssl_results_path):
        with open(ssl_results_path) as f:
            ssl_results = json.load(f)
    elif ssl_results_path:
        print(f"(No SSL results at {ssl_results_path} yet -- skipping that comparison line.)")

    os.makedirs(out_dir, exist_ok=True)
    plot_3d_trajectory(results, os.path.join(out_dir, 'trajectory_3d.png'))
    plot_projection(results, (0, 1), ('Position X [m]', 'Position Y [m]'),
                     'Top-Down XY Path View', os.path.join(out_dir, 'trajectory_xy.png'))
    plot_projection(results, (0, 2), ('Position X [m]', 'Altitude Z [m]'),
                     'Elevation XZ Path View', os.path.join(out_dir, 'trajectory_xz.png'))
    plot_error_over_time(results, os.path.join(out_dir, 'error_over_time.png'), ssl_results)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None)
    ap.add_argument('--ssl-results', default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), '..')
    results_path = args.results or os.path.join(root, 'runs', 'trajectory_eval_imu_norm.json')
    ssl_results_path = args.ssl_results or os.path.join(root, 'runs', 'trajectory_eval_imu_ssl_norm.json')
    out_dir = args.out_dir or os.path.join(root, 'runs', 'figures_single_drone')
    run(results_path, out_dir, ssl_results_path)
