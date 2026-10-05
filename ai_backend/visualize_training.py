"""Training/eval curve figures -- consumes the history_*.json files saved by
train.py and ssl_pretext.py (per-epoch train/val loss, best-epoch marker).
Fixed palette/line-style conventions, see plot_style.py.

Usage:
  python ai_backend/visualize_training.py
"""
import os
import sys
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import (COLOR_GROUND_TRUTH, COLOR_EKF, COLOR_PREDICTED,
                         COLOR_REFERENCE, style_axes)


def plot_training_curve(history: dict, title: str, out_path: str) -> None:
    epochs = list(range(1, len(history['train_losses']) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_losses'], color=COLOR_GROUND_TRUTH, linewidth=1.5,
            linestyle='-', label='Train Loss')
    ax.plot(epochs, history['val_losses'], color=COLOR_EKF, linewidth=1.3,
            linestyle=':', label='Validation Loss')
    best = history.get('best_epoch')
    if best:
        ax.axvline(best, color=COLOR_PREDICTED, linewidth=1.0, linestyle=':', alpha=0.7)
        ax.annotate(f'best (epoch {best})', xy=(best, history['val_losses'][best - 1]),
                    xytext=(10, 18), textcoords='offset points', fontsize=8, color=COLOR_PREDICTED,
                    arrowprops=dict(arrowstyle='-', color=COLOR_PREDICTED, alpha=0.6, linewidth=0.8))
    style_axes(ax, title=title, xlabel='Epoch', ylabel='MSE Loss')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_comparison(histories: dict, out_path: str) -> None:
    """histories: {label: history_dict}"""
    labels = list(histories.keys())
    vals = [h['best_val'] for h in histories.values()]
    palette = [COLOR_PREDICTED, COLOR_REFERENCE, COLOR_EKF]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    style_axes(ax, title='Best Validation Loss by Model', ylabel='MSE Loss (normalized)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run(models_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    histories = {}

    sup_path = os.path.join(models_dir, 'history_imu_norm.json')
    if os.path.exists(sup_path):
        with open(sup_path) as f:
            h = json.load(f)
        plot_training_curve(h, 'Supervised LSTM Training (imu)',
                             os.path.join(out_dir, 'train_supervised.png'))
        histories['Supervised\nLSTM'] = h
    else:
        print(f"(No {sup_path} yet -- skipping.)")

    ssl_pre_path = os.path.join(models_dir, 'history_ssl_pretrain_imu_ssl_norm.json')
    if os.path.exists(ssl_pre_path):
        with open(ssl_pre_path) as f:
            h = json.load(f)
        plot_training_curve(h, 'SSL Pretraining (Masked Reconstruction)',
                             os.path.join(out_dir, 'train_ssl_pretrain.png'))
    else:
        print(f"(No {ssl_pre_path} yet -- skipping.)")

    ssl_ft_path = os.path.join(models_dir, 'history_imu_ssl_norm.json')
    if os.path.exists(ssl_ft_path):
        with open(ssl_ft_path) as f:
            h = json.load(f)
        plot_training_curve(h, 'SSL-Pretrained LSTM Fine-Tuning (imu)',
                             os.path.join(out_dir, 'train_ssl_finetune.png'))
        histories['SSL-Pretrained\nLSTM'] = h
    else:
        print(f"(No {ssl_ft_path} yet -- skipping.)")

    if len(histories) >= 2:
        plot_comparison(histories, os.path.join(out_dir, 'comparison_val_loss.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), '..')
    models_dir = args.models_dir or os.path.join(os.path.dirname(__file__), 'models')
    out_dir = args.out_dir or os.path.join(root, 'runs', 'figures_single_drone')
    run(models_dir, out_dir)
