# -*- coding: utf-8 -*-
"""Regenerate Figure 19 (swarm connectivity method comparison) with the real
method names instead of the generic "Method A/B/C" placeholders.

Supervisor comment [235] asked which method the purple bar corresponds to. The
stored results are keyed by placeholder names, and two different models both
carried the label "Method C", which is why the figure could not be read against
Πίνακας 8.

This reads runs/swarm_connectivity/comparison.json and re-plots it. No training
is re-run, so every value is byte-identical to the reported results. Bars are
ordered by descending Macro-F1 so the figure reads in the same order as
Πίνακας 8; colour stays bound to the method, not to its rank.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import (COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE,
                        COLOR_ACCENT_GREEN, COLOR_ACCENT_PURPLE)

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(os.path.dirname(_HERE), 'runs', 'swarm_connectivity')

# stored key -> (display label, colour). Colour follows the entity, so each
# method keeps the hue it had in the original figure even though bars move.
NAMES = {
    'Method A\n(graph features)': ('Χειροποίητα\nχαρακτηριστικά (MLP)', COLOR_ACCENT_BLUE),
    'Method B\n(Node2Vec)':       ('Node2Vec (MLP)',                    COLOR_ACCENT_ORANGE),
    'Method C\n(GCN)':            ('GCN',                               COLOR_ACCENT_GREEN),
    'Method C\n(GIN)':            ('GIN',                               COLOR_ACCENT_PURPLE),
}


def main():
    with open(os.path.join(_RUNS, 'comparison.json'), encoding='utf-8') as fh:
        data = json.load(fh)

    rows = []
    for key, rec in data.items():
        label, colour = NAMES[key]
        rows.append((rec['mean'], rec['ci_lo'], rec['ci_hi'], label, colour))
    rows.sort(reverse=True)                      # descending, matching Πίνακας 8

    means = [r[0] for r in rows]
    errs = [[m - r[1] for m, r in zip(means, rows)],
            [r[2] - m for m, r in zip(means, rows)]]
    labels = [r[3] for r in rows]
    colours = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=colours, zorder=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f'{m:.3f}'.replace('.', ','),
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold', zorder=4)
    ax.set_ylabel('Test Macro-F1')
    ax.set_ylim(0, 1.0)
    # Greek decimal convention, matching the thesis body and tables
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    out = os.path.join(_RUNS, 'figures', 'method_comparison.png')
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved -> {out}')
    for m, lo, hi, lab, _ in rows:
        print(f'   {lab.replace(chr(10), " "):32s} {m:.4f}  [{lo:.4f}, {hi:.4f}]')


if __name__ == '__main__':
    main()
