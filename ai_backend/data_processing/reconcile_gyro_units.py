"""Diagnostic: is the imu/px4 gyroscope std mismatch a fixable unit-conversion
bug, or a genuine scale/dynamics difference between sources? (Track A, Phase A1
of the coursework-grounded extension roadmap.)

`verify_unit_consistency()` currently blocks combining 'imu' and 'px4' under
one shared scaler because their gyro columns' std ratio exceeds the 3x
threshold. This script sweeps the physically meaningful candidate conversions
(rad/s <-> deg/s, since PX4's `gyro_rad` is documented rad/s and imu_data.csv's
scale was never independently confirmed) against both sources' actual std,
using verify_unit_consistency()'s own ratio formula, and reports whether any
candidate reconciles them under the threshold.

Run from the AI_Recovery folder:
    python ai_backend\\data_processing\\reconcile_gyro_units.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.dataset_parser import (
    FEATURE_COLS, load_and_clean_imu_data, load_and_clean_px4_data, UNIT_MISMATCH_THRESHOLD,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAD2DEG = 57.29577951308232

GYRO_COLS = ['imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']

# Candidate conversions to try on each source, relative to its as-recorded
# value. 1.0 = leave as-is. Only rad<->deg is physically motivated here (both
# sources already store calibrated physical units, not raw ADC ticks, so a
# full-scale-sensitivity guess doesn't apply the way it would for raw sensor
# registers) but the full imu x px4 grid is swept anyway so the result isn't
# assumed, it's checked.
CANDIDATES = {
    'as-is': 1.0,
    '*57.2958 (rad->deg)': RAD2DEG,
    '/57.2958 (deg->rad)': 1.0 / RAD2DEG,
}


def std_ratio(std_a, std_b):
    if std_a < 1e-9 or std_b < 1e-9:
        return float('nan')
    return max(std_a, std_b) / min(std_a, std_b)


def main():
    print("Loading imu_data.csv ...")
    imu_df = load_and_clean_imu_data(os.path.join(BASE_DIR, 'datasets', 'imu_data.csv'))
    print("Loading px4_raw/*.ulg ...")
    px4_df = load_and_clean_px4_data(os.path.join(BASE_DIR, 'datasets', 'px4_raw'))

    print(f"\nimu_data.csv: {len(imu_df)} rows")
    print(f"px4 combined: {len(px4_df)} rows\n")

    imu_std = {c: imu_df[c].std() for c in GYRO_COLS}
    px4_std = {c: px4_df[c].std() for c in GYRO_COLS}

    print("Raw (as-recorded) stats:")
    for c in GYRO_COLS:
        print(f"  {c}: imu std={imu_std[c]:.4f}   px4 std={px4_std[c]:.4f}   "
              f"ratio={std_ratio(imu_std[c], px4_std[c]):.2f}x")

    print(f"\nSweeping candidate conversions (threshold = {UNIT_MISMATCH_THRESHOLD}x):\n")
    header = f"{'imu candidate':<22}{'px4 candidate':<22}" + \
             "".join(f"{c + ' ratio':<16}" for c in GYRO_COLS) + "verdict"
    print(header)
    print("-" * len(header))

    best = None
    for imu_name, imu_scale in CANDIDATES.items():
        for px4_name, px4_scale in CANDIDATES.items():
            ratios = []
            for c in GYRO_COLS:
                r = std_ratio(imu_std[c] * imu_scale, px4_std[c] * px4_scale)
                ratios.append(r)
            worst = max(ratios)
            passed = worst <= UNIT_MISMATCH_THRESHOLD
            verdict = "PASS" if passed else "fail"
            row = f"{imu_name:<22}{px4_name:<22}" + \
                  "".join(f"{r:<16.2f}" for r in ratios) + verdict
            print(row)
            if best is None or worst < best[0]:
                best = (worst, imu_name, px4_name, ratios)

    print(f"\nBest candidate: imu={best[1]}, px4={best[2]}, "
          f"worst-axis ratio={best[0]:.2f}x (threshold {UNIT_MISMATCH_THRESHOLD}x)")

    if best[0] <= UNIT_MISMATCH_THRESHOLD:
        print("RESULT: RECONCILED - a unit conversion brings both sources under threshold.")
    else:
        print("RESULT: NOT RECONCILED by a simple unit conversion, "
              "remaining gap is most likely a genuine dynamics/flight-profile "
              "difference between the two sources, not a units bug. See "
              "GYRO_UNIT_ANALYSIS.md for the full interpretation.")

    # Distributional context: is the imu-side gyro clamped/saturated (a sim
    # artifact) or just a more aggressive flight profile (real dynamics)?
    print("\nDistributional check (imu_data.csv, % of samples with |value| > 95, "
          "near its own +-100-ish max):")
    for c in GYRO_COLS:
        near_max = float((imu_df[c].abs() > 95).mean() * 100)
        p90 = float(imu_df[c].abs().quantile(0.90))
        print(f"  {c}: {near_max:.2f}% near max, 90th-percentile |value| = {p90:.2f}")


if __name__ == '__main__':
    main()
