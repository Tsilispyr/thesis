# Gyroscope Unit Reconciliation - Finding (imu vs px4)

**Tool**: `ai_backend/data_processing/reconcile_gyro_units.py`
**Result**: not reconciled. `imu_data.csv` and `px4_raw/*.ulg` remain two separate,
non-combinable sources under `verify_unit_consistency()`. This is not a regression or a bug to
fix later, it's a checked, honest negative result, the same treatment as the SSL fine-tuning
finding in `ai_backend/models/ssl_pretext.py`.

## The sweep

`verify_unit_consistency()` blocks combining `imu` and `px4` because their gyro columns'
per-axis std ratio exceeds its 3x threshold. The only physically motivated candidate conversion
between the two is rad/s <-> deg/s (PX4's `sensor_combined.gyro_rad[i]` is documented rad/s;
`imu_data.csv`'s scale was never independently confirmed). The diagnostic swept both directions
against the actual recorded std of both sources:

| imu candidate | px4 candidate | x ratio | y ratio | z ratio | verdict |
|---|---|---|---|---|---|
| as-is | as-is | 251.45x | 359.55x | 211.31x | fail |
| as-is | x57.2958 (rad->deg) | **4.39x** | **6.28x** | **3.69x** | fail (closest) |
| /57.2958 (deg->rad) | as-is | 4.39x | 6.28x | 3.69x | fail (closest, same pair) |

Every other combination in the full grid (9 pairs total, see script output) is off by 3-6 orders
of magnitude, clearly wrong units, not close calls. The rad/s<->deg/s pair is the only
physically sensible candidate, and it gets closest, but still fails the 3x threshold on every
axis (3.69x-6.28x, i.e. roughly 1.2x-2.1x over the line depending on axis).

## Why this isn't "just tighten the threshold"

The 3x threshold isn't arbitrary, `verify_unit_consistency()`'s docstring notes real unit
mismatches historically blow past it by 50-100,000x (exactly what the raw as-is/as-is row above
shows: 211-360x). A 3.7-6.3x gap sitting just above the line, on the *one* physically correct
conversion, is a different kind of signal: not "wrong units," but "right units, different
distributions." Loosening the threshold to paper over this would defeat the guard's actual
purpose.

## What the gap actually is: flight-profile dynamics, not a units bug

Two additional checks rule out simulation artifacts as the cause:

1. **Not a clamp/saturation artifact.** `imu_data.csv`'s gyro values do top out near +-100
   (max 94.5-100.0 across axes), which could suggest an artificial clamp. But only 0.00-0.22% of
   samples actually sit near that ceiling (`|value| > 95`), the high std isn't from pinning at a
   limit, it's from sustained moderate-to-high values throughout the flight (90th-percentile
   `|value|` is 33.6-39.8 across axes).
2. **A real difference in flight character.** After converting px4 to deg/s, its gyro std is
   3.5-6.3 (mean near zero, calm bench-verification flights per `SOURCE.md`: "Quadrotor, zero
   recorded errors, EKF2 estimator"). `imu_data.csv`'s gyro std is 20-23 in the same units, the
   simulated flight spends much more of its duration at moderate-to-high angular rates. That's
   consistent with `imu_data.csv` being a maneuver-heavy synthetic flight (turns/orbits exercised
   deliberately to stress the sensor pipeline during development) versus PX4's calmer real-world
   verification flights, not a scale error in either recording.

## Implication for the ablation matrix (Phase A5)

`imu` and `px4` stay separate, non-fused sources, as they already were before this check.
The ablation matrix (Phase A5) proceeds with **imu as the primary training/evaluation source**
and **px4 as the existing held-out chained-trajectory realism check** (as already used in
`evaluate_trajectory.py`), 2 usable roles, not the 3-source fused matrix that would have existed
had this reconciled. No code path changes: `dataset_parser.py`'s `reconciled=` set is
intentionally left untouched, since nothing here was actually verified-and-handled in the sense
that parameter requires, registering it there would suppress a real, still-open discrepancy.

## If this is revisited later

The gap is small enough (under 2x over threshold on the best candidate) that it's plausible
future PX4 logs flown with more aggressive maneuvers, or a larger `imu_data.csv` capture with
calmer segments included, could close it naturally without any code change, worth re-running
`reconcile_gyro_units.py` if either dataset grows.
