# SD-UAV Dead Reckoning - Training Analysis

**Last Updated:** 2026-08-14 (Track A ablation-matrix extension and follow-up experiments -- see Parts 4e-4i)
**Model:** 6-architecture ablation matrix -- LSTM, SSL-LSTM, Transformer, SSL-Transformer, RandomForest, GBT (see Parts 4a-4h), plus a classical EKF baseline. Production default is the LSTM (input=14, hidden=64, layers=2, output=3); the Transformer is the statistically-best architecture (Part 4h) and is selectable live but not the default.
**Task:** Predict Δlat / Δlon / Δalt from a 10-step sensor window (Dead Reckoning)

> **Retraction notice (reads first, on purpose):** every result in this document dated Run #7 or earlier (val loss 0.2264 for Run C, 0.3325 for Run D, and the "8-feature" historical log at the bottom) has been **retracted**. Those numbers were produced by a pipeline with three independent data-integrity bugs, all now found and fixed - see Part 0. They should not be cited anywhere (thesis included). The numbers in this revision (Run #10) are the first ones in this file's history that are honest, leakage-free, and independently cross-validated against `train.py`'s separate implementation.

---

## Part 0 - What Changed Since Run #7 (read this before anything else)

Three real bugs were found in the training pipeline between Run #7 and Run #10, each discovered while investigating why the previous "best" number looked suspiciously good. Full technical narrative and code-level detail: `ai_backend/data_processing/ANALYSIS.md`. Summary:

1. **Sequence-shuffle leakage.** The pipeline used to shuffle overlapping sliding-window sequences *before* splitting into train/val. Adjacent windows share 90% of their timesteps, so this put near-duplicate samples on both sides of the split, producing an artificially low validation loss. Fixed: sequences are now kept in strict temporal order and the split is a plain slice.
2. **Scaler fit-before-split leakage.** `StandardScaler` used to be fit on the *entire* feature/target array before the train/val split - meaning every historical run's normalization statistics were computed partly from validation data. Fixed: the scaler is now fit on the training slice only, and the validation slice is transformed (never re-fit) with those parameters.
3. **Domain-shift scaler contamination (Run D).** Concatenating `nav` (GPS-degree positions) and `imu` (relative-metre positions) without unit conversion created one extreme outlier delta at the join boundary, which inflated the fitted scaler's standard deviation and artificially shrank the normalized error for every other row. Fixed with per-source delta computation before concatenation, source-boundary-aware windowing (`_src_id`), and a new standing guard, `verify_unit_consistency()`, that now blocks any future source combination whose per-column scale differs by more than 3× without an explicit, logged reconciliation.

A fourth, non-code finding compounds Run D's story: **`uav_navigation_dataset.csv` ("nav") was independently found to be physically impossible synthetic data** - a single timestep implies ~1,000–2,200 m of movement while the file's own `speed` column simultaneously reports 7–30 m/s for that same timestep. This is not a unit-conversion problem; the position column has no real spatial continuity for `.diff()` to extract in the first place. `nav` is now formally deprecated (`load_combined_dataset()` prints a warning if requested) and has been replaced as the project's second data source by real hardware flight logs from the public PX4 Flight Review database (`review.px4.io`, CC-BY) - see Part 1a.

**Practical consequence for this document:** Run C (imu-only) is the only run whose number should ever be cited as "the model's performance." Runs A, B, and D are kept in the pipeline and in this document deliberately, as a documented methodology lesson, not as competing candidates for "best model."

---

## Pipeline Run #10 Results (2026-08-04) - Current, Honest, Trustworthy

| Run | Sources | Norm | Sequences | Val Loss | Time | Verdict |
|-----|---------|------|-----------|----------|------|---------|
| A | nav only | No | 4,754 | 447,521.29 | ~2.2s | Unusable - raw GPS scale |
| B | nav only | Yes | 4,754 | 0.9539 | ~2.5s | Caution - flat, nav has no learnable position signal (§Run D discussion) |
| C | imu only | Yes | 544,753 | **0.7348** | ~249s | **Best - the only trustworthy number in this table** |
| D | nav + imu | Yes | 549,507 | 0.0000 | ~252s | Invalid - scaler-contamination artifact, **not a real result, kept for the record** |

The dashboard's own automatic winner-selection logic (`ml_pipeline.py::_select_trustworthy_winner()`) now explicitly prefers `C_imu_norm` regardless of which run reports the numerically lowest loss - added specifically so a future rerun can never again silently promote Run D (or another nav-tainted run) to "best model" the way naive `min()` selection would.

> Historical values for reference (not to be trusted, see Part 0): Run #5 reported C=0.2058; Run #7 reported C=0.2264, D=0.3325. Both were leakage artifacts.

---

## Part 1 - Data Pipeline: How It Works Step by Step

### Step 1: Dataset Audit

Three sources now exist (was two): the two original CSVs plus real PX4 flight logs.

**`uav_navigation_dataset.csv` (nav) - DEPRECATED, do not use for new work**
- 5,000 rows × 15 columns
- Columns: `timestamp`, `latitude`, `longitude`, `altitude`, `imu_acc_x/y/z`, `imu_gyro_x/y/z`, `lidar_distance`, `speed`, `wind_speed`, `battery_level`, `obstacle_detected`
- Its `latitude`/`longitude`/`altitude` columns are **not a real trajectory** (Part 0) - per-row-randomized values arranged behind sequential timestamps, matching the schema of a Kaggle synthetic dataset ("UAV Autonomous Navigation Dataset", ziya07) rather than a physically simulated or recorded flight. Its other columns (`lidar_distance`, `wind_speed`, `battery_level`, `obstacle_detected`) are not affected by this finding and may still be usable for whatever they were originally intended for.

**`imu_data.csv` (imu) - the trustworthy, primary source**
- 544,763 rows × 16 columns
- Columns: `time`, `accel_x/y/z`, `gyro_x/y/z`, `mag_x/y/z`, `pos_x/y/z`, `roll`, `pitch`, `yaw`
- Coordinate system: relative metres from an arbitrary origin point
- Sample rate: uniform ~240 Hz (`dt` std ≈ 1.4×10⁻¹³ s - essentially perfectly regular), smooth frame-to-frame position changes, implied speed maxes at ~35 m/s (no clipping against the 200 m/s guard). Independently verified physically consistent.

**`datasets/px4_raw/*.ulg` (px4) - real hardware flight logs, 46 total across two batches**
- Batch 1 (6 flights): downloaded from the public PX4 Flight Review database (`review.px4.io`, CC-BY, Dronecode Foundation), filtered for zero logged errors, EKF2 estimator, Quadrotor, 120–600s duration. Confirmed genuine hardware (`sys_hw = PX4_FMU_V6X`), human-uploaded via the web UI, not simulation output. Same-day, blank-description, bench/validation-style flights.
- Batch 2 (40 more flights, see `datasets/px4_raw/SOURCE.md`): same filters plus `flight_modes` containing `Mission` (id `3`), biasing toward flights with real translational dynamics - the exact next step this file had scoped but never executed. A real bug was found and fixed getting there: `flight_modes` is a list of PX4 nav_state *integer* codes, not the string this project's own docs originally assumed. Re-running the standalone `sources=px4` training arm on the combined 46-log corpus (42 parse cleanly) took val loss from ~1.5 ("worse than predicting the mean") to **0.2768** - confirms the original result was a data-quantity/diversity problem, not structural.
- Parsed via `pyulog`, multi-rate topics aligned with `pd.merge_asof` onto `sensor_combined`'s timeline. Position comes from `vehicle_local_position` (EKF2's fused estimate, metres, NED) rather than `sensor_gps`, because the first log checked had `sensor_gps.latitude_deg` uniformly `0.0` (an indoor/optical-flow test flight, common in this database).
- **Not currently combinable with `imu`** - `verify_unit_consistency()` correctly blocks it: 13 columns exceed the 3× cross-source scale-ratio threshold, including `imu_gyro_x/y/z` at 211–360× (confirming a real rad/s-vs-different-scale unit mismatch, broader than originally suspected). A physically-motivated synthetic-diversity augmentation (rotation + state-matched segment splicing, real measured values only) was tested as a separate lever within the standalone px4 corpus itself (not requiring the imu/px4 merge) - 3-seed mean 0.2587 vs. 0.2741 real-only, a consistent-direction ~5.6% better point estimate but CIs still overlap at n=3, not yet statistically proven. Full numbers: `SOURCE.md`, `runs/px4_synthetic_diversity/`.

The `imu` dataset remains **109× larger** than `nav`, but the size difference is now the *secondary* reason to prefer it - the primary reason is that `nav`'s position column isn't usable at all (Part 0).

---

### Step 2: Data Cleaning

#### 2a. Column Renaming (IMU dataset)

Unchanged from the original pipeline - the imu CSV uses hardware-native column names, remapped to the canonical schema:

```
accel_x → imu_acc_x    accel_y → imu_acc_y    accel_z → imu_acc_z
gyro_x  → imu_gyro_x   gyro_y  → imu_gyro_y   gyro_z  → imu_gyro_z
pos_x   → latitude     pos_y   → longitude     pos_z   → altitude
```

#### 2b. Obstacle-Avoidance Row Filter (NAV only)

Unchanged in mechanism: rows where `obstacle_detected == 1` are dropped (236 rows, 4.7% of nav) because those trajectories are driven by collision-avoidance logic, not normal flight dynamics. Kept in the pipeline even though `nav` itself is deprecated, since Run A/B still exist as the documented "why normalization matters" / "why this file doesn't work" control cases.

#### 2c. dt Feature Engineering - unchanged

Dead Reckoning depends on time integration; `dt` is computed per-source as documented previously (nav: timestamp diff, default 0.1s; imu: time-column diff, default = median dt ≈ 0.004s at 240Hz). PX4's `dt` is computed the same way from its microsecond `timestamp` column, divided by 1e6.

#### 2d. Speed Feature Engineering - unchanged

`speed = sqrt(dx² + dy² + dz²) / dt`, computed from position diffs, per source, before any concatenation.

#### 2e. Source-Boundary Tagging - NEW since Run #7

Every source-loading function now tags its rows with `_src_id` (nav=0, imu=1, each px4 flight = 1000+i). `create_dead_reckoning_dataset()`'s window-building loop skips any window whose first and last row have different `_src_id`s - this is what makes it safe to eventually combine multiple physically-independent recordings (or multiple PX4 flights) without accidentally building a window that spans two unrelated flights. This mechanism, plus per-source delta computation (Δlat/lon/alt computed independently *inside* each `load_and_clean_*()` function, never across a concatenation boundary), is the direct fix for the Run D bug in Part 0.

#### 2f. Missing Orientation/Magnetometer Columns (NAV only) - unchanged, now lower-stakes

Nav rows still zero-fill `roll/pitch/yaw/mag_x/y/z` (not present in that CSV). Since `nav` is deprecated for training, this approximation now matters only for reproducing the historical negative result, not for any live modeling decision.

---

### Step 3: Normalization

#### Why Normalization Is Mandatory - unchanged

Raw IMU feature scales differ by up to 253× (`dt` std≈0.10 vs. `imu_gyro_z` std≈25.23) - without normalization the largest-magnitude features dominate gradient updates.

#### StandardScaler Mechanics - corrected

> **Correction to a claim in earlier revisions of this document:** an earlier version of this section stated "fit on training data only... prevents data leakage" as if this were already how the pipeline worked. It was not, until Session 5 - see Part 0, Bug 2. The scaler is now genuinely fit on the training split only (`.fit_transform()` on `X_train`/`y_train`) and applied via `.transform()` (never re-fit) to the validation split, in `dataset_parser.py::create_dead_reckoning_dataset()`. Every number in this revision's Part 2 was produced under that corrected discipline.

For each feature column `f`: `f_normalized = (f - μ_f) / σ_f`, with μ, σ computed on the train split only.

#### Run #10 Normalization Statistics (14 features, fitted on IMU dataset)

| Feature | Raw Mean | Raw Std | After: Mean | After: Std |
|---------|----------|---------|-------------|------------|
| imu_acc_x | -0.0026 | 13.02 | 0.000 | 1.000 |
| imu_acc_y | 0.0009 | 13.61 | 0.000 | 1.000 |
| imu_acc_z | 0.0020 | 4.71 | 0.000 | 1.000 |
| imu_gyro_x | 0.2164 | 22.10 | 0.000 | 1.000 |
| imu_gyro_y | 2.9495 | 23.69 | 0.000 | 1.000 |
| imu_gyro_z | 1.3942 | 25.23 | 0.000 | 1.000 |
| roll | 0.0225 | 1.62 | 0.000 | 1.000 |
| pitch | 0.0279 | 0.63 | 0.000 | 1.000 |
| yaw | 0.0081 | 1.59 | 0.000 | 1.000 |
| mag_x | 0.0247 | 0.54 | 0.000 | 1.000 |
| mag_y | -0.0193 | 0.54 | 0.000 | 1.000 |
| mag_z | -0.1725 | 0.63 | 0.000 | 1.000 |
| speed | 15.048 | 10.15 | 0.000 | 1.000 |
| dt | 0.0132 | 0.099 | 0.000 | 1.000 |

(These raw statistics are unchanged from earlier revisions - they describe the raw data, which hasn't changed - only the *procedure* for fitting the scaler was corrected.)

#### Why the Scaler Is Saved to Disk - unchanged

`scaler_X_{tag}.pkl` / `scaler_y_{tag}.pkl` are mandatory runtime companions to the `.pth` weights; inference must denormalize with the exact same parameters used at training time.

---

### Step 4: Feature Engineering

14 features and rationale - unchanged from the original design (see table in earlier revisions / `ai_backend/data_processing/ANALYSIS.md`).

#### Feature Correlations with Targets (Run #10)

```
delta_lat: imu_gyro_z=0.008, imu_gyro_x=0.008, dt=0.002
delta_lon: imu_gyro_y=0.009, imu_gyro_x=0.004, dt=0.003
delta_alt: imu_gyro_x=0.011, dt=0.003, imu_gyro_z=0.002
```

Uniformly low, as before, and still not a concern - dead reckoning is a temporally-integrated, nonlinear process; Pearson correlation only sees first-order, instantaneous relationships. This is direct, honest evidence for why an LSTM (which integrates information across the whole 10-step window) is the right tool, not a simpler per-timestep regressor.

**New observation, not present in earlier revisions:** `pitch` and `mag_x` correlate at 0.99, and `roll`/`mag_y` at −0.75 (feature-feature, not feature-target). This is a known limitation of the simulated magnetometer model behind `imu_data.csv` and `drone_body.gd`'s telemetry: a fixed world-frame magnetic field rotated purely by attitude, with no real geographic declination or independent noise process. It means `mag_x`/`mag_y` currently carry substantially redundant, not independent, information relative to `roll`/`pitch` - worth flagging for anyone using magnetometer features expecting them to be an independent correction source.

---

### Step 5: Sequence Building (Sliding Window) - unchanged, now boundary-aware

Window size 10, `(10, 14) → (3,)`. **New**: windows are additionally dropped if they would straddle a `_src_id` boundary (Step 2e) - relevant once more than one physically independent source/flight is combined.

- **Nav sequences:** 4,764 rows → 4,754 sequences.
- **IMU sequences:** 544,763 rows → 544,753 sequences.

---

## Part 2 - Training Run Analysis (Run #10, honest numbers)

### Run A - Nav Only, No Normalization (baseline)

**Result:** Val Loss final **447,521.29** (MSE on raw GPS-degree deltas), essentially flat/non-converging across 10 epochs. The exact number differs from earlier revisions' 11,451.3 (different pipeline state at the time), but the conclusion is identical and, if anything, stronger now: without normalization, gradients cannot make useful progress at this scale. **Verdict unchanged: normalization is architecturally required, not optional.**

### Run B - Nav Only, Normalized

**Result:** Val Loss final **0.9539**, flat from epoch 1 (previously reported 1.0107 - same conclusion, refined number).

**Revised explanation** (this is the one substantive correction to this section's narrative): earlier revisions attributed this plateau to `nav` being merely **data-starved** (only 4,754 sequences, more parameters than the dataset can train effectively). That explanation is no longer the primary one. Given Part 0's finding that `nav`'s position column is not a real trajectory, the flat ~0.95-1.0 loss (statistically indistinguishable from "predict the mean") is now understood as a direct symptom of there being **no real position signal to learn in the first place** - more data or more epochs would not have fixed this, because the delta-position target derived from `nav` is close to random noise with respect to the IMU input features. Data volume was never the actual bottleneck for this particular file.

### Run C - IMU Only, Normalized - the only trustworthy result

**Result:** Val Loss final **0.7348** (epoch-by-epoch: 0.7442 → 0.7322 → 0.7114 → 0.7189 → 0.7262 → 0.7056 → 0.7111 → 0.7300 → 0.7546 → 0.7348), training loss falling smoothly and continuously from 0.6965 to 0.2080.

This replaces the retracted 0.2264. The gap between the two numbers (0.73 vs. 0.23) is entirely explained by the leakage bugs in Part 0 - this is not a case of the model getting worse, but of the measurement finally being honest. A visible train/val gap (0.21 vs. 0.73 by epoch 10) is now visible and expected - mild overfitting, independently confirmed by `train.py`'s separate implementation (best epoch 1, val=0.7189, before early stopping fires at epoch 6) and consistent across every training script in the project. **This cross-validation between two independently-written training loops (`ml_pipeline.py` and `train.py`) reaching the same ballpark number is itself part of why this result is trustworthy, not just "the new number."**

**Why more epochs no longer straightforwardly help:** unlike the earlier revision's claim ("still converging at epoch 10... extending to 20 or 50 epochs would further reduce val loss"), the honest curve shows val loss *bottoming out around epoch 1 and drifting upward afterward* - see `train.py`'s early-stopping results and `runs/figures_single_drone/train_supervised.png`. This is why early stopping (Part 4a) was added rather than simply running more epochs.

### Run D - Nav + IMU Combined, Normalized - retracted, kept as a warning

**Result:** Val Loss final **0.0000** (previously reported as 0.3325, "domain shift penalty" - that framing is now understood to be incomplete).

**Revised explanation:** the original framing (`nav`'s GPS-degree deltas vs. `imu`'s metre-scale deltas producing a scaler-contaminating outlier at the concatenation boundary) is still mechanically correct and is exactly what `verify_unit_consistency()` was built to catch - the pipeline's own `step9_run_D()` deliberately bypasses that guard (concatenates `nav`+`imu` directly, not through the checked `dataset_parser.load_combined_dataset()` path) specifically so this failure mode stays visible and reproducible in the dashboard as a teaching example. What changed since the earlier revision is the *severity* of the conclusion: because `nav`'s position data is now known to be fake (Part 0), Run D isn't just "penalized by a fixable unit mismatch" - it is combining a valid source with an invalid one, so **no future unit-conversion fix would make Run D a legitimate result**. It is retired permanently from any "candidate for best model" framing; `ml_pipeline.py`'s own winner-selection logic now enforces this automatically (see the Run #10 results table above).

---

## Part 3 - Reinforcement Learning Evaluation (rewired and re-evaluated, see AI_RECOVERY_EXECUTION_PLAN.md §19.11)

### Hand-coded seek formula vs. trained RL policy

The numbers below **replace** an earlier version of this table (rule-based 100% / PPO 28% on a
`[0,10]×[0,10]` toy grid with a fixed target) - that comparison, re-read while doing this work,
turned out to be two independent, mutually disconnected toy simulations, neither touching the real
system's actual seek-bias formula. `eval_metrics.py` was rebuilt to route both methods through the
real `recovery_orchestrator._hand_coded_seek` formula and one shared, realistic environment
(`RecoveryPolicyEnv`), the first genuine apples-to-apples comparison this project has had for this
piece:

| Method | Success Rate | Avg Steps | Mean \|\|seek\|\| |
|--------|-------------|-----------|-----|
| Hand-coded formula | 9.0% | 77.0 | 1.33-1.50 m/s |
| RL policy (10K timesteps) | 2.0% | 79.8 | 0.10-0.25 m/s |

RL's state space is now the reconstructed position (LSTM/EKF's own believed position) plus EKF
confidence, not raw GPS coordinates - the Phase-3 redesign this section previously called for is
done. Timestep budget stayed at 10K (not scaled to 500K-1M) by explicit user call, so any
improvement is attributable to the redesigned environment/reward, not more training compute. Honest
result: RL's raw arrival success rate is *lower* than the hand-coded formula's, but it learned to
command corrections roughly 6-14x smaller on average - a real, large-sample-verified behavioral
difference (n>1500 steps/confidence-bin), not RL simply mimicking the formula at a lower gain. A
controlled acceptance test (holding position/goal/velocity fixed, varying only injected confidence)
confirms the intended mechanism is genuinely present: `‖seek‖` shrinks from 0.116 to 0.093 m/s as
uncertainty rises from 0.3m to 9.0m - that directional effect is masked in naturalistic rollouts by
confidence, distance-to-goal, and elapsed time all growing together within an episode. Reported as
what it is: the mechanism works, demonstrated in isolation; the training budget was not enough for
it to dominate naturalistic behavior yet. Wired into the live hierarchy as an opt-in replacement for
the hand-coded formula (`set_navigation_policy()` / `--rl-seek`), off by default.

---

## Part 4 - Model Architectures

### 4a. LSTM Dead Reckoning Model (unchanged)

```python
class DeadReckoningLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, output_size=3):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
```

Input `(batch, 10, 14)`, output `(batch, 3)`, ~65K parameters. **New since Run #7:** both `train.py::train_model()` and the SSL fine-tuning path (below) now train this architecture with **early stopping** (patience-based, keyed on validation loss, plus `ReduceLROnPlateau`) instead of a fixed epoch count - necessary because the honest val curve (Run C, above) bottoms out around epoch 1 and rises afterward. Best-checkpoint `state_dict` is deep-copied and restored at the end of training, so the saved model is the best-validation checkpoint, not simply whichever epoch happened to run last.

### 4b. NEW - MaskedLSTMAutoencoder (Self-Supervised Pretraining)

`ai_backend/models/ssl_pretext.py`. An encoder architecturally identical to `DeadReckoningLSTM.lstm` plus a per-timestep linear decoder (`nn.Linear(64,14)`), trained to reconstruct randomly-masked whole timesteps (30% mask ratio) within a window - no labels required. The pretrained encoder's weights transfer directly into a fresh `DeadReckoningLSTM` via `load_state_dict()` (identical architecture, no remapping) for fine-tuning on the normal Δposition regression task.

**Results:** pretraining itself works cleanly - validation loss falls from 0.181 to a best of 0.157 (epoch 13), tracking training loss the whole way with no overfitting gap. Fine-tuning from that pretrained encoder, however, reaches best val=0.8097 (epoch 1) - **worse** than the from-scratch supervised baseline's 0.7189. This is reported as a genuine negative ablation result: for this architecture, data, and masking configuration, masked-reconstruction pretraining does not improve - and slightly hurts - the downstream regression task. Independently confirmed at the trajectory level (Part 4d): the SSL-pretrained model's real-world position error (29.5 m) is worse than the from-scratch model's (26.3 m), the same ranking as the normalized-loss comparison.

### 4c. NEW - DeadReckoningEKF (Classical Baseline)

`ai_backend/models/ekf_baseline.py`. A real, reusable 6-state (`[px,py,pz,vx,vy,vz]`) Kalman filter - strapdown inertial integration of specific-force acceleration plus an optional barometric-altitude correction. No learned weights. Serves two roles: (1) the non-ML floor in any dead-reckoning comparison, and (2) the tertiary layer (behind an RL policy and the SSL/LSTM estimator) of the project's fail-safe hierarchy - "degrade toward simplicity": RL Policy → SSL/LSTM → Classic EKF → deterministic kinematic breadcrumb (implemented Godot-side in `simulation/scripts/breadcrumb_recovery.gd`, outside the scope of this Python-training document).

Verified via a synthetic smoke test: a stationary drone integrated for 5s at 10Hz with noisy accelerometer input drifts to 5.20 m of 1-sigma position uncertainty per axis under pure inertial integration; a periodic (2Hz) barometer correction reduces the *vertical-axis* uncertainty specifically to 0.26 m, leaving the uncorrected horizontal axes unchanged - exactly the expected behavior for a filter with only an altitude measurement.

### 4d. NEW - Real Trajectory Evaluation (`ai_backend/evaluate_trajectory.py`)

Rather than relying only on normalized, per-window MSE (which doesn't say how many metres off a model ends up), this reconstructs a real continuous trajectory by **chaining** a model's per-window delta predictions over a genuine, held-out, contiguous 3,000-sample (~12.5s) segment of `imu_data.csv`'s validation tail - errors compound exactly as they would in deployment. Two non-learned baselines are computed over the *same* real data using the EKF class above: with barometer correction (classical EKF) and without (pure inertial integration).

| Method | Final error (m) | Mean error (m) | Max error (m) |
|---|---|---|---|
| Supervised LSTM | **26.3** | **16.5** | 36.9 |
| SSL-Pretrained LSTM | 29.5 | 19.0 | 42.2 |
| Classical EKF | 132.5 | 66.1 | 132.5 |
| Pure Inertial Integration | 143.0 | 71.4 | 143.0 |

Both LSTM variants dramatically outperform the classical baselines, which grow ~linearly (unbounded drift) and wander ~90 m off the true horizontal path within 12.5 s. The EKF barely beats pure inertial integration here because it only corrects altitude - no horizontal correction source exists yet - so the horizontal drift that dominates this segment goes uncorrected. This is the strongest evidence in the project that a "mediocre-looking" normalized loss (~0.73) corresponds to genuinely useful real-world performance. Full figure-by-figure discussion: `runs/figures_single_drone/ANALYSIS.md`.

### 4e. NEW -- DeadReckoningLSTMUncertainty (Gaussian NLL Uncertainty Head)

`ai_backend/models/dead_reckoning_model_uncertainty.py`. Same LSTM backbone, regression head widened to output mean + log-variance, trained with `gaussian_nll_loss()` (Kendall & Gal 2017). A real, non-obvious finding: validation NLL's best epoch (1) and validation MSE-on-the-mean's best epoch (6) disagree, because the variance head shrinks faster than accuracy improvement warrants (increasing overconfidence, which NLL penalizes and plain MSE cannot see). Calibration, checked directly against actual chained error on the held-out segment: **r = 0.203** (weakly positive, real but not yet a usable confidence signal) versus the EKF's native covariance at r = 0.999 (a near-deterministic function of its process model, not a fair comparison of "which uncertainty is better," just of how much harder data-driven confidence estimation is). Retrained three times this pass; chained error and calibration both varied more run-to-run (29.44m/31.69m/38.54m, r=0.203/0.212/0.119) than the plain LSTM's own reproducibility -- a preview of the seed-variance finding formalized in Part 4h below.

### 4f. NEW -- DeadReckoningTransformer (+ SSL Variant)

`ai_backend/models/dr_transformer.py` / `ssl_pretext_transformer.py`. `d_model=64` (matched to the LSTM's `hidden_size` for a fair comparison), `nhead=4`, 4 pre-norm encoder layers, hyperparameters taken from the user's own Computer Vision coursework's `TinyViT` reference. A 10-step IMU window has no 2-D patch structure, so `PatchEmbed`'s `Conv2d` is replaced by a per-timestep `Linear` projection. Time-series data augmentation (jitter/scale/time-warp) and dropout are included from the start given this project's single-continuous-flight data-diversity limitation.

**Result (at the validated `batch_size=64`, see Part 4h for why batch size specifically mattered here): windowed val loss 0.3545, chained-trajectory error 12.46m final / 9.75m mean -- beats the from-scratch LSTM outright.** Attention weights, extracted over 500 real windows: only layer 0 shows real temporal selectivity (U-shaped across the window); layers 1-3 are statistically indistinguishable from uniform -- an honest limitation, the Transformer's advantage does not appear to come from deep multi-layer attention reasoning. The SSL variant reproduces the LSTM's own negative pretraining result: pretraining converges cleanly, but fine-tuning from it does not beat from-scratch.

### 4g. NEW -- Classical ML Baselines (RandomForest, GBT) + Anomaly Gate

`ai_backend/models/classical_baselines.py`. `RandomForestRegressor` and `MultiOutputRegressor(HistGradientBoostingRegressor(...))` on the same flattened, scaled windows (140 columns, temporal order discarded), `GridSearchCV`-tuned. **GBT trains in ~48s and reaches chained error 18.06m** -- beating the from-scratch LSTM despite having zero sequential modeling capability at all, a genuinely striking result. RandomForest's windowed loss looked competitive but did not carry through to chained accuracy (27.24m, essentially tied with the LSTM) -- another instance of the windowed-vs-chained gap this document's Part 4d already established for the SSL comparison. `anomaly_gate.py` adds a dual-detector (`IsolationForestGate` + `PCAReconstructionGate`) input-pathology check, a different signal from either uncertainty source above (flags sensor glitches, not motion uncertainty).

### 4h. NEW -- Formal Multi-Seed Ablation Matrix, and Why Single-Run Numbers Were Never the Whole Story

Every number in Parts 4a-4g above (and in the historical run log) is a **single training run** -- informative, but, as this document's own Part 0 retraction already demonstrated for a different reason (data-integrity bugs), a single run is not automatically representative. `ai_backend/ablation_matrix.py` trains/evaluates all 6 stochastic architectures across multiple random seeds (mean ± t-distribution 95% CI) on the same held-out chained-trajectory segment. Two complete runs exist: `runs/ablation/cpu_run1/` (3 seeds) and `runs/ablation/cpu_5seeds/` (5 seeds, extending the first) -- full tables, figures, and written analysis in each folder's own `ANALYSIS.md`.

**Headline, at 5 seeds**: the Transformer's 12.46m single-run number above turns out to sit near the *favorable* end of a real spread -- its 5 seeds ranged 3.22-15.04m (mean 9.10m), a coefficient of variation of 71%, by far the least consistent model in the matrix. RandomForest, by contrast, is the most consistent (2% CV) despite a worse mean. With 5 seeds, "the Transformer is statistically the best architecture" is a defensible claim (it cleanly beats LSTM, SSL-LSTM, RandomForest, and GBT on non-overlapping confidence intervals) -- but "the Transformer's number is reproducible" is not; a different seed could land anywhere in that 3-15m range. A genuinely new finding only visible at 5 seeds: the LSTM's own seed 4 never triggered early stopping across its full epoch budget and independently had both the best windowed loss and the best chained error of all 5 seeds -- tested directly (Part 5 below), not just noted.

### 4i. NEW -- Testing Five of Part 4h's Own Open Questions Directly

All five reused artifacts already on disk or a single retraining run -- no new flight data.

**Transformer ensemble across its 5 `cpu_5seeds` checkpoints** (`ai_backend/evaluate_transformer_ensemble.py`): averages all 5 seeds' per-step delta prediction before chaining. Result: **3.92m final / 5.88m mean**, against the per-seed range [3.22, 15.04]m and the ablation study's own random-single-seed expectation (9.10m). Roughly halves the expected error of a randomly-deployed single seed, for free -- a real mitigation, not a fix for the underlying variance.

**Two-phase mu/log_var training for the uncertainty head** (`ai_backend/train_uncertainty_two_phase.py`), testing Part 4e's own proposed fix: freeze `mu` after MSE convergence, fit `log_var` alone afterward. `DeadReckoningLSTMUncertainty.fc[2]` is one shared `nn.Linear(32,6)` (mu = rows 0:3, log_var = rows 3:6), so freezing is done via gradient-masking, not `requires_grad`. A real implementation bug was caught and fixed along the way (Adam's `weight_decay` was silently re-introducing gradient on the "frozen" rows from inside its own `step()`; fixed with `weight_decay=0` on phase 2). **Result: an honest negative** -- calibration barely moved (r=0.199 vs. 0.203) and chained error came out worse (38.49m vs. 29.44m, still within that model's already-documented run-to-run range); `log_var` collapsed toward a narrow, still-overconfident range rather than learning real input-dependent variation.

**Order-shuffled-input Transformer** (`train_transformer.py --shuffle-order`, evaluated via `ai_backend/evaluate_transformer_shuffled.py`): a single fixed permutation of the 10 timestep slots, trained and evaluated consistently, testing whether the Transformer's edge over GBT (order-blind) depends on chronological order. Result: **9.78m final / 7.24m mean**, nominally *better* than the real, order-preserving Transformer (12.46m/9.75m). Suggestive against chronological order being the driver, consistent with the attention finding above -- but honestly caveated: one shuffled run against one normal run, and the Transformer's own seed variance (3.22-15.04m from random init alone) is large enough that this isn't conclusive by itself. **Confirmed with the same 5 seeds** (`ai_backend/train_transformer_multiseed_confirmation.py`): 8.28/7.36/4.94/20.67/8.80m final, mean **10.01m, 95% CI [2.39, 17.64]m**, against the real Transformer's 9.10m mean, CI [2.61, 15.60]m -- heavily overlapping, statistically indistinguishable. The single-run result was ordinary seed noise, not a real effect.

**LSTM early-stopping patience, raised from 5 to 10** (`ai_backend/train_lstm_patience_experiment.py`): retrained the same 5 `cpu_5seeds` seeds, same initial weights (`set_global_seed(seed)`). Aggregate: **32.24m mean final error**, worse than patience=5's 29.67m -- no clean win. One seed (1) improved substantially (29.48m -> 26.69m); two got worse despite similar/better windowed loss; two roughly unchanged. **A more important, unplanned finding**: seed 4's rerun stopped at epoch 3 (val 0.7200), nothing like its own original (never early-stopped, val 0.6846 by epoch 30) -- and patience cannot affect epochs 1-3's actual numbers, only the stop decision, so the divergence isn't from patience itself. An isolated-process rerun of seed 4 alone (`ai_backend/train_lstm_seed4_patience10_isolated.py`) reproduced the sequential run's epoch 1-3 values bit-for-bit identical, ruling out same-process state leakage. Remaining explanation: `set_global_seed()` reproduces results *within* one environment (confirmed here) but not bit-identical CPU floating-point results *across* sessions (a documented PyTorch limitation, thread-count-dependent BLAS reduction order, not a codebase bug) -- the original seed-4 trajectory cannot be reproduced here to test the patience hypothesis against it specifically, an honest limit on cross-session seed reproducibility.

**Transformer without time-series augmentation** (`train_transformer.py --no-augment`, a flag that existed but had never been run): **a second real bug caught before damage** -- the flag had no distinct tag suffix, so running it as originally written would have overwritten `dr_transformer_imu_norm.pth`, the production checkpoint; caught within seconds (confirmed via timestamps), fixed with a composite tag-suffix, relaunched safely. Windowed val loss: 0.2212 (vs. 0.3545 with augmentation). Chained result: **6.47m final / 4.23m mean** -- nearly half the with-augmentation result, the best single-run Transformer number in this project, beating every one of the 5 `cpu_5seeds` seeds individually. Same caveat as the shuffle finding: one run against one run, not distinguishable from luck without multi-seed confirmation given the Transformer's 71% CV -- but the first direct evidence that augmentation (included from the start to mitigate data-diversity risk, never isolated until now) may be net-harmful rather than helpful here. **Confirmed with the same 5 seeds, and this time the exciting number did not survive**: 9.51/10.32/10.50/15.32/5.76m final, mean **10.28m, 95% CI [6.05, 14.52]m**, against the real Transformer's 9.10m mean, CI [2.61, 15.60]m -- heavily overlapping, not a defensible improvement, nominally slightly worse on the mean than the real Transformer, the opposite of what the lucky single run suggested. One secondary finding survives: the no-augment variant is meaningfully more consistent seed-to-seed (CV ~41%) than the real Transformer's 71%, just not enough to move the mean.

---

## Part 5 - Production Model Status

### Current State (post ablation-matrix update -- see Part 4h)

| File | input_size | Status |
|------|-----------|--------|
| `dr_lstm_imu_norm.pth` | 14 | Current -- **no longer the original Run #10 checkpoint.** Replaced by ablation-matrix seed 4 (Part 4h), which beat it on a fresh chained-trajectory evaluation (24.78m vs. 26.34m). Outgoing checkpoint preserved, not deleted: `dr_lstm_imu_norm_pre_seed4_continuation.pth`. |
| `scaler_X/y_imu_norm.pkl` | 14 features | Current, matches the model above |
| `ssl_pretrain_imu_ssl_norm.pth` | 14 (encoder only) | Current -- SSL-pretrained encoder checkpoint |
| `dr_lstm_imu_ssl_norm.pth` | 14 | Current -- SSL-fine-tuned model (underperforms the plain-supervised one, kept for the ablation record, not as the production choice) |
| `dr_transformer_imu_norm.pth`, `dr_transformer_imu_ssl_norm.pth` | 14 | Current -- production Transformer/SSL-Transformer checkpoints, selectable live via the model picker (Part 4h, see execution plan §19.8) but not the default |
| `rf_imu_norm.pkl`, `gbt_imu_norm.pkl` | 14 (flattened) | Current -- production RandomForest/GBT checkpoints |
| `dr_lstm_px4_norm.pth` (Batch 1, 6 logs), `dr_lstm_px4_norm_mission_v2.pth` (46-log corpus) | 14 | Current -- standalone px4-source checkpoints (val loss 1.5 -> 0.2768, not combinable with `imu`, evaluation-only, see Part 1) |
| `ppo_recovery_seek.zip` | -- (11-dim RL obs) | Current -- trained RL seek-bias policy, wired into `recovery_orchestrator.py` as an opt-in replacement for the hand-coded seek formula (Part 3) |
| `dr_lstm_nav_imu_norm.pth` (old, 8-feature) | 8 | Deleted during Session 5 housekeeping -- was blocking `udp_server.py`'s fallback load path |

**Production recommendation, revised**: `dr_lstm_imu_norm.pth` (now the seed-4-continued checkpoint) remains the *default* -- it is what `recovery_orchestrator.py` loads absent an explicit `--model` choice. But it is **not the best-known architecture**: the ablation matrix (Part 4h) found the Transformer statistically beats it on chained accuracy. It was not promoted to the default because it is also, by a wide margin, the least consistent architecture in the matrix (71% relative variability vs. the LSTM's 16%) -- "best on average, least predictable per run" is a real, unresolved tradeoff, not an oversight. Both are available side by side via the live model picker for hands-on comparison.

---

## Part 6 - Improvement Roadmap (updated)

### Done since the last revision of this document

1. ~~Fix train/val leakage (sequence shuffle + scaler-fit-order)~~ - done, Part 0.
2. ~~Fix domain-shift scaler contamination~~ - done via per-source deltas + `_src_id` boundaries + `verify_unit_consistency()` guard, Part 0.
3. ~~Retrain production models with 14 features~~ - done, Part 5.
4. ~~Add early stopping~~ - done, Part 4a.
5. ~~Attempt self-supervised pretraining, ablate against from-scratch~~ - done, Part 4b (honest negative result).
6. ~~Add a non-ML baseline (classical filter)~~ - done, Part 4c.
7. ~~Evaluate on a real, continuous trajectory (not just windowed MSE)~~ - done, Part 4d.
8. ~~Integrate a real second/replacement data source~~ - done via PX4 public flight logs, Part 1 (not yet combinable with `imu` - see below).

### Done since Run #10 (Track A ablation-matrix extension, see Part 4e-4h)

1. ~~Formal 5-model × 3-dataset ablation matrix~~ -- done, at multi-seed statistical rigor (mean ± 95% CI), twice (`cpu_run1` 3-seed, `cpu_5seeds` 5-seed extension). Scoped to `imu` only, not the 3-dataset fusion originally envisioned -- see the gyro-unit item below for why.
2. ~~Transformer encoder ablation arm~~ -- done, now the statistically-best architecture in the comparison, with an honest reliability caveat (Part 4h).
3. ~~Uncertainty output (Gaussian NLL) for the LSTM~~ -- done, Part 4e. Real but weak calibration (r=0.203) -- not yet a signal an RL layer could usefully consume as-is.
4. ~~Reconcile the imu/px4 gyro-unit mismatch~~ -- **investigated, found genuinely irreconcilable**, not merely unattempted: a full candidate-conversion sweep (`ai_backend/data_processing/reconcile_gyro_units.py`) found no unit-conversion error; the mismatch is a real dynamics difference between the simulated `imu` flight and real PX4 hardware. `px4` stays evaluation-only; a fused A+B benchmark is not blocked by a bug, it is blocked by the data itself.
5. ~~Ensemble the Transformer's 5 already-trained seeds~~ -- done, Part 4i: 3.92m final / 5.88m mean, vs. a 9.10m random-single-seed expectation. A real mitigation for the reliability problem, not a fix for its cause.
6. ~~Test the two-phase mu/log_var fix for uncertainty calibration~~ -- done, Part 4i: **did not help** (r=0.199, essentially unchanged from 0.203), an honest negative result.
7. ~~Test whether the Transformer's edge over GBT depends on chronological order~~ -- done and confirmed with the same 5 seeds, Part 4i: **not distinguishable from normal order** (mean 10.01m, 95% CI [2.39, 17.64]m, vs. the real Transformer's 9.10m, CI [2.61, 15.60]m -- heavily overlapping). The single-run "nominally better" result was ordinary seed noise.
8. ~~Test whether raising LSTM early-stopping patience closes seed 4's gap~~ -- done, Part 4i: no clean win (aggregate mean got worse, 32.24m vs. 29.67m), one seed improved, and a genuine cross-session reproducibility gap was discovered along the way (`set_global_seed()` is not bit-identical across sessions on CPU, confirmed via an isolated-process control) -- the original seed-4 trajectory cannot be reproduced here to test against directly, reported as a real limitation, not glossed over.
9. ~~Test whether removing time-series augmentation changes the Transformer's real result~~ -- done and confirmed with the same 5 seeds, Part 4i: **the exciting single-run number (6.47m) did not survive** confirmation (mean 10.28m, 95% CI [6.05, 14.52]m, vs. the real Transformer's 9.10m, CI [2.61, 15.60]m -- heavily overlapping, nominally slightly worse on the mean). One secondary finding survives: the no-augment variant is meaningfully more consistent seed-to-seed (CV ~41%) than the real Transformer's 71%, just not enough to move the mean.

### Highest ROI (do next)

1. **Try a different SSL configuration** to address the Part 4b/4f negative results: a contrastive (SimCLR-style) objective instead of/alongside masked reconstruction, a different mask ratio, or a larger/more diverse pretraining corpus -- now a documented finding for *both* architectures, strengthening the case that the pretext task or data diversity, not either architecture specifically, is the limiting factor.
2. **Extend the feature set with motor-thrust/barometer/magnetometer channels** once a `flight_recorder.py`-captured dataset of meaningful size exists (the Godot telemetry now emits these fields correctly, but no bulk recording has been done yet).

~~More diverse training data for the Transformer's reliability problem~~ -- the analogous
experiment was run (§19.12/Part 1), but on the standalone px4 corpus, not `imu_data.csv` directly
(the two remain uncombinable, see the gyro-unit finding). What it showed: a larger, Mission-filtered
px4 batch (6 -> 46 logs) fixed that corpus's own data-diversity problem (val loss 1.5 -> 0.2768);
physically-motivated synthetic diversity (rotation + segment splicing) on top of that gave a
promising-but-not-yet-proven 3-seed result (0.2587 vs. 0.2741, CIs overlap). The Transformer's own
71%-CV reliability problem on its actual `imu_data.csv` training data remains untested by this --
there's still no path to new, diverse `imu_data.csv` recordings without Godot access.

### Medium Term

1. ~~RL redesign: state = reconstructed position + uncertainty (not raw GPS)~~ -- **done** (§19.11).
   Uses the EKF's own confidence signal, not A2's uncertainty head specifically (that head's own
   calibration was too weak to trust here). Timestep budget kept at 10K, not scaled to 500K-1M, per
   explicit user call. Honest result in Part 3 above.
2. Swarm connectivity graph & topology-similarity analysis (Track B) -- fully scoped, deliberately deferred until single-drone work was verified; it now is. Next and last item on the user's own prioritization.
3. A showcase/dashboard layer folding the ablation-matrix results into the existing reporting pattern (Track D) -- deferred until Track A had results worth showing; it does now.

### Long Term

1. 3D real-time trajectory visualization inside Godot (`ImmediateMesh`-based), lower priority - a demonstration/debugging tool, not a dependency for any scientific claim above.
2. Multi-drone swarm extension and UDP multi-node protocol - explicitly deferred until the single-drone pipeline above is fully verified, per project scoping decision.

---

## Historical Run Log (superseded - kept for provenance only)

### Session 1–3 Runs (8-feature model)

| Run | Sources | Norm | Epochs | Val Loss | Notes |
|-----|---------|------|--------|----------|-------|
| 1 | nav | No | 10 | ~11,000 | Baseline |
| 2 | PPO RL | - | 10K steps | 33% success | RL proof-of-concept |
| 3 | nav | Yes | 5 | 0.965 | Underfitting plateau |
| 4 | imu | Yes | 2 | 0.558 | Partial convergence |
| 5 | nav+imu | Yes | 3 | 0.074 | Best at 3 epochs with 8 features - later shown to be a leakage/domain-shift artifact pattern |
| 6 | nav+imu | Yes | 20 | 0.054 | Production model - later deleted (stale, then superseded) |

### Session 4–5 Runs (14-feature model, pre-fix - retracted, see Part 0)

| Run | Sources | Norm | Val Loss | Status |
|-----|---------|------|----------|--------|
| #5 | imu | Yes | 0.2058 | Retracted - leakage artifact |
| #7 A | nav | No | 11,451.3 | Superseded (Run #10: 447,521.29) |
| #7 B | nav | Yes | 1.0107 | Superseded (Run #10: 0.9539) |
| #7 C | imu | Yes | 0.2264 | **Retracted - leakage artifact.** Honest value: 0.7348 (Run #10) |
| #7 D | nav+imu | Yes | 0.3325 | **Retracted.** Now understood as combining valid data with fabricated data - permanently non-citable, any future value included |

All of the above predate the fixes in Part 0 and must not be cited. **Run #10 (this revision) is the first trustworthy entry in this file's history.**
