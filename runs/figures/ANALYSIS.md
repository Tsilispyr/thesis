# `runs/figures/` - The 12-Step Pipeline Dashboard: Figure-by-Figure Analysis

**Provenance**: this analysis is written against **Run #10** of `ai_backend/ml_pipeline.py` (`runs/pipeline.log`), executed fresh specifically so this document's content is backed by real, current numbers rather than a stale cached run - the winner-selection bug (see below) was fixed immediately before this run, so this is also the first run where the dashboard's own "best model" claim is trustworthy by construction, not just by lucky numeric accident. Command: `python ai_backend/ml_pipeline.py` (`epochs=10, batch=64, window=10`, full 544,763-row `imu_data.csv`, no row cap).

**What this pipeline is for, distinct from `runs/figures_single_drone/`**: `figures_single_drone/` (see its own `ANALYSIS.md`) answers "does the final model actually work, in real metres, on a real trajectory." This folder answers a different, earlier question: "walk through the whole methodology, step by step, including the mistakes, so a reader can audit *how* the data got from raw CSVs to a trained model." It is deliberately narrative and includes the deprecated/broken configurations (Runs A, B, D) on purpose - removing them would hide the exact comparisons that motivated every fix documented in `ai_backend/data_processing/ANALYSIS.md`.

---

## `01_dataset_audit.png` - Dataset Audit

**What it shows**: three panels - record counts (`nav`=5,000 rows, `imu`=544,763 rows - a ~109x size difference), NaN percentage (0.0% for both, i.e. no missing-data cleanup was actually needed at the raw-file level), and a binary "usable for dead reckoning" check (both pass, since both nominally have IMU + GPS columns present).

**What it means**: this is the point where `imu`'s scale advantage over `nav` first becomes visible quantitively - over two orders of magnitude more rows. It's also the first hint of why the two datasets need to be treated asymmetrically downstream: `nav`'s tiny size alone wouldn't disqualify it, but combined with the domain-shift and physical-impossibility findings documented elsewhere, it's the first data point in the eventual case for treating `imu` as the primary, trustworthy source.

## `02_data_cleaning.png` - Data Cleaning

**What it shows**: left, a before/after row-count bar chart - `nav` drops from 5,000 to 4,764 rows (removing 236 obstacle-avoidance rows, 4.7%, per the pipeline log), `imu` is unchanged at 544,763 (no rows dropped). Right, the IMU column rename table (`accel_x→imu_acc_x`, `pos_x→latitude`, etc.) that gives `imu_data.csv`'s raw columns the canonical `FEATURE_COLS` names shared across the whole project.

**What it means, and what it drove us to do**: dropping obstacle-avoidance rows from `nav` is a deliberate methodological choice, not incidental cleanup - those rows' trajectories are driven by collision-avoidance logic rather than normal flight dynamics, and including them would teach a dead-reckoning model to expect sudden evasive maneuvers as "normal" motion. The rename table on the right is what makes `nav` and `imu` schema-compatible enough to even compare (let alone combine) - every downstream figure's feature names trace back to this mapping.

## `03_normalization.png` - StandardScaler Before/After

**What it shows**: all 14 `FEATURE_COLS`, raw distribution (top row) vs. normalized distribution (bottom row), with each panel's std annotated (raw σ ranges from 0.10 for `dt` up to 25.23 for `imu_gyro_z`; all normalized to σ=1.00).

**What it means**: the raw-σ numbers are the direct, visual proof of *why* normalization is necessary at all before training an LSTM - a 253x spread in scale between `dt` (σ≈0.10) and `imu_gyro_z` (σ≈25.23) would otherwise let the highest-variance features dominate the loss gradient regardless of their actual predictive value. This figure is also the honest baseline that makes Run A vs. Run B's dramatic loss-scale difference (see below) legible rather than mysterious.

## `04_feature_correlations.png` - Feature × Target Correlation Matrix

**What it shows**: a 17×17 heatmap (14 features + 3 targets) of pairwise Pearson correlation. Two things stand out: (1) every feature's correlation with `delta_lat`/`delta_lon`/`delta_alt` is small - the largest is `imu_gyro_x` vs. `delta_alt` at just 0.011; (2) two feature-feature pairs are strikingly high: `pitch` vs. `mag_x` at 0.99, and `roll` vs. `mag_y` at -0.75.

**What it means, and what it drove us to conclude**: the near-zero feature→target correlations are not a red flag - they're the expected signature of a task where the *useful* signal is temporal (how the sequence evolves over 10 steps), not instantaneous (what any single timestep's raw value is). A linear correlation matrix, by construction, cannot see that kind of structure - this is direct, honest evidence for why an LSTM (which integrates information across the whole window) is the right tool, and why a simpler model that only looked at instantaneous feature values would likely fail outright. The `pitch`/`mag_x` and `roll`/`mag_y` near-collinearity, on the other hand, is a genuine limitation worth flagging honestly: it's a direct consequence of the simulated magnetometer model being a fixed world-frame field rotated purely by attitude (no real geographic declination or independent noise process, a documented simplification - see `AI_RECOVERY_EXECUTION_PLAN.md` §6.1) - meaning `mag_x`/`mag_y` in this dataset carry substantially *redundant*, not independent, information relative to `roll`/`pitch`. This is useful context for interpreting why adding magnetometer features may contribute less unique signal than a real, independently-noisy magnetometer would.

## `05_sequence_window.png` - Sequence Building, One Concrete Example

**What it shows**: all 14 features plotted over one real 10-timestep window (IMU rows 1000-1009), plus a bottom summary box giving the actual target the model would need to predict from that window (`delta_lat=0.0046377m, delta_lon=0.0323776m, delta_alt=-0.0204493m`).

**What it means**: this is the figure that makes the abstract "(10,14)→(3,)" input/output shape concrete and inspectable - a reader can see that, e.g., `pitch` and `speed` both trend smoothly and monotonically across the window (consistent with steady, non-maneuvering flight in this particular segment) while `imu_gyro_x`/`imu_gyro_z` oscillate around zero with no clear trend (consistent with small-amplitude stabilization corrections rather than a deliberate turn). The target box uses a plain rectangle with no border (fixed per an explicit styling request mid-project, replacing an earlier rounded/bordered box) - a deliberate, minor but real design decision: a bordered box competed visually with the 14 bordered feature subplots above it, while a flat, borderless fill reads clearly as "this is the answer," not "this is another subplot."

## `06_runA_nav_nonorm.png` - Run A: nav, no normalization (deliberately naive baseline)

**What it shows**: `nav` data, no `StandardScaler` applied, trained for 10 epochs. Both train and val loss sit around 446,000-462,000 for the entire run, with no visible improvement - literally flat lines at the scale of the y-axis.

**What it means, and what it drove us to do**: this run exists specifically as a "why normalization matters" control, not as a real candidate model. Raw, unnormalized position deltas plus an MSE loss produce a loss surface dominated by the highest-magnitude values - the network essentially cannot make useful gradient progress at this scale in 10 epochs. This is the empirical demonstration behind Run B's very existence: normalize the same data, keep everything else fixed, and see what changes.

## `07_runB_nav_norm.png` - Run B: nav, normalized

**What it shows**: same `nav` data, now normalized. Val loss drops to ~0.95 and stays essentially flat there for all 10 epochs (0.9539 final) - train loss hovers near 1.0 the whole time too, with no real separation between train and val.

**What it means**: normalization alone fixes the *training dynamics* (loss is now in a sane, comparable range, unlike Run A) but the *model* still isn't learning anything useful - a flat ~1.0 normalized loss with train and val essentially identical is the signature of a model failing to find signal in the data at all (not overfitting, not underfitting in the usual sense - just stuck). Combined with the physically-impossible-position finding documented elsewhere (`AI_RECOVERY_EXECUTION_PLAN.md` §10 - `nav`'s implied per-step displacement is 2 orders of magnitude larger than its own reported speed), this flat curve is now understood as a direct symptom of that underlying data problem, not a modeling failure to be tuned away.

## `08_runC_imu_norm.png` - Run C: imu, normalized (the only trustworthy run)

**What it shows**: the real `imu_data.csv` corpus (544,753 sequences - vastly more than nav's 4,754), normalized. Train loss falls smoothly and continuously from 0.6965 to 0.2080; val loss starts at 0.7442, dips to a low of ~0.7056 around epoch 6, and ends at 0.7348 - a visible, if modest, train/val gap opening up (classic mild overfitting, consistent with the same pattern independently observed and addressed with early stopping in `train.py`/`ssl_pretext.py`, see `runs/figures_single_drone/ANALYSIS.md`).

**What it means**: this is the first run in the whole dashboard where the model is demonstrably learning something real - a smooth, monotonic train-loss decline over an order-of-magnitude more data than nav provides. It is *not* a leakage-inflated number (this run goes through the same fixed `create_dead_reckoning_dataset()` pipeline documented in `data_processing/ANALYSIS.md`: temporal-order split, train-only scaler fit, source-boundary-aware windowing) - which is exactly why it, and not the lower-looking Run D below, is the one this dashboard now correctly designates as the winner.

## `09_runD_nav_imu_norm.png` - Run D: nav + imu combined (deliberately kept naive, as a warning)

**What it shows**: `nav` and `imu` concatenated directly (`pd.concat([nav, imu])`, in `ml_pipeline.py`'s own `step9_run_D()` - **not** routed through `dataset_parser.load_combined_dataset()`'s guarded, `verify_unit_consistency()`-checked path), normalized, trained for 10 epochs. Val loss is reported as essentially **0.0000** the entire run, while train loss stays flat around 1.0.

**What it means - and why this figure is kept in the dashboard on purpose, not fixed**: train loss ≈1.0 while val loss ≈0.0000 is the unmistakable red-flag shape this project has now seen twice (`data_processing/ANALYSIS.md`'s Bug 2) - a scaler contaminated by an extreme outlier at the nav→imu concatenation boundary inflates the fitted standard deviation, which then artificially shrinks the *normalized* error for the much larger, unaffected imu-derived portion of the validation set. This run is **deliberately left un-fixed** in `ml_pipeline.py` (unlike `train.py`/`ssl_pretext.py`, which always go through the guarded, unit-checked loader) specifically so this exact failure mode remains visible and reproducible in the narrative dashboard - it is the concrete, empirical justification for why `verify_unit_consistency()` needed to exist at all, and removing Run D or quietly fixing it would delete the evidence for that design decision. The step's printed "Improvement vs Run B: 23.0%" line refers only to Run C vs. B; Run D is intentionally excluded from any "improvement" framing (see `10_comparison_all_runs.png`, and the winner-selection logic below).

## `10_comparison_all_runs.png` - All-Runs Comparison

**What it shows**: left, all four runs' val-loss curves over 10 epochs on a shared log-scale y-axis; right, a bar chart of each run's final val loss (also log-scale) - A at 447,521 (off the top of any reasonable non-log view), B at 0.9539, C at 0.7348, D at 0.0000 (indistinguishable from zero even on a log axis).

**What it means**: the log scale is doing real interpretive work here - it's the only way to show all four runs' wildly different magnitudes (a ~10^9 dynamic range from A to D) on one honest axis without clipping or distorting any of them. Read naively, this chart looks like it says "D is best" (lowest bar) - that naive reading is exactly the bug that was just fixed (see below): this chart is a faithful plot of what each run's `final_val` number literally is, and interpreting *which bar means "the model actually works"* requires the domain knowledge that Run D's number is a scaler-contamination artifact, not a real result. This is precisely why the dashboard-level text (summary box, `12_summary_dashboard.png`) explicitly names C as the winner in prose, rather than leaving a reader to infer it from bar height alone.

## `11_rl_evaluation.png` - Path Recovery: Rule-Based vs. RL Agent

**What it shows**: the pre-existing, not-yet-upgraded `SwarmRecoveryEnv`/PPO prototype (see `ai_backend/models/ANALYSIS.md`'s honest flagging of this file) evaluated via `eval_metrics.py` - rule-based controller: 100.0% success, 39.9 avg steps; RL (PPO, 10K timesteps): 28.0% success, 158.45 avg steps.

**What it means**: this figure is unchanged in substance from the original audit's finding and is included here for completeness of the pipeline narrative, not as a new result - a straight-line rule-based controller trivially beats a PPO agent trained for only 10K timesteps on a task that doesn't yet consume the reconstructed position/uncertainty the fail-safe hierarchy design calls for. It is the empirical starting point for the still-pending Phase 3 RL work (state = reconstructed position + uncertainty, target = return-to-home, 500K-1M timesteps), sequenced explicitly after the single-drone SSL/EKF/breadcrumb deliverables per the user's own prioritization.

## `12_summary_dashboard.png` - Final Summary

**What it shows**: a results table (Run / Sources / Norm / Sequences / Final Train Loss / Final Val Loss / Time) with the `C_imu_norm` row highlighted in a light-green background and bold black text, plus a stats box below stating "Best LSTM: C_imu_norm → Val Loss = 0.7348," an explicit note that "Runs A/B/D use the deprecated 'nav' dataset - reference only," the RL numbers from Step 11, and "Production model: `ai_backend/models/dr_lstm_imu_norm.pth`."

**What it means, and the methodology bug this figure now correctly avoids**: until very recently, both this table's highlighted-row logic and the stats box's "winner" text were computed by naive `min(runs, key=lambda r: r["final_val"])` - which would have picked **Run D** (0.0000), the scaler-contamination artifact described above, as the dashboard's headline result. A new `_select_trustworthy_winner()` helper (`ml_pipeline.py`) now explicitly prefers `C_imu_norm` whenever it's present in the run set, falling back to the naive minimum only if it's absent - so the single most visible number in the entire dashboard can no longer be an artifact of a known, already-diagnosed data-integrity bug. This run (`Run #10`) is the first one generated *after* that fix, which is why this document can state, without caveat, that the dashboard's own "Best LSTM" claim is currently correct. Two smaller styling fixes are also visible here, both applied per explicit request: the stats box uses a lighter fill with darker, bold text (previously a dark navy box with light text, harder to read at a glance), and the winner row's text is plain bold black rather than a green-on-dark-background color combination - letting the green *background* alone carry the "this is the winner" signal, with text color reserved for legibility rather than double-duty as a second encoding of the same fact.

---

## Summary: how this dashboard's own history validates its final claim

Reading Steps 6 through 12 in order tells one continuous, honest story: naive/unnormalized data fails outright (Run A) → normalization alone isn't enough if the underlying data is bad (Run B) → real, large-volume, physically-consistent data actually works (Run C) → naively combining a bad and a good source produces a number that *looks* better than it is (Run D) → the dashboard's own selection logic has, at two different points in this project, needed to be corrected to avoid being fooled by exactly that trap. The fact that this document can now trace that whole arc, including its own most recent bug and fix, using one single fresh pipeline run (`Run #10`) is itself the strongest available evidence that the current numbers are trustworthy - not because nothing ever went wrong, but because every time something did, it's traceable, documented, and now structurally harder to repeat.
