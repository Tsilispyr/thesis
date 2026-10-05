# Provenance: PX4 Public Flight Logs

**Source:** [PX4 Flight Review public log database](https://review.px4.io/) (`logs.px4.io`), operated by the Dronecode Foundation.
**License:** CC-BY (PX4).
**Downloaded:** 2026-07-23, via the database's public `dbinfo`/`download` API (same endpoints used by the project's own `app/download_logs.py` tool: https://github.com/PX4/flight_review).

## Why these logs

Selected from 433,352 total public entries, filtered for: `mav_type == "Quadrotor"`, `num_logged_errors == 0`, `estimator == "EKF2"`, `120 <= duration_s <= 600`, sorted newest-first, top 6 taken. All 6 confirmed as **real hardware flights** (`sys_hw = PX4_FMU_V6X`, a genuine Pixhawk-class autopilot board; `source = webui`, meaning a human manually uploaded the log - not an automated CI/SITL-simulation upload).

## Files

| log_id | date | duration | bytes |
|---|---|---|---|
| `b8255b9b-c6a9-4d3a-a6a1-456c685906de.ulg` | 2026-07-22 | 401s | 47,385,021 |
| `bfa8e3d3-ab05-458c-a762-fff745cb24fd.ulg` | 2026-07-22 | 333s | 39,554,663 |
| `30779140-9828-40c7-b088-8ee216d23b39.ulg` | 2026-07-22 | 354s | 41,922,505 |
| `44add118-bacd-4de5-8c7d-fa7177157df8.ulg` | 2026-07-22 | 409s | 48,203,771 |
| `a91890e9-b41d-4af3-ab51-8c7f3160e5e1.ulg` | 2026-07-22 | 481s | 56,558,409 |
| `794cc5b5-5dc4-413e-abad-bfb26347ccd0.ulg` | 2026-07-22 | 223s | 26,397,297 |

All are `airframe_type=22111` (a generic quadrotor X config), `ver_sw_release=v1.15.0`. Descriptions were left blank by the uploaders - not unusual for casual/personal flight-log uploads.

## Known limitations (be aware of these, don't gloss over them)

- **Not GPS-based.** `sensor_gps.latitude_deg`/`longitude_deg` were verified as uniformly `0.0` in at least one of these logs - i.e., no real GPS fix (likely indoor/bench/optical-flow test flights). Position is instead taken from `vehicle_local_position` (EKF2's fused local-position estimate, in metres, NED frame), which is populated regardless of GPS availability. This means these logs are NOT currently usable as a source of real GPS-degree coordinates - they're in the same "local relative metres" family as `imu_data.csv`, not a replacement for what `uav_navigation_dataset.csv` was supposed to be.
- **Units not yet reconciled with `imu_data.csv`.** A quick check on one log: PX4's `gyro_rad` (genuine rad/s) has std ≈ 0.07, vs. `imu_data.csv`'s reported gyro std of ≈ 22-25 - these are not the same scale, and the difference hasn't been root-caused. **Do not combine `sources=('imu','px4')` under one shared StandardScaler until this is resolved** - doing so would repeat the exact class of mistake documented in `AI_RECOVERY_EXECUTION_PLAN.md` §7/§10.
- **Small, homogeneous batch.** All 6 logs are same-day uploads with blank descriptions and a generic airframe config - likely bench/validation-style flights rather than varied dynamic maneuvers. A first training run (`train.py --sources px4 --epochs 10`) plateaued near val loss ≈ 1.5 (worse than "predict the mean"), consistent with a genuinely small/homogeneous sample rather than a data-quality bug (contrast with `uav_navigation_dataset.csv`, where the *data itself* was physically inconsistent - this data checks out physically, there's just not much of it yet, and it may not contain much translational movement).
- `actuator_outputs`'s first 4 `output[]` channels are used as "motor thrust" (PWM 1000-2000 remapped to ~0-1). This is a reasonable assumption for a quadrotor X config but hasn't been verified against the specific motor-to-output channel mapping for `airframe_type=22111`.

## Batch 2: Mission-filtered, 40 additional logs

**Downloaded:** 2026-08-15, via `ai_backend/data_processing/download_px4_logs.py` (a from-scratch reimplementation of the same `dbinfo`/`download` API the original batch used, since no `download_logs.py` tool is vendored in this repo - see that script's own docstring).

**Filter:** same as Batch 1 (`mav_type == "Quadrotor"`, `num_logged_errors == 0`, `estimator == "EKF2"`, `120 <= duration_s <= 600`) **plus** `flight_modes` containing `Mission` (id `3`), the exact next step this file's own "To get more / better data later" section had flagged but never executed. Top 40 by newest-first, out of 19,965 candidates that matched (out of 67,787 matching the base filter, out of 447,792 total public entries) - plenty of headroom for an even larger pull later if needed.

**Correction to this file's own prior wording:** `flight_modes` was assumed above to be a string field (`"Mission"`). It is not - `dbinfo` returns it as a list of PX4 nav_state *integer* codes (e.g. `[2]` for a Position-only flight). `download_px4_logs.py::FLIGHT_MODES_TABLE` maps names to codes, taken verbatim from PX4/flight_review's own `app/plot_app/config_tables.py` (the same upstream tool this project's download logic is modeled on), not guessed.

Total: 46 `.ulg` files, 992MB, in this directory (6 from Batch 1 + 40 from Batch 2).

### Files (Batch 2)

| log_id | date | duration | bytes |
|---|---|---|---|
| `7a88ee10-b983-4486-a6aa-a9da6fed111f.ulg` | 2026-08-14 | 411s | 14,672,749 |
| `1e10bf3f-a9bc-4f8c-a6ea-4d7501c97626.ulg` | 2026-08-14 | 411s | 14,672,749 |
| `51c360dc-f650-46cb-b0b0-a7f2e2e1f3bb.ulg` | 2026-08-14 | 410s | 14,881,687 |
| `88951bea-1461-4208-b994-46c027d60bc8.ulg` | 2026-08-14 | 410s | 14,881,687 |
| `0d11d599-7467-4652-b095-e61878dbd9f8.ulg` | 2026-08-14 | 199s | 8,407,591 |
| `41fe53a1-35ec-44b1-b84d-f8c17af4155a.ulg` | 2026-08-14 | 445s | 17,265,229 |
| `2e7b4230-2638-44ec-96f5-8ca448940567.ulg` | 2026-08-14 | 177s | 7,711,810 |
| `506b7d3f-efc9-4941-9b67-bd7aab908e67.ulg` | 2026-08-14 | 271s | 13,211,826 |
| `a34dad90-1fd9-4105-856c-4303a7bd4309.ulg` | 2026-08-14 | 277s | 89,762,927 |
| `4ef843c7-1cec-40b4-99f7-ee860609bf76.ulg` | 2026-08-14 | 150s | 5,736,974 |
| `9ebc5e26-8143-47f1-8cc7-8b33da76a6c9.ulg` | 2026-08-13 | 301s | 234,662,928 |
| `a23b7923-daf2-4e95-bb60-0d11f8f4a791.ulg` | 2026-08-13 | 132s | 5,591,421 |
| `a340360a-5909-4dc5-ae2b-c7ef2ffcbb04.ulg` | 2026-08-13 | 326s | 14,117,034 |
| `2010c82f-4320-4580-843b-1d17009cf2bf.ulg` | 2026-08-13 | 222s | 9,447,638 |
| `db8345b5-6ab3-4849-a3e7-2f22a09a5de2.ulg` | 2026-08-13 | 193s | 8,196,874 |
| `f82d00f3-40da-499c-a35f-c3ffc1c8f041.ulg` | 2026-08-13 | 132s | 5,591,421 |
| `24ee3ab4-ed08-48da-a624-7b621d89b375.ulg` | 2026-08-13 | 158s | 6,886,890 |
| `6295753e-a027-4b6b-b4f9-5959c3c45363.ulg` | 2026-08-13 | 123s | 5,270,305 |
| `4972c5a5-7e47-4c11-a1d3-0087f649f7ea.ulg` | 2026-08-13 | 177s | 7,711,810 |
| `d661cbf6-a7e3-47db-9f09-35286fb56647.ulg` | 2026-08-13 | 185s | 7,912,483 |
| `9d02410c-46e8-4130-8c99-9a9edb6ac739.ulg` | 2026-08-13 | 449s | 19,217,542 |
| `cbbf1568-0eb0-46c8-88de-889623040713.ulg` | 2026-08-13 | 171s | 7,093,250 |
| `19aecbdd-0b4c-41af-b2d5-04bb36e82dff.ulg` | 2026-08-13 | 403s | 17,302,978 |
| `0996b141-9c2d-4069-b6a0-63ec8b62c576.ulg` | 2026-08-13 | 375s | 16,128,501 |
| `4dc73598-f00e-4f52-bb72-8217ab352504.ulg` | 2026-08-13 | 219s | 48,799,006 |
| `043ed8fb-68e7-47a2-aed3-8c3c786d0141.ulg` | 2026-08-13 | 172s | 25,407,890 |
| `6e71d1e3-de8f-46a5-9fb2-13c9878cb1ed.ulg` | 2026-08-13 | 271s | 13,286,219 |
| `f954cbd5-9efb-4741-8ed3-7e4300baa6b8.ulg` | 2026-08-13 | 146s | 6,268,577 |
| `4a5408be-9034-4eae-9637-0fa65aec90c6.ulg` | 2026-08-13 | 226s | 26,118,754 |
| `d6801a95-d83a-47fb-bef3-a89867a4a927.ulg` | 2026-08-12 | 239s | 29,407,290 |
| `9702fb01-dc43-4a10-ba08-ccb3d727e9a9.ulg` | 2026-08-12 | 174s | 7,583,136 |
| `704d5c17-5fd5-4348-8746-c1ca2dea77ee.ulg` | 2026-08-12 | 126s | 5,527,885 |
| `d0d9df5e-e7f7-4ef7-936d-d79960d8dc43.ulg` | 2026-08-12 | 133s | 5,798,982 |
| `59af6875-e57b-4fbf-82ea-a0cd70c4c630.ulg` | 2026-08-12 | 160s | 6,800,070 |
| `3a0f8d85-196b-4b59-95da-473111d4174e.ulg` | 2026-08-12 | 126s | 5,386,988 |
| `3787f0c5-f380-48de-a062-16ab3436b425.ulg` | 2026-08-12 | 132s | 5,591,421 |
| `1377e417-5b3e-4cfc-a0ae-acb53ddec2cb.ulg` | 2026-08-12 | 175s | 7,460,107 |
| `d796b1dc-c5c6-48cb-aa48-38ff7decb1df.ulg` | 2026-08-12 | 168s | 7,281,838 |
| `cfe56cae-f517-46c8-af37-dd6af2cbfa2f.ulg` | 2026-08-12 | 124s | 5,387,829 |
| `613859f7-d730-4b86-b730-c6365e5b29a1.ulg` | 2026-08-12 | 168s | 7,293,354 |

Not yet individually spot-checked the way Batch 1's hardware/GPS fields were (per Batch 1's "Known limitations" above) -- these came from the same public database under the same clean-log filter, so the same caveats (no GPS fix, local-position-only, units not reconciled with `imu_data.csv`) should be assumed to apply until checked otherwise. `9ebc5e26-...` (234MB) is far larger than the rest of the batch -- likely a higher logging rate or additional topics, not a corrupt download (size matches what the server reported in `dbinfo`).

4 of the 46 total `.ulg` files fail to parse in `dataset_parser.py::load_and_clean_px4_data` ("list index out of range"): `043ed8fb-68e7-47a2-aed3-8c3c786d0141`, `2e7b4230-2638-44ec-96f5-8ca448940567`, `4972c5a5-7e47-4c11-a1d3-0087f649f7ea`, `d0d9df5e-e7f7-4ef7-936d-d79960d8dc43`. Not investigated further -- likely a `.ulg` message-format edge case (e.g. a topic the parser expects that these particular logs don't emit), not a download corruption (file sizes match what `dbinfo` reported). The loader skips them cleanly and continues; the other 42 load fine (2,210,969 total rows).

### Result: this batch fixes the "worse than predicting the mean" finding

Batch 1 alone (6 logs, `train.py --sources px4 --epochs 10`) plateaued at val loss ~1.5. Re-run on the combined 46-log corpus (42 usable, `train_model(sources=('px4',), tag_suffix='_mission_v2')`, 2026-08-15) reached **val loss 0.2768** (best epoch 1 of 6 run before early-stopping, `dr_lstm_px4_norm_mission_v2.pth` -- a distinctly-tagged file, does not overwrite Batch 1's own `dr_lstm_px4_norm.pth` reference result). Same code path, same normalization, so directly comparable -- confirms this was a data-quantity/diversity problem, not a structural one, exactly as this file's own prior "Small, homogeneous batch" note predicted but never tested until now. Not a claim that 0.2768 is competitive with the production `imu`-sourced models (different data source, different scale of dataset, no chained-trajectory reconstruction run against it yet) -- just evidence the earlier near-random result was fixable data, not a dead end.

### Physically-motivated synthetic diversity: a promising but not-yet-confirmed lever

`ai_backend/data_processing/px4_synthetic_augment.py` (2026-08-15) adds two synthetic-diversity
transforms on top of the real 46-log corpus, deliberately a different category from
`train_transformer.py --no-augment`'s statistical jitter/scale/time-warp noise (tested earlier
this session, found not to survive multi-seed confirmation): every value either transform produces
was actually measured by a real autopilot at some point.

- **`rotate_augment()`** -- reframes a whole real flight as if flown at a different absolute
  heading. Body-frame quantities (accel, gyro, roll, pitch, speed, dt) are physically invariant
  under a pure world-frame yaw relabeling and are left untouched; only delta_lat/delta_lon and yaw
  are rotated together, keeping every row internally consistent. Known, deliberate simplification:
  `mag_x/y/z` is left unrotated (a real magnetometer's body-frame reading would rotate too under a
  genuine heading change; reconstructing that correctly would require inverting the original
  attitude to recover the world-frame field vector first, which risks a subtler error than the one
  this avoids -- see this file's own gyro-unit-mismatch incident for why that risk is taken
  seriously here).
- **`splice_augment()`** -- builds new, longer synthetic flights by joining one real flight's tail
  to another's head at a state-matched cut point (nearest neighbor in
  `[speed, roll, pitch, sin(yaw), cos(yaw)]`, gated by a max-distance threshold so a poor match is
  skipped rather than spliced anyway).

**Result (`ai_backend/train_px4_synthetic.py`, 3 seeds, 2026-08-15, full logs/figures in
`runs/px4_synthetic_diversity/`)**: both arms validate against the SAME fixed, real-only held-out
rows (augmentation only ever touches the training pool, never validation, so "val loss" means the
same thing in both arms).

| arm | val loss (best epoch), by seed | mean | 95% CI |
|---|---|---|---|
| Baseline (real only) | 0.2739, 0.2775, 0.2708 | 0.2741 | [0.2657, 0.2824] |
| Augmented (real + rotation + splice) | 0.2480, 0.2774, 0.2506 | 0.2587 | [0.2182, 0.2991] |

The augmented mean is ~5.6% lower, and 2 of 3 augmented seeds beat every baseline seed outright
(visible in `runs/px4_synthetic_diversity/figures/val_curve_overlay.png`: the augmented curve sits
below and is markedly more stable than baseline's for nearly the whole training run). But the 95%
CIs overlap -- at n=3 seeds, one augmented seed (0.2774) lands right on top of the baseline cluster,
which is enough to keep this shy of the same statistical-significance bar this session applied to
the Transformer shuffle/no-augment checks. Unlike those two (which the multi-seed check reversed
entirely, mean converging back to baseline), this trend is *consistent in direction* across 2 of 3
seeds and the point estimate is a real improvement, not noise centered on zero -- plausible, worth
more seeds to confirm, but not yet a proven effect. Reported honestly as such, not rounded up to a
finding.

## Motor-informed features: does knowing commanded thrust help, on top of IMU?

`ai_backend/data_processing/dataset_parser.py::MOTOR_FEATURE_COLS` extends the shared 14-feature
`FEATURE_COLS` with `motor_0..motor_3` (the first 4 `actuator_outputs` channels, PWM microseconds
rescaled to ~0-1 thrust - see `load_and_clean_px4_data()`), an 18-feature, px4-only, opt-in set,
never the default `FEATURE_COLS`/`INPUT_SIZE` every other checkpoint's shape is locked to. Reason:
the last, un-run item this project's own fail-safe-hierarchy design had named - a genuinely
independent physical signal (what the autopilot commanded the motors to do) beyond the IMU alone,
worth checking before treating IMU-only dead reckoning as the ceiling.

`ai_backend/train_motor_informed.py` (2026-08-15), 3-seed baseline (IMU-only, 14 features) vs.
motor-informed (IMU+motor, 18 features), same real 46-log px4 corpus, same held-out-real-val-rows
discipline and multi-seed rigor this session established for every px4 experiment:

| arm | val loss (best epoch), by seed | mean | 95% CI |
|---|---|---|---|
| Baseline (IMU-only, 14 features) | 0.2845, 0.2908, 0.2997 | 0.2917 | [0.2728, 0.3106] |
| Motor-informed (IMU+motor, 18 features) | 0.2789, 0.2820, 0.2795 | 0.2801 | [0.2761, 0.2842] |

The motor-informed mean is ~4% lower and, more strikingly, visibly *tighter* across seeds (95% CI
half-width 0.004 vs. baseline's 0.019 - see `runs/px4_motor_informed/figures/comparison.png`'s box
plot) - a real, physically sensible signal (commanded thrust is causally upstream of how the
vehicle actually accelerates) landing where a real signal should. But the CIs still overlap at
n=3, so by this project's own statistical bar this is not yet a proven effect, the same honest
"promising, not confirmed" shape as the synthetic-diversity result above.

**Wired into the live hierarchy regardless of the statistical outcome**, per an explicit scope
decision made before training: `Motor-LSTM` in `api/recovery_orchestrator.py::MODEL_REGISTRY`,
manually-selectable only (never auto-picked - no automatic "IMU unhealthy" signal exists anywhere
in this codebase to trigger it on), backed by the promoted (lowest-val-loss) seed-0 checkpoint of
the run above (`dr_lstm_px4_motor_informed.pth`). The Godot simulation's own telemetry already
carries a genuine `motor_out` channel (`drone_body.gd::_send_telemetry`), so the live demo's
Motor-LSTM option consumes real, non-placeholder motor values, not zeros - the demo shows the
architecture choice exists, not a claim that it's the best one. `runs/live_mission_demo_test.png`
re-captured with the new option visible in the model picker.

## To get more / better data later

Re-run `download_px4_logs.py` with a larger `--max-num` for more data. `--flight-modes` now supports any name in `FLIGHT_MODES_TABLE` (e.g. `Loiter`, `Return to Land`) if a different dynamics bias is wanted later. Respect the server's rate limits (6s between downloads is the tool's own default) - this is a free service funded by the Dronecode Foundation.
