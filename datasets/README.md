# `datasets/` - Training Data

All loading/cleaning logic for these files lives in
`../ai_backend/data_processing/dataset_parser.py` - this README covers what
each file *is*, not how it's parsed.

| File / folder | Status | What it is |
|---|---|---|
| `imu_data.csv` | **Validated, default source** | 544,763 rows of high-frequency (240 Hz) IMU data. Independently checked, not just assumed trustworthy: uniform sampling, smooth frame-to-frame position changes, realistic implied speed (max ~35 m/s, nowhere near any clipping threshold). Origin/simulator that produced it is not itself verified - only that it's *internally physically consistent*. |
| `uav_navigation_dataset.csv` | **Deprecated - do not use for new work** | 5,000 rows that look like GPS telemetry but aren't. Consecutive 1-second rows imply ~1,000–2,200 m of movement while the file's own `speed` column reports 7–30 m/s - physically impossible. Traced to a likely-synthetic Kaggle dataset (`ziya07/uav-autonomous-navigation-dataset`). Kept only so the historical negative result (`AI_RECOVERY_EXECUTION_PLAN.md` §10) can be reproduced; `dataset_parser.py` prints a deprecation warning if you load it. |
| `px4_raw/` | **Real hardware data, standalone use only** | 6 real flight logs (`.ulg` format) downloaded from the public [PX4 Flight Review database](https://review.px4.io/) via `../ai_backend/data_processing/download_px4_logs.py`. **See [`px4_raw/SOURCE.md`](px4_raw/SOURCE.md) for full provenance** (log IDs, dates, filter criteria, known limitations) - that file is the source of truth for this folder, not this README. Not yet combinable with `imu_data.csv` - their sensor units haven't been reconciled (`AI_RECOVERY_EXECUTION_PLAN.md` §11.2/§13). |
| `recorded/` | **Doesn't exist yet - created on first use** | Where `../ai_backend/data_processing/flight_recorder.py` writes new CSVs when `udp_server.py` is run with `--record`. This is how a genuinely new, motor/barometer/magnetometer-inclusive dataset gets built from live Godot flights - none of the files above have a motor-thrust column at all. |

## Which source should I use?

- **Training a model right now**: `imu_data.csv` (the default - `train.py`
  and `ml_pipeline.py` both default to `--sources imu`).
- **Wanting real hardware data**: `px4_raw/` (`--sources px4`), on its own,
  not combined with `imu`.
- **Never**: `uav_navigation_dataset.csv` as a delta-position training
  target, for any new work - see the Status column above.
