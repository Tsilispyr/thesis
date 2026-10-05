# `ai_backend/data_processing/` - Loading, Cleaning, and Fetching Data

Everything that turns raw files (CSVs, `.ulg` flight logs) into the windowed,
scaled tensors the models in `../models/` train on. `dataset_parser.py` is
the one file everything else in the project imports from - it's the single
source of truth for `FEATURE_COLS`, `TARGET_COLS`, and how a dataset gets
turned into train/val sequences. Read it first.

| File | What it is |
|---|---|
| `dataset_parser.py` | **The core module.** Per-source loaders (`load_and_clean_navigation_data`, `load_and_clean_imu_data`, `load_and_clean_px4_data`), `load_combined_dataset()` (merges sources, tags each row with a `_src_id` so a sliding window never spans two unrelated recordings, and calls the unit-consistency gate before combining), `verify_unit_consistency()` (blocks combining sources whose columns differ >3x in scale - see `AI_RECOVERY_EXECUTION_PLAN.md` §13), and `create_dead_reckoning_dataset()` (builds the sliding windows, splits train/val *before* fitting the scaler, returns everything `train.py`/`ml_pipeline.py`/`ssl_pretext.py` need). `python dataset_parser.py` runs a self-contained smoke test of all three sources. |
| `flight_recorder.py` | `FlightRecorder` class - appends every UDP telemetry packet `udp_server.py` receives to a timestamped CSV under `datasets/recorded/`. This is how a **new**, motor/baro/mag-inclusive training dataset gets built from the Godot simulation (the existing CSVs have no motor-thrust column at all). Off by default; enable with `udp_server.py --record`. |
| `download_px4_logs.py` | Reproducible downloader for real flight logs from the public [PX4 Flight Review database](https://review.px4.io/) (free, CC-BY). Filters by airframe type / error count / duration / estimator, writes `.ulg` files into `datasets/px4_raw/`. Rerun this to fetch a larger or differently-filtered batch - see `datasets/px4_raw/SOURCE.md` for what's already been downloaded and why those filters were chosen. |
| `osm_terrain_fetcher.py` | **Placeholder, not wired to real data.** Generates a synthetic random heightmap instead of fetching real OpenStreetMap/SRTM terrain - its own docstring says so. Intended for Godot mission terrain variety; not part of the dead-reckoning training pipeline. Left as-is; flagged here so nobody mistakes its output for real terrain data. |

## Data sources this feeds (see `AI_RECOVERY_EXECUTION_PLAN.md` for the full history)

| Source key | Loader | Status |
|---|---|---|
| `'imu'` | `load_and_clean_imu_data` | **Validated, default.** 544,763 rows, internally physically consistent (checked, not assumed - §10/§11). |
| `'nav'` | `load_and_clean_navigation_data` | **Deprecated.** Its position columns are not a real trajectory (§10) - kept loadable only to reproduce the documented historical negative result. |
| `'px4'` | `load_and_clean_px4_data` | **Real hardware data, standalone only.** 6 real flight logs (§11). Not yet combinable with `'imu'` - `verify_unit_consistency()` will refuse the combination until the two sources' units are reconciled (§13). |

## Why the unit-consistency gate lives here

Three separate incidents this project hit (§7 nav/imu domain shift, §8/§9 the
scaler-fit-before-split bug, §11.2 the still-open imu/px4 gyro mismatch) were
all the same underlying failure: combining two sources whose numeric scales
didn't actually match. `verify_unit_consistency()` in `dataset_parser.py`
turns that from "a thing to remember" into a hard precondition enforced by
`load_combined_dataset()` itself.
