import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# imu_data.csv  ->  canonical column names
_IMU_RENAME = {
    'accel_x': 'imu_acc_x', 'accel_y': 'imu_acc_y', 'accel_z': 'imu_acc_z',
    'gyro_x':  'imu_gyro_x','gyro_y':  'imu_gyro_y','gyro_z':  'imu_gyro_z',
    'pos_x':   'latitude',  'pos_y':   'longitude',  'pos_z':   'altitude',
}

# 14 features - full IMU sensor suite + time step
# roll/pitch/yaw: orientation (Euler angles, deg)
# mag_x/y/z:      magnetometer → compass heading
# speed:           scalar velocity (m/s), computed from position diffs / dt
# dt:              sample period (s) - essential for integration
FEATURE_COLS = [
    'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
    'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
    'roll', 'pitch', 'yaw',
    'mag_x', 'mag_y', 'mag_z',
    'speed', 'dt',
]
TARGET_COLS = ['delta_lat', 'delta_lon', 'delta_alt']

INPUT_SIZE = len(FEATURE_COLS)   # 14 - single source of truth for model input_size

# px4-only, opt-in feature set (never the default FEATURE_COLS/INPUT_SIZE above -- those are a
# global single source of truth every existing checkpoint's shape is locked to; silently adding
# motor columns there would break all 6 MODEL_REGISTRY architectures, not just a new one). Only
# 'px4' carries real actuator_outputs (see load_and_clean_px4_data()) -- imu_data.csv and
# uav_navigation_dataset.csv have no motor/actuator column at all, confirmed by reading their
# headers directly, so this is never usable with those sources. motor_0..3 are the first 4
# actuator_outputs channels, PWM microseconds rescaled to ~0-1 thrust (see load_and_clean_px4_data).
MOTOR_FEATURE_COLS = FEATURE_COLS + ['motor_0', 'motor_1', 'motor_2', 'motor_3']

# Equirectangular approximation for converting nav's GPS-degree lat/lon deltas
# into metres, so they share imu's relative-metre scale (imu's pos_x/y/z are
# already metres). Altitude in the nav CSV is already metres, no conversion
# needed. Fine for the small (~kilometre-scale) areas a single flight covers.
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 111_320.0


def load_and_clean_navigation_data(filepath):
    """Primary nav dataset - already in canonical schema.
    Fills orientation/magnetometer columns with 0 (not available in nav CSV).
    """
    print(f"Loading Navigation Data from {filepath}...")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.ffill(inplace=True)

    # Drop rows where the drone was in obstacle-avoidance mode - those trajectories
    # are driven by collision logic, not normal flight dynamics, and would corrupt DR training.
    if 'obstacle_detected' in df.columns:
        before = len(df)
        df = df[df['obstacle_detected'] == 0].copy()
        print(f"  Dropped {before - len(df)} obstacle-avoidance rows "
              f"({(before - len(df)) / before * 100:.1f}% of nav data).")

    # dt from timestamp differences (seconds)
    df['dt'] = (
        df['timestamp'].diff()
        .dt.total_seconds()
        .fillna(0.1)
        .clip(1e-3, 10.0)
    )

    # speed already present in nav dataset - keep it
    if 'speed' not in df.columns:
        df['speed'] = 0.0

    # Orientation & magnetometer not in nav CSV → fill with 0
    for col in ('roll', 'pitch', 'yaw', 'mag_x', 'mag_y', 'mag_z'):
        if col not in df.columns:
            df[col] = 0.0

    # Targets, computed here (per-source, before any concatenation with imu)
    # so a delta never spans two unrelated recordings. lat/lon deltas are
    # converted from GPS degrees to metres so they're on the same physical
    # scale as imu's already-metric delta_lat/lon (see _M_PER_DEG_* above).
    lat_rad = np.radians(df['latitude'])
    df['delta_lat'] = df['latitude'].diff().fillna(0.0) * _M_PER_DEG_LAT
    df['delta_lon'] = (df['longitude'].diff().fillna(0.0)
                        * _M_PER_DEG_LON * np.cos(lat_rad))
    df['delta_alt'] = df['altitude'].diff().fillna(0.0)

    print(f"  Loaded {len(df)} records.")
    return df


def load_and_clean_imu_data(filepath):
    """High-frequency IMU dataset (544 K rows).
    All 16 original columns used:
      time | pos_x/y/z → lat/lon/alt | roll | pitch | yaw
      accel_x/y/z → imu_acc_x/y/z | gyro_x/y/z → imu_gyro_x/y/z
      mag_x | mag_y | mag_z
    """
    print(f"Loading IMU Data from {filepath}...")
    df = pd.read_csv(filepath)
    df.sort_values('time', inplace=True)
    df.dropna(inplace=True)
    df.rename(columns=_IMU_RENAME, inplace=True)

    # dt from time column (seconds between samples)
    df['dt'] = (
        df['time'].diff()
        .fillna(df['time'].diff().median())
        .clip(1e-3, 10.0)
    )

    # Speed: magnitude of velocity derived from position diffs / dt
    # pos_x/y/z (→ lat/lon/alt) are in metres (relative coordinates)
    dx = df['latitude'].diff().fillna(0.0)
    dy = df['longitude'].diff().fillna(0.0)
    dz = df['altitude'].diff().fillna(0.0)
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    df['speed'] = (dist / df['dt']).fillna(0.0).clip(0.0, 200.0)

    # Targets: already relative metres, no conversion needed. Reuses the
    # diffs computed above for speed - same per-source-only diff, no boundary
    # crossing once concatenated with nav.
    df['delta_lat'] = dx
    df['delta_lon'] = dy
    df['delta_alt'] = dz

    print(f"  Loaded {len(df)} records.")
    return df


def load_and_clean_px4_data(raw_dir):
    """Real-hardware flight logs from the public PX4 Flight Review database
    (https://review.px4.io, CC-BY), downloaded as .ulg files into raw_dir.

    Position comes from vehicle_local_position (EKF2's fused local-position
    estimate, in metres, NED frame) rather than sensor_gps: many public logs
    are indoor/optical-flow test flights with no real GPS fix (confirmed on
    the first batch downloaded for this project - sensor_gps.latitude_deg
    was uniformly 0.0), while vehicle_local_position is populated regardless
    of position source. This keeps px4 in the same "already-metres" family
    as imu_data.csv rather than requiring a degrees-to-metres conversion.

    Each .ulg file is one physically independent flight - rows are tagged
    with a distinct '_src_id' per file (offset by 1000 to never collide with
    nav=0/imu=1 from load_combined_dataset) so create_dead_reckoning_dataset()
    never builds a window spanning two different flights.

    NOTE: PX4's native units have NOT been verified to match imu_data.csv's
    (e.g. gyro_rad is genuine rad/s with std ~0.07 on a sample log, vs.
    imu_data.csv's reported gyro std of ~22-25 - those are not the same
    units/scale). Do not combine 'px4' with 'imu' under one shared scaler
    until that mismatch is reconciled; treat 'px4' as its own standalone
    source for now (see AI_RECOVERY_EXECUTION_PLAN.md).
    """
    import glob
    from pyulog import ULog

    def _get_dataset_any_instance(u, name):
        """u.get_dataset(name) defaults to multi_instance=0 and raises if that
        specific instance doesn't exist on this log, even when the topic is
        genuinely present under a different instance. Found on 4 of the 46
        logs in this project's own corpus: actuator_outputs only exists under
        multi_instance=1 or 2 there (confirmed directly, not guessed) -- a
        different flight controller/logging configuration, not corrupt data.
        Tries 0/1/2 (PX4 logs practically never carry more instances than
        that for any one topic) and returns the first that exists."""
        last_err = None
        for mi in (0, 1, 2):
            try:
                return u.get_dataset(name, mi)
            except Exception as e:
                last_err = e
        raise last_err

    ulg_files = sorted(glob.glob(os.path.join(raw_dir, '*.ulg')))
    if not ulg_files:
        raise FileNotFoundError(f"No .ulg files found in {raw_dir}")

    frames = []
    for i, path in enumerate(ulg_files):
        fname = os.path.basename(path)
        try:
            u = ULog(path)
            sc  = pd.DataFrame(u.get_dataset('sensor_combined').data).sort_values('timestamp')
            att = pd.DataFrame(u.get_dataset('vehicle_attitude').data).sort_values('timestamp')
            mag = pd.DataFrame(u.get_dataset('sensor_mag').data).sort_values('timestamp')
            pos = pd.DataFrame(u.get_dataset('vehicle_local_position').data).sort_values('timestamp')
            act = pd.DataFrame(_get_dataset_any_instance(u, 'actuator_outputs').data).sort_values('timestamp')
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        df = sc.rename(columns={
            'accelerometer_m_s2[0]': 'imu_acc_x', 'accelerometer_m_s2[1]': 'imu_acc_y',
            'accelerometer_m_s2[2]': 'imu_acc_z',
            'gyro_rad[0]': 'imu_gyro_x', 'gyro_rad[1]': 'imu_gyro_y', 'gyro_rad[2]': 'imu_gyro_z',
        })[['timestamp', 'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
            'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']]

        # merge_asof onto sensor_combined's timeline (the highest-rate topic) -
        # standard technique for aligning multi-rate sensor topics by timestamp.
        df = pd.merge_asof(df, att[['timestamp', 'q[0]', 'q[1]', 'q[2]', 'q[3]']],
                            on='timestamp')
        df = pd.merge_asof(df, mag[['timestamp', 'x', 'y', 'z']]
                            .rename(columns={'x': 'mag_x', 'y': 'mag_y', 'z': 'mag_z'}),
                            on='timestamp')
        df = pd.merge_asof(df, pos[['timestamp', 'x', 'y', 'z']]
                            .rename(columns={'x': 'latitude', 'y': 'longitude', 'z': 'altitude'}),
                            on='timestamp')
        motor_cols = [c for c in act.columns if c.startswith('output[')][:4]
        act_small = act[['timestamp'] + motor_cols].copy()
        for j, c in enumerate(motor_cols):
            act_small[f'motor_{j}'] = (act_small[c] - 1000.0) / 1000.0  # PWM us -> ~0-1 thrust
        df = pd.merge_asof(df, act_small[['timestamp'] + [f'motor_{j}' for j in range(len(motor_cols))]],
                            on='timestamp')
        df.dropna(inplace=True)

        # Quaternion (PX4 order: w,x,y,z) -> Euler angles, degrees, matching
        # the roll/pitch/yaw convention used elsewhere in FEATURE_COLS.
        w, x, y, z = df['q[0]'], df['q[1]'], df['q[2]'], df['q[3]']
        df['roll']  = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        df['pitch'] = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        df['yaw']   = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        df.drop(columns=['q[0]', 'q[1]', 'q[2]', 'q[3]'], inplace=True)

        # dt: ULog timestamps are microseconds
        df['dt'] = (df['timestamp'].diff()
                    .fillna(df['timestamp'].diff().median()) / 1e6).clip(1e-4, 10.0)

        # Position already in metres (vehicle_local_position, NED) - plain
        # diff, same as imu_data.csv, no degrees-to-metres conversion needed.
        dx = df['latitude'].diff().fillna(0.0)
        dy = df['longitude'].diff().fillna(0.0)
        dz = df['altitude'].diff().fillna(0.0)
        df['speed'] = (np.sqrt(dx**2 + dy**2 + dz**2) / df['dt']).fillna(0.0).clip(0.0, 200.0)
        df['delta_lat'], df['delta_lon'], df['delta_alt'] = dx, dy, dz

        df['_src_id'] = 1000 + i  # offset: never collides with nav=0/imu=1
        frames.append(df)
        print(f"  {fname}: {len(df)} synchronized rows")

    if not frames:
        raise ValueError(f"No usable PX4 logs found in {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined)} total rows from {len(frames)} PX4 flight logs.")
    return combined


# Max allowed cross-source std ratio (max_std/min_std) for any shared
# FEATURE_COLS/TARGET_COLS column before verify_unit_consistency() blocks
# combining two sources. Real sensor-noise/scale differences between
# genuine recordings rarely exceed this; an actual unit mismatch blows
# straight past it (rad/s vs deg/s ~57x, GPS degrees vs metres ~100,000x -
# see AI_RECOVERY_EXECUTION_PLAN.md §7, §11.2).
UNIT_MISMATCH_THRESHOLD = 3.0


def verify_unit_consistency(source_frames, threshold=UNIT_MISMATCH_THRESHOLD,
                             reconciled=frozenset()):
    """Guard against combining sources whose columns are on different
    physical scales/units under one shared StandardScaler.

    This is the direct code-level fix for the failure class behind three
    separate incidents in this project: the nav/imu GPS-degrees-vs-metres
    domain shift (§7), the scaler-fit-before-split leakage (§8/§9), and the
    still-open imu/px4 gyro rad/s-vs-unknown-scale mismatch (§11.2). Call
    this with the INDIVIDUAL per-source frames (before pd.concat) whenever
    load_combined_dataset() is about to combine 2+ sources.

    source_frames: {source_name: DataFrame}, the not-yet-concatenated
                   per-source frames.
    threshold:     max allowed cross-source std ratio for any shared
                   FEATURE_COLS or TARGET_COLS column. Checking TARGET_COLS
                   too (not just features) is deliberate - the worst
                   historical incident (§7) was a target-column mismatch,
                   not a feature one.
    reconciled:    set of (source_a, source_b, column) tuples (either
                   order) known to differ in raw scale but already handled
                   elsewhere (e.g. an explicit unit conversion applied
                   before this point) - exempted from raising. Use this,
                   not a higher threshold, to silence one specific known-
                   and-handled case without weakening the check generally.

    Raises ValueError listing every offending (source_a, source_b, column,
    ratio) triple if any unreconciled mismatch is found. Columns that are
    all-zero/near-constant placeholders in a source (e.g. nav's zero-filled
    roll/pitch/yaw/mag_x/y/z) are skipped - those are documented stand-ins,
    not a unit mismatch, and would otherwise trigger a spurious
    division-by-near-zero flag on every comparison involving them.
    """
    check_cols = FEATURE_COLS + TARGET_COLS
    names = list(source_frames.keys())
    stats = {
        name: {col: (float(df[col].mean()), float(df[col].std()))
                for col in check_cols if col in df.columns}
        for name, df in source_frames.items()
    }

    violations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            for col in check_cols:
                if col not in stats[a] or col not in stats[b]:
                    continue
                std_a, std_b = stats[a][col][1], stats[b][col][1]
                if std_a < 1e-9 or std_b < 1e-9:
                    continue  # placeholder/near-constant column, not a real mismatch
                ratio = max(std_a, std_b) / min(std_a, std_b)
                if ratio <= threshold:
                    continue
                if (a, b, col) in reconciled or (b, a, col) in reconciled:
                    continue
                violations.append((a, b, col, ratio, std_a, std_b))

    if violations:
        lines = [f"  {a} vs {b}: '{col}' std ratio = {ratio:.1f}x "
                 f"({a}={std_a:.4g}, {b}={std_b:.4g})"
                 for a, b, col, ratio, std_a, std_b in violations]
        raise ValueError(
            "verify_unit_consistency() blocked this dataset combination - "
            f"{len(violations)} column(s) exceed the {threshold}x cross-source "
            "std-ratio threshold (a likely unit/scale mismatch, not real signal). "
            "Fix the mismatch at the source (see load_and_clean_navigation_data()'s "
            "degrees->metres conversion for the pattern), or pass the column "
            "explicitly via `reconciled` once verified and handled:\n"
            + "\n".join(lines)
        )


def load_combined_dataset(base_dir, sources=('imu',)):
    """Load and merge requested dataset sources into one DataFrame.

    Tags each row with '_src_id' (a distinct int per source) so that
    create_dead_reckoning_dataset() can avoid building a window/target that
    spans two different, physically unrelated recordings.

    Sources:
      'imu' - imu_data.csv (544K rows, verified physically consistent -
              uniform 240Hz sampling, smooth positions, realistic speeds).
              Default and currently the only recommended standalone source.
      'nav' - uav_navigation_dataset.csv. DEPRECATED: its lat/lon/altitude
              columns are not a real trajectory (a single timestep implies
              ~1000-2200 m of movement while its own 'speed' column reports
              7-30 m/s - physically impossible). Kept loadable only for
              reproducing the historical negative result documented in
              AI_RECOVERY_EXECUTION_PLAN.md §10; do not use for new work.
      'px4' - real PX4 public flight logs (see load_and_clean_px4_data).
              Do not combine with 'imu' yet - native units haven't been
              reconciled between the two sources (see that function's
              docstring). verify_unit_consistency() below will refuse this
              combination until that's fixed (or explicitly reconciled).
    """
    named_frames = {}   # source name -> its DataFrame, for verify_unit_consistency()
    frames = []

    if 'nav' in sources:
        print("  WARNING: 'nav' (uav_navigation_dataset.csv) is deprecated - "
              "its position data is not a real trajectory. See "
              "AI_RECOVERY_EXECUTION_PLAN.md §10.")
        nav_path = os.path.join(base_dir, 'uav_navigation_dataset.csv')
        nav_df = load_and_clean_navigation_data(nav_path)
        nav_df['_src_id'] = len(frames)
        named_frames['nav'] = nav_df
        frames.append(nav_df)

    if 'imu' in sources:
        imu_path = os.path.join(base_dir, 'imu_data.csv')
        imu_df = load_and_clean_imu_data(imu_path)
        imu_df['_src_id'] = len(frames)
        named_frames['imu'] = imu_df
        frames.append(imu_df)

    if 'px4' in sources:
        px4_dir = os.path.join(base_dir, 'px4_raw')
        px4_df = load_and_clean_px4_data(px4_dir)
        named_frames['px4'] = px4_df
        frames.append(px4_df)  # already carries its own per-flight _src_id (offset 1000+)

    if not frames:
        raise ValueError(f"No valid sources specified. Got: {sources}")

    if len(named_frames) > 1:
        verify_unit_consistency(named_frames)

    df = pd.concat(frames, ignore_index=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    print(f"Combined dataset: {len(df)} total records from {list(sources)}.")
    return df


def _window_df(df, window_size, feature_cols=None):
    """Raw (unscaled) sliding-window sequences from a single df, respecting
    '_src_id' boundaries -- the shared core of create_dead_reckoning_dataset(),
    factored out so a caller needing separately-windowed pools (e.g. a
    held-out real validation set that must never see augmented/synthetic
    rows, see held_out_val_df below) doesn't have to reimplement this loop.

    feature_cols, if given, overrides the module-level FEATURE_COLS (e.g.
    MOTOR_FEATURE_COLS for a px4-only motor-informed arm) -- defaults to
    None so every existing caller's behavior is unchanged."""
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    missing = [c for c in feature_cols + TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after mapping: {missing}")

    feat = df[feature_cols].values.astype(np.float32)
    targ = df[TARGET_COLS].values.astype(np.float32)
    src_id = (df['_src_id'].values if '_src_id' in df.columns
              else np.zeros(len(df), dtype=np.int64))

    X, y = [], []
    for i in range(len(df) - window_size):
        if src_id[i] != src_id[i + window_size]:
            continue  # window/target would straddle two unrelated recordings
        X.append(feat[i:i + window_size])
        y.append(targ[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_dead_reckoning_dataset(df, window_size=10, normalize=True, val_split=0.2,
                                   held_out_val_df=None, feature_cols=None):
    """Build sliding-window sequences for LSTM Dead Reckoning, split into
    train/val, and (if normalize) fit StandardScaler on the train split only.

    Expects delta_lat/delta_lon/delta_alt to already be present on df,
    computed per-source by load_and_clean_navigation_data()/
    load_and_clean_imu_data() before any concatenation - so a delta never
    spans two unrelated recordings. If df carries a '_src_id' column (set by
    load_combined_dataset() for multi-source data), windows that would
    straddle a source boundary are dropped for the same reason.

    Sequences are kept in temporal order and split sequentially (not
    shuffled) - shuffling overlapping sliding windows before a split leaks
    validation data into training, since adjacent windows share most of
    their timesteps.

    held_out_val_df, if given, replaces the internal sequential val_split
    entirely: ALL of df's windows become the train pool, and held_out_val_df
    is windowed separately to become the val pool (still scaled with the
    scaler fit on df's windows only). For px4_synthetic_augment.py's
    experiments, where synthetic rows get concatenated onto real training
    rows -- a plain row-sequential split on the combined df would put
    synthetic (not real) rows in the tail, silently validating against
    synthetic dynamics instead of the same real held-out flights every
    other px4 run is scored against. Pass the real-only validation rows
    here instead of concatenating them into df, to make comparisons across
    real-only vs. augmented runs apples-to-apples.

    feature_cols, if given, overrides FEATURE_COLS for this call (e.g.
    MOTOR_FEATURE_COLS) -- passed straight through to _window_df(). Defaults
    to None, unchanged behavior for every existing caller. Note the printed
    "features=" count below still reports the module-level INPUT_SIZE, not
    this override -- see train.py::train_model()'s own feature_cols handling
    for where the actual per-call feature count is computed and used.

    Returns (X_train, y_train, X_val, y_val, scaler_X, scaler_y).
    X shape: (N, window_size, INPUT_SIZE=14)
    y shape: (N, 3)  - delta_lat, delta_lon, delta_alt
    """
    print("Creating Dead Reckoning sequences...")

    if held_out_val_df is not None:
        X_train, y_train = _window_df(df, window_size, feature_cols=feature_cols)
        X_val, y_val = _window_df(held_out_val_df, window_size, feature_cols=feature_cols)
    else:
        X, y = _window_df(df, window_size, feature_cols=feature_cols)
        split = int(len(X) * (1.0 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

    scaler_X = scaler_y = None
    if normalize:
        n_tr, win, n_feat = X_train.shape
        n_va = X_val.shape[0]

        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train.reshape(-1, n_feat)).reshape(n_tr, win, n_feat)
        X_val   = scaler_X.transform(X_val.reshape(-1, n_feat)).reshape(n_va, win, n_feat)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_val   = scaler_y.transform(y_val)

    X_train = X_train.astype(np.float32); X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.float32); y_val = y_val.astype(np.float32)

    print(f"  Created {len(X_train):,} train / {len(X_val):,} val sequences  "
          f"(window={window_size}, features={X_train.shape[-1]}, normalized={normalize}).")
    return X_train, y_train, X_val, y_val, scaler_X, scaler_y


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')

    # Single-source loads: expected to succeed.
    for combo in [('nav',), ('imu',)]:
        print(f"\n=== Sources: {combo} ===")
        df = load_combined_dataset(base_dir, sources=combo)
        X_tr, y_tr, X_va, y_va, sx, sy = create_dead_reckoning_dataset(df, window_size=10)
        print(f"  X_train: {X_tr.shape}  y_train: {y_tr.shape}  "
              f"X_val: {X_va.shape}  y_val: {y_va.shape}")

    # Multi-source loads: expected to be BLOCKED by verify_unit_consistency()
    # until each pair's unit mismatch is reconciled (§11.2/§12) - demonstrates
    # the gate working, not a broken pipeline.
    for combo in [('nav', 'imu'), ('imu', 'px4')]:
        print(f"\n=== Sources: {combo}  (expected: blocked by verify_unit_consistency) ===")
        try:
            df = load_combined_dataset(base_dir, sources=combo)
            print(f"  UNEXPECTED: combination succeeded, {len(df)} rows - "
                  f"was a unit mismatch reconciled without updating this comment?")
        except ValueError as e:
            print(f"  Correctly blocked: {str(e).splitlines()[0]}")
