"""Physically-motivated synthetic diversity for the PX4 corpus (AI_RECOVERY
plan, Part B.2b) -- deliberately a different category of augmentation from
train_transformer.py --no-augment's statistical jitter/scale/time-warp noise,
which was tested and found not to survive multi-seed confirmation this
session (mean 10.28m vs 9.10m, statistically indistinguishable). That
augmentation perturbed real samples with synthetic noise; these two
transforms instead recombine or re-frame real recorded dynamics, so every
value a model sees was actually measured by a real autopilot at some point.

Two transforms:

1. rotate_augment() -- reframes a whole real flight as if flown at a
   different absolute heading. Body-frame quantities (accel, gyro, roll,
   pitch, speed, dt) are physically invariant under a pure world-frame yaw
   relabeling and are left untouched; only the world-frame quantities
   (delta_lat/delta_lon horizontal position deltas, and yaw itself) are
   rotated by the same angle, keeping every (feature, target) row
   internally consistent. mag_x/y/z is a KNOWN, DELIBERATE simplification:
   it is left unrotated. A real magnetometer senses Earth's externally
   fixed field, so a genuine heading change would rotate its body-frame
   reading too -- correctly reconstructing that would require inverting
   the original attitude to recover the world-frame field vector first,
   which risks a subtler error than the one this is trying to fix (this
   project has been burned by exactly this class of frame/unit mistake
   before, see dataset_parser.py's verify_unit_consistency() and its
   docstring's incident list). Leaving mag alone costs the model a
   slightly-stale compass cue on augmented copies; it does not corrupt
   accel/gyro/attitude/position, which carry the actual dead-reckoning
   signal.

2. splice_augment() -- builds new, longer synthetic "flights" by joining
   the real tail of one flight to the real head of another at a state-
   matched cut point (nearest neighbor in [speed, roll, pitch, sin(yaw),
   cos(yaw)] space, gated by a max-distance threshold so a poor match is
   skipped rather than spliced anyway). Every individual row stays exactly
   as recorded; only which real row follows which changes. This is what
   lets a training window span two different flights' dynamics as if they
   were one continuous recording -- genuinely new transitions, built
   entirely out of real measurements.

Both return a DataFrame in the same schema load_and_clean_px4_data()
produces (same FEATURE_COLS/TARGET_COLS, its own '_src_id' range so
create_dead_reckoning_dataset()'s window-boundary logic treats each
synthetic copy/splice as its own recording, same as any two real flights).
"""
import numpy as np
import pandas as pd

_ROTATE_SRC_OFFSET = 200_000   # kth rotation round -> _src_id + k*this; real px4 ids are 1000+i, i<200
_SPLICE_SRC_BASE = 900_000     # spliced synthetic flights get ids from here up


def rotate_augment(df: pd.DataFrame, n_copies: int = 2, seed: int = 0) -> pd.DataFrame:
    """Returns n_copies additional rotated copies of df (NOT including the
    original -- call site is responsible for concatenating the original
    back in if it wants both). One random heading offset per copy per
    _src_id (i.e. each real flight gets its own independent random
    rotation within a copy, not one shared angle for the whole corpus) --
    otherwise every flight in a given copy would share the exact same
    rotation, adding far less diversity than independent per-flight angles."""
    rng = np.random.default_rng(seed)
    src_ids = df['_src_id'].unique()
    copies = []
    for k in range(n_copies):
        angles = {sid: rng.uniform(0.0, 2 * np.pi) for sid in src_ids}
        theta = df['_src_id'].map(angles).values
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        copy_df = df.copy()
        dlat = df['delta_lat'].values
        dlon = df['delta_lon'].values
        copy_df['delta_lat'] = dlat * cos_t - dlon * sin_t
        copy_df['delta_lon'] = dlat * sin_t + dlon * cos_t
        # yaw wrapped back to (-180, 180], matching arctan2-derived yaw's own range
        copy_df['yaw'] = ((df['yaw'].values + np.degrees(theta) + 180.0) % 360.0) - 180.0
        copy_df['_src_id'] = df['_src_id'] + _ROTATE_SRC_OFFSET * (k + 1)
        copies.append(copy_df)
    return pd.concat(copies, ignore_index=True)


def _state_vectors(df: pd.DataFrame) -> np.ndarray:
    """[speed, roll, pitch, sin(yaw), cos(yaw)], each independently
    z-scored (state matching cares about relative closeness, not raw
    units, and roll/pitch in degrees would otherwise be swamped by speed
    in m/s in a raw Euclidean distance) -- sin/cos avoids the wraparound
    discontinuity a raw yaw-degrees distance would have at +-180."""
    yaw_rad = np.radians(df['yaw'].values)
    raw = np.stack([
        df['speed'].values, df['roll'].values, df['pitch'].values,
        np.sin(yaw_rad), np.cos(yaw_rad),
    ], axis=1).astype(np.float64)
    std = raw.std(axis=0)
    std[std < 1e-9] = 1.0
    return (raw - raw.mean(axis=0)) / std


def splice_augment(df: pd.DataFrame, n_splices: int = 20, stride: int = 200,
                    max_state_dist: float = 1.5, seed: int = 0) -> pd.DataFrame:
    """Builds n_splices synthetic flights, each = real_flight_A[:cut_a] +
    real_flight_B[cut_b:], where (cut_a, cut_b) is a nearest-neighbor match
    in state space among candidate cut points (every `stride` rows, to keep
    the search tractable across flights with 80K+ rows) between two
    different, randomly chosen flights. Pairs whose best match exceeds
    max_state_dist (z-scored Euclidean, see _state_vectors) are skipped
    rather than spliced anyway -- a handful of skipped attempts is expected
    and fine, a bad splice baked into training data is not. May return
    fewer than n_splices rows worth of flights if too many attempts get
    skipped; callers should check how many were actually produced."""
    rng = np.random.default_rng(seed)
    src_ids = df['_src_id'].unique()
    if len(src_ids) < 2:
        return df.iloc[0:0].copy()

    by_flight = {sid: df[df['_src_id'] == sid].reset_index(drop=True) for sid in src_ids}
    states = {sid: _state_vectors(fl) for sid, fl in by_flight.items()}
    candidates = {sid: np.arange(0, len(fl), stride) for sid, fl in by_flight.items()}

    spliced = []
    attempts = 0
    while len(spliced) < n_splices and attempts < n_splices * 5:
        attempts += 1
        sid_a, sid_b = rng.choice(src_ids, size=2, replace=False)
        fl_a, fl_b = by_flight[sid_a], by_flight[sid_b]
        cand_a, cand_b = candidates[sid_a], candidates[sid_b]
        if len(cand_a) == 0 or len(cand_b) == 0:
            continue
        # nearest neighbor over the small candidate grids, not every row
        sa, sb = states[sid_a][cand_a], states[sid_b][cand_b]
        d = np.linalg.norm(sa[:, None, :] - sb[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] > max_state_dist:
            continue
        cut_a, cut_b = int(cand_a[i]), int(cand_b[j])
        if cut_a < 5 or (len(fl_b) - cut_b) < 5:
            continue   # too little of either side to be a meaningful splice

        joined = pd.concat([fl_a.iloc[:cut_a], fl_b.iloc[cut_b:]], ignore_index=True)
        joined['_src_id'] = _SPLICE_SRC_BASE + len(spliced)
        spliced.append(joined)

    if not spliced:
        return df.iloc[0:0].copy()
    return pd.concat(spliced, ignore_index=True)


if __name__ == '__main__':
    import os
    from dataset_parser import load_and_clean_px4_data

    raw_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'px4_raw')
    real = load_and_clean_px4_data(raw_dir)
    print(f"\nReal: {len(real)} rows, {real['_src_id'].nunique()} flights")

    rot = rotate_augment(real, n_copies=2, seed=0)
    print(f"Rotated: {len(rot)} rows, {rot['_src_id'].nunique()} synthetic flights")
    # Sanity: rotation must preserve per-row horizontal displacement magnitude
    real_mag = np.sqrt(real['delta_lat']**2 + real['delta_lon']**2).values
    rot_mag = np.sqrt(rot['delta_lat']**2 + rot['delta_lon']**2).values[:len(real)]
    assert np.allclose(real_mag, rot_mag, atol=1e-3), "rotation changed horizontal displacement magnitude!"
    print("  OK: horizontal displacement magnitude preserved under rotation.")
    assert rot['yaw'].between(-180.0, 180.0).all(), "rotated yaw escaped (-180,180] range"
    print("  OK: rotated yaw stays in range.")

    splice = splice_augment(real, n_splices=20, seed=0)
    print(f"Spliced: {len(splice)} rows, {splice['_src_id'].nunique()} synthetic flights "
          f"(requested 20, some pairs may be skipped -- see max_state_dist)")
    assert splice['_src_id'].nunique() <= 20
    print("\nAll px4_synthetic_augment smoke checks passed.")
