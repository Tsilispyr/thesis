"""Downloads real flight logs from the public PX4 Flight Review database
(https://review.px4.io, CC-BY, Dronecode Foundation) into datasets/px4_raw/.

This is the exact selection logic used to build the batch documented in
datasets/px4_raw/SOURCE.md: filters for clean (error-free), EKF2-estimated,
Quadrotor flights of moderate duration, newest first. Re-run with a larger
--max-num for more data, and/or --flight-modes Mission (SOURCE.md's own
"to get more/better data later" note, never previously executed) to bias
toward flights with more translational dynamics than the first batch's
bench/hover-heavy sample.

dbinfo's own 'flight_modes' field is a list of PX4 nav_state integer codes,
NOT strings (SOURCE.md's original "e.g. require 'Mission'" phrasing implied
otherwise -- checked directly against a live dbinfo response, e.g.
flight_modes: [2] for a POSCTL-only flight). FLIGHT_MODES_TABLE below maps
the human names this script's --flight-modes flag accepts to those integer
codes, taken verbatim from PX4/flight_review's own app/plot_app/config_tables.py
(the same upstream tool SOURCE.md credits for the original download's
dbinfo/download API usage) -- not independently guessed.

Respects the server's own rate-limit convention (6s between downloads,
matching the project's official app/download_logs.py tool) - this is a free
service funded by the Dronecode Foundation, don't hammer it.

Existing files already present in --out-dir are left untouched and
skipped -- safe to re-run against a larger --max-num without re-downloading
or overwriting the original 6-log batch.

Usage (from AI_Recovery/):
    python ai_backend/data_processing/download_px4_logs.py --max-num 6
    python ai_backend/data_processing/download_px4_logs.py --max-num 40 --flight-modes Mission
"""
import argparse
import os
import time

import requests

DB_INFO_API = "https://review.px4.io/dbinfo"
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'px4_raw')

# Verbatim from PX4/flight_review's app/plot_app/config_tables.py
# (flight_modes_table) -- only the name->id direction is needed here, colors
# dropped. All AUTO-modes (Mission/Loiter/RTL/...) share id ranges typical of
# waypoint-following, sustained-translation flight, as opposed to
# Manual/Altitude/Position (0/1/2), which cover manual and bench/hover-style
# stick-flying -- the latter is what the original 6-log batch turned out to
# be (SOURCE.md's "Known limitations").
FLIGHT_MODES_TABLE = {
    'Manual': 0, 'Altitude': 1, 'Altitude Cruise': 8, 'Position': 2,
    'Position (Slow)': 6, 'Acro': 10, 'Offboard': 14, 'Stabilized': 15,
    'Mission': 3, 'Loiter': 4, 'Return to Land': 5, 'Descend': 12,
    'Terminate': 13, 'Takeoff': 17, 'Land': 18, 'Follow Target': 19,
    'Precision Land': 20, 'Orbit': 21, 'VTOL Takeoff': 22,
}


def fetch_candidates(mav_type="Quadrotor", max_errors=0, estimator="EKF2",
                      min_duration_s=120, max_duration_s=600, flight_modes=None):
    """flight_modes, if given (list of names from FLIGHT_MODES_TABLE, e.g.
    ['Mission']), keeps only entries whose own 'flight_modes' field (a list
    of PX4 nav_state integer codes, see FLIGHT_MODES_TABLE's docstring)
    contains any of the requested modes' codes -- i.e. the flight passed
    through that mode at some point, not that it was flown exclusively in
    it."""
    print("Fetching PX4 public log database info...")
    entries = requests.get(DB_INFO_API, timeout=120).json()
    print(f"  {len(entries)} total public logs in database.")

    candidates = [e for e in entries
                  if e.get('mav_type', '').lower() == mav_type.lower()
                  and e.get('num_logged_errors', 0) <= max_errors
                  and e.get('estimator', '') == estimator
                  and min_duration_s <= e.get('duration_s', 0) <= max_duration_s]
    print(f"  {len(candidates)} candidates match mav_type/errors/estimator/duration.")

    if flight_modes:
        unknown = [m for m in flight_modes if m not in FLIGHT_MODES_TABLE]
        if unknown:
            raise ValueError(f"Unknown --flight-modes {unknown}; choose from "
                              f"{sorted(FLIGHT_MODES_TABLE)}")
        wanted_ids = {FLIGHT_MODES_TABLE[m] for m in flight_modes}
        filtered = [e for e in candidates
                    if wanted_ids & set(e.get('flight_modes', []))]
        print(f"  {len(filtered)}/{len(candidates)} candidates passed through "
              f"flight_modes={flight_modes} (ids {sorted(wanted_ids)}) at some point.")
        candidates = filtered

    candidates.sort(key=lambda x: x['log_date'], reverse=True)
    return candidates


def download(candidates, max_num, out_dir, delay_s=6.0):
    os.makedirs(out_dir, exist_ok=True)
    picked = candidates[:max_num]
    for i, entry in enumerate(picked):
        fp = os.path.join(out_dir, entry['log_id'] + '.ulg')
        if os.path.exists(fp):
            print(f"  [{i+1}/{len(picked)}] already downloaded: {entry['log_id']}")
            continue
        print(f"  [{i+1}/{len(picked)}] downloading {entry['log_id']} "
              f"({entry['log_date']}, {entry['duration_s']}s)...")
        resp = requests.get(entry['download_url'], timeout=300)
        with open(fp, 'wb') as f:
            f.write(resp.content)
        print(f"    saved {len(resp.content):,} bytes")
        if i < len(picked) - 1:
            time.sleep(delay_s)
    print(f"Done. Logs in {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-num', type=int, default=6)
    ap.add_argument('--mav-type', type=str, default='Quadrotor')
    ap.add_argument('--min-duration-s', type=int, default=120)
    ap.add_argument('--max-duration-s', type=int, default=600)
    ap.add_argument('--out-dir', type=str, default=OUT_DIR)
    ap.add_argument('--flight-modes', type=str, nargs='+', default=None,
                     choices=sorted(FLIGHT_MODES_TABLE),
                     help="Only keep flights that passed through at least one of these "
                          "modes (e.g. --flight-modes Mission), biasing the batch toward "
                          "translational-dynamics flights over bench/hover-only ones.")
    args = ap.parse_args()

    cands = fetch_candidates(mav_type=args.mav_type,
                              min_duration_s=args.min_duration_s,
                              max_duration_s=args.max_duration_s,
                              flight_modes=args.flight_modes)
    download(cands, args.max_num, args.out_dir)
