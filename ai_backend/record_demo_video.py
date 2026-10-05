"""
Record the two live demos to MP4 for the thesis defense deck.

Both demos are matplotlib FuncAnimation dashboards driven by a `tick()` method,
and both already have a deterministic, assertion-backed scripted scenario in
their own `run_headless_test()`. This script drives *those same scenarios* and
captures a frame per tick into an MP4 via matplotlib's ffmpeg writer, so the
recorded video shows exactly the run the thesis reports -- not a re-enactment.

Why this exists: a pre-recorded video makes demo timing deterministic. A live
demo's runtime is the one number the presentation budget cannot control.

Usage:
  python ai_backend/record_demo_video.py --demo mission
  python ai_backend/record_demo_video.py --demo swarm
  python ai_backend/record_demo_video.py --demo both --fps 10

Output: runs/video/mission_control_demo.mp4, runs/video/swarm_demo.mp4
Requires ffmpeg on PATH (verified present: `ffmpeg -version`).
"""
import os
import sys
import argparse

# The demo modules pick their matplotlib backend from sys.argv at import time:
# '--headless-test' present -> Agg. We need Agg (no GUI window while recording),
# so inject the flag before importing either module.
if '--headless-test' not in sys.argv:
    sys.argv.append('--headless-test')

import numpy as np
import matplotlib
from matplotlib.animation import FFMpegWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(os.path.dirname(_HERE), 'runs', 'video')


def _writer(fps, bitrate=2400):
    return FFMpegWriter(
        fps=fps, bitrate=bitrate,
        metadata={'title': 'SD-UAV thesis demo', 'artist': 'Tsilimpokos Spyridon'},
    )


def record_mission(out_path, fps=10, dpi=120):
    """Mission Control: mission start -> GPS cut -> RTL abort -> GPS restored.

    Mirrors live_mission_demo.LiveDashboard.run_headless_test() exactly:
    420 ticks, cut at tick 3, restore at tick 300, target (30, 4, 25).
    At DT=0.1 s/tick the arc is 42 simulated seconds, so fps=10 plays back
    at real time.
    """
    import live_mission_demo as lmd

    dash = lmd.LiveDashboard()
    target = [30.0, 4.0, 25.0]
    dash.mission_start_time = dash.drone.t
    dash.drone.start_mission(target)
    dash.log(f"MISSION START -> target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = _writer(fps)
    n_frames = 0
    with writer.saving(dash.fig, out_path, dpi):
        for i in range(420):
            dash.tick()
            if i == 3:
                dash.drone.cut_connection()
                dash.log("GPS SIGNAL CUT -- mission autopilot suspended, awaiting recovery guidance")
            if i == 300:
                dash.drone.restore_gps()
                dash.log("GPS SIGNAL RESTORED")
            dash._redraw()
            writer.grab_frame()
            n_frames += 1
        # hold the final frame so the closing state is readable on screen
        for _ in range(fps * 2):
            writer.grab_frame()
            n_frames += 1

    dist = float(np.linalg.norm(dash.drone.pos - np.array(target)))
    rtl = any("RTL ENGAGED" in line for line in dash.log_lines)
    assert rtl, "expected the target-then-RTL hybrid to abort to home in this scenario"
    return {'frames': n_frames, 'seconds': n_frames / fps,
            'final_distance_m': dist, 'rtl_engaged': rtl}


def record_swarm(out_path, fps=6, dpi=120):
    """Swarm: healthy topology -> all-drone GPS cut -> drift -> fragmentation.

    Mirrors live_swarm_demo.LiveSwarmDashboard.run_headless_test(): seed=3,
    15 healthy ticks, cut all, 60 drift ticks.
    """
    import live_swarm_demo as lsd

    dash = lsd.LiveSwarmDashboard()
    dash.sim = lsd.SwarmSim(seed=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = _writer(fps)
    n_frames = 0
    with writer.saving(dash.fig, out_path, dpi):
        for _ in range(15):
            dash.tick(); dash._redraw(); writer.grab_frame(); n_frames += 1
        risk_before = dash.sim.risk
        for _ in range(fps):          # hold on the healthy state
            writer.grab_frame(); n_frames += 1

        dash._on_cut_all(None)
        dash.log("-- all-drone GPS cut --")
        for _ in range(60):
            dash.tick(); dash._redraw(); writer.grab_frame(); n_frames += 1
        risk_after = dash.sim.risk
        for _ in range(fps * 2):      # hold on the fragmented state
            writer.grab_frame(); n_frames += 1

    assert risk_after > risk_before, "expected predicted fragmentation risk to rise after sustained GPS loss"
    return {'frames': n_frames, 'seconds': n_frames / fps,
            'risk_before': round(float(risk_before), 3),
            'risk_after': round(float(risk_after), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', choices=['mission', 'swarm', 'both'], default='both')
    ap.add_argument('--fps', type=int, default=None,
                    help='Playback fps. Mission defaults to 10 (= real time); swarm to 6.')
    ap.add_argument('--dpi', type=int, default=120)
    ap.add_argument('--outdir', default=_OUT_DIR)
    args, _ = ap.parse_known_args()

    if args.demo in ('mission', 'both'):
        p = os.path.join(args.outdir, 'mission_control_demo.mp4')
        info = record_mission(p, fps=args.fps or 10, dpi=args.dpi)
        print(f"mission -> {p}\n   {info}")
    if args.demo in ('swarm', 'both'):
        p = os.path.join(args.outdir, 'swarm_demo.mp4')
        info = record_swarm(p, fps=args.fps or 6, dpi=args.dpi)
        print(f"swarm   -> {p}\n   {info}")


if __name__ == '__main__':
    main()
