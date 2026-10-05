"""Standalone, hands-on live mission demo -- no Godot required.

A simple point-mass "drone" is simulated directly in Python, driven by the
same RecoveryOrchestrator (api/recovery_orchestrator.py) that talks to the
real Godot simulation. Enter a mission target, press START MISSION to watch
it fly there, press CUT CONNECTION to trigger GPS loss on demand and watch
the live LSTM -> EKF -> rule-based recovery hierarchy take over -- rendered
as a 3D trajectory plot styled exactly like the offline thesis figures
(ai_backend/plot_style.py's palette), with a scrolling decision log beside it.

This exists specifically because the Godot version can only be tested and
debugged by a human running the editor -- this tool is self-contained enough
to be built, run, and *visually verified* end-to-end without any GUI at all
(see --headless-test below), which is what made several real Godot bugs
(wrong camera target, silently-zero mission fields, an out-of-map target)
hard to catch without a human in the loop first.

Usage:
  python ai_backend/live_mission_demo.py                    # interactive window
  python ai_backend/live_mission_demo.py --headless-test     # scripted run, saves a PNG, no display needed
"""
import os
import sys
import math
import random
import argparse

import numpy as np

import matplotlib
if '--headless-test' in sys.argv:
    matplotlib.use('Agg')
else:
    # Force a real GUI backend explicitly -- some terminals (VS Code's
    # integrated terminal among them, via its Python extension's plot-
    # capture integration) set MPLBACKEND=agg in the environment even for
    # a plain script run, not just notebook cells. An explicit use() call
    # wins over that env var (see live_swarm_demo.py's identical fix,
    # checked directly there), which is what actually makes the live
    # window appear instead of silently degrading to Agg.
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass   # no Tk available -- run_interactive() below reports this clearly instead of failing silently
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons, CheckButtons
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import COLOR_REFERENCE, COLOR_PREDICTED, COLOR_GROUND_TRUTH, style_axes
from models.ekf_baseline import rotation_from_euler_deg
from api.recovery_orchestrator import RecoveryOrchestrator, MODEL_REGISTRY

DT = 0.1                        # 10 Hz, matches the real telemetry rate elsewhere in the project
GRAVITY = 9.8

# -- Mission-seek controller (the "GPS-based autopilot", honest: suspended the
#    instant GPS is lost, exactly like drone_body.gd's _update_mission_guidance) --
SEEK_KP = 0.6
SEEK_KD = 1.2
MAX_ACCEL = 3.0                 # m/s^2 -- also the "specific force" scale used to derive body-frame accel
ARRIVE_DIST = 0.4
DRAG_PER_S = 0.6

# -- Attitude model: bank angle proportional to commanded horizontal accel,
#    the same physically-motivated coupling drone_body.gd's real mixer produces --
MAX_BANK_DEG = 25.0
ATTITUDE_RATE = 4.0             # 1/s, how fast attitude tracks its target bank angle

# -- AI correction blending -- same rationale/constant as drone_body.gd's
#    CORRECTION_BLEND_RATE: corrections arrive every tick once GPS is lost;
#    blending avoids a visually jarring snap. --
CORRECTION_BLEND_RATE = 3.0

# -- Mission-target bounds -- mirrors hud_layer.gd's MISSION_XZ_LIMIT/ALT so
#    both live tools share one notion of "a reasonable demo target". --
MISSION_XZ_LIMIT = 40.0
MISSION_ALT_MIN = 1.0
MISSION_ALT_MAX = 25.0

MAG_WORLD_DIR = np.array([0.0, -0.6, 0.8])   # same fixed "Earth field" as drone_body.gd's WORLD_MAG_DIR


class SimDrone:
    """Point-mass kinematic drone. Axes match Godot's convention (Y = up) so
    packets can be fed into RecoveryOrchestrator unchanged -- x<->lon,
    y<->alt, z<->lat, exactly as recovery_orchestrator.py already assumes.
    A deliberately simpler abstraction than drone_body.gd's full rigid-body
    sim: commanded acceleration IS the net specific force directly (no
    separate thrust-vs-weight bookkeeping) -- enough to derive physically
    consistent IMU-style telemetry without simulating a motor mixer."""

    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.attitude = np.zeros(3)          # roll, pitch, yaw, degrees
        self._prev_attitude = np.zeros(3)
        self.mission_target = None
        self.mission_active = False
        self.mission_done = False
        self.gps_signal = True
        self._baro_bias = 0.0
        self.t = 0.0
        self.active_layer = "NONE"
        self.active_goal = "NONE"      # "TARGET" | "HOME" | "NONE" -- see recovery_orchestrator.py's RTL_* hybrid
        self._last_goal = ""           # tracks TARGET->HOME transitions for a one-time log line
        self._log_tick_counter = 0     # throttles the routine per-tick correction log line, see step()

        self.executed_path: list[np.ndarray] = []
        self.reference_path: list[np.ndarray] = []
        self.predicted_path: list[np.ndarray] = []
        self._predicted_pos = np.zeros(3)

    def start_mission(self, target) -> None:
        self.mission_target = np.array(target, dtype=float)
        self.mission_active = True
        self.mission_done = False
        self.executed_path.clear()
        self.reference_path.clear()
        self.predicted_path.clear()

    def cut_connection(self) -> None:
        if not self.gps_signal:
            return
        self.gps_signal = False
        if (self.mission_active or self.mission_done) and self.mission_target is not None:
            self.reference_path = [self.pos.copy(), self.mission_target.copy()]
        self.predicted_path = [self.pos.copy()]
        self._predicted_pos = self.pos.copy()
        self._last_goal = ""

    def restore_gps(self) -> None:
        self.gps_signal = True

    def step(self, dt: float, orchestrator: RecoveryOrchestrator, force_layer: str, log) -> None:
        self.t += dt

        # -- Mission guidance: TRUE position, only while GPS is actually live.
        #    The instant gps_signal is false this contributes nothing -- guidance
        #    never secretly keeps steering from ground truth once "GPS is lost"
        #    is supposed to be in effect (same rule as drone_body.gd). --
        accel_cmd = np.zeros(3)
        if self.mission_active and not self.mission_done and self.gps_signal and self.mission_target is not None:
            err = self.mission_target - self.pos
            horiz = np.array([err[0], 0.0, err[2]])
            if np.linalg.norm(horiz) < ARRIVE_DIST and abs(err[1]) < ARRIVE_DIST:
                self.mission_done = True
                self.mission_active = False
                log("MISSION COMPLETE -- holding position")
            else:
                accel_cmd = np.clip(SEEK_KP * err - SEEK_KD * self.vel, -MAX_ACCEL, MAX_ACCEL)
                self.vel = self.vel + accel_cmd * dt

        self.vel *= math.exp(-DRAG_PER_S * dt)
        self.pos = self.pos + self.vel * dt

        target_roll = float(np.clip(accel_cmd[0] / MAX_ACCEL, -1.0, 1.0) * MAX_BANK_DEG)
        target_pitch = float(np.clip(-accel_cmd[2] / MAX_ACCEL, -1.0, 1.0) * MAX_BANK_DEG)
        blend = min(1.0, ATTITUDE_RATE * dt)
        self.attitude[0] += (target_roll - self.attitude[0]) * blend
        self.attitude[1] += (target_pitch - self.attitude[1]) * blend
        speed = float(np.linalg.norm(self.vel))
        if speed > 0.2:
            self.attitude[2] = math.degrees(math.atan2(self.vel[0], self.vel[2]))

        gyro = (self.attitude - self._prev_attitude) / dt
        self._prev_attitude = self.attitude.copy()

        R = rotation_from_euler_deg(self.attitude[0], self.attitude[1], self.attitude[2])
        # Specific force in body frame -- accel_cmd already excludes gravity
        # (this sim's "hover" abstraction), so no separate gravity subtraction
        # is needed here, unlike drone_body.gd's full rigid-body formula.
        accel_body = R.T @ accel_cmd
        # Small sensor noise (same role as drone_body.gd's baro/mag noise
        # terms below) -- without it, a coasting drone's IMU window is
        # perfectly static and the LSTM's per-window prediction never varies,
        # which reads as "stuck" rather than "live" during GPS loss.
        accel_body = accel_body + np.random.uniform(-0.03, 0.03, size=3)
        gyro = gyro + np.random.uniform(-0.02, 0.02, size=3)

        self._baro_bias = float(np.clip(self._baro_bias + random.uniform(-0.02, 0.02) * dt, -1.0, 1.0))
        baro_alt = self.pos[1] + self._baro_bias + random.uniform(-0.03, 0.03)
        mag_body = R.T @ MAG_WORLD_DIR

        packet = {
            "gps_signal": self.gps_signal,
            "lat": float(self.pos[2]), "lon": float(self.pos[0]), "alt": max(0.0, float(self.pos[1])),
            "imu_acc_x": float(accel_body[0]), "imu_acc_y": float(accel_body[1]), "imu_acc_z": float(accel_body[2]),
            "imu_gyro_x": float(gyro[0]), "imu_gyro_y": float(gyro[1]), "imu_gyro_z": float(gyro[2]),
            "roll": float(self.attitude[0]), "pitch": float(self.attitude[1]), "yaw": float(self.attitude[2]),
            "mag_x": float(mag_body[0]), "mag_y": float(mag_body[1]), "mag_z": float(mag_body[2]),
            "baro_alt": float(baro_alt), "speed": speed,
        }
        if self.mission_active and self.mission_target is not None:
            packet["mission_target"] = self.mission_target.tolist()

        self.executed_path.append(self.pos.copy())

        if not self.gps_signal:
            forced = force_layer if force_layer != "Auto" else None
            result = orchestrator.decide("SimDrone", packet, now=self.t, forced_layer=forced)
            self.active_layer = result["layer"]
            self.active_goal = result["goal"]
            v = np.array(result["velocity_cmd"])
            corr_blend = min(1.0, dt * CORRECTION_BLEND_RATE)
            self.vel = self.vel * (1.0 - corr_blend) + v * corr_blend
            self._predicted_pos = self._predicted_pos + v * dt
            self.predicted_path.append(self._predicted_pos.copy())
            if self.active_goal == "HOME" and self._last_goal != "HOME":
                log("RTL ENGAGED -- mission target abandoned, heading home instead")
            self._last_goal = self.active_goal
            # Routine per-tick correction lines are throttled to ~1/s (DT is
            # ~0.1s/tick) -- logging every tick during a long GPS-loss window
            # floods the capped log buffer and pushes one-time events like
            # "RTL ENGAGED" out before anyone (or this file's own headless
            # test) gets to see them.
            self._log_tick_counter += 1
            if self._log_tick_counter % 10 == 0:
                log(f"{result['layer']} correction (goal={self.active_goal}) -> "
                    f"v=({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
        else:
            orchestrator.decide("SimDrone", packet, now=self.t)   # keeps the buffer warm for when GPS drops
            self.active_layer = "NONE"
            self.active_goal = "NONE"


class LiveDashboard:
    def __init__(self, model_name: str = "LSTM", use_rl_seek: bool = False):
        self.drone = SimDrone()
        self.model_name = model_name
        # Checkbox state is tracked separately from whatever
        # set_navigation_policy() actually achieved -- mirrors self.model_name
        # vs. set_model()'s own success/failure split above, same reasoning:
        # a missing checkpoint should be reported, not silently pretended away.
        self.rl_seek_enabled = use_rl_seek
        self.orchestrator = RecoveryOrchestrator(model_name=model_name, use_rl_seek=use_rl_seek)
        self.force_layer = "Auto"
        self.log_lines: list[str] = []
        # Display-only elapsed-time offset -- set to self.drone.t whenever
        # START MISSION is pressed, so the title/log timestamps count from
        # 0 at that moment (see _display_t()). self.drone.t itself is never
        # reset: it's the orchestrator's own absolute clock, passed to
        # decide()'s `now` on every packet, and RecoveryOrchestrator uses it
        # to detect stale reconnect gaps -- resetting it mid-session would
        # corrupt that, independent of what the UI happens to display.
        self.mission_start_time: float | None = None
        self.paused = False

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.fig.patch.set_facecolor("white")
        # Sits above both axes' own titles (which end near y=0.86) with clear room to spare.
        self.suptitle = self.fig.suptitle("", fontsize=11, fontweight="bold", y=0.985)

        self.ax3d = self.fig.add_axes([0.04, 0.30, 0.55, 0.56], projection="3d")
        style_axes(self.ax3d, title="Live Mission - 3D Trajectory",
                   xlabel="Position X [m]", ylabel="Position Y [m]", zlabel="Altitude Z [m]")
        (self.line_ref,) = self.ax3d.plot([], [], [], color=COLOR_REFERENCE, linewidth=1.5,
                                           linestyle="-", label="Reference / Mission Path")
        (self.line_pred,) = self.ax3d.plot([], [], [], color=COLOR_PREDICTED, linewidth=1.3,
                                            linestyle="-", label="Recovery Estimate (Predicted)")
        (self.line_exec,) = self.ax3d.plot([], [], [], color=COLOR_GROUND_TRUTH, linewidth=1.5,
                                            linestyle="-", label="Executed (True) Path")
        self.ax3d.legend(loc="upper left", fontsize=8, frameon=False)

        self.ax_log = self.fig.add_axes([0.63, 0.30, 0.34, 0.56])
        self.ax_log.axis("off")
        # "DECISION LOG" is the first line of the text block itself, not an
        # axes title -- axes titles float just above the axes' bounding box
        # and collided with the figure suptitle at this layout's proportions.
        self.log_text = self.ax_log.text(0.0, 1.0, "", va="top", ha="left", family="monospace",
                                          fontsize=7.5, transform=self.ax_log.transAxes)

        # Widgets are always built, even in headless mode -- Agg still draws
        # static Button/TextBox/RadioButtons artists correctly (they just
        # don't respond to clicks), so a headless snapshot shows the same
        # layout a real interactive session would, letting layout issues be
        # caught from a saved PNG instead of only by a human running it live.
        self._build_widgets()
        # RadioButtons' active= only sets which option is drawn selected at
        # construction -- it does not fire on_clicked, so the startup model
        # (unlike every later switch) would otherwise never appear in the
        # decision log, making it look like nothing was chosen yet.
        self.log(f"Model -> {self.model_name} (starting default)")

    # -- widgets -------------------------------------------------------------
    def _build_widgets(self) -> None:
        self.tb_x = TextBox(self.fig.add_axes([0.06, 0.16, 0.07, 0.05]), "X ", initial="10.0")
        self.tb_z = TextBox(self.fig.add_axes([0.19, 0.16, 0.07, 0.05]), "Z ", initial="8.0")
        self.tb_alt = TextBox(self.fig.add_axes([0.32, 0.16, 0.07, 0.05]), "Alt ", initial="4.0")

        # x=0.46-0.63 (right of the Reset/End/Pause/Resume column, left of
        # the Model picker) is open the whole way down -- moved here rather
        # than squeezed into the ~0.02 gap directly above the new button
        # row, which visibly touched RESET's top edge at this text's
        # previous position.
        self.fig.text(0.46, 0.115,
                       f"local metres -- X/Z within ±{MISSION_XZ_LIMIT:.0f}, "
                       f"Alt {MISSION_ALT_MIN:.0f}-{MISSION_ALT_MAX:.0f}",
                       fontsize=7.5, color="#666666")

        self.btn_start = Button(self.fig.add_axes([0.44, 0.16, 0.13, 0.05]), "START MISSION",
                                 color="#1d5c2b", hovercolor="#237536")
        self.btn_start.label.set_color("white")
        self.btn_start.on_clicked(self._on_start)

        self.btn_cut = Button(self.fig.add_axes([0.59, 0.16, 0.13, 0.05]), "CUT CONNECTION",
                               color="#7a1f1f", hovercolor="#932727")
        self.btn_cut.label.set_color("white")
        self.btn_cut.on_clicked(self._on_cut)

        self.btn_restore = Button(self.fig.add_axes([0.74, 0.16, 0.13, 0.05]), "RESTORE GPS",
                                   color="#1f4e79", hovercolor="#26629c")
        self.btn_restore.label.set_color("white")
        self.btn_restore.on_clicked(self._on_restore)

        # Secondary run-control row -- deliberately smaller than the
        # primary Start/Cut/Restore row above (utility actions, not the
        # main flow), sitting below it and clear of the Model picker (which
        # only starts at x=0.63). RESUME sits directly beneath PAUSE, same
        # x-column, as its natural pair.
        self.btn_reset = Button(self.fig.add_axes([0.06, 0.095, 0.10, 0.045]), "RESET",
                                 color="#555555", hovercolor="#6e6e6e")
        self.btn_reset.label.set_color("white")
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._on_reset)

        self.btn_end_mission = Button(self.fig.add_axes([0.18, 0.095, 0.13, 0.045]), "END MISSION",
                                       color="#8a5a00", hovercolor="#a86f00")
        self.btn_end_mission.label.set_color("white")
        self.btn_end_mission.label.set_fontsize(8)
        self.btn_end_mission.on_clicked(self._on_end_mission)

        self.btn_pause = Button(self.fig.add_axes([0.33, 0.095, 0.09, 0.045]), "PAUSE",
                                 color="#3d5a6b", hovercolor="#4d7086")
        self.btn_pause.label.set_color("white")
        self.btn_pause.label.set_fontsize(8)
        self.btn_pause.on_clicked(self._on_pause)

        self.btn_resume = Button(self.fig.add_axes([0.33, 0.03, 0.09, 0.045]), "RESUME",
                                  color="#2d6b3a", hovercolor="#379149")
        self.btn_resume.label.set_color("white")
        self.btn_resume.label.set_fontsize(8)
        self.btn_resume.on_clicked(self._on_resume)

        self.fig.text(0.94, 0.235, "Force layer", fontsize=8, ha="center")
        self.radio = RadioButtons(self.fig.add_axes([0.89, 0.02, 0.10, 0.20]),
                                   ["Auto", "LSTM", "EKF", "RULE_BASED"])
        self.radio.on_clicked(self._on_force_layer)

        # Model picker -- which trained architecture answers for the "LSTM"
        # layer (see recovery_orchestrator.py's MODEL_REGISTRY; the reported
        # layer name stays "LSTM" regardless of which one is selected here,
        # this only changes which checkpoint is actually driving it). Sits
        # in the free strip below the mission-control button row and to the
        # left of "Force layer" -- both are on the same row, own separate
        # columns, no shared axes.
        # y=0.175's label previously sat inside the RESTORE GPS button's own
        # bounding box (button: x 0.74-0.87, y 0.16-0.21; label x was 0.75,
        # squarely inside that y-range too) -- visually overlapping it, not
        # just close to it. Whole widget shifted down to sit fully below the
        # button row (bottom edge 0.16) instead, with a small gap.
        self.fig.text(0.75, 0.148, "Model", fontsize=8, ha="center")
        # Six labels (one of them "SSL-Transformer", the longest) in the
        # tallest box this slot allows without reaching the button row
        # above -- small label font is what actually keeps them from
        # overlapping each other at this height, ncols is not available on
        # this matplotlib's RadioButtons.
        self.radio_model = RadioButtons(
            self.fig.add_axes([0.63, 0.005, 0.24, 0.13]), list(MODEL_REGISTRY),
            active=list(MODEL_REGISTRY).index(self.model_name),
            label_props={'fontsize': [6] * len(MODEL_REGISTRY)},
            radio_props={'s': [24] * len(MODEL_REGISTRY)})
        self.radio_model.on_clicked(self._on_model_change)

        # RL seek toggle -- a binary yes/no axis (does _apply_seek_bias use
        # the learned RL policy or the original hand-coded formula),
        # deliberately CheckButtons rather than another RadioButtons: this
        # isn't a third architecture choice alongside Force layer/Model, it's
        # orthogonal to both (RL replaces only the seek-bias term, on top of
        # whichever estimator layer/model is currently active). Sits in the
        # free strip above the Model picker, same x-column -- raised clear of
        # the START/CUT/RESTORE button row's top edge (y=0.21), which it used
        # to overlap by about 0.01 (found directly: the checkbox's own axes
        # bottom sat below the button row's top).
        self.fig.text(0.75, 0.29, "Navigation", fontsize=8, ha="center")
        self.check_rl_seek = CheckButtons(
            self.fig.add_axes([0.66, 0.23, 0.18, 0.05]), ["RL seek policy"],
            actives=[self.rl_seek_enabled], label_props={'fontsize': [7]})
        self.check_rl_seek.on_clicked(self._on_rl_seek_toggle)

    def _on_start(self, _event) -> None:
        try:
            x = float(self.tb_x.text)
            z = float(self.tb_z.text)
            alt = float(self.tb_alt.text)
        except ValueError:
            self.log("Invalid target -- X/Z/Alt must be numbers")
            return
        x = float(np.clip(x, -MISSION_XZ_LIMIT, MISSION_XZ_LIMIT))
        z = float(np.clip(z, -MISSION_XZ_LIMIT, MISSION_XZ_LIMIT))
        alt = float(np.clip(alt, MISSION_ALT_MIN, MISSION_ALT_MAX))
        self.mission_start_time = self.drone.t   # display clock zeroes here -- see _display_t()
        self.drone.start_mission([x, alt, z])
        self.log(f"MISSION START -> target ({x:.1f}, {alt:.1f}, {z:.1f})")

    def _on_cut(self, _event) -> None:
        if self.drone.gps_signal:
            self.drone.cut_connection()
            self.log("GPS SIGNAL CUT -- mission autopilot suspended, awaiting recovery guidance")

    def _on_restore(self, _event) -> None:
        if not self.drone.gps_signal:
            self.drone.restore_gps()
            self.log("GPS SIGNAL RESTORED")

    def _on_pause(self, _event) -> None:
        if self.paused:
            return
        self.paused = True
        self.log("PAUSED")

    def _on_resume(self, _event) -> None:
        if not self.paused:
            return
        self.paused = False
        self.log("RESUMED")

    def _on_end_mission(self, _event) -> None:
        """Aborts the active mission without touching flight state -- the
        drone keeps its current position/velocity/GPS status, it just stops
        seeking a target. Distinct from RESET, which discards everything.
        Dropping mission_target makes SimDrone.step() stop attaching it to
        outgoing packets, so recovery_orchestrator.py's decide() naturally
        falls back to steering toward home instead (its own target-is-None
        branch), no extra plumbing needed here."""
        if not self.drone.mission_active:
            self.log("No active mission to end")
            return
        self.drone.mission_active = False
        self.drone.mission_done = False
        self.drone.mission_target = None
        self.log("MISSION TERMINATED -- target cleared, defaulting to home-seeking")

    def _on_reset(self, _event) -> None:
        """Full reset: fresh drone, fresh orchestrator (flight/recovery
        state discarded -- unlike a model switch via set_model(), which
        deliberately preserves it, a reset is supposed to discard it, that
        is the whole point of the button), GPS restored, force-layer back
        to Auto, timer and log cleared. Keeps the currently-selected model
        architecture rather than reverting to LSTM -- RESET clears the
        flight, not a deliberate model choice made via the picker. Same
        reasoning for the RL seek toggle: RESET discards flight state, not
        a deliberate navigation-policy choice made via the checkbox."""
        self.drone = SimDrone()
        self.orchestrator = RecoveryOrchestrator(model_name=self.model_name,
                                                   use_rl_seek=self.rl_seek_enabled)
        self.force_layer = "Auto"
        self.radio.set_active(0)   # sync the Force-layer widget's own display back to Auto
        self.mission_start_time = None
        self.paused = False
        self.log_lines = []
        self.log("SIMULATION RESET")
        self._redraw()

    def _on_force_layer(self, label: str) -> None:
        self.force_layer = label
        self.log(f"Force layer -> {label}")

    def _on_model_change(self, label: str) -> None:
        """Swaps the orchestrator's model in place (RecoveryOrchestrator.
        set_model(), not a new RecoveryOrchestrator instance) -- deliberately
        preserves any in-progress recovery episode's home position, RTL
        status, and EKF/believed-position state, so switching architectures
        mid-flight compares them fairly on the same episode rather than
        silently resetting it (see set_model()'s docstring for the real bug
        this replaced). Loading can take a moment for the larger arms
        (Transformer/SSL-Transformer) -- logged so it's clear the pause is
        expected, not a hang. Reports whether the checkpoint actually
        loaded rather than assuming success -- a missing production
        checkpoint degrades to EKF/rule-based only, silently claiming
        "switched" either way would mislead whoever is watching the log."""
        if label == self.model_name:
            return
        self.log(f"Switching model -> {label} (reloading checkpoint)...")
        ok = self.orchestrator.set_model(label)
        if ok:
            self.log(f"Model switched -> {label}")
        else:
            self.log(f"Model switch -> {label} FAILED: no checkpoint found, "
                     f"degraded to EKF/rule-based only")

    def _on_rl_seek_toggle(self, label: str) -> None:
        """Toggles whether _apply_seek_bias() uses the learned RL policy or
        the original hand-coded proportional formula, via
        set_navigation_policy() (mirrors set_model()'s exact contract: an
        in-place swap that preserves any in-progress recovery episode's
        state, reports whether the checkpoint actually loaded rather than
        assuming success -- same reasoning as _on_model_change() above)."""
        enabled = not self.rl_seek_enabled   # CheckButtons reports the clicked label, not the new state
        ok = self.orchestrator.set_navigation_policy(enabled)
        if enabled and not ok:
            self.log("RL seek ON requested -- no trained checkpoint found, staying on the hand-coded formula")
            self.check_rl_seek.set_active(0, False)   # resync the widget's own display to the actual (failed) state
        else:
            self.rl_seek_enabled = enabled
            self.log(f"RL seek policy -> {'ON' if enabled else 'OFF'}")

    # -- simulation / rendering loop ----------------------------------------
    def _display_t(self) -> float:
        """Elapsed time since START MISSION was last pressed (0.0 before
        any mission has started, or right after RESET) -- what the title
        and every log line's timestamp actually shows. See
        self.mission_start_time's docstring for why this is a display-only
        offset, not a reset of self.drone.t itself."""
        if self.mission_start_time is None:
            return 0.0
        return self.drone.t - self.mission_start_time

    def log(self, msg: str) -> None:
        line = f"[T+{self._display_t():6.2f}s] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)
        print(line)

    def tick(self, dt: float = DT) -> None:
        if not self.paused:
            self.drone.step(dt, self.orchestrator, self.force_layer, self.log)
        self._redraw()

    def _redraw(self) -> None:
        def as_xyz(path: list[np.ndarray]):
            if len(path) < 2:
                return [], [], []
            p = np.array(path)
            # display mapping: sim (x, y=up, z) -> plot (X, Y, Altitude Z), matching trajectory_3d.png's axis roles
            return p[:, 0].tolist(), p[:, 2].tolist(), p[:, 1].tolist()

        self.line_ref.set_data_3d(*as_xyz(self.drone.reference_path))
        self.line_pred.set_data_3d(*as_xyz(self.drone.predicted_path))
        self.line_exec.set_data_3d(*as_xyz(self.drone.executed_path))

        all_pts = [*self.drone.reference_path, *self.drone.predicted_path, *self.drone.executed_path]
        if all_pts:
            arr = np.array(all_pts)
            xs, ys, zs = arr[:, 0], arr[:, 2], arr[:, 1]

            def bounds(lo, hi):
                if hi - lo < 1.0:
                    lo, hi = lo - 1.0, hi + 1.0
                pad = (hi - lo) * 0.15
                return lo - pad, hi + pad

            self.ax3d.set_xlim(*bounds(xs.min(), xs.max()))
            self.ax3d.set_ylim(*bounds(ys.min(), ys.max()))
            self.ax3d.set_zlim(*bounds(zs.min(), zs.max()))

        self.log_text.set_text("\n".join(self.log_lines[-26:]))

        gps_txt = "OK" if self.drone.gps_signal else "LOST"
        mission_txt = "ACTIVE" if self.drone.mission_active else ("DONE" if self.drone.mission_done else "-")
        paused_txt = "    [PAUSED]" if self.paused else ""
        self.suptitle.set_text(
            f"t = {self._display_t():6.1f}s    GPS = {gps_txt}    Active layer = {self.drone.active_layer}    "
            f"Goal = {self.drone.active_goal}    Mission = {mission_txt}{paused_txt}")

    def run_interactive(self):
        backend = matplotlib.get_backend()
        # A live FuncAnimation needs a real GUI event loop. Plain 'Agg' or
        # matplotlib_inline's inline-agg backend (what VS Code's Jupyter
        # "Run in Interactive Window" / "Run Cell" auto-selects instead of a
        # plain terminal) can only render static frames -- this, not a
        # missing GUI toolkit, is what produces a cryptic "FigureCanvasAgg
        # is non-interactive" / "Animation was deleted" warning instead of
        # an actual window (see live_swarm_demo.py's identical check).
        if backend.lower() == 'agg' or 'inline' in backend.lower():
            print(f"\nNo interactive display available (matplotlib backend = '{backend}').\n"
                  f"Run this from a plain terminal instead of a Jupyter/VS Code "
                  f"'Run in Interactive Window' cell, e.g.:\n"
                  f"    python ai_backend/live_mission_demo.py\n"
                  f"Or use --headless-test to generate a PNG snapshot with no display needed.\n")
            return None
        ani = FuncAnimation(self.fig, lambda _f: self.tick(), interval=100, cache_frame_data=False)
        plt.show()
        return ani

    def run_headless_test(self, out_path: str) -> None:
        """Scripted, no-GUI run: start a mission, cut the connection almost
        immediately (before any real progress) toward a target far enough
        that the capped seek-bias can't close it within RTL_TIMEOUT_S, hold
        the loss long enough to watch the hybrid abort to RTL, then restore
        GPS and let the original mission resume -- self-contained enough to
        verify visually without a display or a human at the controls."""
        target = [30.0, 4.0, 25.0]
        self.mission_start_time = self.drone.t   # mirrors _on_start() -- see _display_t()
        self.drone.start_mission(target)
        self.log(f"MISSION START -> target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

        for i in range(420):
            self.tick()
            if i == 3:
                self.drone.cut_connection()
                self.log("GPS SIGNAL CUT -- mission autopilot suspended, awaiting recovery guidance")
            if i == 300:
                self.drone.restore_gps()
                self.log("GPS SIGNAL RESTORED")

        self._redraw()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.fig.savefig(out_path, dpi=140)

        print(f"\nSaved snapshot -> {out_path}")
        print(f"Final position (x, y=alt, z): {self.drone.pos}")
        print(f"Mission target:               {np.array(target)}")
        print(f"Final distance to target:     {np.linalg.norm(self.drone.pos - np.array(target)):.2f} m")
        print(f"Executed-path points:  {len(self.drone.executed_path)}")
        print(f"Predicted-path points: {len(self.drone.predicted_path)}")
        rtl_engaged = any("RTL ENGAGED" in line for line in self.log_lines)
        print(f"RTL engaged during this run:   {rtl_engaged}")
        assert len(self.drone.executed_path) > 300, "executed path should have ~one point per tick"
        assert len(self.drone.predicted_path) > 100, "predicted path should have accumulated during the GPS-loss window"
        assert rtl_engaged, ("expected the hybrid target-then-RTL logic to abort to home for a target this far, "
                              "cut this early -- see recovery_orchestrator.py's RTL_TIMEOUT_S/RTL_SKIP_IF_CLOSE_M")

        # -- Exercise pause/resume/end-mission/reset too, not just their
        #    static widget layout -- doesn't touch the snapshot already
        #    saved above, pure logic/state assertions from here on.
        print("\n--- Pause / resume / end mission / reset ---")
        pos_before_pause = self.drone.pos.copy()
        self._on_pause(None)
        assert self.paused
        for _ in range(5):
            self.tick()
        assert np.allclose(self.drone.pos, pos_before_pause), "position moved while paused"
        print("Pause: position held steady, as expected.")

        self._on_resume(None)
        assert not self.paused
        for _ in range(5):
            self.tick()
        assert not np.allclose(self.drone.pos, pos_before_pause), "position should change once resumed"
        print("Resume: position advanced again, as expected.")

        self._on_end_mission(None)
        assert not self.drone.mission_active
        assert self.drone.mission_target is None
        print("End mission: mission_active cleared, target dropped, as expected.")

        self._on_reset(None)
        assert self.mission_start_time is None
        assert np.allclose(self.drone.pos, np.zeros(3))
        assert self.drone.gps_signal
        assert self.force_layer == "Auto"
        assert self._display_t() == 0.0
        print("Reset: drone/orchestrator/state back to a fresh baseline, as expected.")

        print("\nOK -- headless test completed without error.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless-test", action="store_true",
                     help="Run a scripted mission/cut/recover/restore sequence with no GUI, "
                          "saving a PNG snapshot for visual inspection.")
    ap.add_argument("--out", default=None, help="Snapshot output path (headless-test only).")
    ap.add_argument("--model", type=str, default="LSTM", choices=list(MODEL_REGISTRY),
                     help="Which trained architecture backs the LSTM fail-safe layer at "
                          "startup (default: LSTM). Can also be changed live via the "
                          "'Model' radio buttons.")
    ap.add_argument("--rl-seek", action="store_true",
                     help="Start with the learned RL seek policy active instead of the "
                          "hand-coded formula (default: off -- opt-in, same reasoning as "
                          "RecoveryOrchestrator's own use_rl_seek default). Degrades cleanly "
                          "to the hand-coded formula if no checkpoint exists. Can also be "
                          "toggled live via the 'RL seek policy' checkbox.")
    args = ap.parse_args()

    dashboard = LiveDashboard(model_name=args.model, use_rl_seek=args.rl_seek)

    if args.headless_test:
        out = args.out or os.path.join(os.path.dirname(__file__), "..", "runs", "live_mission_demo_test.png")
        dashboard.run_headless_test(out)
    else:
        _animation_ref = dashboard.run_interactive()
