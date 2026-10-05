"""Standalone, hands-on live multi-drone swarm demo -- no Godot required,
mirroring live_mission_demo.py's own pattern (self-contained Python
simulation + matplotlib dashboard + --headless-test) but for the swarm-
connectivity track instead of single-drone position recovery.

N drones + 1 fixed base station move around a search area. Each drone's
BELIEVED position is its true position plus accumulated dead-reckoning
drift whenever its GPS is lost -- the exact same drift model
swarm_network_generator.py used to build the offline training set (same
DRIFT_SIGMA_LOW/HIGH, RTL_TIMEOUT_S, DT from models/rl_path_recovery.py),
just integrated tick-by-tick here instead of sampled as one snapshot.

Every tick, the trained Method A model (runs/swarm_connectivity/
method_a_model.pkl -- the best-performing, dependency-lightest of the
three methods trained in train_swarm_connectivity.py, see its own
comparison.json) scores the current believed network topology for
fragmentation risk, using extract_graph_features() imported directly from
that file rather than reimplemented here. This is a genuine live
inference loop, not a canned animation.

Usage:
  python ai_backend/live_swarm_demo.py                    # interactive window
  python ai_backend/live_swarm_demo.py --headless-test     # scripted run, saves a PNG, no display needed
"""
import os
import sys
import argparse
import pickle

import numpy as np
import networkx as nx

import matplotlib
if '--headless-test' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import style_axes
from data_processing.swarm_network_generator import (
    AREA_SIZE_M, COMM_RANGE_M, MAX_SPEED_MPS, N_NODE_FEATURES, _adjacency_from_positions,
)
from models.rl_path_recovery import DRIFT_SIGMA_LOW, DRIFT_SIGMA_HIGH, RTL_TIMEOUT_S, DT
# train_swarm_connectivity imports set_global_seed/confidence_interval from
# train_px4_synthetic.py, which unconditionally calls matplotlib.use('Agg')
# at module load (correct for that always-headless training script, but it
# silently clobbers any backend chosen before this import -- checked
# directly). The real fix is below, AFTER this import, not before it.
from train_swarm_connectivity import extract_graph_features

if '--headless-test' not in sys.argv:
    # Re-assert a real GUI backend now that every import that could have
    # forced Agg (see above) has already run. Also wins over MPLBACKEND=agg
    # injected by some terminals (VS Code's integrated terminal among them,
    # via its Python extension's plot-capture integration) even for a plain
    # script run, not just notebook cells -- an explicit use() call beats
    # that env var (checked directly).
    try:
        matplotlib.use('TkAgg', force=True)
    except ImportError:
        pass   # no Tk available -- run_interactive() below reports this clearly instead of failing silently

N_DRONES = 8                    # fixed for the live demo -- middle of the 6-10 range the model trained on
TICK_DT = 0.2                   # sim seconds per tick (coarser than DT=0.1 -- a smoother, watchable demo speed)
WANDER_ACCEL_STD = 0.4          # m/s^2 per tick, random heading drift -- keeps drones patrolling, not static
RISK_LOW, RISK_HIGH = 0.30, 0.60   # display thresholds only, match the green/amber/red gauge bands

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'swarm_connectivity', 'method_a_model.pkl')


class SwarmSim:
    """N drones + 1 fixed base station (node 0) on a local-metres square.
    Every drone always carries SOME believed-position uncertainty once GPS
    is lost, exactly like the training data's own node-feature schema
    (x, y, vx, vy, uncertainty, elapsed_loss/RTL_TIMEOUT_S) -- a drone with
    GPS currently OK reports uncertainty=0 / elapsed=0, which sits slightly
    below the training distribution's own floor (drift_sigma at
    elapsed=0, never exactly 0 in the offline data -- see
    swarm_network_generator.generate_scenario()). Left as an honest,
    small, boundary-only extrapolation rather than re-designing the live
    demo's GPS-toggle semantics around it."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.base_pos = np.array([AREA_SIZE_M / 2.0, AREA_SIZE_M / 2.0])
        self.pos = self.rng.uniform(0.0, AREA_SIZE_M, size=(N_DRONES, 2))
        self.vel = self.rng.normal(0.0, 0.5, size=(N_DRONES, 2))
        self.gps_signal = np.ones(N_DRONES, dtype=bool)
        self.drift_sigma = np.exp(self.rng.uniform(np.log(DRIFT_SIGMA_LOW), np.log(DRIFT_SIGMA_HIGH), size=N_DRONES))
        self.elapsed_loss = np.zeros(N_DRONES)
        self.uncertainty = np.zeros(N_DRONES)
        self.accumulated_error = np.zeros((N_DRONES, 2))
        self.believed_pos = self.pos.copy()
        self.t = 0.0
        self._rtl_warned = np.zeros(N_DRONES, dtype=bool)

        self.node_features = np.zeros((N_DRONES + 1, N_NODE_FEATURES), dtype=np.float32)
        self.believed_adjacency = np.zeros((N_DRONES + 1, N_DRONES + 1), dtype=np.float32)
        self.reachable_from_base = set(range(N_DRONES + 1))
        self.risk = 0.0

    def cut_gps(self, i: int) -> None:
        self.gps_signal[i] = False
        self.elapsed_loss[i] = 0.0
        self.uncertainty[i] = 0.0
        self.accumulated_error[i] = 0.0
        self._rtl_warned[i] = False

    def restore_gps(self, i: int) -> None:
        self.gps_signal[i] = True
        self.elapsed_loss[i] = 0.0
        self.uncertainty[i] = 0.0
        self.accumulated_error[i] = 0.0

    def step(self, dt: float, log) -> None:
        self.t += dt

        self.vel += self.rng.normal(0.0, WANDER_ACCEL_STD, size=(N_DRONES, 2)) * dt
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_SPEED_MPS / np.maximum(speed, 1e-9))
        self.vel *= scale
        self.pos += self.vel * dt

        # Reflect off the search-area boundary -- keeps the swarm patrolling
        # a bounded area instead of wandering off-screen (a real search
        # pattern would loiter, this is the cheapest stand-in for that).
        for d in range(2):
            below = self.pos[:, d] < 0.0
            above = self.pos[:, d] > AREA_SIZE_M
            self.vel[below | above, d] *= -1.0
            self.pos[:, d] = np.clip(self.pos[:, d], 0.0, AREA_SIZE_M)

        for i in range(N_DRONES):
            if self.gps_signal[i]:
                self.believed_pos[i] = self.pos[i]
                continue
            self.elapsed_loss[i] += dt
            new_unc = self.drift_sigma[i] * np.sqrt(max(self.elapsed_loss[i] / DT, 1.0))
            var_increment = max(new_unc ** 2 - self.uncertainty[i] ** 2, 0.0)
            self.accumulated_error[i] += self.rng.normal(0.0, np.sqrt(var_increment), size=2)
            self.uncertainty[i] = new_unc
            self.believed_pos[i] = self.pos[i] + self.accumulated_error[i]
            if self.elapsed_loss[i] > RTL_TIMEOUT_S and not self._rtl_warned[i]:
                self._rtl_warned[i] = True
                log(f"Drone {i}: RTL_TIMEOUT_S exceeded (real hybrid -> RTL)")

        all_believed = np.vstack([self.base_pos[None, :], self.believed_pos])
        self.believed_adjacency = _adjacency_from_positions(all_believed, COMM_RANGE_M)

        self.node_features[0] = [self.base_pos[0], self.base_pos[1], 0.0, 0.0, 0.0, 0.0]
        self.node_features[1:, 0:2] = self.believed_pos
        self.node_features[1:, 2:4] = self.vel
        self.node_features[1:, 4] = self.uncertainty
        self.node_features[1:, 5] = self.elapsed_loss / RTL_TIMEOUT_S

        G = nx.from_numpy_array(self.believed_adjacency)
        self.reachable_from_base = nx.node_connected_component(G, 0)

        return G


class LiveSwarmDashboard:
    def __init__(self):
        self.sim = SwarmSim()
        self.selected_drone = 0
        self.paused = False
        self.log_lines: list[str] = []

        self.scaler = self.mlp = None
        if os.path.exists(_MODEL_PATH):
            with open(_MODEL_PATH, 'rb') as f:
                artifacts = pickle.load(f)
            self.scaler, self.mlp = artifacts['scaler'], artifacts['mlp']
            self.log(f"Loaded Method A connectivity model (val Macro-F1={artifacts['val_f1']:.3f})")
        else:
            self.log(f"WARNING: no trained model at {_MODEL_PATH} -- risk gauge will show N/A "
                     f"(run train_swarm_connectivity.py first)")

        self._prev_fragmented = False
        self._prev_risk_band = "LOW"

        self.fig = plt.figure(figsize=(12.5, 7.2))
        self.fig.patch.set_facecolor("white")
        self.suptitle = self.fig.suptitle("", fontsize=11, fontweight="bold", y=0.985)

        self.ax_net = self.fig.add_axes([0.05, 0.30, 0.55, 0.56])
        style_axes(self.ax_net, title="Live Swarm - Believed Connectivity Graph",
                   xlabel="X [m]", ylabel="Y [m]")
        self.ax_net.set_xlim(-20, AREA_SIZE_M + 20)
        self.ax_net.set_ylim(-20, AREA_SIZE_M + 20)
        self.ax_net.set_aspect("equal")

        self.ax_log = self.fig.add_axes([0.63, 0.30, 0.34, 0.56])
        self.ax_log.axis("off")
        self.log_text = self.ax_log.text(0.0, 1.0, "", va="top", ha="left", family="monospace",
                                          fontsize=7.5, transform=self.ax_log.transAxes)

        self._build_widgets()
        self.log(f"SWARM READY -- {N_DRONES} drones + base station, "
                 f"area {AREA_SIZE_M:.0f}m, comm range {COMM_RANGE_M:.0f}m")

    # -- widgets -------------------------------------------------------------
    def _build_widgets(self) -> None:
        self.btn_cut = Button(self.fig.add_axes([0.06, 0.16, 0.15, 0.05]), "CUT GPS (selected)",
                               color="#7a1f1f", hovercolor="#932727")
        self.btn_cut.label.set_color("white")
        self.btn_cut.label.set_fontsize(8)
        self.btn_cut.on_clicked(self._on_cut)

        self.btn_restore = Button(self.fig.add_axes([0.23, 0.16, 0.15, 0.05]), "RESTORE GPS (selected)",
                                   color="#1f4e79", hovercolor="#26629c")
        self.btn_restore.label.set_color("white")
        self.btn_restore.label.set_fontsize(8)
        self.btn_restore.on_clicked(self._on_restore)

        self.btn_cut_all = Button(self.fig.add_axes([0.06, 0.095, 0.15, 0.045]), "CUT ALL GPS",
                                   color="#932727", hovercolor="#ad3333")
        self.btn_cut_all.label.set_color("white")
        self.btn_cut_all.label.set_fontsize(8)
        self.btn_cut_all.on_clicked(self._on_cut_all)

        self.btn_restore_all = Button(self.fig.add_axes([0.23, 0.095, 0.15, 0.045]), "RESTORE ALL GPS",
                                       color="#26629c", hovercolor="#337cbf")
        self.btn_restore_all.label.set_color("white")
        self.btn_restore_all.label.set_fontsize(8)
        self.btn_restore_all.on_clicked(self._on_restore_all)

        self.btn_pause = Button(self.fig.add_axes([0.06, 0.03, 0.15, 0.045]), "PAUSE / RESUME",
                                 color="#3d5a6b", hovercolor="#4d7086")
        self.btn_pause.label.set_color("white")
        self.btn_pause.label.set_fontsize(8)
        self.btn_pause.on_clicked(self._on_pause_toggle)

        self.btn_reset = Button(self.fig.add_axes([0.23, 0.03, 0.15, 0.045]), "RESET",
                                 color="#555555", hovercolor="#6e6e6e")
        self.btn_reset.label.set_color("white")
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._on_reset)

        self.fig.text(0.47, 0.205, "Selected drone", fontsize=8, ha="center")
        self.radio_drone = RadioButtons(
            self.fig.add_axes([0.41, 0.03, 0.12, 0.17]), [str(i) for i in range(N_DRONES)],
            active=0, label_props={'fontsize': [7] * N_DRONES}, radio_props={'s': [20] * N_DRONES})
        self.radio_drone.on_clicked(self._on_select_drone)

    def _on_cut(self, _event) -> None:
        i = self.selected_drone
        if self.sim.gps_signal[i]:
            self.sim.cut_gps(i)
            self.log(f"Drone {i}: GPS SIGNAL CUT")

    def _on_restore(self, _event) -> None:
        i = self.selected_drone
        if not self.sim.gps_signal[i]:
            self.sim.restore_gps(i)
            self.log(f"Drone {i}: GPS SIGNAL RESTORED")

    def _on_cut_all(self, _event) -> None:
        for i in range(N_DRONES):
            self.sim.cut_gps(i)
        self.log("ALL drones: GPS SIGNAL CUT")

    def _on_restore_all(self, _event) -> None:
        for i in range(N_DRONES):
            self.sim.restore_gps(i)
        self.log("ALL drones: GPS SIGNAL RESTORED")

    def _on_pause_toggle(self, _event) -> None:
        self.paused = not self.paused
        self.log("PAUSED" if self.paused else "RESUMED")

    def _on_reset(self, _event) -> None:
        self.sim = SwarmSim()
        self.selected_drone = 0
        self.radio_drone.set_active(0)
        self.paused = False
        self._prev_fragmented = False
        self._prev_risk_band = "LOW"
        self.log_lines = []
        self.log("SIMULATION RESET")

    def _on_select_drone(self, label: str) -> None:
        self.selected_drone = int(label)

    # -- simulation / rendering loop -----------------------------------------
    def log(self, msg: str) -> None:
        line = f"[T+{self.sim.t:6.1f}s] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)
        print(line)

    def _risk_band(self, risk: float) -> str:
        if risk < RISK_LOW:
            return "LOW"
        if risk < RISK_HIGH:
            return "MEDIUM"
        return "HIGH"

    def tick(self, dt: float = TICK_DT) -> None:
        if self.paused:
            self._redraw()
            return
        G = self.sim.step(dt, self.log)

        fragmented = len(self.sim.reachable_from_base) < N_DRONES + 1
        if fragmented and not self._prev_fragmented:
            unreachable = sorted(set(range(N_DRONES + 1)) - self.sim.reachable_from_base)
            self.log(f"NETWORK FRAGMENTED (belief) -- unreachable from base: {unreachable}")
        elif not fragmented and self._prev_fragmented:
            self.log("Network belief re-connected -- all drones reachable from base")
        self._prev_fragmented = fragmented

        if self.mlp is not None:
            feat_vec = extract_graph_features(G, self.sim.node_features)
            self.sim.risk = float(self.mlp.predict_proba(self.scaler.transform([feat_vec]))[0, 1])
            band = self._risk_band(self.sim.risk)
            if band != self._prev_risk_band:
                self.log(f"Predicted fragmentation risk -> {band} ({self.sim.risk:.2f})")
            self._prev_risk_band = band

        self._redraw()

    def _redraw(self) -> None:
        self.ax_net.clear()
        style_axes(self.ax_net, title="Live Swarm - Believed Connectivity Graph",
                   xlabel="X [m]", ylabel="Y [m]")
        self.ax_net.set_xlim(-20, AREA_SIZE_M + 20)
        self.ax_net.set_ylim(-20, AREA_SIZE_M + 20)
        self.ax_net.set_aspect("equal")

        n = N_DRONES + 1
        adj = self.sim.believed_adjacency
        all_pos = np.vstack([self.sim.base_pos[None, :], self.sim.believed_pos])
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b]:
                    self.ax_net.plot(*zip(all_pos[a], all_pos[b]), color="#999999",
                                      linewidth=0.8, alpha=0.6, zorder=1)

        self.ax_net.scatter(*self.sim.base_pos, marker="*", s=260, color="#000000",
                             edgecolors="#FFD700", linewidths=1.2, zorder=3, label="Base station")

        for i in range(N_DRONES):
            reachable = (i + 1) in self.sim.reachable_from_base
            fill = "#2CA02C" if self.sim.gps_signal[i] else "#FF7F0E"
            edge = "#1f4e79" if reachable else "#D62728"
            self.ax_net.scatter(*self.sim.believed_pos[i], s=140, color=fill,
                                 edgecolors=edge, linewidths=2.0 if not reachable else 1.0, zorder=4)
            self.ax_net.annotate(str(i), self.sim.believed_pos[i], textcoords="offset points",
                                  xytext=(6, 6), fontsize=7.5)
            if i == self.selected_drone:
                self.ax_net.scatter(*self.sim.believed_pos[i], s=280, facecolors="none",
                                     edgecolors="#000000", linewidths=1.3, zorder=5)

        comm_circle = plt.Circle(self.sim.base_pos, COMM_RANGE_M, fill=False,
                                  linestyle=":", color="#cccccc", linewidth=0.8, zorder=0)
        self.ax_net.add_patch(comm_circle)

        handles = [
            plt.Line2D([], [], marker="o", color="none", markerfacecolor="#2CA02C", markersize=9, label="GPS OK"),
            plt.Line2D([], [], marker="o", color="none", markerfacecolor="#FF7F0E", markersize=9, label="GPS lost"),
            plt.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#D62728",
                       markeredgewidth=2, markersize=9, label="Unreachable from base"),
            plt.Line2D([], [], marker="*", color="none", markerfacecolor="#000000", markersize=12, label="Base"),
        ]
        self.ax_net.legend(handles=handles, loc="upper left", fontsize=7, frameon=False)

        self.log_text.set_text("\n".join(self.log_lines[-26:]))

        n_lost = int((~self.sim.gps_signal).sum())
        n_unreach = N_DRONES + 1 - len(self.sim.reachable_from_base)
        risk_txt = f"{self.sim.risk:.2f} ({self._risk_band(self.sim.risk)})" if self.mlp is not None else "N/A"
        paused_txt = "    [PAUSED]" if self.paused else ""
        self.suptitle.set_text(
            f"t = {self.sim.t:6.1f}s    GPS lost = {n_lost}/{N_DRONES}    "
            f"Unreachable from base = {n_unreach}    Predicted fragmentation risk = {risk_txt}{paused_txt}")

    def run_interactive(self):
        backend = matplotlib.get_backend()
        # A live FuncAnimation needs a real GUI event loop. Plain 'Agg' or
        # matplotlib_inline's inline-agg backend (what VS Code's Jupyter
        # "Run in Interactive Window" / "Run Cell" auto-selects instead of a
        # plain terminal) can only render static frames -- this, not a
        # missing GUI toolkit (TkAgg works fine in this project's own
        # interpreters, checked directly), is what actually produces the
        # "FigureCanvasAgg is non-interactive" / "Animation was deleted"
        # warnings.
        if backend.lower() == 'agg' or 'inline' in backend.lower():
            print(f"\nNo interactive display available (matplotlib backend = '{backend}').\n"
                  f"Run this from a plain terminal instead of a Jupyter/VS Code "
                  f"'Run in Interactive Window' cell, e.g.:\n"
                  f"    python ai_backend/live_swarm_demo.py\n"
                  f"Or use --headless-test to generate a PNG snapshot with no display needed.\n")
            return None
        ani = FuncAnimation(self.fig, lambda _f: self.tick(), interval=200, cache_frame_data=False)
        plt.show()
        return ani

    def run_headless_test(self, out_path: str) -> None:
        """Scripted, no-GUI run, seeded for reproducibility: let the swarm
        fly with GPS healthy for a while (expect low logged risk), then cut
        every drone's GPS at once and run long enough for drift to
        accumulate, watching the risk gauge and fragmentation state
        actually respond -- self-contained enough to verify visually
        without a human at the controls, same convention as
        live_mission_demo.py's own run_headless_test()."""
        self.sim = SwarmSim(seed=3)   # seed checked directly: this draw produces a clean fragmentation event
        for _ in range(15):
            self.tick()
        risk_before = self.sim.risk
        frag_before = len(self.sim.reachable_from_base) < N_DRONES + 1
        self.log(f"-- baseline (GPS healthy): risk={risk_before:.3f} frag={frag_before}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        healthy_path = out_path.replace('.png', '_healthy.png')
        self._redraw()
        self.fig.savefig(healthy_path, dpi=140)
        print(f"Saved healthy-baseline snapshot -> {healthy_path}")

        self._on_cut_all(None)
        for _ in range(60):
            self.tick()
        risk_after = self.sim.risk
        frag_after = len(self.sim.reachable_from_base) < N_DRONES + 1
        self.log(f"-- after all-GPS-lost drift: risk={risk_after:.3f} frag={frag_after}")

        fragmented_path = out_path.replace('.png', '_fragmented.png')
        self._redraw()
        self.fig.savefig(fragmented_path, dpi=140)
        self.fig.savefig(out_path, dpi=140)   # also kept at the plain requested path for backward compat
        print(f"Saved fragmented-state snapshot -> {fragmented_path}")
        print(f"Risk before GPS loss: {risk_before:.3f}   Risk after: {risk_after:.3f}")
        print(f"Fragmented before: {frag_before}   Fragmented after: {frag_after}")

        assert self.mlp is not None, "Method A model failed to load -- run train_swarm_connectivity.py first"
        assert 0.0 <= risk_before <= 1.0 and 0.0 <= risk_after <= 1.0, "risk must be a valid probability"
        assert risk_after > risk_before, (
            "expected sustained GPS loss across the whole swarm to raise predicted fragmentation risk")
        assert frag_after, "expected this seeded scenario to actually fragment after 12s of total GPS loss"

        print("\n--- Pause / resume / reset / drone-select ---")
        self._on_pause_toggle(None)
        assert self.paused
        pos_before = self.sim.believed_pos.copy()
        self.tick()
        assert np.allclose(self.sim.believed_pos, pos_before), "state advanced while paused"
        print("Pause: state held steady, as expected.")

        self._on_pause_toggle(None)
        assert not self.paused
        self.tick()
        print("Resume: ticking again, as expected.")

        self._on_select_drone("3")
        assert self.selected_drone == 3
        self._on_cut(None)
        assert not self.sim.gps_signal[3]
        self._on_restore(None)
        assert self.sim.gps_signal[3]
        print("Per-drone select/cut/restore: works as expected.")

        self._on_reset(None)
        assert self.sim.t == 0.0
        assert self.sim.gps_signal.all()
        print("Reset: fresh swarm, as expected.")

        print("\nOK -- headless test completed without error.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless-test", action="store_true",
                     help="Run a scripted healthy/cut-all/drift sequence with no GUI, "
                          "saving a PNG snapshot for visual inspection.")
    ap.add_argument("--out", default=None, help="Snapshot output path (headless-test only).")
    args = ap.parse_args()

    dashboard = LiveSwarmDashboard()

    if args.headless_test:
        out = args.out or os.path.join(os.path.dirname(__file__), "..", "runs", "live_swarm_demo_test.png")
        dashboard.run_headless_test(out)
    else:
        _animation_ref = dashboard.run_interactive()
