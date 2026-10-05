# `simulation/scripts/` - GDScript Sources

Short index only - **`../CODEBASE_DOCS.md` is the authoritative deep-dive**
(per-constant values, exact control-flow, UDP port map, coordinate
conventions). Keeping the detailed description in one place and this file as
a pointer avoids the two drifting out of sync with each other.

| File | Role |
|---|---|
| `drone_state.gd` | Autoload singleton - shared telemetry/flight-state data bus every other script reads/writes. |
| `serial_bridge.gd` | Autoload - UDP :14550 listener for real-hardware telemetry (SERIAL mode). Live, not dead code, despite what older top-level docs used to claim. |
| `main.gd` | Scene builder - constructs the entire world (environment, ground, helipad, 3-drone swarm, camera, HUD, UDP listener) procedurally in code. |
| `drone_body.gd` | The flight controller - `RigidBody3D` physics, motor mixer, auto-hover/stabilisation, mission autopilot (`start_mission`), real sensor telemetry (`_send_telemetry`), AI-correction intake (`apply_ai_correction`, blended not raw), and the fail-safe hierarchy wiring (health tracking, breadcrumb hand-off). |
| `breadcrumb_recovery.gd` | Zero-AI emergency fail-safe - the last-resort layer of the RL→SSL/LSTM→EKF→this hierarchy. Plain `RefCounted`, owned by each `drone_body.gd`, runs independent of the Python link. |
| `mission_log.gd` | Autoload - event bus (`log_event()`/`event_logged`) feeding the HUD's Decision Terminal panel. |
| `trajectory_visualizer.gd` | `Node3D` - live `ImmediateMesh` rendering of the mission demo's reference/predicted/executed paths (cyan/magenta/black). |
| `udp_bridge.gd` | UDP :14552 listener - dispatches incoming `ai_correction` commands (`{layer, velocity_cmd}`) to the named drone (SIM-mode counterpart to `serial_bridge.gd`). |
| `camera_controller.gd` | Third-person camera, TRACKING (fixed-angle + deadzone) / FREE_ROAM (fly-free) modes, toggled with **C**. |
| `hud_layer.gd` | Builds/refreshes the full 2D HUD in code, incl. the Mission Control panel (drives `Drone_1`'s mission demo) and Decision Terminal. |
| `artificial_horizon.gd` | Attitude instrument (`Control`, `_draw()` only). |
| `compass_rose.gd` | Heading instrument (`Control`, `_draw()` only). |
| `axis_graph.gd` | Scrolling pitch/roll/throttle history graph. |

## Deleted (confirmed dead code, do not resurrect without checking `AI_RECOVERY_EXECUTION_PLAN.md` first)

`godot_ws_client.gd` (Godot 3 syntax, won't parse under Godot 4) and
`swarm_manager.gd` (superseded by `main.gd`'s `_build_swarm()`) were removed
during the Phase-1 housekeeping pass (their leftover `.uid` sidecar files
were cleaned up too).
