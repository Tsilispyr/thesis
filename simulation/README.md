# `simulation/` - Godot 4 Drone Swarm Simulator

A fully code-driven 3-drone quadcopter swarm simulation ("Atlas 4 LR"), no
external meshes - every visual is built from Godot primitives at runtime.
This is the "environment" half of the project; the AI backend
(`../ai_backend/`) is the "brain" half, connected over UDP.

**Open this in Godot 4.x by importing `project.godot`** (not any file inside
`scripts/` directly). `scenes/main.tscn` is the only scene - it's a single
empty `Node3D` with `main.gd` attached, which builds the entire world in
code (`_build_environment`, `_build_swarm`, `_build_hud`, etc.).

## Where to look

| What you want | Where |
|---|---|
| Full technical reference (per-constant values, control flow, UDP port map, coordinate conventions) | [`CODEBASE_DOCS.md`](CODEBASE_DOCS.md) - read this for anything beyond a quick orientation |
| Short per-file index | [`scripts/README.md`](scripts/README.md) |
| Flight controller / physics / telemetry / mission autopilot | `scripts/drone_body.gd` |
| Hands-on Mission Control panel + Decision Terminal | `scripts/hud_layer.gd` |
| Live 3D trajectory rendering (reference/predicted/executed) | `scripts/trajectory_visualizer.gd` |
| Decision-log event bus feeding the Decision Terminal | `scripts/mission_log.gd` (autoload) |
| Zero-AI emergency fail-safe | `scripts/breadcrumb_recovery.gd` |
| How this talks to the Python AI backend | `scripts/udp_bridge.gd` (receives), `drone_body.gd::_send_telemetry()` (sends) |

## Running it

1. Start the Python AI server first: `python ../ai_backend/api/udp_server.py` (from `AI_Recovery/`).
2. Open Godot 4.x, import `simulation/project.godot`, press F5.
3. Drone_0 is player-controlled (Enter to arm, WASD/Space/Ctrl to fly), and the
   camera tracks **Drone_1** by default - the dedicated Mission Control demo
   drone. Use the HUD's Mission Control panel to enter a target (local
   metres), press START MISSION to watch it fly there, then CUT CONNECTION to
   manually trigger GPS loss on demand and watch the live LSTM → EKF →
   rule-based → breadcrumb recovery hierarchy take over, with the Decision
   Terminal panel showing exactly what's happening. Drone_2 remains a plain
   autonomous hover drone. Press **C** to switch the camera to FREE_ROAM if
   you want to fly Drone_0 or look elsewhere.

See the project-root `AI_Recovery/README.md` and `EXECUTION_GUIDE.md` for the
full walkthrough, including how to trigger and observe GPS-loss recovery. A
Python-only equivalent that needs no Godot at all is
`ai_backend/live_mission_demo.py` - see its screenshot in
`../ai_backend/README.md`.

## Fail-safe hierarchy (see `AI_RECOVERY_EXECUTION_PLAN.md` §12)

This simulation is where the **bottom** of the project's degrade-toward-
simplicity fail-safe stack actually lives - `breadcrumb_recovery.gd` runs
entirely inside Godot, independent of the Python backend or the UDP link
being alive, by design. Everything above it (RL policy, SSL/LSTM, classic
EKF) lives in `../ai_backend/` and reaches this simulation only through the
UDP correction channel.
