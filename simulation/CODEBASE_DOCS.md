# Simulation Codebase Reference

> Godot 4.6 · Forward Plus · Jolt Physics · 1280x720 · D3D12 (Windows)

A fully code-driven drone flight simulator for a 3-drone "Atlas 4 LR" quadcopter swarm.
No external meshes or assets - every visual is built from Godot primitives at runtime.

## Project Overview

| Mode | Description |
|------|-------------|
| **SIM** | Full physics-based flight, keyboard/gamepad control, autonomous hover for non-player drones |
| **SERIAL** | Real-hardware telemetry mode - `serial_bridge.gd` switches `DroneState.mode` to `"SERIAL"` when it receives a UDP packet on port 14550 from a real drone via a Python serial-to-UDP bridge |

The swarm consists of three drones spawned by `main.gd`:

| Drone | Role |
|-------|------|
| `Drone_0` | Player-controlled (keyboard/gamepad); not the default camera target anymore - switch to it via the **C** FREE_ROAM toggle or by flying manually |
| `Drone_1` | **Mission demo drone**, and the default camera target - driven by the HUD's Mission Control panel: enter a local-metre target, START MISSION to fly there (true-GPS position hold), CUT CONNECTION to manually trigger GPS loss and watch the live LSTM → EKF → rule-based → breadcrumb recovery hierarchy take over (see `ai_backend/api/recovery_orchestrator.py` and the Decision Terminal panel) |
| `Drone_2` | Autonomous - pre-armed, holds hover at spawn altitude |

## File Map

```
simulation/
├── project.godot              Engine config, autoloads, renderer settings
├── scenes/
│   └── main.tscn              Minimal root scene (Node3D "Main" + main.gd)
└── scripts/
    ├── drone_state.gd         Autoload - shared flight/telemetry data bus
    ├── serial_bridge.gd       Autoload - UDP :14550 real-hardware telemetry ingestion (SERIAL mode)
    ├── main.gd                Scene builder - constructs world + swarm in code
    ├── drone_body.gd          RigidBody3D flight controller + physics + UDP telemetry sender + mission autopilot
    ├── breadcrumb_recovery.gd Zero-AI emergency fail-safe (RefCounted, owned by drone_body.gd) - see below
    ├── mission_log.gd         Autoload - event bus feeding the HUD's Decision Terminal panel
    ├── trajectory_visualizer.gd  3D ImmediateMesh renderer for the mission demo's reference/predicted/executed paths
    ├── udp_bridge.gd          UDP :14552 listener - dispatches AI corrections to drones (SIM mode)
    ├── camera_controller.gd   TRACKING / FREE_ROAM third-person camera
    ├── hud_layer.gd           Full 2D HUD layout and refresh, incl. Mission Control + Decision Terminal panels
    ├── artificial_horizon.gd  Attitude instrument (Canvas draw)
    ├── compass_rose.gd        Heading instrument (Canvas draw)
    └── axis_graph.gd          Scrolling pitch/roll/throttle graph
```

## `project.godot`

| Setting | Value | Effect |
|---------|-------|--------|
| `run/main_scene` | `res://scenes/main.tscn` | Entry point |
| `DroneState` autoload | `res://scripts/drone_state.gd` | Global singleton, available as `DroneState` everywhere |
| `SerialBridge` autoload | `res://scripts/serial_bridge.gd` | UDP :14550 listener starts immediately on launch - live for real-hardware SERIAL telemetry, independent of the SIM-mode UDP path below |
| `MissionLog` autoload | `res://scripts/mission_log.gd` | Global event bus (`log_event()`/`event_logged` signal) feeding the HUD's Decision Terminal panel - any script can log to it, keeping the terminal's content tied to real state transitions |
| `viewport_width/height` | `1280 x 720` | Fixed resolution; HUD positions are hardcoded to this |
| `stretch/mode` | `canvas_items` | Scales the entire canvas (HUD included) on resize |
| `3d/physics_engine` | `Jolt Physics` | Replaces default GodotPhysics |
| `rendering_device/driver.windows` | `d3d12` | Direct3D 12 on Windows for Forward Plus rendering |

## `scenes/main.tscn`

Minimal scene: a single `Node3D` root named `"Main"` with `main.gd` attached. Everything else
(environment, ground, helipad, swarm, camera, HUD, UDP listener) is spawned in code - no scene
editor dependency.

## `scripts/main.gd`

Procedural scene builder, `extends Node3D`.

`_ready()` call order:
```
_build_environment()            -> WorldEnvironment (procedural sky, directional sun, filmic tonemap)
_build_ground()                 -> 200x200 m green PlaneMesh + WorldBoundaryShape3D collision
_build_helipad()                -> Dark disc + yellow torus "H" ring at world origin
_build_swarm()                  -> Spawns Drone_0/1/2 via _build_drone() (RigidBody3D + drone_body.gd)
_build_camera()                 -> Camera3D with camera_controller.gd, targets Drone_1 (the mission demo drone)
_build_hud()                    -> CanvasLayer with hud_layer.gd
_build_udp_listener()           -> Node with udp_bridge.gd (AI-correction listener)
_build_trajectory_visualizer()  -> Node3D with trajectory_visualizer.gd, tracks Drone_1
_print_controls()               -> Console cheat-sheet
```

GPS loss on `Drone_1` is triggered manually via the HUD's Mission Control CUT CONNECTION button
(`hud_layer.gd` sets `gps_signal = false` directly on the node) rather than an automatic timer -
this lets the user run the recovery demo on demand, repeatably (RESTORE GPS resets it).

Each drone's visual frame (`_build_frame_mesh`) is built from primitives: centre stack (FC PCB,
ESC), 4 arms in X-config with motor markers, semi-transparent props (red-tinted front, dark rear -
shows orientation from above), aviation-standard nav LEDs (red front / green rear), an orange
emissive nose-arrow, an RPi Zero 2W mock, an FPV camera mock, and a battery pack. Motor identity
colours (must match `MOTOR_COLORS` in `hud_layer.gd`):

| Motor | Position | Colour |
|-------|----------|--------|
| FL (0) | Front-Left | Vivid Red |
| FR (1) | Front-Right | Lime Green |
| RL (2) | Rear-Left | Dodger Blue |
| RR (3) | Rear-Right | Vivid Orange |

## `scripts/drone_body.gd`

`RigidBody3D` script - complete flight controller: reads input, runs the motor mixer, applies
physics forces, manages auto-stabilisation/hover-hold, and sends UDP telemetry.

### Physical constants (read directly from source)

| Constant | Value | Meaning |
|----------|-------|---------|
| `ARM_M` | 0.085 m | Centre-to-motor distance |
| `MAX_THRUST` | 4.5 N/motor | XING2 1404 3800KV on 4S ~450 g thrust each |
| `PROP_DIR` | `[-1, 1, 1, -1]` | FL=CCW, FR=CW, RL=CW, RR=CCW (X-config) |
| `mass` | 0.281 kg | Set in `_ready()` |
| `linear_damp` | 0.7 | Baseline air-drag approximation (boosted to `HOVER_LIN_DAMP=6.0` during hover) |
| `angular_damp` | 7.0 | Baseline gyroscopic stability (boosted to `STAB_ANG_DAMP=16.0` during hover) |
| `gravity_scale` | 1.0 | Matches Godot's default 3D gravity (9.8 m/s^2) |
| `IDLE_GRACE` | 2.5 s | No attitude input before auto-hover engages |
| `STAB_RATE` | 4.0 | Yaw-command washout speed (1/s) while stabilising |
| `HOVER_ALT_KP` / `HOVER_ALT_KI` | 0.30 / 0.15 | Altitude-hold P+I gains (direct-throttle form) |
| `LEVEL_KP` | 0.018 | Degrees-of-tilt -> mixer command gain while auto-levelling |

Non-player drones (`is_player == false`) skip `_input()` entirely, are pre-armed in `_ready()`,
and start in manual hover at their spawn altitude - they never read keyboard/gamepad state.

### Motor mixer (`_mix_motors`, X-config)

```
FL (CCW) = t + r + p + y
FR (CW)  = t - r + p - y
RL (CW)  = t + r - p - y
RR (CCW) = t - r - p + y
```
`t`=throttle, `p`=pitch_cmd x `pitch_sens`, `r`=roll_cmd x `roll_sens`, `y`=yaw_cmd x `yaw_sens`.
All outputs clamped 0-1.

### Auto-hover / stabilisation (`_update_stabilisation`)

Triggered only by attitude-input inactivity (`_attitude_active`) - throttle changes alone never
cancel or trigger it. After `IDLE_GRACE` (2.5 s) with no pitch/roll/yaw input:
1. Latches current altitude as `_hover_target_alt`.
2. Ramps `linear_damp` -> `HOVER_LIN_DAMP` (6.0) and `angular_damp` -> `STAB_ANG_DAMP` (16.0) over
   a 0.8 s blend (`_hover_blend`) to avoid a snap.
3. Auto-levels pitch/roll toward 0 via `LEVEL_KP`, washes out yaw at `STAB_RATE`.
4. Holds altitude with a direct-output P+I loop (`HOVER_ALT_KP`/`HOVER_ALT_KI`).

Pressing **H** manually toggles hover on/off (`_toggle_hover`) independent of the idle timer.
Throttle input while hovering shifts `_hover_target_alt` instead of cancelling hover.

### Telemetry (SIM mode, `_send_telemetry`)

Each drone independently sends its own UDP JSON telemetry packet (no bind needed) to
`127.0.0.1:14551` at 10 Hz (`TELEM_RATE = 0.1`). All sensor fields are **physically real**,
not placeholder noise (this was not always true - see the "Simulated-sensor realism" note below):

| Field(s) | Source | Notes |
|---|---|---|
| `drone_id`, `timestamp`, `gps_signal` | identity/state | `gps_signal` is flipped manually via the HUD's Mission Control CUT CONNECTION / RESTORE GPS buttons (or any other code setting the drone's `gps_signal` property) |
| `lat`, `lon`, `alt` | position | `alt` is `global_position.y`; `lat`/`lon` are `DroneState.gps_lat/gps_lon` offset by world position |
| `imu_acc_x/y/z` | `_accel_body` | True specific force: `R^-1 * ((v_t - v_{t-1})/dt - g)`, computed every physics frame regardless of arm state |
| `imu_gyro_x/y/z` | `angular_velocity` | RigidBody3D's real angular velocity, rotated into body frame |
| `pitch`, `roll`, `yaw` | `global_transform.basis.get_euler()` | Degrees, aviation sign convention (same as `_push_state()`) |
| `mag_x/y/z` | `WORLD_MAG_DIR` | Fixed world-frame "Earth field" vector rotated into body frame + noise - not yet a true geographic-declination model, documented simplification |
| `baro_alt` | `_baro_bias` | `alt` + slow random-walk drift + sensor noise |
| `motor_out` | `_motor_out` | The 4 real per-motor thrust values (0-1), already computed by `_mix_motors()` |
| `speed`, `battery` | `linear_velocity.length()`, `DroneState.battery_v` | |
| `mission_target` *(optional)* | `mission_target` | `[x, y, z]` local metres, included only while `mission_active` - lets `recovery_orchestrator.py` compute the rule-based vector-to-target fallback |
| `force_layer` *(optional)* | `force_layer` | `"LSTM"`/`"EKF"`/`"RULE_BASED"`, set by the HUD's Force Layer dropdown for demo purposes - included only when non-empty |

When `gps_signal` is false, the Python AI backend (`ai_backend/api/udp_server.py`, delegating to
`ai_backend/api/recovery_orchestrator.py`) responds with a live recovery correction - see **AI
correction intake** below - and if those stop arriving (or the whole Python process dies), the
fail-safe hierarchy degrades toward `breadcrumb_recovery.gd` (see its own section).

### Simulated-sensor realism (`GRAVITY_MPS2`, `WORLD_MAG_DIR`, `_prev_lin_vel`, `_accel_body`, `_baro_bias`)

Earlier revisions of this script sent fabricated telemetry (`randf_range()` calls for
accelerometer, orientation-angle-derived gyro) - fixed; see `AI_RECOVERY_EXECUTION_PLAN.md` §6.1
for the full history and the exact specific-force formula used.

### AI correction intake (`apply_ai_correction`)

Called by `udp_bridge.gd` when an `ai_correction` command arrives for this drone's `drone_id`.
Payload shape: `{"layer": "LSTM"|"EKF"|"RULE_BASED", "goal": "TARGET"|"HOME"|"NONE",
"velocity_cmd": [vx, vy, vz]}` - a world-frame m/s velocity, the same language
`breadcrumb_recovery.gd`'s `pop_next_reversal_command()` already uses internally. `goal` reports
`recovery_orchestrator.py`'s target-then-RTL hybrid state: `TARGET` while still trying to reach
the mission target blind, `HOME` once that's been abandoned in favor of the launch point after
`RTL_TIMEOUT_S` of continuous loss (logged here as a one-time "RTL ENGAGED" line via `_log()`, the
same one-shot pattern used for breadcrumb's own trigger-start log below). First calls
`_breadcrumb.mark_healthy()` (proof a higher layer of the fail-safe hierarchy is alive - see
`breadcrumb_recovery.gd` below), then **smoothly blends** `linear_velocity` toward `velocity_cmd`
(`linear_velocity.lerp(target_v, dt * CORRECTION_BLEND_RATE)`) rather than overriding it -
corrections arrive roughly every telemetry tick (~10 Hz) once GPS is lost, and a raw per-tick
assignment would fight the RigidBody3D solver and read as jitter. (`breadcrumb_recovery.gd`'s own
replay stays a raw override - an intentionally rare, minimal command-replay, not a continuous
correction stream, so the same concern doesn't apply there.) Also integrates `velocity_cmd` over
time into `predicted_path` - a visualization-only "what the recovery layer currently believes"
track, consumed by `trajectory_visualizer.gd`; it never feeds back into physics.

### Mission autopilot (`start_mission`, `_update_mission_guidance`)

Non-player drones (in practice, `Drone_1`) can be given a local-metre target via
`start_mission(target: Vector3)`, called by the HUD's Mission Control panel. While
`mission_active` and `gps_signal` is true, `_update_mission_guidance()` (called from
`_physics_process()` for non-player drones) computes the true position error against
`global_position`, transforms it into body-frame forward/right components, and sets
`_pitch_cmd`/`_roll_cmd` via a small-signal proportional controller (`MISSION_KP`, clamped to
`MISSION_MAX_TILT`) plus `_hover_target_alt = mission_target.y` - reusing the existing
`_update_stabilisation()`/`_mix_motors()`/`_apply_forces()` pipeline, the same way keyboard input
drives the player. **The moment `gps_signal` goes false, this function returns immediately**
without touching pitch/roll - guidance never keeps secretly using true `global_position` once
"GPS is lost" is supposed to be in effect. The drone instead holds attitude via the normal
auto-hover path until a `velocity_cmd` correction (LSTM/EKF/rule-based) or breadcrumb takes over.
Also maintains three point-history buffers (`executed_path`, `reference_path`, `predicted_path`,
capped at `PATH_MAX_POINTS`) purely for `trajectory_visualizer.gd` to render.

Note: `_process()` gates its keyboard/gamepad reads (`_read_keys`/`_read_controller`) behind
`is_player` - `Input.is_key_pressed()` is global state, not per-node, so without this guard every
drone (not just the player) would react to the same keypresses, fighting the mission autopilot's
own `_pitch_cmd`/`_roll_cmd` on non-player drones.

## `scripts/breadcrumb_recovery.gd`

Plain `RefCounted` class (`class_name BreadcrumbRecovery`, no scene-tree dependency), owned by
each `drone_body.gd` instance (`_breadcrumb := BreadcrumbRecoveryScript.new()`, preloaded rather
than referenced by bare class name so it works even before Godot's global script-class cache has
refreshed). The **zero-AI emergency floor** of the fail-safe hierarchy: RL policy → SSL/LSTM →
classic EKF → this. Runs entirely locally - no dependency on `udp_server.py` or the UDP link
being alive, by design (see `AI_RECOVERY_EXECUTION_PLAN.md` §12 Goal 3 / §14).

| Method | Purpose |
|---|---|
| `record_sample(v_world, dt)` | Buffers one decimated world-frame velocity sample every 0.5 s (2 Hz) into a capped ring buffer (`MAX_BUFFER_SAMPLES = 240`, ~2 min of history) |
| `mark_healthy()` | Resets the health timer - called every frame while `gps_signal` is true, and at the top of `apply_ai_correction()` whenever a fresh AI correction arrives |
| `should_trigger()` | True once 500 ms have elapsed with no `mark_healthy()` call *and* the buffer isn't empty - the real, computed trigger signal |
| `pop_next_reversal_command()` | Pops the buffer from the back, returning the **negated** stored velocity - pure command-replay, no controller loop |

`drone_body.gd::_physics_process()` wiring: while `gps_signal` is true, calls `mark_healthy()` +
`reset()` every frame (only the *current* loss episode's history is kept) and always
`record_sample()`s. Once `gps_signal` is false and `should_trigger()` fires, control branches
into `_run_breadcrumb_recovery()` instead of the normal stabilisation/motor-mixing path - direct
`linear_velocity` override, replaying buffered commands at their original decimated intervals.
When the buffer drains, `should_trigger()` naturally goes false again and control falls straight
back through to normal stabilisation/hover on its own - no explicit "recovery complete" state
needed. Deliberate simplification: no attitude/orientation correction is applied during replay
(pure command-replay per spec), so the drone may look visually tilted during retrace.

## `scripts/udp_bridge.gd`

Plain `Node`, spawned by `main.gd` under `Main/UDPBridge`. Binds UDP port **14552**, listens for
JSON commands from the Python AI backend. For `{"action": "ai_correction", "drone_id": "...",
"layer": "...", "goal": "...", "velocity_cmd": [...]}` messages, looks up `/root/Main/<drone_id>`
and calls its `apply_ai_correction()`. This is the SIM-mode counterpart to the SERIAL-mode ingestion path in
`serial_bridge.gd`.

## `scripts/trajectory_visualizer.gd`

Plain `Node3D`, spawned by `main.gd` (`_build_trajectory_visualizer()`) tracking `Drone_1`. Live
3D rendering of the mission demo's three trajectory tracks (Goal 4 Category A -
`AI_RECOVERY_EXECUTION_PLAN.md` §12): three `MeshInstance3D`+`ImmediateMesh` line-strips, redrawn
every `_process()` frame from `target`'s point-history buffers (`drone_body.gd`'s
`reference_path`/`predicted_path`/`executed_path`, read via `target.get(...)` since the static
type is a generic `Node3D`). Colors match `ai_backend/plot_style.py`'s established role palette:
cyan = reference/mission path (snapshotted once at the instant GPS is cut), magenta = predicted
path (integral of incoming `velocity_cmd`s - what the active recovery layer believes), black =
executed path (the drone's real position history). Solid, thin, uniform-width lines throughout -
`ImmediateMesh` line-strips don't support per-segment dash gaps without materially more geometry
complexity, and three distinct colors are already unambiguous without them.

## `scripts/serial_bridge.gd`

Autoload singleton (`SerialBridge`), always running. Binds UDP port **14550** and listens for JSON
telemetry from a real-hardware serial-to-UDP bridge (`{"pitch","roll","yaw","altitude","battery",
"motors"}`). Stays silent (SIM mode unaffected) until the first valid packet arrives, at which
point it sets `DroneState.mode = "SERIAL"` and calls `DroneState.update_from_serial(data)`. This
is a live, separate ingestion path from the SIM-mode telemetry/correction UDP flow above - it is
not dead code, despite older docs claiming otherwise.

## `scripts/drone_state.gd`

Autoload singleton - shared data bus that every script reads/writes.

| Field | Description |
|-------|-------------|
| `pitch/roll/yaw` | Degrees, aviation sign convention (Drone_0 / player only) |
| `altitude_m`, `speed_ms`, `battery_v` | Player telemetry |
| `gps_lat/gps_lon` | Athens default (37.9755, 23.7348) offset by player position |
| `motor[4]` | Per-motor output 0.0-1.0 (FL, FR, RL, RR) |
| `armed` | Player armed state |
| `mode` | `"SIM"` or `"SERIAL"` |
| `cam_mode` | `0` = TRACKING, `1` = FREE_ROAM (set by `camera_controller.gd`) |
| `hover_target_alt` | Player hover target altitude, -1 = inactive |
| `throttle_rate`, `pitch_sens`, `roll_sens`, `yaw_sens`, `yaw_torque`, `expo` | Runtime-tunable via the HUD Tab settings panel |

`telemetry_updated` signal is rate-limited to ~30 Hz (`EMIT_INTERVAL`) when driven by
`update_from_physics()`, but emitted immediately (unthrottled) by `update_from_serial()`.

## `scripts/mission_log.gd`

Autoload singleton (`MissionLog`) - a small event bus for the hands-on mission demo's Decision
Terminal panel. Any script calls `MissionLog.log_event("...")` from anywhere; `drone_body.gd`'s
`_log()` helper routes GPS cut/restore, mission start/complete, AI-correction-applied, and
breadcrumb-trigger events through it (alongside a plain `print()`). Keeps a capped `history` array
(`MAX_HISTORY = 200`) timestamped against a running sim-clock (`[T+12.34s] ...`) and emits
`event_logged(text)` on every new line - `hud_layer.gd`'s terminal panel just re-renders
`history` whenever that signal fires, so the terminal reflects real state transitions rather than
anything fabricated for display purposes.

## `scripts/camera_controller.gd`

`Camera3D` script, two modes toggled with **C** (or gamepad Back/Select). Mode is published to
`DroneState.cam_mode`.

### Mode 0 - TRACKING (fixed-angle + deadzone)

The camera sits at a fixed world-space pitch/yaw offset (`pitch_deg`, `yaw_deg`,
`camera_height`, `camera_distance`) from an invisible anchor point. The anchor only moves when
the target leaves a configurable screen-space deadzone rectangle (`deadzone_width` x
`deadzone_height`, centred); inside it the camera holds perfectly still even if the target
rotates or jitters. Target rotation is never consulted. Anchor catch-up uses `tracking_speed`
as a time-scaled lerp.

### Mode 1 - FREE_ROAM

Camera detaches and flies freely: WASD/Q/E to move (camera-local space), mouse or gamepad
right-stick to look (`freeroam_mouse_sens`, `freeroam_stick_sens`), never rolls. Player drone
input is locked while this mode is active (`drone_body.gd` checks `DroneState.cam_mode` in its
`_input()`). Mouse is captured (`MOUSE_MODE_CAPTURED`) while active and released on mode switch
or window close.

## `scripts/hud_layer.gd`

`CanvasLayer` script - builds the entire 2D HUD in code, refreshes on `DroneState.telemetry_updated`
(rate-limited to ~30 Hz).

| Element | Position | Content |
|---------|----------|---------|
| Status strip | Top-left | ARMED/DISARMED, MODE, KB/PAD, TRACK/FLY |
| Artificial horizon | Left, 160x160 | `artificial_horizon.gd` |
| Compass rose | Left, 120x120 | `compass_rose.gd` |
| Telemetry | Bottom-left | ALT / SPD / BAT + hover-hold indicator |
| Motor bars | Bottom-centre | 4x colour-coded progress bars (FL/FR/RL/RR) |
| Axis graph | Bottom-right, 372x96 | `axis_graph.gd` |
| GPS | Top-right | Lat/Lon |
| Controls hint | Far-right | Key reference list |
| **Mission Control** | Top-centre, 497x128 | Target X/Z/Alt fields, START MISSION / CUT CONNECTION / RESTORE GPS buttons, Force Layer dropdown - drives `Drone_1` via `get_node_or_null("/root/Main/Drone_1")` |
| **Decision Terminal** | Centre, 715x250 | `RichTextLabel` subscribed to `MissionLog.event_logged` - live log of GPS/mission/recovery-layer events |
| Settings panel (Tab) | Centre | 6 sliders bound to `DroneState` sensitivity fields - added last so it renders above Mission Control/Decision Terminal when toggled |

## `scripts/artificial_horizon.gd`

`Control`, `_draw()` only. Renders an AHRS-style horizon in a circle from `DroneState.pitch`/
`roll`: solves the circle/horizon-line intersection to fill sky (blue) vs. ground (brown) via
`draw_colored_polygon`, plus pitch tick marks, a roll arc with pointer, and a fixed aircraft
symbol. Redraws only when `DroneState.telemetry_updated` fires.

## `scripts/compass_rose.gd`

`Control`, `_draw()` only. Rotating compass showing `DroneState.yaw`: 8 cardinal/intercardinal
ticks placed at `world_deg - yaw - 90deg`, a fixed heading pointer at the top, numeric heading text
at centre. Redraws on `telemetry_updated`.

## `scripts/axis_graph.gd`

`Control`, samples in `_process()` independent of the telemetry signal (own ~30 Hz timer,
`RATE_S = 0.033`), draws in `_draw()`. Keeps up to `MAX_PTS = 300` samples (~10 s) per channel for
pitch, roll, and average-motor throttle, normalised to -1..1, rendered as three coloured
polylines with grid lines and a legend.

## UDP Port Map

| Port | Direction | Script | Purpose |
|------|-----------|--------|---------|
| 14550 | inbound | `serial_bridge.gd` (autoload) | Real-hardware telemetry -> SERIAL mode |
| 14551 | outbound (from Godot) | `drone_body.gd` (`_send_telemetry`) | Per-drone SIM telemetry -> Python AI backend |
| 14552 | inbound | `udp_bridge.gd` | AI corrections from Python -> `apply_ai_correction()` on the named drone |

## Coordinate System

Godot uses Y-up, right-handed, -Z forward.

| Axis | World meaning | Drone meaning |
|------|---------------|----------------|
| +X | Right | Roll right |
| +Y | Up | Throttle / altitude |
| +Z | Backward | Pitch back / move backward |
| -Z | Forward | Pitch forward / move forward |

Euler angles from `get_euler()` are YXZ order (Godot default). `_push_state()` in `drone_body.gd`
negates X and Z before converting to degrees to match aviation sign convention (`pitch = -eu.x`,
`roll = -eu.z`, `yaw = (-eu.y + 360) mod 360`).
