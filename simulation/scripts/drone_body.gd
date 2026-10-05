## Atlas 4 LR flight simulation - SIM mode only (SERIAL mode = visualise only).
## Controls (SIM, after pressing Enter to ARM):
##   Space / Ctrl   - throttle up / down
##   W / S          - pitch forward / back
##   A / D          - roll left / right
##   Q / E          - yaw left / right
##   R / R1         - flip upright
##   Auto-hover     - when no stick input, drone levels then holds altitude
extends RigidBody3D

const ARM_M      := 0.085   # centre-to-motor distance (m)
const MAX_THRUST := 4.5     # N per motor  (XING2 1404 3800KV on 4S ≈ 450 g each)
# X-config spin direction:  FL=CCW(-) FR=CW(+) RL=CW(+) RR=CCW(-)
const PROP_DIR   := [-1.0, 1.0, 1.0, -1.0]

# ── Simulated-sensor constants (telemetry realism) ─────────────────────────
const GRAVITY_MPS2  := 9.8                     # matches Godot's default 3D gravity; gravity_scale=1.0
const WORLD_MAG_DIR := Vector3(0.0, -0.6, 0.8) # fixed unit-length "Earth field" direction, world frame

# Zero-AI emergency fail-safe (last resort of the RL->SSL/LSTM->EKF->this
# hierarchy). preload (not bare `BreadcrumbRecovery.new()`) so this works
# even before Godot's global script-class cache has been refreshed by
# opening the project in-editor.
const BreadcrumbRecoveryScript := preload("res://scripts/breadcrumb_recovery.gd")

# Arm offsets in body-local space (Y-up, +Z = nose/front)
var _arms := [
	Vector3(-ARM_M, 0.0,  ARM_M),   # FL  (front-left)
	Vector3( ARM_M, 0.0,  ARM_M),   # FR  (front-right)
	Vector3(-ARM_M, 0.0, -ARM_M),   # RL  (rear-left)
	Vector3( ARM_M, 0.0, -ARM_M),   # RR  (rear-right)
]

## Swarm identity - set by main.gd before add_child()
@export var drone_id:  String = "Drone_0"
@export var is_player: bool   = true

## GPS signal state - external code (main.gd, hud_layer.gd) can set false to
## trigger DR; transition is detected in _physics_process() regardless of
## who flips it.
var gps_signal: bool = true
var _prev_gps_signal := true

## Mission autopilot (non-player drones only) - see start_mission().
var mission_target: Vector3 = Vector3.ZERO
var mission_active: bool    = false
var _mission_done          := false
const MISSION_ARRIVE_DIST  := 0.5    # metres - close enough to call it "arrived"
const MISSION_KP           := 0.10   # position-error -> tilt-command gain (small-signal, same spirit as LEVEL_KP)
const MISSION_MAX_TILT     := 0.35   # clamp on the resulting pitch/roll command

## Demo control - HUD's "Force Layer" selector, forwarded verbatim in
## telemetry so recovery_orchestrator.py can skip auto-selection and compute
## that layer's genuine output on demand (never fabricates a result, just
## picks which real computation runs).
var force_layer: String = ""   # "" = auto, else "LSTM" / "EKF" / "RULE_BASED"

## Live trajectory visualization buffers (consumed by trajectory_visualizer.gd).
## Purely cosmetic/logging state - never fed back into physics.
var executed_path:  PackedVector3Array = []
var reference_path: PackedVector3Array = []
var predicted_path: PackedVector3Array = []
var _predicted_pos:  Vector3 = Vector3.ZERO
const PATH_MAX_POINTS   := 600
const PATH_RECORD_EVERY := 0.2   # seconds between executed-path samples
var _path_accum := 0.0

## Telemetry sender (send-only UDP, no bind needed)
var _udp         := PacketPeerUDP.new()
var _telem_accum := 0.0
const TELEM_RATE := 0.1   # 10 Hz

var _throttle  := 0.0
var _pitch_cmd := 0.0
var _roll_cmd  := 0.0
var _yaw_cmd   := 0.0
var _armed     := false
var _motor_out := [0.0, 0.0, 0.0, 0.0]
var _prop_angle:= [0.0, 0.0, 0.0, 0.0]
var _props: Array = []

# ── Simulated-sensor state ─────────────────────────────────────────────────
var _prev_lin_vel := Vector3.ZERO   # last frame's world-space linear_velocity, for accel differencing
var _accel_body   := Vector3.ZERO   # last-sampled body-frame specific force ("accelerometer" reading)
var _baro_bias    := 0.0            # slow random-walk barometer bias (m)

# ── Zero-AI emergency fail-safe state ───────────────────────────────────────
var _breadcrumb          := BreadcrumbRecoveryScript.new()
var _recovery_cmd         # Dictionary {"velocity": Vector3, "dt": float} or null
var _recovery_cmd_accum  := 0.0
var _breadcrumb_was_triggered := false

# ── AI correction blending ───────────────────────────────────────────────
# Corrections arrive roughly every telemetry tick (~10Hz) once GPS is lost --
# a raw per-tick assignment to linear_velocity would fight the RigidBody3D
# solver and read as jitter, so it's blended in smoothly instead. Deliberately
# different from breadcrumb's own raw override below, which stays a pure
# command-replay since it's an intentionally rare, minimal last resort, not a
# continuous correction stream.
const CORRECTION_BLEND_RATE := 3.0   # 1/s
var _last_goal := ""   # tracks TARGET->HOME transitions for the one-time "RTL ENGAGED" log line

# ── Auto-hover / stabilisation state ──────────────────────────────────────
# _attitude_active: only pitch/roll/yaw inputs set this - throttle does NOT.
# This means the pilot can hold/change altitude without cancelling hover.
var _attitude_active  := false
var _input_idle_timer := 0.0

const IDLE_GRACE         := 2.5    # seconds no attitude input before auto-hover
const STAB_RATE          := 4.0    # attitude washout speed (1/s)
const STAB_ANG_DAMP      := 16.0   # angular damping in hover
const HOVER_LIN_DAMP     := 6.0    # linear damping in hover (brakes XZ drift)
# Altitude-hold PI - DIRECT output (not step-based):
#   _throttle = integrator + KP * alt_err
#   integrator += KI * alt_err * dt   (integrates toward hover trim)
# KP: how aggressively P corrects per metre  KI: how fast trim is learned
const HOVER_ALT_KP       := 0.30   # proportional gain (throttle per metre)
const HOVER_ALT_KI       := 0.15   # integral gain   (trim learning rate)
const LEVEL_KP           := 0.018  # leveling gain: degrees of tilt → mixer command

var _hover_target_alt := -1.0
var _hover_throttle_i := 0.0
var _in_hover         := false
var _hover_manual     := false
var _hover_allowed    := true    # set false by H-off; only H-on restores it
var _hover_blend      := 0.0    # 0=off 1=full; ramps over 0.8s to prevent snap

func _ready() -> void:
	mass          = 0.281
	linear_damp   = 0.7
	angular_damp  = 7.0
	gravity_scale = 1.0
	for i in range(4):
		var p := get_node_or_null("Frame/Prop_%d" % i)
		_props.append(p)
	if is_player:
		DroneState.hover_target_alt = -1.0
	else:
		# Autonomous swarm drone: pre-arm and enter manual hover immediately
		_armed            = true
		_in_hover         = true
		_hover_allowed    = true
		_hover_manual     = true
		_hover_target_alt = position.y
		_hover_throttle_i = 0.5
		_input_idle_timer = IDLE_GRACE + 1.0   # skip auto-hover grace period

func _input(e: InputEvent) -> void:
	if not is_player:
		return
	if DroneState.cam_mode == 1:   # FREE ROAM - camera owns all input
		return
	if DroneState.mode != "SIM":
		return
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_ENTER: _toggle_arm()
			KEY_R:     _flip_upright()
			KEY_H:     _toggle_hover()
	elif e is InputEventJoypadButton and e.pressed:
		match e.button_index:
			JOY_BUTTON_A:             _toggle_arm()
			JOY_BUTTON_RIGHT_SHOULDER: _flip_upright()
			JOY_BUTTON_Y:             _toggle_hover()

func _toggle_arm() -> void:
	_armed = not _armed
	if is_player:
		DroneState.armed = _armed
	print("[%s] Motors %s" % [drone_id, "ARMED" if _armed else "DISARMED"])

func _flip_upright() -> void:
	var eu := global_transform.basis.get_euler()
	global_transform.basis = Basis(Vector3.UP, eu.y)
	angular_velocity = Vector3.ZERO
	linear_velocity  = Vector3.ZERO
	_throttle = 0.0
	print(">>> Flip upright")

func _toggle_hover() -> void:
	if _in_hover:
		# ── Turn hover OFF ──
		_in_hover         = false
		_hover_manual     = false
		_hover_allowed    = false   # blocks auto-hover until H pressed again
		_hover_blend      = 0.0
		_hover_target_alt = -1.0
		_hover_throttle_i = 0.0
		_input_idle_timer = 0.0
		DroneState.hover_target_alt = -1.0
		print(">>> HOVER OFF")
	else:
		# ── Turn hover ON ──
		_hover_allowed    = true    # re-enable auto-hover too
		_in_hover         = true
		_hover_manual     = true
		_input_idle_timer = IDLE_GRACE
		var cur_alt := maxf(global_position.y, 0.05)
		_hover_target_alt = cur_alt
		DroneState.hover_target_alt = _hover_target_alt
		_hover_throttle_i = _throttle
		print(">>> HOVER ON  (%.1f m)" % cur_alt)

# ── Mission autopilot (hands-on demo) ───────────────────────────────────────

## Called by hud_layer.gd's Mission Control panel. Only meaningful for
## non-player drones (the dedicated mission drone) -- the player flies by hand.
func start_mission(target: Vector3) -> void:
	mission_target = target
	mission_active = true
	_mission_done  = false
	executed_path.clear()
	reference_path.clear()
	predicted_path.clear()
	_log("MISSION START -> target (%.1f, %.1f, %.1f)" % [target.x, target.y, target.z])

## While GPS is live, steers toward mission_target using TRUE global_position --
## honest, since GPS really is available at this point. The moment gps_signal
## goes false this function returns immediately without touching _pitch_cmd/
## _roll_cmd, so guidance never secretly keeps using ground truth once "GPS is
## lost" -- the drone holds attitude via the normal auto-hover path until a
## velocity_cmd correction (LSTM/EKF/rule-based) or breadcrumb takes over.
func _update_mission_guidance(_dt: float) -> void:
	if not mission_active or _mission_done or not gps_signal:
		return

	var err := mission_target - global_position
	if Vector2(err.x, err.z).length() < MISSION_ARRIVE_DIST and absf(err.y) < MISSION_ARRIVE_DIST:
		_mission_done  = true
		mission_active = false
		_log("MISSION COMPLETE -- holding position")
		return

	var local_err := global_transform.basis.inverse() * Vector3(err.x, 0.0, err.z)
	_pitch_cmd = clampf(-local_err.z * MISSION_KP, -MISSION_MAX_TILT, MISSION_MAX_TILT)
	_roll_cmd  = clampf( local_err.x * MISSION_KP, -MISSION_MAX_TILT, MISSION_MAX_TILT)
	_attitude_active  = true
	_hover_target_alt = mission_target.y

# ── _process: reset cmds, then read keyboard, then read controller ──────────
# Only the PLAYER drone reads keyboard/gamepad state -- Input.is_key_pressed()
# is global, not per-node, so without this guard every drone would react to
# the same keypresses. Non-player drones get their _pitch_cmd/_roll_cmd from
# _update_mission_guidance() instead (in _physics_process, below).
func _process(dt: float) -> void:
	if DroneState.mode != "SIM":
		return
	if is_player:
		_pitch_cmd = 0.0
		_roll_cmd  = 0.0
		_yaw_cmd   = 0.0
		_read_keys(dt)
		_read_controller(dt)
	_spin_props(dt)

func _read_keys(dt: float) -> void:
	if not _armed:
		_throttle = 0.0
		return

	# Throttle - does NOT cancel hover; during hover it shifts the target altitude
	if Input.is_key_pressed(KEY_SPACE):
		_throttle = clampf(_throttle + DroneState.throttle_rate * dt, 0.0, 1.0)
		if _in_hover: _hover_target_alt += DroneState.throttle_rate * 2.0 * dt
	if Input.is_key_pressed(KEY_CTRL):
		_throttle = clampf(_throttle - DroneState.throttle_rate * dt, 0.0, 1.0)
		if _in_hover: _hover_target_alt = maxf(0.3, _hover_target_alt - DroneState.throttle_rate * 2.0 * dt)

	# W = nose down (forward), S = nose up (backward)
	var p := float(Input.is_key_pressed(KEY_S)) - float(Input.is_key_pressed(KEY_W))
	# A = roll right, D = roll left
	var r := float(Input.is_key_pressed(KEY_A)) - float(Input.is_key_pressed(KEY_D))
	# E = yaw right (CW), Q = yaw left (CCW)
	var y := float(Input.is_key_pressed(KEY_E)) - float(Input.is_key_pressed(KEY_Q))

	if p != 0.0: _pitch_cmd = p; _attitude_active = true
	if r != 0.0: _roll_cmd  = r; _attitude_active = true
	if y != 0.0: _yaw_cmd   = y; _attitude_active = true

## PS4 / gamepad - Mode 2 layout. Only overrides a command when stick is
## OUTSIDE the deadband - so keyboard inputs are NOT zeroed by an idle stick.
func _read_controller(dt: float) -> void:
	var joys := Input.get_connected_joypads()
	if joys.is_empty():
		return
	var id: int = joys[0]
	const DB := 0.08

	# Left Y → throttle (does NOT cancel hover; shifts target altitude)
	if _armed:
		var ly := Input.get_joy_axis(id, JOY_AXIS_LEFT_Y)
		if abs(ly) > DB:
			_throttle = clampf(_throttle - ly * DroneState.throttle_rate * dt, 0.0, 1.0)
			if _in_hover:
				_hover_target_alt = maxf(0.3,
					_hover_target_alt + ly * DroneState.throttle_rate * -2.0 * dt)

	# Left X → yaw
	var lx := Input.get_joy_axis(id, JOY_AXIS_LEFT_X)
	if abs(lx) > DB:
		_yaw_cmd       = _expo(lx)
		_attitude_active = true

	# Right Y → pitch
	var ry := Input.get_joy_axis(id, JOY_AXIS_RIGHT_Y)
	if abs(ry) > DB:
		_pitch_cmd     = _expo(ry)
		_attitude_active = true

	# Right X → roll
	var rx := Input.get_joy_axis(id, JOY_AXIS_RIGHT_X)
	if abs(rx) > DB:
		_roll_cmd      = _expo(-rx)
		_attitude_active = true

# Expo curve: soft centre, full authority at edges.
func _expo(v: float) -> float:
	var e := DroneState.expo
	return v * (e * v * v + (1.0 - e))

func _physics_process(dt: float) -> void:
	# Real "accelerometer" reading: specific force = coordinate acceleration minus
	# gravity, rotated into body frame. Sampled every physics frame regardless of
	# arm state so _send_telemetry() always has a physically consistent value.
	var world_accel := (linear_velocity - _prev_lin_vel) / maxf(dt, 0.0001)
	var gravity_vec := Vector3(0.0, -GRAVITY_MPS2 * gravity_scale, 0.0)
	_accel_body = global_transform.basis.inverse() * (world_accel - gravity_vec)
	_prev_lin_vel = linear_velocity

	var sim_ok := not is_player or DroneState.mode == "SIM"
	if not sim_ok or not _armed:
		_motor_out        = [0.0, 0.0, 0.0, 0.0]
		_hover_target_alt = -1.0
		_hover_throttle_i = 0.0
		_in_hover         = false
		_hover_blend      = 0.0
		_attitude_active  = false
		_push_state()
		return

	# ── GPS transition detection (drives the demo's visible behavior + trajectory buffers) ──
	if gps_signal != _prev_gps_signal:
		if not gps_signal:
			reference_path = (PackedVector3Array([global_position, mission_target])
							  if (mission_active or _mission_done) else PackedVector3Array())
			predicted_path = PackedVector3Array([global_position])
			_predicted_pos = global_position
			_last_goal = ""
			_log("GPS SIGNAL LOST -- mission autopilot suspended, awaiting recovery guidance")
		else:
			_log("GPS SIGNAL RESTORED")
		_prev_gps_signal = gps_signal

	# ── Executed-path sampling (visualization only, while armed) ──
	_path_accum += dt
	if _path_accum >= PATH_RECORD_EVERY:
		_path_accum = 0.0
		executed_path.append(global_position)
		if executed_path.size() > PATH_MAX_POINTS:
			executed_path.remove_at(0)

	# ── Fail-safe hierarchy health tracking ──────────────────────────────────
	# GPS present means every layer above the breadcrumb (RL/SSL/LSTM/EKF) is
	# presumed reachable via the normal telemetry loop, so keep the breadcrumb
	# "healthy" and only retain history from the CURRENT loss episode. Once
	# gps_signal goes false, only a fresh apply_ai_correction() call (proof a
	# higher layer is still alive) resets the health timer from here on.
	if gps_signal:
		_breadcrumb.mark_healthy()
		_breadcrumb.reset()
		_recovery_cmd = null
		_recovery_cmd_accum = 0.0
	_breadcrumb.record_sample(linear_velocity, dt)

	if not gps_signal and _breadcrumb.should_trigger():
		if not _breadcrumb_was_triggered:
			_breadcrumb_was_triggered = true
			_log("BREADCRUMB RECOVERY ACTIVE (no AI correction for >500ms)")
		_run_breadcrumb_recovery(dt)
		_push_state()
		return
	elif _breadcrumb_was_triggered:
		_breadcrumb_was_triggered = false

	if not is_player:
		_update_mission_guidance(dt)

	_update_stabilisation(dt)
	_mix_motors()
	_apply_forces()
	_push_state()
	_attitude_active = false

	_telem_accum += dt
	if _telem_accum >= TELEM_RATE:
		_telem_accum = 0.0
		_send_telemetry()

# ── Zero-AI emergency fail-safe: replay the breadcrumb buffer in reverse ────
# Pure command-replay - direct velocity override, no controller loop, no
# attitude correction, deliberately minimal so this layer stays trustworthy
# even if every learned/estimated layer above it has failed. Runs only while
# _breadcrumb.should_trigger() is true; once the buffer drains, should_trigger()
# naturally returns false and _physics_process() falls back to normal
# stabilisation/hover control on its own.
func _run_breadcrumb_recovery(dt: float) -> void:
	_recovery_cmd_accum += dt
	if _recovery_cmd == null or _recovery_cmd_accum >= float(_recovery_cmd["dt"]):
		_recovery_cmd_accum = 0.0
		_recovery_cmd = _breadcrumb.pop_next_reversal_command()
		if _recovery_cmd == null:
			_log("Breadcrumb recovery: buffer exhausted, handing off to hover.")
			return
		var v: Vector3 = _recovery_cmd["velocity"]
		_log("Breadcrumb recovery: replaying v=(%.2f, %.2f, %.2f)" % [v.x, v.y, v.z])
	linear_velocity = _recovery_cmd["velocity"]

# ── X-config motor mixer ───────────────────────────────────────────────────
# Top view (+Z = nose, +X = right):
#   FL(0)  FR(1)      Roll RIGHT (r>0): left motors (FL,RL) MORE thrust
#   RL(2)  RR(3)      Pitch FWD  (p<0): front motors LESS thrust
#                     Yaw RIGHT  (y>0): CCW motors (FL,RR) MORE thrust
#
#  FL(CCW) = t + r + p + y    FR(CW)  = t - r + p - y
#  RL(CW)  = t + r - p - y    RR(CCW) = t - r - p + y
func _mix_motors() -> void:
	var t := _throttle
	var p := _pitch_cmd * DroneState.pitch_sens
	var r := _roll_cmd  * DroneState.roll_sens
	var y := _yaw_cmd   * DroneState.yaw_sens
	_motor_out[0] = clampf(t + r + p + y, 0.0, 1.0)  # FL  CCW
	_motor_out[1] = clampf(t - r + p - y, 0.0, 1.0)  # FR  CW
	_motor_out[2] = clampf(t + r - p - y, 0.0, 1.0)  # RL  CW
	_motor_out[3] = clampf(t - r - p + y, 0.0, 1.0)  # RR  CCW

func _apply_forces() -> void:
	var up := global_transform.basis.y
	var net_yaw := 0.0
	for i in range(4):
		var force: Vector3 = up * (float(_motor_out[i]) * MAX_THRUST)
		var arm_w: Vector3 = global_transform.basis * (_arms[i] as Vector3)
		apply_force(force, arm_w)
		net_yaw += _motor_out[i] * PROP_DIR[i]
	apply_torque(up * net_yaw * DroneState.yaw_torque)

func _spin_props(dt: float) -> void:
	for i in range(4):
		if _props[i] == null: continue
		_prop_angle[i] += _motor_out[i] * PROP_DIR[i] * 2200.0 * dt
		_props[i].rotation.y = deg_to_rad(_prop_angle[i])

# ── Auto-stabilisation & hover-hold ─────────────────────────────────────────
# Triggered by ATTITUDE inactivity only (throttle changes are fine).
# On hover entry:
#   1. Latches current altitude as target
#   2. Boosts angular + linear damping → levels attitude AND brakes XZ drift
#      so the drone stops at the spot where input was released
#   3. P+I throttle loop holds altitude with rate-limited adjustments
# Throttle stick/keys shift _hover_target_alt instead of cancelling hover.
func _update_stabilisation(dt: float) -> void:
	# Use each drone's own euler angles - DroneState.pitch/roll only belong to Drone_0
	var _eu          := global_transform.basis.get_euler()
	var _local_pitch := rad_to_deg(-_eu.x)
	var _local_roll  := rad_to_deg(-_eu.z)

	if _attitude_active:
		if not _hover_manual:
			_input_idle_timer = 0.0
			_hover_target_alt = -1.0
			_hover_throttle_i = 0.0
			_in_hover         = false
			_hover_blend      = 0.0
			if is_player: DroneState.hover_target_alt = -1.0
			linear_damp  = lerpf(linear_damp,  0.7, 5.0 * dt)
			angular_damp = lerpf(angular_damp, 7.0, 5.0 * dt)
			return
		else:
			linear_damp  = lerpf(linear_damp,  0.7, 5.0 * dt)
			angular_damp = lerpf(angular_damp, 7.0, 5.0 * dt)
			_hover_blend = maxf(0.0, _hover_blend - dt * 2.0)
			if is_player: DroneState.hover_target_alt = _hover_target_alt
			var ca2 := maxf(global_position.y, 0.05)
			var ae2 := _hover_target_alt - ca2
			_hover_throttle_i = clampf(_hover_throttle_i + HOVER_ALT_KI * ae2 * dt, 0.02, 0.95)
			_throttle = clampf(_hover_throttle_i + HOVER_ALT_KP * ae2, 0.02, 0.98)
			return

	_input_idle_timer += dt
	if not _hover_allowed or _input_idle_timer < IDLE_GRACE:
		return

	var cur_alt := maxf(global_position.y, 0.05)

	if not _in_hover:
		_in_hover         = true
		_hover_blend      = 0.0
		_hover_target_alt = cur_alt
		if is_player: DroneState.hover_target_alt = _hover_target_alt
		_hover_throttle_i = _throttle

	_hover_blend = minf(1.0, _hover_blend + dt / 0.8)

	angular_damp = lerpf(angular_damp, STAB_ANG_DAMP, 6.0 * dt)
	linear_damp  = lerpf(linear_damp,  HOVER_LIN_DAMP, 4.0 * dt)

	_pitch_cmd = clampf(-_local_pitch * LEVEL_KP, -0.5, 0.5) * _hover_blend
	_roll_cmd  = clampf(-_local_roll  * LEVEL_KP, -0.5, 0.5) * _hover_blend
	_yaw_cmd   = lerpf(_yaw_cmd, 0.0, STAB_RATE * dt)

	if is_player: DroneState.hover_target_alt = _hover_target_alt
	var alt_err := _hover_target_alt - cur_alt
	_hover_throttle_i = clampf(_hover_throttle_i + HOVER_ALT_KI * alt_err * dt, 0.02, 0.95)
	_throttle = clampf(_hover_throttle_i + HOVER_ALT_KP * alt_err, 0.02, 0.98)

func _push_state() -> void:
	var eu := global_transform.basis.get_euler()
	if is_player:
		DroneState.update_from_physics(
			rad_to_deg(-eu.x),
			rad_to_deg(-eu.z),
			fmod(rad_to_deg(-eu.y) + 360.0, 360.0),
			maxf(0.0, global_position.y),
			linear_velocity.length(),
			_motor_out
		)

func _send_telemetry() -> void:
	var eu  := global_transform.basis.get_euler()
	var lat := DroneState.gps_lat + global_position.z * 0.000009
	var lon := DroneState.gps_lon + global_position.x * 0.000009
	var alt := maxf(0.0, global_position.y)

	# Gyroscope: true body-frame angular velocity (rad/s) - RigidBody3D's own
	# angular_velocity is global-frame, so rotate it into the body frame.
	var gyro_body := global_transform.basis.inverse() * angular_velocity

	# Barometer: altitude + slow random-walk bias + sensor noise.
	_baro_bias = clampf(_baro_bias + randf_range(-0.02, 0.02) * TELEM_RATE, -1.0, 1.0)
	var baro_alt := alt + _baro_bias + randf_range(-0.03, 0.03)

	# Magnetometer: fixed Earth-field direction rotated into body frame + noise.
	var mag_body := global_transform.basis.inverse() * WORLD_MAG_DIR
	mag_body += Vector3(randf_range(-0.02, 0.02), randf_range(-0.02, 0.02), randf_range(-0.02, 0.02))

	var payload := {
		"drone_id":   drone_id,
		"timestamp":  Time.get_unix_time_from_system(),
		"gps_signal": gps_signal,
		"lat": lat,   "lon": lon,   "alt": alt,
		"imu_acc_x":  _accel_body.x,
		"imu_acc_y":  _accel_body.y,
		"imu_acc_z":  _accel_body.z,
		"imu_gyro_x": gyro_body.x,
		"imu_gyro_y": gyro_body.y,
		"imu_gyro_z": gyro_body.z,
		"pitch": rad_to_deg(-eu.x),
		"roll":  rad_to_deg(-eu.z),
		"yaw":   fmod(rad_to_deg(-eu.y) + 360.0, 360.0),
		"mag_x": mag_body.x, "mag_y": mag_body.y, "mag_z": mag_body.z,
		"baro_alt":   baro_alt,
		"motor_out":  _motor_out.duplicate(),
		"speed":      linear_velocity.length(),
		"battery":    DroneState.battery_v,
	}
	if mission_active:
		payload["mission_target"] = [mission_target.x, mission_target.y, mission_target.z]
	if force_layer != "":
		payload["force_layer"] = force_layer

	var json := JSON.stringify(payload)
	_udp.set_dest_address("127.0.0.1", 14551)
	_udp.put_packet(json.to_utf8_buffer())

## Applies a live recovery correction from recovery_orchestrator.py (via
## udp_bridge.gd). Payload shape: {"layer": "LSTM"|"EKF"|"RULE_BASED",
## "goal": "TARGET"|"HOME"|"NONE", "velocity_cmd": [vx, vy, vz]} -- a
## world-frame m/s velocity, the same language breadcrumb_recovery.gd's
## pop_next_reversal_command() already uses. "goal" reports the orchestrator's
## hybrid target-then-RTL state: it keeps seeking the mission target for a
## while, then abandons it for the launch/home position if GPS loss drags on
## (see recovery_orchestrator.py's RTL_* constants) -- logged here as a
## one-time "RTL ENGAGED" event, the same pattern used for breadcrumb's own
## trigger-start logging below.
func apply_ai_correction(data: Dictionary) -> void:
	# A correction arrived, meaning a higher layer of the fail-safe hierarchy
	# (LSTM / EKF / rule-based, whichever produced it) is alive - resets the
	# breadcrumb's health timer so the zero-AI emergency floor stands down.
	_breadcrumb.mark_healthy()

	var layer: String = data.get("layer", "?")
	var goal: String = data.get("goal", "?")
	var vc: Array = data.get("velocity_cmd", [])
	if vc.size() < 3:
		return
	var target_v := Vector3(float(vc[0]), float(vc[1]), float(vc[2]))
	var pdt := get_physics_process_delta_time()

	# Smooth blend toward the commanded velocity - see CORRECTION_BLEND_RATE's
	# comment above for why this isn't a raw assignment.
	var blend := clampf(pdt * CORRECTION_BLEND_RATE, 0.0, 1.0)
	linear_velocity = linear_velocity.lerp(target_v, blend)

	# Visualization-only "what the recovery layer currently believes" track --
	# does not feed back into physics. The live equivalent of
	# evaluate_trajectory.py's offline chained reconstruction.
	_predicted_pos += target_v * pdt
	predicted_path.append(_predicted_pos)
	if predicted_path.size() > PATH_MAX_POINTS:
		predicted_path.remove_at(0)

	if goal == "HOME" and _last_goal != "HOME":
		_log("RTL ENGAGED -- mission target abandoned, heading home instead")
	_last_goal = goal

	_log("%s correction (goal=%s) -> v=(%.2f, %.2f, %.2f)" %
		 [layer, goal, target_v.x, target_v.y, target_v.z])

func _log(text: String) -> void:
	var line := "[%s] %s" % [drone_id, text]
	print(line)
	MissionLog.log_event(line)
