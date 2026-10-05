extends Node

signal telemetry_updated

# Attitude (degrees)
var pitch:      float = 0.0
var roll:       float = 0.0
var yaw:        float = 0.0

# Navigation
var altitude_m: float = 0.0
var gps_lat:    float = 37.9755   # Athens default
var gps_lon:    float = 23.7348
var speed_ms:   float = 0.0

# Power  (4S pack = 14.8 V nominal)
var battery_v:  float = 14.8

# Motors (0.0–1.0), index: 0=FL 1=FR 2=RL 3=RR
var motor: Array = [0.0, 0.0, 0.0, 0.0]

# Status
var armed:    bool   = false
var mode:     String = "SIM"   # "SIM" or "SERIAL"
var cam_mode: int    = 0        # 0 = ORBIT  1 = FOLLOW
var hover_target_alt: float = -1.0  # -1 = not in hover-hold

# Control sensitivities (runtime-adjustable via HUD Settings panel)
var throttle_rate: float = 1.50   # throttle change per second (faster climb)
var pitch_sens:    float = 0.38   # pitch authority in mixer
var roll_sens:     float = 0.38   # roll authority in mixer
var yaw_sens:      float = 0.18   # yaw authority in mixer
var yaw_torque:    float = 0.12   # prop reaction torque scale (N⋅m)
var expo:          float = 0.82   # stick expo: 0=linear, 1=full cubic
								  # 0.82 = soft centre, full throw at edges (log-like)

var _emit_accum: float = 0.0
const EMIT_INTERVAL := 1.0 / 30.0   # max 30 HUD refreshes per second

func update_from_physics(p: float, r: float, y: float,
		alt: float, spd: float, m: Array) -> void:
	pitch = p; roll = r; yaw = y
	altitude_m = alt; speed_ms = spd; motor = m
	# Throttle signal to ~30 Hz to prevent per-physics-tick HUD redraws
	# _emit_accum is incremented in _process so we can't rely on dt here;
	# use a flag that _process resets after emitting.
	_needs_emit = true

var _needs_emit: bool = false

func _process(dt: float) -> void:
	_emit_accum += dt
	if _needs_emit and _emit_accum >= EMIT_INTERVAL:
		_emit_accum = 0.0
		_needs_emit = false
		telemetry_updated.emit()

func update_from_serial(data: Dictionary) -> void:
	if "pitch"    in data: pitch      = float(data["pitch"])
	if "roll"     in data: roll       = float(data["roll"])
	if "yaw"      in data: yaw        = float(data["yaw"])
	if "altitude" in data: altitude_m = float(data["altitude"])
	if "battery"  in data: battery_v  = float(data["battery"])
	if "motors"   in data: motor      = data["motors"]
	telemetry_updated.emit()
