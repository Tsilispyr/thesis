## Zero-AI emergency fail-safe: the last-resort layer of the fail-safe
## hierarchy (RL policy -> SSL/LSTM -> classic EKF -> this).
##
## Pure kinematics, no learned model, no dependency on the Python backend or
## the UDP link being alive - runs entirely inside drone_body.gd's own
## _physics_process(). Maintains a decimated ring buffer of world-frame
## velocity samples; when the rest of the hierarchy has been unavailable for
## more than TRIGGER_TIMEOUT_MS, replays the buffer in reverse as negated
## velocity commands, retracing exactly the path just flown.
##
## Deliberately a plain RefCounted, not a Node - no scene-tree dependency,
## so it's directly unit-testable in isolation (instantiate, feed synthetic
## samples via record_sample(), assert the reconstructed reverse path).
class_name BreadcrumbRecovery
extends RefCounted

const DECIMATION_HZ         := 2.0                    # buffer one sample every 0.5s
const DECIMATION_INTERVAL   := 1.0 / DECIMATION_HZ
const TRIGGER_TIMEOUT_MS    := 500                     # ms of no "healthy" signal before triggering
const MAX_BUFFER_SAMPLES    := 240                     # 2 min of history at 2Hz

## Each entry: {"dt": float, "v_world": Vector3} - the world-frame velocity
## held for that decimated interval. Negating and replaying these in reverse
## retraces the path; see pop_next_reversal_command().
var _buffer: Array = []
var _accum_dt := 0.0
var _last_healthy_ms := 0

func _init() -> void:
	mark_healthy()

## Call every physics frame while flying normally (armed, not already in
## recovery). v_world is the world-frame velocity for this frame - in this
## simulation that's RigidBody3D.linear_velocity directly; on real hardware
## this would be reconstructed as R(q) * v_body from IMU-derived body-frame
## velocity and the attitude estimate. Buffers one decimated sample every
## DECIMATION_INTERVAL seconds, not every frame, to bound memory and avoid
## integrating high-frequency sensor noise into the backtracking path.
func record_sample(v_world: Vector3, dt: float) -> void:
	_accum_dt += dt
	if _accum_dt < DECIMATION_INTERVAL:
		return
	_buffer.append({"dt": _accum_dt, "v_world": v_world})
	if _buffer.size() > MAX_BUFFER_SAMPLES:
		_buffer.pop_front()
	_accum_dt = 0.0

## Call whenever a higher layer of the fail-safe hierarchy is confirmed
## alive: GPS signal present, or a fresh AI correction was just received.
## Resets the trigger timeout - this subsystem only activates once nothing
## above it has checked in for TRIGGER_TIMEOUT_MS.
func mark_healthy() -> void:
	_last_healthy_ms = Time.get_ticks_msec()

## True once no higher layer has reported healthy for > TRIGGER_TIMEOUT_MS
## and there is buffered history to retrace. This is the actual, computed
## health-check signal - not a placeholder - that the rest of the hierarchy
## degrades against.
func should_trigger() -> bool:
	return (Time.get_ticks_msec() - _last_healthy_ms > TRIGGER_TIMEOUT_MS
		and not _buffer.is_empty())

## Pops and returns the next reversal step as {"velocity": Vector3, "dt": float}
## - the negated world-frame velocity to command directly (pure command-
## replay, no position-setpoint controller, no target-seeking logic).
## Returns null once the buffer is exhausted (back near the origin) - the
## caller should hand off to normal hover/land behavior at that point.
func pop_next_reversal_command():
	if _buffer.is_empty():
		return null
	var entry: Dictionary = _buffer.pop_back()
	return {"velocity": -(entry["v_world"] as Vector3), "dt": entry["dt"]}

func has_history() -> bool:
	return not _buffer.is_empty()

## Full reset - call after a successful recovery (or a fresh arm) so a new
## flight starts with a clean buffer and health timer.
func reset() -> void:
	_buffer.clear()
	_accum_dt = 0.0
	mark_healthy()
