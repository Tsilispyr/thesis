## camera_controller.gd
## Third-person camera - two modes, toggle with C (or gamepad Select/Back).
##
## MODE 0 - TRACKING (Fixed-Angle + Deadzone)
##   Camera sits at a fixed world-space pitch/yaw offset from an invisible
##   "anchor" point.  The anchor only moves when the target exits a configurable
##   screen-space deadzone; inside the box the camera is perfectly still even if
##   the target spins or jitters.  Target rotation is completely ignored.
##
## MODE 1 - FREE ROAM
##   Camera detaches from the target and flies freely.
##   WASD / Q / E  +  mouse  or  gamepad sticks.
##   Player drone inputs are locked while this mode is active.
##
## All tunable values are exposed via @export so they appear in the Inspector.
extends Camera3D

# ════════════════════════════════════════════════════════════════════════════
#  INSPECTOR SETTINGS
# ════════════════════════════════════════════════════════════════════════════

@export_group("Target")
## Node to track.  Assign in the Inspector or set via code (main.gd does this).
@export var target: Node3D = null

# ── Fixed-angle ──────────────────────────────────────────────────────────────
@export_group("Camera Angle  (world-space, ignores target rotation)")
## Downward tilt in degrees.  0 = horizon, 90 = straight down.
@export_range(0.0, 89.0, 0.5, "degrees") var pitch_deg: float = 30.0
## Horizontal rotation offset in world space.  0 = camera looks toward -Z.
@export_range(-180.0, 180.0, 1.0, "degrees") var yaw_deg: float = 0.0

@export_group("Camera Position")
## Extra height above the target's world Y position (metres).
@export_range(0.0, 50.0, 0.1, "suffix:m") var camera_height: float = 4.0
## Pull-back distance along the camera's outward vector (metres).
@export_range(0.1, 100.0, 0.1, "suffix:m") var camera_distance: float = 8.0

# ── Deadzone ─────────────────────────────────────────────────────────────────
@export_group("Deadzone")
## Fraction of the viewport width the target may roam without moving the
## camera.  0 = no deadzone (always tracks), 1 = entire screen.
@export_range(0.0, 1.0, 0.01) var deadzone_width:  float = 0.25
## Fraction of the viewport height for the deadzone.
@export_range(0.0, 1.0, 0.01) var deadzone_height: float = 0.20
## Show the deadzone as a yellow debug rectangle on screen.
@export var show_deadzone: bool = false

# ── Tracking ─────────────────────────────────────────────────────────────────
@export_group("Tracking")
## Lerp rate for anchor catch-up when the target exits the deadzone.
## Higher = tighter.  Time-scaled so behaviour is frame-rate independent.
@export_range(0.5, 30.0, 0.5) var tracking_speed: float = 6.0

# ── Free-Roam ────────────────────────────────────────────────────────────────
@export_group("Free-Roam")
## Translation speed while flying freely (m/s).
@export_range(1.0, 100.0, 1.0, "suffix:m/s") var freeroam_move_speed: float = 12.0
## Mouse look sensitivity (degrees per pixel of mouse movement).
@export_range(0.02, 1.0, 0.01) var freeroam_mouse_sens: float = 0.20
## Gamepad right-stick look multiplier (scales on top of mouse sensitivity).
@export_range(0.5, 10.0, 0.1) var freeroam_stick_sens: float = 80.0

# ════════════════════════════════════════════════════════════════════════════
#  INTERNAL STATE
# ════════════════════════════════════════════════════════════════════════════

const _MODE_TRACKING  := 0
const _MODE_FREE_ROAM := 1

var _mode:        int     = _MODE_TRACKING
var _anchor:      Vector3 = Vector3.ZERO   # world-point the camera orbits around
var _ready_done:  bool    = false
var _dz_overlay:  Control = null           # optional debug rectangle

# Free-roam Euler angles (maintained separately to avoid gimbal issues)
var _fr_yaw:   float = 0.0
var _fr_pitch: float = 0.0

# ════════════════════════════════════════════════════════════════════════════
#  LIFECYCLE
# ════════════════════════════════════════════════════════════════════════════

func _ready() -> void:
	_fr_yaw   = rotation_degrees.y
	_fr_pitch = rotation_degrees.x

	if is_instance_valid(target):
		_anchor     = target.global_position
		_ready_done = true
		_apply_tracking_transform(_anchor)   # prevent one-frame snap on start

	if show_deadzone:
		_build_deadzone_overlay()


func _notification(what: int) -> void:
	# Always release the mouse when the window closes or the node is freed
	if what in [NOTIFICATION_WM_CLOSE_REQUEST, NOTIFICATION_PREDELETE]:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

# ════════════════════════════════════════════════════════════════════════════
#  INPUT
# ════════════════════════════════════════════════════════════════════════════

func _input(e: InputEvent) -> void:
	# ── Mode toggle: C key or gamepad Select/Back button ─────────────────────
	var is_toggle: bool = (
		(e is InputEventKey and e.pressed and not e.echo and e.keycode == KEY_C)
		or
		(e is InputEventJoypadButton and e.pressed
		 and e.button_index == JOY_BUTTON_BACK)
	)
	if is_toggle:
		_set_mode(_MODE_FREE_ROAM if _mode == _MODE_TRACKING else _MODE_TRACKING)
		return

	# ── Free-roam mouse look (only while captured) ────────────────────────────
	if _mode == _MODE_FREE_ROAM and e is InputEventMouseMotion:
		_fr_yaw   -= e.relative.x * freeroam_mouse_sens
		_fr_pitch  = clampf(
			_fr_pitch - e.relative.y * freeroam_mouse_sens, -89.0, 89.0)


func _set_mode(new_mode: int) -> void:
	_mode = new_mode

	if _mode == _MODE_FREE_ROAM:
		# Capture mouse and seed free-roam angles from current camera facing
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		_fr_yaw   = rotation_degrees.y
		_fr_pitch = rotation_degrees.x
	else:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		# Snap anchor to target so the camera doesn't jump on return
		if is_instance_valid(target):
			_anchor = target.global_position

	# Notify HUD + any other listeners
	DroneState.cam_mode = _mode
	DroneState.telemetry_updated.emit()

# ════════════════════════════════════════════════════════════════════════════
#  PER-FRAME UPDATE
# ════════════════════════════════════════════════════════════════════════════

func _process(dt: float) -> void:
	match _mode:
		_MODE_TRACKING:  _process_tracking(dt)
		_MODE_FREE_ROAM: _process_free_roam(dt)

	if is_instance_valid(_dz_overlay):
		_dz_overlay.queue_redraw()

# ════════════════════════════════════════════════════════════════════════════
#  MODE 0 - TRACKING
# ════════════════════════════════════════════════════════════════════════════

func _process_tracking(dt: float) -> void:
	if not is_instance_valid(target):
		return

	# First-frame initialisation: anchor at target, camera snapped in place
	if not _ready_done:
		_anchor     = target.global_position
		_ready_done = true
		_apply_tracking_transform(_anchor)
		return

	# ── Step 1: Position the camera at the current anchor ────────────────────
	# This must happen first so unproject_position() returns accurate coords.
	_apply_tracking_transform(_anchor)

	# ── Step 2: Project target to normalised screen space [0 … 1] ───────────
	var target_world := target.global_position
	var vp_size      := get_viewport().get_visible_rect().size
	var screen_px    := unproject_position(target_world)
	var norm         := screen_px / vp_size

	# Detect whether the target is behind the camera (projection invalid)
	var to_target  := target_world - global_position
	var in_front   := to_target.dot(-global_transform.basis.z) > 0.0

	# ── Step 3: Deadzone test ────────────────────────────────────────────────
	# The deadzone is a centred rectangle in normalised screen space.
	# While the target sits inside it, the anchor (and camera) do not move.
	var half_w := deadzone_width  * 0.5
	var half_h := deadzone_height * 0.5
	var in_dz   := (
		in_front
		and norm.x >= 0.5 - half_w and norm.x <= 0.5 + half_w
		and norm.y >= 0.5 - half_h and norm.y <= 0.5 + half_h
	)

	# ── Step 4: Move anchor if target left the deadzone ──────────────────────
	if not in_dz:
		_anchor = _anchor.lerp(target_world, tracking_speed * dt)
		# Re-apply immediately so this frame already shows the updated position
		_apply_tracking_transform(_anchor)


func _apply_tracking_transform(pivot: Vector3) -> void:
	## Position the camera at [pivot + fixed world-space offset] and aim at pivot.
	## Target rotation is never consulted - only the pivot world position matters.

	# Build offset using a pure world-space rotation (pitch then yaw, no roll)
	var rot    := Quaternion.from_euler(
		Vector3(deg_to_rad(-pitch_deg), deg_to_rad(yaw_deg), 0.0))
	var offset := rot * Vector3(0.0, 0.0, camera_distance)
	offset.y  += camera_height

	global_position = pivot + offset

	# Look slightly above the pivot for more natural framing
	var look_pt := pivot + Vector3.UP * maxf(camera_height * 0.15, 0.3)
	if global_position.distance_squared_to(look_pt) > 1e-4:
		look_at(look_pt, Vector3.UP)

# ════════════════════════════════════════════════════════════════════════════
#  MODE 1 - FREE ROAM
# ════════════════════════════════════════════════════════════════════════════

func _process_free_roam(dt: float) -> void:
	# ── Look: gamepad right stick ─────────────────────────────────────────────
	var joy_rx := Input.get_joy_axis(0, JOY_AXIS_RIGHT_X)
	var joy_ry := Input.get_joy_axis(0, JOY_AXIS_RIGHT_Y)
	const DEADBAND := 0.12
	if abs(joy_rx) > DEADBAND or abs(joy_ry) > DEADBAND:
		_fr_yaw   -= joy_rx * freeroam_stick_sens * dt
		_fr_pitch  = clampf(
			_fr_pitch - joy_ry * freeroam_stick_sens * dt, -89.0, 89.0)

	# Apply rotation - never roll
	rotation_degrees = Vector3(_fr_pitch, _fr_yaw, 0.0)

	# ── Move: keyboard + gamepad left stick ──────────────────────────────────
	# Note: WASD here does NOT conflict with the drone because drone_body.gd
	# skips its _input() when DroneState.cam_mode == _MODE_FREE_ROAM.
	var dir := Vector3.ZERO

	if Input.is_key_pressed(KEY_W): dir.z -= 1.0   # forward
	if Input.is_key_pressed(KEY_S): dir.z += 1.0   # back
	if Input.is_key_pressed(KEY_A): dir.x -= 1.0   # strafe left
	if Input.is_key_pressed(KEY_D): dir.x += 1.0   # strafe right
	if Input.is_key_pressed(KEY_E): dir.y += 1.0   # ascend
	if Input.is_key_pressed(KEY_Q): dir.y -= 1.0   # descend

	var joy_lx := Input.get_joy_axis(0, JOY_AXIS_LEFT_X)
	var joy_ly := Input.get_joy_axis(0, JOY_AXIS_LEFT_Y)
	if abs(joy_lx) > DEADBAND: dir.x += joy_lx
	if abs(joy_ly) > DEADBAND: dir.z += joy_ly

	if dir.length_squared() > 1e-4:
		dir = dir.normalized()

	# Translate in camera-local space (WASD always moves where you look)
	global_position += global_transform.basis * dir * freeroam_move_speed * dt

# ════════════════════════════════════════════════════════════════════════════
#  DEBUG OVERLAY - deadzone rectangle
# ════════════════════════════════════════════════════════════════════════════

## Inner Control node that draws the deadzone rectangle each frame.
## Created only when show_deadzone = true.
class _DeadzoneOverlay extends Control:
	var cam: Camera3D   # reference set by _build_deadzone_overlay()

	func _draw() -> void:
		if not is_instance_valid(cam):
			return
		var vp := cam.get_viewport().get_visible_rect().size
		var hw: float = cam.deadzone_width  * 0.5 * vp.x
		var hh: float = cam.deadzone_height * 0.5 * vp.y
		var r   := Rect2(vp.x * 0.5 - hw, vp.y * 0.5 - hh, hw * 2.0, hh * 2.0)
		# Border
		draw_rect(r, Color(1.0, 0.9, 0.0, 0.85), false, 2.0)
		# Corner ticks for readability
		const T := 8.0
		for corner in [r.position,
					   Vector2(r.end.x, r.position.y),
					   Vector2(r.position.x, r.end.y),
					   r.end]:
			draw_circle(corner, 3.0, Color(1.0, 0.9, 0.0, 1.0))
		# Crosshair at screen centre
		draw_line(Vector2(vp.x*0.5 - 6, vp.y*0.5),
				  Vector2(vp.x*0.5 + 6, vp.y*0.5), Color(1,1,1,0.4), 1.0)
		draw_line(Vector2(vp.x*0.5, vp.y*0.5 - 6),
				  Vector2(vp.x*0.5, vp.y*0.5 + 6), Color(1,1,1,0.4), 1.0)


func _build_deadzone_overlay() -> void:
	var canvas := CanvasLayer.new()
	canvas.layer = 10   # render above HUD
	add_child(canvas)

	var overlay := _DeadzoneOverlay.new()
	overlay.cam = self
	overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	canvas.add_child(overlay)

	_dz_overlay = overlay
