## Full HUD - creates all child nodes in _ready() so nothing needs scene setup.
## Layout (1280×720 viewport):
##   Top-left:     status strip (armed / mode)
##   Left:         artificial horizon (160×160)
##   Right:        compass rose (120×120)
##   Bottom-left:  telemetry labels
##   Bottom-centre:motor bars (M1–M4)
##   Top-right:    GPS
extends CanvasLayer

var _lbl_arm:  Label
var _lbl_mode: Label
var _lbl_ctrl: Label
var _lbl_cam:  Label
var _lbl_alt:  Label
var _lbl_spd:  Label
var _lbl_bat:  Label
var _lbl_gps:  Label
var _motor_bars:     Array = []   # ProgressBar nodes
var _motor_pct_lbls: Array = []   # "xx%" Label nodes

# Motor identity colours - shared with drone mesh markers
const MOTOR_COLORS := [
	Color(1.00, 0.12, 0.12),   # FL - vivid red
	Color(0.10, 1.00, 0.15),   # FR - lime green
	Color(0.15, 0.50, 1.00),   # RL - dodger blue
	Color(1.00, 0.55, 0.05),   # RR - vivid orange
]

var _settings_visible := false
var _settings_panel:    Panel
var _sliders:           Array = []
var _slider_labels:     Array = []

# ── Hands-on mission demo (drives Drone_1) ─────────────────────────────────
const MISSION_DRONE_PATH := "/root/Main/Drone_1"
# Keeps mission targets within a range the fixed camera and the drone's small
# thrust budget can actually cover in a reasonable demo time -- the ground
# plane is 200x200m, but a target anywhere near its edges would fly off
# camera and take minutes to reach.
const MISSION_XZ_LIMIT := 40.0
const MISSION_ALT_MIN  := 1.0
const MISSION_ALT_MAX  := 25.0
var _le_target_x:    LineEdit
var _le_target_z:    LineEdit
var _le_target_alt:  LineEdit
var _opt_force_layer: OptionButton
var _terminal:        RichTextLabel

# [property_name, display_label, min, max, step]
const SETTINGS := [
	["throttle_rate", "Throttle Rate",         0.20, 4.00, 0.10],
	["expo",          "Stick Expo (0=linear)",  0.00, 0.97, 0.01],
	["pitch_sens",    "Pitch Max Authority",    0.05, 0.70, 0.01],
	["roll_sens",     "Roll Max Authority",     0.05, 0.70, 0.01],
	["yaw_sens",      "Yaw Max Authority",      0.02, 0.40, 0.01],
	["yaw_torque",    "Yaw Torque",             0.02, 0.60, 0.01],
]

func _ready() -> void:
	_add_horizon()
	_add_compass()
	_add_status_strip()
	_add_telemetry()
	_add_motor_bars()
	_add_axis_graph()
	_add_gps()
	_add_controls_hint()
	_add_mission_control()
	_add_decision_terminal()
	_add_settings_panel()   # added last so it renders on top when Tab is toggled
	DroneState.telemetry_updated.connect(_refresh)
	_refresh()

func _input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed and e.keycode == KEY_TAB:
		_settings_visible = not _settings_visible
		_settings_panel.visible = _settings_visible
		if _settings_visible:
			_sync_sliders_from_state()
		get_viewport().set_input_as_handled()   # prevent Tab reaching sliders

# ── Axis graph (bottom-right) ──────────────────────────────────────────────

func _add_axis_graph() -> void:
	var bg := _panel(Vector2(895, 532), Vector2(372, 96))
	var g  := Control.new()
	g.position = Vector2(2, 2)
	g.size     = Vector2(368, 92)
	g.set_script(load("res://scripts/axis_graph.gd"))
	bg.add_child(g)

# ── Instruments ────────────────────────────────────────────────────────────

func _add_horizon() -> void:
	var bg := _panel(Vector2(8, 48), Vector2(164, 164))
	var h  := Control.new()
	h.position = Vector2(2, 2)
	h.size     = Vector2(160, 160)
	h.set_script(load("res://scripts/artificial_horizon.gd"))
	bg.add_child(h)

func _add_compass() -> void:
	var bg := _panel(Vector2(8, 220), Vector2(124, 124))
	var c  := Control.new()
	c.position = Vector2(2, 2)
	c.size     = Vector2(120, 120)
	c.set_script(load("res://scripts/compass_rose.gd"))
	bg.add_child(c)

# ── Status strip (top-left) ────────────────────────────────────────────────

func _add_status_strip() -> void:
	var bg := _panel(Vector2(8, 4), Vector2(390, 40))
	_lbl_arm  = _label(bg, Vector2(6,    4), "", 14, Color.LIME_GREEN)
	_lbl_mode = _label(bg, Vector2(110,  4), "", 13, Color(0.6, 0.9, 1.0))
	_lbl_ctrl = _label(bg, Vector2(218,  4), "", 12, Color(0.9, 0.85, 0.4))
	_lbl_cam  = _label(bg, Vector2(265,  4), "", 12, Color(0.7, 0.7, 1.0))

# ── Telemetry (bottom-left) ────────────────────────────────────────────────

func _add_telemetry() -> void:
	var bg := _panel(Vector2(8, 590), Vector2(240, 76))
	_lbl_alt = _label(bg, Vector2(6, 4),  "", 13, Color.WHITE)
	_lbl_spd = _label(bg, Vector2(6, 22), "", 13, Color.WHITE)
	_lbl_bat = _label(bg, Vector2(6, 40), "", 13, Color.WHITE)

# ── Motor bars (bottom-centre) ─────────────────────────────────────────────

func _add_motor_bars() -> void:
	var names := ["FL", "FR", "RL", "RR"]
	var bg := _panel(Vector2(400, 626), Vector2(480, 74))
	for i in range(4):
		var cx := Vector2(8.0 + i * 118.0, 4.0)
		var mc: Color = MOTOR_COLORS[i]

		# Name label in motor colour
		_label(bg, cx, names[i], 11, mc)

		# Percentage label (right of name)
		var pct := _label(bg, cx + Vector2(26, 0), "0%", 10, mc)
		_motor_pct_lbls.append(pct)

		# Progress bar below
		var bar := ProgressBar.new()
		bar.position        = cx + Vector2(0, 16)
		bar.size            = Vector2(110, 14)
		bar.min_value       = 0.0
		bar.max_value       = 100.0
		bar.value           = 0.0
		bar.show_percentage = false
		bg.add_child(bar)
		_motor_bars.append(bar)

		var sty := StyleBoxFlat.new()
		sty.bg_color = mc
		bar.add_theme_stylebox_override("fill", sty)

# ── GPS (top-right) ────────────────────────────────────────────────────────

func _add_gps() -> void:
	var bg := _panel(Vector2(900, 4), Vector2(372, 40))
	_lbl_gps = _label(bg, Vector2(6, 6), "", 13, Color(0.7, 1.0, 0.7))

# ── Mission Control (top-centre) - hands-on demo: enter a target, start the
#    mission, cut the connection, watch which recovery layer takes over ────

func _add_mission_control() -> void:
	var bg := _panel(Vector2(398, 4), Vector2(497, 138))
	_label(bg, Vector2(6, 4), "MISSION CONTROL  (drives Drone_1)", 11, Color(0.75, 0.85, 1.0))
	_label(bg, Vector2(6, 20),
		   "Target, local metres  -  X/Z within ±%d, Alt %d-%d" %
		   [int(MISSION_XZ_LIMIT), int(MISSION_ALT_MIN), int(MISSION_ALT_MAX)],
		   9, Color(0.6, 0.6, 0.6))

	_le_target_x = _line_edit(bg, Vector2(6, 38), 60, "12.0")
	_le_target_x.text = "12.0"
	_label(bg, Vector2(70, 40), "X", 9, Color(0.7, 0.7, 0.7))
	_le_target_z = _line_edit(bg, Vector2(96, 38), 60, "0.0")
	_le_target_z.text = "0.0"
	_label(bg, Vector2(160, 40), "Z", 9, Color(0.7, 0.7, 0.7))
	_le_target_alt = _line_edit(bg, Vector2(186, 38), 60, "6.0")
	_le_target_alt.text = "6.0"
	_label(bg, Vector2(250, 40), "Alt", 9, Color(0.7, 0.7, 0.7))

	var btn_start := _button(bg, Vector2(6, 68), 150, "START MISSION", Color(0.15, 0.45, 0.20))
	btn_start.pressed.connect(_on_start_mission_pressed)

	_label(bg, Vector2(164, 72), "Force layer:", 9, Color(0.7, 0.7, 0.7))
	_opt_force_layer = OptionButton.new()
	_opt_force_layer.position = Vector2(250, 67)
	_opt_force_layer.size     = Vector2(120, 24)
	for opt in ["Auto", "LSTM", "EKF", "Rule-Based"]:
		_opt_force_layer.add_item(opt)
	_opt_force_layer.item_selected.connect(_on_force_layer_selected)
	bg.add_child(_opt_force_layer)

	var btn_cut := _button(bg, Vector2(6, 104), 150, "CUT CONNECTION", Color(0.55, 0.15, 0.15))
	btn_cut.pressed.connect(_on_cut_connection_pressed)
	var btn_restore := _button(bg, Vector2(164, 104), 150, "RESTORE GPS", Color(0.15, 0.35, 0.55))
	btn_restore.pressed.connect(_on_restore_gps_pressed)

func _mission_drone() -> Node:
	return get_node_or_null(MISSION_DRONE_PATH)

func _on_start_mission_pressed() -> void:
	var d := _mission_drone()
	if d == null:
		return
	var x   := clampf(_le_target_x.text.to_float(),   -MISSION_XZ_LIMIT, MISSION_XZ_LIMIT)
	var z   := clampf(_le_target_z.text.to_float(),   -MISSION_XZ_LIMIT, MISSION_XZ_LIMIT)
	var alt := clampf(_le_target_alt.text.to_float(), MISSION_ALT_MIN, MISSION_ALT_MAX)
	d.start_mission(Vector3(x, alt, z))

func _on_cut_connection_pressed() -> void:
	var d := _mission_drone()
	if d != null:
		d.gps_signal = false

func _on_restore_gps_pressed() -> void:
	var d := _mission_drone()
	if d != null:
		d.gps_signal = true

func _on_force_layer_selected(idx: int) -> void:
	var d := _mission_drone()
	if d == null:
		return
	var names := ["", "LSTM", "EKF", "RULE_BASED"]
	d.force_layer = names[idx]

# ── Decision Terminal (centre) - live log of GPS/mission/recovery-layer events ──

func _add_decision_terminal() -> void:
	var bg := _panel(Vector2(180, 270), Vector2(715, 250))
	_label(bg, Vector2(6, 4), "DECISION TERMINAL", 11, Color(0.75, 0.85, 1.0))

	_terminal = RichTextLabel.new()
	_terminal.position       = Vector2(6, 22)
	_terminal.size           = Vector2(703, 222)
	_terminal.bbcode_enabled = false
	_terminal.scroll_following = true
	_terminal.add_theme_font_size_override("normal_font_size", 11)
	_terminal.add_theme_color_override("default_color", Color(0.65, 1.0, 0.65))
	bg.add_child(_terminal)

	_rebuild_terminal()
	MissionLog.event_logged.connect(_on_log_event)

func _on_log_event(_line: String) -> void:
	_rebuild_terminal()

## RichTextLabel has no remove_line() API to trim from the top, so the
## terminal is simply re-rendered from MissionLog.history (already capped at
## MissionLog.MAX_HISTORY) whenever a new event arrives -- events are state
## transitions, not per-frame noise, so this is cheap in practice.
func _rebuild_terminal() -> void:
	_terminal.clear()
	for line in MissionLog.history:
		_terminal.append_text(line + "\n")

# ── Controls hint (bottom-right) ───────────────────────────────────────────

func _add_controls_hint() -> void:
	var lines := [
		"Enter / × ARM/DISARM",
		"Space / L↑  Throttle ↑",
		"Ctrl  / L↓  Throttle ↓",
		"W/S   / R↑↓ Pitch fwd/back",
		"A/D   / R←→ Roll left/right",
		"Q/E   / L←→ Yaw left/right",
		"H / △   Hover ON/OFF",
		"R / R1   Flip upright",
		"C      Cam: Track/FreeRoam",
		"  WASD/QE  Fly cam",
		"  Mouse    Look",
		"Tab    Sensitivity",
		"(idle) Auto-hover & stab",
	]
	var bg := _panel(Vector2(1100, 48), Vector2(178, 8 + 16 * lines.size()))
	for i in range(lines.size()):
		_label(bg, Vector2(5, 4 + i * 16), lines[i], 10, Color(0.6, 0.6, 0.6))

# ── Sensitivity settings panel (Tab to toggle) ────────────────────────────

func _add_settings_panel() -> void:
	var row_h  := 44
	var pad    := 10
	var w      := 370
	var h      := 34 + SETTINGS.size() * row_h
	_settings_panel = _panel(Vector2(455, 195), Vector2(w, h))

	# Override to fully opaque so it clearly sits above other HUD elements
	var sty := StyleBoxFlat.new()
	sty.bg_color     = Color(0.06, 0.07, 0.12, 0.97)
	sty.border_color = Color(0.45, 0.50, 0.70, 1.00)
	sty.set_border_width_all(2)
	sty.set_corner_radius_all(5)
	_settings_panel.add_theme_stylebox_override("panel", sty)
	_settings_panel.clip_contents = true
	_settings_panel.visible = false

	_label(_settings_panel, Vector2(pad, 7), "SENSITIVITY  [Tab] close", 11,
		   Color(0.75, 0.85, 1.0))

	for i in range(SETTINGS.size()):
		var s: Array = SETTINGS[i]
		var yo := float(30 + i * row_h)

		var lbl := _label(_settings_panel, Vector2(pad, yo), "", 11, Color.WHITE)
		_slider_labels.append(lbl)

		var slider := HSlider.new()
		slider.position  = Vector2(float(pad), yo + 16.0)
		slider.size      = Vector2(float(w - pad * 2), 16.0)
		slider.min_value = s[2]
		slider.max_value = s[3]
		slider.step      = s[4]
		slider.value     = DroneState.get(s[0])
		slider.value_changed.connect(_on_slider_changed.bind(i))
		_settings_panel.add_child(slider)
		_sliders.append(slider)

	_update_slider_labels()

func _on_slider_changed(v: float, idx: int) -> void:
	DroneState.set(SETTINGS[idx][0], v)
	_update_slider_labels()

func _sync_sliders_from_state() -> void:
	for i in range(_sliders.size()):
		_sliders[i].value = DroneState.get(SETTINGS[i][0])
	_update_slider_labels()

func _update_slider_labels() -> void:
	for i in range(_slider_labels.size()):
		var s: Array = SETTINGS[i]
		_slider_labels[i].text = "%-16s %.2f" % [s[1], DroneState.get(s[0])]

# ── Refresh ────────────────────────────────────────────────────────────────

func _refresh() -> void:
	_lbl_arm.text  = "ARMED" if DroneState.armed else "DISARMED"
	_lbl_arm.modulate = Color.LIME_GREEN if DroneState.armed else Color.RED
	_lbl_mode.text = "MODE: %s" % DroneState.mode
	var has_pad := not Input.get_connected_joypads().is_empty()
	_lbl_ctrl.text = "PAD" if has_pad else "KB"
	_lbl_cam.text  = "TRACK" if DroneState.cam_mode == 0 else "FLY"
	_lbl_alt.text  = "ALT  %6.1f m" % DroneState.altitude_m
	if DroneState.hover_target_alt >= 0.0:
		_lbl_alt.text += "  [HOVER %.1fm]" % DroneState.hover_target_alt
		_lbl_alt.modulate = Color(0.4, 1.0, 0.6)
	else:
		_lbl_alt.modulate = Color.WHITE
	_lbl_spd.text  = "SPD  %6.1f m/s"  % DroneState.speed_ms
	_lbl_bat.text  = "BAT  %5.1f V"    % DroneState.battery_v
	_lbl_gps.text  = "GPS  %.4f°N  %.4f°E" % [DroneState.gps_lat, DroneState.gps_lon]
	for i in range(4):
		if i < _motor_bars.size():
			var v: float = float(DroneState.motor[i]) * 100.0 if i < DroneState.motor.size() else 0.0
			_motor_bars[i].value = v
			if i < _motor_pct_lbls.size():
				_motor_pct_lbls[i].text = "%d%%" % int(v)

# ── Helpers ────────────────────────────────────────────────────────────────

func _panel(pos: Vector2, sz: Vector2) -> Panel:
	var pc := Panel.new()   # Panel doesn't reposition children (PanelContainer does)
	pc.position = pos
	pc.size     = sz
	var sty := StyleBoxFlat.new()
	sty.bg_color         = Color(0.04, 0.04, 0.06, 0.72)
	sty.border_color     = Color(0.25, 0.25, 0.30, 0.80)
	sty.set_border_width_all(1)
	sty.set_corner_radius_all(4)
	pc.add_theme_stylebox_override("panel", sty)
	add_child(pc)
	return pc

func _label(parent: Node, pos: Vector2, text: String,
		size: int, col: Color) -> Label:
	var l := Label.new()
	l.position    = pos
	l.text        = text
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", col)
	parent.add_child(l)
	return l

func _line_edit(parent: Node, pos: Vector2, width: float, placeholder: String) -> LineEdit:
	var le := LineEdit.new()
	le.position          = pos
	le.size              = Vector2(width, 24)
	le.placeholder_text  = placeholder
	le.add_theme_font_size_override("font_size", 11)
	parent.add_child(le)
	return le

func _button(parent: Node, pos: Vector2, width: float, text: String, col: Color) -> Button:
	var b := Button.new()
	b.position = pos
	b.size     = Vector2(width, 28)
	b.text     = text
	var sty := StyleBoxFlat.new()
	sty.bg_color = col
	sty.set_corner_radius_all(3)
	b.add_theme_stylebox_override("normal", sty)
	b.add_theme_font_size_override("font_size", 11)
	parent.add_child(b)
	return b
