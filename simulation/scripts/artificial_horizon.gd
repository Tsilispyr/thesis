## Artificial horizon - simple fill approach.
## Brown = ground (full circle), Blue = sky cap traced through screen top.
extends Control

const SKY := Color(0.14, 0.44, 0.82)
const GND := Color(0.52, 0.35, 0.12)
const STEPS := 52

func _ready() -> void:
	DroneState.telemetry_updated.connect(queue_redraw)

func _draw() -> void:
	var cx  := size.x * 0.5
	var cy  := size.y * 0.5
	var r   := minf(cx, cy) - 2.0

	# Roll: positive = bank right → right end of horizon drops
	var rr  := deg_to_rad(DroneState.roll)
	# Pitch: positive = nose up → horizon drops below centre (more sky above)
	# In screen coords: positive Y = down.  pitch_shift > 0 → hy > cy → horizon lower ✓
	var pp  := clampf(DroneState.pitch * r / 65.0, -r * 0.95, r * 0.95)

	# Horizon centre: shift in "ground" direction (down when level)
	# Ground direction (screen): (sin rr, cos rr)  - at rr=0 this is (0,1) = down ✓
	var hx  := cx + sin(rr) * pp
	var hy  := cy + cos(rr) * pp

	# Horizon tangent (direction along horizon line in screen space)
	# At rr=0: (1,0) = horizontal.  Roll right: right end goes down → (cos rr, sin rr) ✓
	var hdx := cos(rr)
	var hdy := sin(rr)

	# Intersect horizon line with circle
	var ox  := hx - cx;  var oy := hy - cy
	var bv  := ox * hdx + oy * hdy
	var cv  := ox*ox + oy*oy - r*r
	var dsc := bv*bv - cv

	# Always start by filling everything ground
	draw_circle(Vector2(cx, cy), r, GND)

	if dsc < 0.0:
		# Horizon entirely off-screen: all sky or all ground
		# Sky direction (screen-up rotated by roll): (-sin rr, -cos rr)
		# If circle centre is on sky side of horizon → all sky
		var dot := ox * (-sin(rr)) + oy * (-cos(rr))
		if dot >= 0.0:
			draw_circle(Vector2(cx, cy), r, SKY)
		# else: already full ground
		_draw_overlays(cx, cy, r, rr)
		return

	var sq  := sqrt(dsc)
	var p1  := Vector2(hx + hdx * (-bv - sq), hy + hdy * (-bv - sq))
	var p2  := Vector2(hx + hdx * (-bv + sq), hy + hdy * (-bv + sq))
	var a1  := atan2(p1.y - cy, p1.x - cx)
	var a2  := atan2(p2.y - cy, p2.x - cx)

	# Build two arcs: CCW (a1→a2) and CW (a1→a2 the other way)
	var da_ccw := a2 - a1
	if da_ccw < 0.0: da_ccw += TAU

	# Sky arc = whichever arc passes through screen TOP (angle = -π/2, i.e. 3π/2 normalised)
	# A normalised angle of 3π/2 corresponds to sin = -1 = minimum Y = top of screen
	var top := 3.0 * PI / 2.0
	var dt  := top - a1
	if dt < 0.0: dt += TAU
	var ccw_has_top := (dt <= da_ccw)   # does the CCW arc pass through 3π/2?

	var sky_pts := PackedVector2Array()
	sky_pts.append(p1)
	var da_sky := da_ccw if ccw_has_top else (da_ccw - TAU)   # positive=CCW, negative=CW
	for i in range(1, STEPS):
		var ang := a1 + da_sky * (float(i) / float(STEPS))
		sky_pts.append(Vector2(cx + cos(ang) * r, cy + sin(ang) * r))
	sky_pts.append(p2)

	# Sky polygon = arc + chord (p2 → p1 closing via chord is implicit in polygon)
	if sky_pts.size() >= 3:
		draw_colored_polygon(sky_pts, SKY)

	_draw_overlays(cx, cy, r, rr)

func _draw_overlays(cx: float, cy: float, r: float, rr: float) -> void:
	var font := ThemeDB.fallback_font
	var snx  := -sin(rr)
	var sny  := -cos(rr)
	var hdx  :=  cos(rr)
	var hdy  :=  sin(rr)

	# Pitch tick marks every 10° (above and below horizon)
	for deg in [-30, -20, -10, 10, 20, 30]:
		# Positive deg = above horizon = sky direction = snx,sny direction
		# Shift by (deg - current_pitch) in sky direction
		var off: float = (float(deg) - DroneState.pitch) * r / 65.0
		var tcx := cx + snx * off
		var tcy := cy + sny * off
		var hw  := 14.0 if deg % 20 == 0 else 8.0
		draw_line(Vector2(tcx - hdx * hw, tcy - hdy * hw),
				  Vector2(tcx + hdx * hw, tcy + hdy * hw),
				  Color(1.0, 1.0, 1.0, 0.7), 1.2)

	# Horizon centre line
	var hp := DroneState.pitch * r / 65.0
	var lhx := cx - snx * hp;  var lhy := cy - sny * hp
	draw_line(Vector2(lhx - hdx * r * 0.9, lhy - hdy * r * 0.9),
			  Vector2(lhx + hdx * r * 0.9, lhy + hdy * r * 0.9),
			  Color(1.0, 1.0, 0.6, 0.9), 1.5)

	# Roll arc (top of instrument) and pointer
	draw_arc(Vector2(cx, cy), r - 5.0, deg_to_rad(-150), deg_to_rad(-30), 32,
			 Color(0.7, 0.7, 0.7, 0.6), 1.0)
	var pa := -PI * 0.5 + rr   # pointer rotates WITH roll (screen-space)
	draw_line(Vector2(cx + cos(pa) * (r - 5),  cy + sin(pa) * (r - 5)),
			  Vector2(cx + cos(pa) * (r - 14), cy + sin(pa) * (r - 14)),
			  Color.WHITE, 2.0)

	# Fixed aircraft symbol
	draw_line(Vector2(cx - 28.0, cy), Vector2(cx - 8.0, cy), Color.YELLOW, 2.0)
	draw_line(Vector2(cx +  8.0, cy), Vector2(cx + 28.0, cy), Color.YELLOW, 2.0)
	draw_circle(Vector2(cx, cy), 3.0, Color.YELLOW)

	# Border
	draw_arc(Vector2(cx, cy), r, 0.0, TAU, 64, Color(0.3, 0.3, 0.3), 2.0)

	# Values
	draw_string(font, Vector2(3.0, 13.0), "P %+5.1f°" % DroneState.pitch,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color.LIME_GREEN)
	draw_string(font, Vector2(3.0, 26.0), "R %+5.1f°" % DroneState.roll,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color.LIME_GREEN)
