## Compass rose instrument - reads DroneState.yaw each frame.
extends Control

const LABELS := ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

func _ready() -> void:
	DroneState.telemetry_updated.connect(queue_redraw)

func _draw() -> void:
	var cx  := size.x * 0.5
	var cy  := size.y * 0.5
	var r   := minf(cx, cy) - 2.0
	var yaw := DroneState.yaw
	var font := ThemeDB.fallback_font

	# Background
	draw_circle(Vector2(cx, cy), r, Color(0.06, 0.06, 0.10, 0.82))

	# Cardinal + inter-cardinal ticks + labels
	for i in range(8):
		var world_deg := float(i) * 45.0
		# Angular position on rose = world_deg - yaw (rose rotates with heading)
		var screen_rad := deg_to_rad(world_deg - yaw - 90.0)
		var is_main    := (i % 2 == 0)
		var tick_len   := 12.0 if is_main else 6.0
		var outer := Vector2(cx + cos(screen_rad) * r,
							 cy + sin(screen_rad) * r)
		var inner := Vector2(cx + cos(screen_rad) * (r - tick_len),
							 cy + sin(screen_rad) * (r - tick_len))
		var col   := (Color.RED if LABELS[i] == "N" else Color.WHITE)
		draw_line(inner, outer, col, 1.8 if is_main else 1.0)

		if is_main:
			var lpos := Vector2(cx + cos(screen_rad) * (r - tick_len - 12),
								cy + sin(screen_rad) * (r - tick_len - 12))
			draw_string(font, lpos - Vector2(6.0, 6.0), LABELS[i],
						HORIZONTAL_ALIGNMENT_LEFT, -1, 11,
						Color.RED if LABELS[i] == "N" else Color.WHITE)

	# Heading pointer (fixed triangle at top)
	var tip   := Vector2(cx, cy - r + 2.0)
	var left  := Vector2(cx - 6.0, cy - r + 14.0)
	var right := Vector2(cx + 6.0, cy - r + 14.0)
	draw_colored_polygon(PackedVector2Array([tip, left, right]), Color.ORANGE)

	# Border
	draw_arc(Vector2(cx, cy), r, 0.0, TAU, 64, Color(0.25, 0.25, 0.25), 2.0)

	# Heading text
	draw_string(font, Vector2(cx - 18.0, cy + 8.0),
				"%03d°" % (int(DroneState.yaw) % 360),
				HORIZONTAL_ALIGNMENT_LEFT, -1, 14, Color.WHITE)
