## Scrolling real-time graph - Pitch, Roll, Throttle (avg motors).
## Sampled at ~30 Hz; stores 300 samples (~10 s of history).
extends Control

const MAX_PTS  := 300
const RATE_S   := 0.033   # sample every ~33 ms

var _hist := {"pitch": [], "roll": [], "thr": []}
var _t    := 0.0

const CH := [
	{"key": "pitch", "label": "Pitch",    "color": Color(0.35, 0.60, 1.00)},
	{"key": "roll",  "label": "Roll",     "color": Color(0.30, 1.00, 0.45)},
	{"key": "thr",   "label": "Throttle", "color": Color(1.00, 0.55, 0.20)},
]

func _ready() -> void:
	pass   # no signal needed - we sample in _process for consistent time axis

func _process(dt: float) -> void:
	_t += dt
	if _t < RATE_S:
		return
	_t = 0.0

	_push("pitch", DroneState.pitch / 90.0)
	_push("roll",  DroneState.roll  / 90.0)
	var avg := 0.0
	for m in DroneState.motor:
		avg += float(m)
	# Map throttle 0..1 → -1..+1 so zero throttle sits at centre line
	_push("thr", avg / 4.0 * 2.0 - 1.0)
	queue_redraw()

func _push(key: String, val: float) -> void:
	_hist[key].append(val)
	if _hist[key].size() > MAX_PTS:
		_hist[key].pop_front()

func _draw() -> void:
	var w   := size.x
	var h   := size.y
	var mid := h * 0.5
	var font := ThemeDB.fallback_font

	# Background + grid
	draw_rect(Rect2(0.0, 0.0, w, h), Color(0.04, 0.04, 0.06, 0.82))
	draw_line(Vector2(0.0, mid * 0.5), Vector2(w, mid * 0.5), Color(0.13, 0.13, 0.13), 1.0)
	draw_line(Vector2(0.0, mid),       Vector2(w, mid),       Color(0.22, 0.22, 0.22), 1.0)
	draw_line(Vector2(0.0, mid * 1.5), Vector2(w, mid * 1.5), Color(0.13, 0.13, 0.13), 1.0)

	# Channel traces
	for ch in CH:
		var arr: Array = _hist[ch.key]
		var n := arr.size()
		if n < 2:
			continue
		var pts := PackedVector2Array()
		for i in range(n):
			pts.append(Vector2(
				w * float(i) / float(MAX_PTS - 1),
				mid - float(arr[i]) * mid * 0.92))
		draw_polyline(pts, ch.color, 1.5)

	# Legend (bottom-left)
	var xi := 4.0
	for ch in CH:
		draw_string(font, Vector2(xi, h - 4.0), ch.label,
					HORIZONTAL_ALIGNMENT_LEFT, -1, 10, ch.color)
		xi += 68.0

	# Y-axis labels
	draw_string(font, Vector2(w - 26.0, 10.0),   "+90°", HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.4,0.4,0.4))
	draw_string(font, Vector2(w - 18.0, mid + 4.0), "0",  HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.4,0.4,0.4))
	draw_string(font, Vector2(w - 26.0, h - 4.0), "-90°", HORIZONTAL_ALIGNMENT_LEFT, -1, 9, Color(0.4,0.4,0.4))
