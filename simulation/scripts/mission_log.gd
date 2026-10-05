## Small event bus for the hands-on mission demo's Decision Terminal panel.
## Any script can call MissionLog.log_event("...") from anywhere -- the HUD's
## terminal panel just listens to event_logged and renders whatever arrives,
## so the log reflects real state transitions (GPS cut, mission start, which
## recovery layer is active, breadcrumb engaging) rather than anything
## fabricated for display purposes.
extends Node

signal event_logged(text: String)

const MAX_HISTORY := 200

var history: Array[String] = []

var _clock_start := 0

func _ready() -> void:
	_clock_start = Time.get_ticks_msec()

func log_event(text: String) -> void:
	var elapsed_s := float(Time.get_ticks_msec() - _clock_start) / 1000.0
	var line := "[T+%6.2fs] %s" % [elapsed_s, text]
	history.append(line)
	if history.size() > MAX_HISTORY:
		history.remove_at(0)
	event_logged.emit(line)
