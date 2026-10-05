## Listens on UDP 14550 for JSON telemetry from the Python serial bridge.
## Packet format: {"pitch":0.0,"roll":0.0,"yaw":0.0,"altitude":0.0,
##                 "battery":14.8,"motors":[0,0,0,0]}
## Stays in SIM mode silently when no sender is present.
extends Node

const UDP_PORT := 14550
var _udp  := PacketPeerUDP.new()
var _live := false

func _ready() -> void:
	if _udp.bind(UDP_PORT) == OK:
		_live = true
		print("[SerialBridge] listening UDP :%d (still SIM until first packet)" % UDP_PORT)
	else:
		print("[SerialBridge] port busy - SIM mode")

func _process(_dt: float) -> void:
	if not _live:
		return
	while _udp.get_available_packet_count() > 0:
		var raw  := _udp.get_packet()
		var data  = JSON.parse_string(raw.get_string_from_utf8())
		if data is Dictionary:
			DroneState.mode = "SERIAL"   # switch only on real data
			DroneState.update_from_serial(data)

func _exit_tree() -> void:
	_udp.close()
