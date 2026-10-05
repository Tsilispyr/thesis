extends Node

## UDP Bridge - Godot 4
## LISTENER only: binds port 14552, receives AI commands from Python, dispatches to drones.
## Each drone sends telemetry independently via its own PacketPeerUDP (no bind needed).

var _udp      := PacketPeerUDP.new()
var _listen_port := 14552

func _ready() -> void:
	var err := _udp.bind(_listen_port)
	if err == OK:
		print("UDPBridge: listening for AI commands on port ", _listen_port)
	else:
		print("UDPBridge: FAILED to bind port ", _listen_port, " (err=", err, ")")

func _process(_delta: float) -> void:
	while _udp.get_available_packet_count() > 0:
		var msg := _udp.get_packet().get_string_from_utf8()
		_handle_command(msg)

func _handle_command(json_str: String) -> void:
	var dict = JSON.parse_string(json_str)
	if not dict is Dictionary:
		return
	var drone_id: String = dict.get("drone_id", "")
	var action:   String = dict.get("action",   "")
	if action == "ai_correction":
		var drone := get_node_or_null("/root/Main/" + drone_id)
		if drone and drone.has_method("apply_ai_correction"):
			drone.apply_ai_correction(dict)
