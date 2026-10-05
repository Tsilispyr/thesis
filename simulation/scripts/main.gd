## Scene builder - constructs the entire simulation world in code.
## Atlas 4 LR frame geometry is built from primitives (no external meshes needed).
extends Node3D

func _ready() -> void:
	_build_environment()
	_build_ground()
	_build_helipad()
	var swarm := _build_swarm()
	add_child(_build_camera(swarm[1]))   # tracks Drone_1 -- the Mission Control demo drone
	_build_hud()
	_build_udp_listener()
	_build_trajectory_visualizer(swarm[1])   # tracks Drone_1, the mission demo drone
	_print_controls()

# ── Environment ────────────────────────────────────────────────────────────

func _build_environment() -> void:
	var env_node := WorldEnvironment.new()
	var env      := Environment.new()
	var sky      := Sky.new()
	var sky_mat  := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color      = Color(0.08, 0.28, 0.68)
	sky_mat.sky_horizon_color  = Color(0.55, 0.70, 0.85)
	sky_mat.ground_horizon_color = Color(0.55, 0.50, 0.40)
	sky_mat.ground_bottom_color  = Color(0.28, 0.24, 0.18)
	sky.sky_material           = sky_mat
	env.background_mode        = Environment.BG_SKY
	env.sky                    = sky
	env.ambient_light_source   = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy   = 0.6
	env.tonemap_mode           = Environment.TONE_MAPPER_FILMIC
	env_node.environment       = env
	add_child(env_node)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-52.0, 38.0, 0.0)
	sun.light_energy     = 1.3
	sun.shadow_enabled   = true
	add_child(sun)

# ── Ground ─────────────────────────────────────────────────────────────────

func _build_ground() -> void:
	var body := StaticBody3D.new()

	var mi   := MeshInstance3D.new()
	var mesh := PlaneMesh.new()
	mesh.size           = Vector2(200.0, 200.0)
	mesh.subdivide_width  = 1
	mesh.subdivide_depth  = 1
	var mat              := StandardMaterial3D.new()
	mat.albedo_color     = Color(0.22, 0.38, 0.16)
	mat.roughness        = 0.95
	mesh.material        = mat
	mi.mesh              = mesh
	body.add_child(mi)

	var col := CollisionShape3D.new()
	col.shape = WorldBoundaryShape3D.new()
	body.add_child(col)
	add_child(body)

func _build_helipad() -> void:
	# Flat dark disc on the ground - takeoff/landing reference
	var mi   := MeshInstance3D.new()
	var mesh := CylinderMesh.new()
	mesh.top_radius    = 0.60
	mesh.bottom_radius = 0.60
	mesh.height        = 0.005
	var mat            := StandardMaterial3D.new()
	mat.albedo_color   = Color(0.12, 0.12, 0.14)
	mesh.material      = mat
	mi.mesh            = mesh
	mi.position        = Vector3(0.0, 0.003, 0.0)
	add_child(mi)

	# "H" ring
	var ring  := MeshInstance3D.new()
	var rmesh := TorusMesh.new()
	rmesh.inner_radius = 0.52
	rmesh.outer_radius = 0.60
	var rmat           := StandardMaterial3D.new()
	rmat.albedo_color  = Color(0.85, 0.85, 0.15)
	rmesh.material     = rmat
	ring.mesh          = rmesh
	ring.position      = Vector3(0.0, 0.006, 0.0)
	add_child(ring)

# ── Swarm ──────────────────────────────────────────────────────────────────

## Spawns 3 drones.  Drone_0 = player-controlled; Drone_1 = mission-control
## demo drone (Mission Control HUD panel drives it); Drone_2 = autonomous hover.
func _build_swarm() -> Array:
	var ids       := ["Drone_0", "Drone_1", "Drone_2"]
	var positions := [Vector3(0.0, 1.8, 0.0), Vector3(4.0, 1.8, 0.0), Vector3(-4.0, 1.8, 0.0)]
	var result    := []
	for i in range(3):
		var d := _build_drone(ids[i], positions[i], i == 0)
		add_child(d)
		result.append(d)
	print("Swarm online: Drone_0 (player) + Drone_1 (mission demo) + Drone_2 (autonomous hover)")
	return result

func _build_drone(id: String, pos: Vector3, player: bool) -> RigidBody3D:
	var body := RigidBody3D.new()
	body.name     = id
	body.position = pos

	body.set_script(load("res://scripts/drone_body.gd"))
	body.set("drone_id",  id)
	body.set("is_player", player)

	var col   := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = Vector3(0.22, 0.05, 0.22)
	col.shape  = shape
	body.add_child(col)

	var frame := Node3D.new()
	frame.name = "Frame"
	body.add_child(frame)
	_build_frame_mesh(frame)

	return body

func _build_udp_listener() -> void:
	var bridge := Node.new()
	bridge.name = "UDPBridge"
	bridge.set_script(load("res://scripts/udp_bridge.gd"))
	add_child(bridge)

## Live 3D rendering of the mission demo's reference/predicted/executed paths
## (Goal 4 Category A) -- reads the point-history buffers drone_body.gd
## maintains on `target`, redraws every frame. 3D world-space geometry, so it
## lives here as a Node3D child of Main, not inside hud_layer.gd's 2D CanvasLayer.
func _build_trajectory_visualizer(target: Node3D) -> void:
	var viz := Node3D.new()
	viz.name = "TrajectoryVisualizer"
	viz.set_script(load("res://scripts/trajectory_visualizer.gd"))
	viz.call_deferred("set", "target", target)
	add_child(viz)

func _build_frame_mesh(frame: Node3D) -> void:
	# Motor identity colours - must match MOTOR_COLORS in hud_layer.gd
	const MOTOR_COLORS := [
		Color(1.00, 0.12, 0.12),   # FL - vivid red
		Color(0.10, 1.00, 0.15),   # FR - lime green
		Color(0.15, 0.50, 1.00),   # RL - dodger blue
		Color(1.00, 0.55, 0.05),   # RR - vivid orange
	]

	# --- Materials ---
	var dark  := _mat(Color(0.12, 0.12, 0.14))
	var metal := _mat(Color(0.55, 0.55, 0.60)); metal.metallic = 0.9
	var green := _mat(Color(0.04, 0.42, 0.08))
	# Front props = red-tinted, rear = dark - instantly shows orientation from above
	var prop_front := _mat(Color(0.85, 0.12, 0.12, 0.55))
	prop_front.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	var prop_rear  := _mat(Color(0.08, 0.08, 0.08, 0.55))
	prop_rear.transparency  = BaseMaterial3D.TRANSPARENCY_ALPHA

	# --- Centre stack ---
	frame.add_child(_box(Vector3(0.068, 0.006, 0.068), Vector3.ZERO, dark))
	frame.add_child(_box(Vector3(0.060, 0.003, 0.060), Vector3(0,0.010,0), dark))
	frame.add_child(_box(Vector3(0.036, 0.003, 0.036), Vector3(0,0.014,0), green))  # FC PCB
	frame.add_child(_box(Vector3(0.038, 0.003, 0.038), Vector3(0,-0.005,0), _mat(Color(0.05,0.05,0.25))))  # ESC

	# --- Arms (X-config at 45°) + motors + props + LEDs ---
	var arm_data := [
		[Vector3(-0.083, 0.0,  0.083), true ],  # FL
		[Vector3( 0.083, 0.0,  0.083), true ],  # FR
		[Vector3(-0.083, 0.0, -0.083), false],  # RL
		[Vector3( 0.083, 0.0, -0.083), false],  # RR
	]
	var arm_angles := [-PI*0.25, PI*0.25, -PI*0.75, PI*0.75]

	for i in range(4):
		var tip: Vector3 = arm_data[i][0]
		var front: bool  = arm_data[i][1]
		var ang: float   = arm_angles[i]

		var arm := _box(Vector3(0.014, 0.008, 0.092), tip * 0.5, dark)
		arm.rotation.y = ang
		frame.add_child(arm)

		frame.add_child(_cyl(0.0115, 0.0115, 0.016, tip + Vector3(0, 0.010, 0), metal))
		# Motor identity marker - emissive ring in motor colour
		frame.add_child(_cyl(0.013, 0.013, 0.004, tip + Vector3(0, 0.020, 0),
							 _emit_mat(MOTOR_COLORS[i], 3.0)))

		var prop := _cyl(0.0508, 0.0508, 0.002,
						 tip + Vector3(0, 0.020, 0),
						 prop_front if front else prop_rear)
		prop.name = "Prop_%d" % i
		frame.add_child(prop)

		# Navigation LEDs - red front, green rear (aviation standard)
		var led_col := Color(1.0, 0.1, 0.1) if front else Color(0.1, 1.0, 0.3)
		frame.add_child(_led(tip + Vector3(0, 0.022, 0), led_col))

	# --- Forward arrow - bright orange emissive cone, unmistakable nose marker ---
	# CylinderMesh top_radius=0 = cone; default axis is Y, rotate -90° X to point +Z
	var arrow := _cyl(0.0, 0.012, 0.032, Vector3(0, 0.014, 0.072),
					  _emit_mat(Color(1.0, 0.45, 0.0), 4.0))
	arrow.rotation.x = -PI * 0.5
	frame.add_child(arrow)

	# --- RPi Zero 2W (rear) ---
	frame.add_child(_box(Vector3(0.065, 0.003, 0.030), Vector3(0, 0.020, -0.030), _mat(Color(0.03,0.35,0.06))))

	# --- FPV camera (front) ---
	frame.add_child(_box(Vector3(0.020, 0.014, 0.014), Vector3(0, 0.008, 0.048), dark))
	frame.add_child(_cyl(0.006, 0.006, 0.005, Vector3(0, 0.008, 0.056), metal))

	# --- Battery (underside) ---
	frame.add_child(_box(Vector3(0.024, 0.015, 0.052), Vector3(0, -0.020, -0.010), _mat(Color(0.05,0.05,0.55))))

# ── Camera ─────────────────────────────────────────────────────────────────

func _build_camera(target: Node3D) -> Camera3D:
	var cam := Camera3D.new()
	cam.set_script(load("res://scripts/camera_controller.gd"))
	# target is set after add_child so _ready() runs first; we assign directly
	cam.call_deferred("set", "target", target)
	return cam

# ── HUD ────────────────────────────────────────────────────────────────────

func _build_hud() -> void:
	var hud := CanvasLayer.new()
	hud.set_script(load("res://scripts/hud_layer.gd"))
	add_child(hud)

# ── Mesh helpers ───────────────────────────────────────────────────────────

func _emit_mat(col: Color, energy: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color            = col
	m.emission_enabled        = true
	m.emission                = col
	m.emission_energy_multiplier = energy
	return m

func _led(pos: Vector3, col: Color) -> MeshInstance3D:
	var mi   := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius  = 0.006
	mesh.height  = 0.012
	mesh.material = _emit_mat(col, 3.0)
	mi.mesh     = mesh
	mi.position = pos
	return mi

func _mat(col: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	if col.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return m

func _box(sz: Vector3, pos: Vector3, mat: Material) -> MeshInstance3D:
	var mi   := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size     = sz
	mesh.material = mat
	mi.mesh       = mesh
	mi.position   = pos
	return mi

func _cyl(r_top: float, r_bot: float, h: float, pos: Vector3,
		mat: Material) -> MeshInstance3D:
	var mi   := MeshInstance3D.new()
	var mesh := CylinderMesh.new()
	mesh.top_radius    = r_top
	mesh.bottom_radius = r_bot
	mesh.height        = h
	mesh.material      = mat
	mi.mesh            = mesh
	mi.position        = pos
	return mi

# ── Startup info ───────────────────────────────────────────────────────────

func _print_controls() -> void:
	print("""
=== Project Atlas - SIM mode ===
  Enter        ARM / DISARM motors
  Space / Ctrl Throttle up / down
  W / S        Pitch
  A / D        Roll
  Q / E        Yaw
  Right-drag   Orbit camera
  Scroll       Zoom      F = reset
  Serial telemetry: UDP JSON → port 14550
  Camera defaults to TRACKING Drone_1 (the Mission Control demo drone) --
  press C for FREE_ROAM if you want to fly Drone_0 or look elsewhere.
  Mission Control HUD panel drives Drone_1 -- enter a target, START MISSION,
  then CUT CONNECTION to watch the live LSTM/EKF/rule-based/breadcrumb
  recovery hierarchy take over (see the Decision Terminal panel).
================================""")
