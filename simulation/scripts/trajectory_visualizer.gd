## Live 3D rendering of the mission demo's three trajectory tracks (Goal 4
## Category A -- AI_RECOVERY_EXECUTION_PLAN.md sec 12). Reads the point-history
## buffers `target` (a drone_body.gd instance) maintains and redraws them every
## frame as solid ImmediateMesh line-strips. Colors match
## ai_backend/plot_style.py's established role palette so the live view and the
## offline matplotlib figures share one visual language:
##   cyan    = reference/mission path (snapshotted the instant GPS is cut)
##   magenta = predicted path (what the active recovery layer believes)
##   black   = executed path (the drone's real position history)
## Solid, thin, uniform-width lines throughout -- ImmediateMesh line-strips
## don't support per-segment dash gaps without materially more geometry-
## building complexity, and three distinct colors are already unambiguous.
extends Node3D

var target: Node3D = null

const COLOR_REFERENCE    := Color(0.0, 1.0, 1.0)   # cyan
const COLOR_PREDICTED    := Color(1.0, 0.0, 1.0)   # magenta
const COLOR_GROUND_TRUTH := Color(0.0, 0.0, 0.0)   # black

var _mi_reference: MeshInstance3D
var _mi_predicted: MeshInstance3D
var _mi_executed:  MeshInstance3D

func _ready() -> void:
	_mi_reference = _make_line_instance(COLOR_REFERENCE)
	_mi_predicted = _make_line_instance(COLOR_PREDICTED)
	_mi_executed  = _make_line_instance(COLOR_GROUND_TRUTH)

func _make_line_instance(color: Color) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = ImmediateMesh.new()
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = color
	add_child(mi)
	return mi

func _process(_dt: float) -> void:
	if target == null or not is_instance_valid(target):
		return
	_draw_line(_mi_reference, target.get("reference_path"))
	_draw_line(_mi_predicted, target.get("predicted_path"))
	_draw_line(_mi_executed,  target.get("executed_path"))

func _draw_line(mi: MeshInstance3D, points) -> void:
	var mesh: ImmediateMesh = mi.mesh
	mesh.clear_surfaces()
	if points == null or points.size() < 2:
		return
	mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	for p in points:
		mesh.surface_add_vertex(p)
	mesh.surface_end()
