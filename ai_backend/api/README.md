# `ai_backend/api/` - Runtime Server & Rule-Based Logic

The live, running side of the AI backend - what's actually listening on a
port while the Godot simulation is playing. Everything else in `ai_backend/`
(`data_processing/`, `models/`) is used to *prepare* what these files serve;
these files are what actually talks to Godot in real time.

| File | What it is |
|---|---|
| `udp_server.py` | The UDP server. Binds `0.0.0.0:14551`, receives per-drone telemetry JSON from `drone_body.gd`, and delegates every packet to `recovery_orchestrator.py`'s `RecoveryOrchestrator.decide()`. Once GPS is lost, sends whatever layer the orchestrator picked back to Godot on port `14552` as `{"action": "ai_correction", "layer": ..., "goal": ..., "velocity_cmd": [...]}`. Also owns optional flight recording (`--record` flag, uses `data_processing/flight_recorder.py`). |
| `recovery_orchestrator.py` | **The live LSTM → EKF → rule-based decision logic**, factored out with no socket/threading dependencies of its own so it's directly unit-testable: `python ai_backend/api/recovery_orchestrator.py` runs a built-in smoke test (no Godot, no network) covering all three layer transitions, a staleness-reconnect guard, the seek-bias, and the RTL hybrid below. Runs the LSTM (`models/dead_reckoning_model.py`) once its 10-step buffer is warmed up, else steps a per-drone `models/ekf_baseline.py::DeadReckoningEKF` (using the telemetry's `baro_alt` for periodic correction), else falls back to a rule-based vector toward the current goal - the live, generalized descendant of the vector-toward-target idea in `path_recovery.py`. **Seek bias**: LSTM/EKF are pure dead-reckoning estimators (they report likely current motion, not "how to get home"), so the orchestrator adds a small, distance-proportional, capped (`SEEK_MAX_CONTRIB`) pull toward the current goal on top of the raw estimate, computed from a running per-drone position *belief* (anchored at the last known-good position, integrated forward each tick by whatever velocity was just commanded) - legitimate because the destination coordinate is a stored waypoint, independent of the live GPS fix that's actually missing. Without it, a coasting drone's estimate converges to a small, near-constant value and the drone just drifts. **Target-then-RTL hybrid**: the "current goal" starts at the mission's `mission_target` (if any) the moment GPS is lost, and stays there while there's still a reasonable chance of reaching it - but after `RTL_TIMEOUT_S` (8s) of continuous loss with the estimated distance still above `RTL_SKIP_IF_CLOSE_M` (3m), the goal switches to the drone's `home_pos` (latched from the very first telemetry packet ever seen for that drone, independent of GPS state, the same way a real autopilot latches home at arm time) and stays there - sticky - for the rest of that loss episode. No target at all -> home is the only goal from the start. Reported back as `"goal": "TARGET"\|"HOME"\|"NONE"`. Also accepts an optional `force_layer` (forwarded from the Godot HUD's demo dropdown) to compute a specific layer's real output on demand. |
| `path_recovery.py` | The original rule-based (non-learned) recovery baseline - deterministic Haversine-distance vector toward a hardcoded list of known-safe signal zones. Predates the live mission-drone demo; `recovery_orchestrator.py`'s rule-based layer is a from-scratch, local-metre reimplementation of the same normalize-and-point idea for a single dynamic target, not a caller of this file. Still used as the "boring baseline" `eval_metrics.py` benchmarks the RL agent against. |
| `swarm_logic.py` | Swarm interlinking: given the live signal state of every drone in the swarm, finds the nearest drone that still has signal so a cut-off drone can relay through it instead of returning all the way home. Not yet wired into the live UDP loop. |

## How this fits together

```
Godot (drone_body.gd)  --UDP 14551-->  udp_server.py  --UDP 14552-->  Godot (udp_bridge.gd)
                                             |
                                             +--> recovery_orchestrator.py  (LSTM -> EKF -> rule-based selection)
                                                        |
                                                        +--> models/dead_reckoning_model.py  (LSTM)
                                                        +--> models/ekf_baseline.py          (EKF)
```

`udp_server.py` is the file you run to serve a live Godot session:
`python ai_backend/api/udp_server.py` (see the top-level `AI_Recovery/README.md` for the full
walkthrough). `recovery_orchestrator.py` is also directly runnable on its own for headless
verification of the decision logic, without Godot or a network connection at all.

## Note on the fail-safe hierarchy (see `AI_RECOVERY_EXECUTION_PLAN.md` §12)

This directory is now genuinely the **live top three layers** of the project's degrade-toward-
simplicity fail-safe stack (Mission autopilot → LSTM → EKF → rule-based → Godot-side kinematic
breadcrumb) - not just LSTM as before. If this server or the UDP link between it and Godot goes
down entirely, the last layer (`simulation/scripts/breadcrumb_recovery.gd`, which runs
independently inside Godot, no dependency on this directory at all) is what keeps the drone
recoverable - that's by design, not an oversight, and is exactly what stopping `udp_server.py`
mid-flight demonstrates.
