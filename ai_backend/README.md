# `ai_backend/` - Python AI Backend

The full Python side of AI_Recovery: data pipeline, models, training scripts,
and the live UDP server that talks to the Godot simulation. See
`AI_RECOVERY_EXECUTION_PLAN.md` (project root) for the full narrative history
of how this got to its current state - this README is the map, that file is
the story.

## Subfolders

| Folder | What's in it |
|---|---|
| [`api/`](api/README.md) | The live UDP server (`udp_server.py`) and rule-based fallback logic - what's actually running while Godot plays. |
| [`data_processing/`](data_processing/README.md) | Loading/cleaning/fetching data - `dataset_parser.py` is the single source of truth for feature/target columns and the leakage-safe train/val split. |
| [`models/`](models/README.md) | Model architectures (`DeadReckoningLSTM`, `MaskedLSTMAutoencoder`, `DeadReckoningEKF`, PPO env), plus every trained artifact (`.pth`/`.pkl`/`.zip`) they've produced. |

## Top-level scripts

| File | What it is |
|---|---|
| `train.py` | The main production-training entry point. `python train.py --sources imu --epochs 20` - from-scratch supervised training of `DeadReckoningLSTM`. `--sources` accepts `nav`/`imu`/`px4` (space-separated for multiple; blocked by `verify_unit_consistency()` if the sources' units don't reconcile). Saves `models/dr_lstm_{tag}.pth` + matching scalers. |
| `eval_metrics.py` | Benchmarks the rule-based path-recovery baseline (`api/path_recovery.py`) against the trained RL agent (`models/rl_path_recovery.py`) over N episodes - success rate, average steps to recovery. |
| `ml_pipeline.py` | The full 12-step reproducible analysis pipeline: dataset audit → cleaning → normalization → feature engineering → sequence visualization → 4 training runs (A/B/C/D across sources) → comparison → RL eval → summary dashboard. Every run appends to `../runs/results.json`/`pipeline.log` and regenerates `../runs/figures/*.png`. This is what produces the numbers/figures actually worth citing (once rerun with all current fixes - see `AI_RECOVERY_EXECUTION_PLAN.md` §9/§10 for why older `runs/` output predates several bug fixes and shouldn't be trusted as-is). |
| `generate_uav_doc.py` | Generates a Greek-language project-overview `.docx` (`../SD_UAV_Network_Overview_GR.docx`) from a hardcoded narrative description. Documentation generator, not part of the training/inference pipeline. |
| `evaluate_trajectory.py` / `visualize_trajectory.py` / `visualize_training.py` / `plot_style.py` | Offline evaluation and figure generation for `runs/figures_single_drone/` - chained real-trajectory reconstruction, training curves, and the shared color palette/style helpers every figure (including the live tools below) draws from. |
| `live_mission_demo.py` | **Hands-on live demo, no Godot required.** A simple point-mass drone simulated directly in Python, driven by the same `api/recovery_orchestrator.py` that talks to the real Godot sim. `python ai_backend/live_mission_demo.py` opens an interactive window: enter a mission target, START MISSION, CUT CONNECTION to trigger GPS loss on demand and watch the live LSTM → EKF → rule-based recovery hierarchy (including the target-then-RTL hybrid, see below) take over - rendered as a 3D trajectory plot styled like the offline figures, with a live decision log beside it. `--headless-test` runs a scripted scenario with no display and saves a PNG snapshot - see the screenshot below. |
| `requirements.txt` | Python dependencies. Notably includes `pyulog` + `requests` (for `data_processing/download_px4_logs.py`) and `stable-baselines3`/`gymnasium` (for the RL agent). |
| `Dockerfile` | Container build for `api/udp_server.py`; see `../docker-compose.yml` at the project root for the orchestration. |

## `live_mission_demo.py` in action

![live_mission_demo](../screenshots/live_mission_demo.png)

A mission starts toward a far target (cyan reference line); the connection is cut almost
immediately, well before arrival. The recovery layer (here: LSTM, `goal=TARGET`) pulls toward the
original target for a few seconds - then, once 8s of continuous GPS loss elapses without getting
close, the hybrid logic in `api/recovery_orchestrator.py` engages RTL (`RTL ENGAGED` in the log):
the mission target is abandoned and the same layer instead pulls back toward the launch/home
position, visible as the correction's sign flipping and the magenta predicted-path line curving
back on itself. Regenerate with:

```powershell
python ai_backend/live_mission_demo.py --headless-test --out screenshots/live_mission_demo.png
```

## Quick start

```powershell
# from AI_Recovery/
pip install -r ai_backend/requirements.txt

# retrain the production model (imu-only, current default/recommended source)
python ai_backend/train.py --sources imu --epochs 20

# start the live server (loads whatever model exists in ai_backend/models/)
python ai_backend/api/udp_server.py
```

See the project-root `README.md` for the full walkthrough including the Godot
side, and `EXECUTION_GUIDE.md` for a more detailed step-by-step.
