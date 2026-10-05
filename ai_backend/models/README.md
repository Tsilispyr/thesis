# `ai_backend/models/` - Model Architectures, Baselines, and Trained Artifacts

This folder holds both the **code that defines/trains models** and the
**trained weight files** those scripts produce. The two are easy to conflate
by filename alone, so: anything ending in `.py` is code; anything ending in
`.pth`/`.pkl`/`.zip` is a saved artifact, safe to delete and regenerate by
rerunning the corresponding script (nothing here is hand-authored data).

## Code

| File | What it is |
|---|---|
| `dead_reckoning_model.py` | `DeadReckoningLSTM` - the production architecture. LSTM(input=14, hidden=64, layers=2) → FC(64→32→3), predicting `[Δlat, Δlon, Δalt]` from a 10-timestep window. `INPUT_SIZE=14` here is the single source of truth other files import for the model's input width. |
| `ssl_pretext.py` | Self-supervised **masked-reconstruction pretext task** (`AI_RECOVERY_EXECUTION_PLAN.md` §15). `MaskedLSTMAutoencoder`'s encoder is architecturally identical to `DeadReckoningLSTM.lstm`, so `transfer_encoder_weights()` copies pretrained weights straight across (verified bit-identical). Two-stage CLI: `python ssl_pretext.py --sources imu --ssl-epochs N --finetune-epochs N` pretrains on masked-reconstruction (no labels needed), then fine-tunes the transferred encoder on the normal Δposition regression task - the "SSL-pretrained LSTM" arm of the Goal 2 ablation matrix. |
| `ekf_baseline.py` | `DeadReckoningEKF` - the classical (non-learned) baseline (§16). Real strapdown Kalman filter: predicts via double-integrated body-frame specific-force acceleration, corrects via barometric altitude, exposes its covariance as a genuine uncertainty estimate (an EKF gets this for free; the LSTM doesn't, without an extra output head). `python ekf_baseline.py` runs a synthetic smoke test showing why this beats pure inertial integration. **Also live**, not just an offline baseline: `ai_backend/api/recovery_orchestrator.py` imports `DeadReckoningEKF` directly and runs it as the second layer of the live LSTM → EKF → rule-based recovery hierarchy (in both the Godot integration and `ai_backend/live_mission_demo.py`) whenever the LSTM's 10-step buffer isn't warmed up yet. |
| `rl_path_recovery.py` | `SwarmRecoveryEnv` (Gymnasium) + PPO training via Stable-Baselines3 - the reinforcement-learning path-recovery agent, top of the fail-safe hierarchy (§12 Goal 3). Currently trained for only 10K timesteps (a proof-of-concept run, 28% success vs. the rule-based baseline's 100% - see `../eval_metrics.py`); scaling this to 500K–1M timesteps is a pending Phase-3 item. |

## Trained artifacts (regeneratable - not source-controlled data)

Filenames follow a `{prefix}_{tag}.{ext}` convention; the tag encodes
*which data source(s) and training method* produced the file.

| Tag pattern | Produced by | Meaning |
|---|---|---|
| `dr_lstm_imu_norm.pth` + `scaler_X/y_imu_norm.pkl` | `../train.py --sources imu` | **Current production model** - from-scratch supervised, imu-only. This is what `udp_server.py` loads by default. |
| `dr_lstm_px4_norm.pth` + `scaler_X/y_px4_norm.pkl` | `../train.py --sources px4` | Supervised, real-hardware-only (§11.3) - early result, val loss plateaus ≈1.5, likely needs more/more-varied flight logs, not a data-quality problem. |
| `ssl_pretrain_imu_ssl_norm.pth` | `ssl_pretext.py --sources imu` (stage 1) | Encoder-only checkpoint from the SSL pretext task - reload with `MaskedLSTMAutoencoder`, not `DeadReckoningLSTM`. |
| `dr_lstm_imu_ssl_norm.pth` + `scaler_X/y_imu_ssl_norm.pkl` | `ssl_pretext.py --sources imu` (stage 2) | Fine-tuned `DeadReckoningLSTM`, SSL-pretrained. **Currently only a 1-epoch smoke-test artifact - not a meaningful trained model yet**; rerun with real epoch counts before citing any result from it (§15). |
| `ppo_swarm_recovery.zip` | `rl_path_recovery.py` | Stable-Baselines3 PPO checkpoint (10K timesteps - proof-of-concept only, see above). |

Every `.pth`/`.pkl` pair must be loaded together - the scaler defines the
normalization the model was trained under; using a model with the wrong
scaler silently produces garbage predictions with no error.
