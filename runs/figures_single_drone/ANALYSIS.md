# Single-Drone Path Reconstruction - Figure-by-Figure Analysis

**Purpose of this document**: a thesis-ready walkthrough of every figure in this folder - what it shows, what it means, what decision or finding it represents, and how it connects to the ones before it. Written to be liftable directly into a Results/Discussion chapter.

**Context**: these figures answer one question - *does the trained dead-reckoning model actually reconstruct a real flight path better than doing nothing (pure inertial integration) or classical filtering (an EKF)?* - using **real** data throughout: a genuine held-out 12.5-second segment from `imu_data.csv` (544,763 real IMU rows, independently verified physically consistent - see `AI_RECOVERY_EXECUTION_PLAN.md` §10/§11), not synthetic data, not a cherry-picked window, not a 1-epoch smoke test. All numbers below come from `runs/trajectory_eval_imu_norm.json` and `runs/trajectory_eval_imu_ssl_norm.json`, produced by `ai_backend/evaluate_trajectory.py`.

## Why this evaluation methodology, not just windowed MSE

Every earlier result in this project (`STATUS.md`, `analysis.md`, the `ml_pipeline.py` figures) reports **normalized, per-window MSE loss** - a number like "0.72" that is honest but hard to interpret physically: it doesn't say *how many metres off* the model ends up after flying for a while. This folder's figures instead **chain** the model's per-window predictions into a running position estimate, exactly as it would behave in real deployment - errors compound over time the way they actually would. This is a stricter, more meaningful test than isolated windowed MSE, and it's what makes the numbers below directly comparable to "how far off will this actually be."

---

## `trajectory_3d.png` - 3D Trajectory Reconstruction

**What it shows**: three paths in 3D space over the 12.5 s segment - Ground Truth (black, solid), the trained LSTM's chained reconstruction (magenta, solid), and the classical EKF's reconstruction (slate gray, dotted). A black dot marks the shared starting point.

**What it means**: the ground-truth path is a real, fairly tight maneuvering loop (visible as the small black spiral/loop shape) - not a straight line, so this is a genuine test of tracking real 3D motion, not a trivial case. The LSTM's magenta path stays in the same general region and volume as the ground truth. The EKF's gray path visibly separates early and drifts into a completely different region of space, ending up dozens of metres away.

**Why the EKF drifts so much**: the EKF here only receives a (simulated) barometer correction, which constrains altitude - nothing constrains horizontal (X/Y) drift. Since this segment's dominant error is horizontal, the EKF's correction barely helps. This is expected, physically correct behavior for a single-sensor EKF, not an implementation bug (see `error_over_time.png` below, and `ai_backend/models/ANALYSIS.md` for the EKF's design).

**What it led to**: this is the first time in the project a model's output was evaluated as an actual reconstructed path rather than an abstract loss number - it's what made the LSTM's real-world usefulness visible and concrete, informing the decision to build the full `evaluate_trajectory.py` + visualization pipeline rather than continuing to reason only from MSE values.

## `trajectory_xy.png` - Top-Down XY Path View

**What it shows**: the same three paths projected onto the horizontal (X/Y) plane - the clearest single view of *how far off course* each method ends up, since altitude differences (which are smaller here) don't visually compress the horizontal detail.

**What it means**: this is the most visually dramatic figure in the set. The ground-truth path is a real spiral maneuver near the origin. The LSTM's reconstruction stays within roughly the same 20-30 m neighborhood, tracking the general shape reasonably (not perfectly - dead reckoning always accumulates some error). The EKF's path visibly launches away from the origin and ends up roughly 90 m north-east of where the drone actually was - nowhere near the true spiral. This single image is the clearest evidence in the project that a horizontally-uncorrected classical filter is not viable alone for this task, and that the learned model, despite an unremarkable-looking training loss, has learned real, useful horizontal-motion structure that pure physics-based integration cannot recover on its own.

## `trajectory_xz.png` - Elevation XZ Path View

**What it shows**: the same three paths projected onto the vertical (X/Z) plane - isolates altitude behavior from horizontal drift, which is the point of splitting this out from the 3D view rather than relying on it alone (the two error sources have different physical origins: horizontal drift comes from uncorrected accelerometer double-integration in both the LSTM and EKF cases, vertical behavior is additionally shaped by the EKF's barometer correction).

**What it means**: altitude stays much more bounded across all three methods (roughly 10-20 m) than horizontal position does - consistent with this being a real flight where the drone mostly held a level cruising altitude rather than climbing/descending aggressively. The EKF's altitude tracking is visibly closer to ground truth here than its horizontal tracking is in `trajectory_xy.png` - direct visual confirmation that the barometer correction is doing its one job (bounding vertical drift) even while horizontal drift goes uncorrected.

## `error_over_time.png` - Position Error Growth over Time

**What it shows**: the single most information-dense figure in the set - Euclidean position error (metres) vs. time (seconds) for all four methods: Pure Inertial Integration (muted coral, dotted), Classical EKF (slate gray, dotted), Supervised LSTM (magenta, solid), SSL-Pretrained LSTM (cyan, solid).

**What it means, precisely**:

| Method | Final error | Mean error | Max error |
|---|---|---|---|
| Supervised LSTM | **26.3 m** | **16.5 m** | 36.9 m |
| SSL-Pretrained LSTM | 29.5 m | 19.0 m | 42.2 m |
| Classical EKF | 132.5 m | 66.1 m | 132.5 m |
| Pure Inertial Integration | 143.0 m | 71.4 m | 143.0 m |

The two baselines grow **almost linearly** with time and reach 130-140 m by 12.5 s - the textbook signature of unconstrained (or barely-constrained) double-integration drift compounding. Both LSTM variants stay **bounded**, oscillating in a 0-40 m band rather than growing without limit. This is the clearest, most quantitative evidence in the whole project that the learned dead-reckoning approach is doing something classical inertial navigation alone fundamentally cannot: constraining drift using learned structure in the sensor data, not just integrating it.

**The SSL finding, stated plainly**: the SSL-pretrained line (cyan) tracks slightly *worse* than the supervised line (magenta) throughout - consistent with, and a direct real-world consequence of, the training-loss finding in `train_ssl_finetune.png` below (SSL fine-tuning's best val loss, 0.810, was worse than the from-scratch supervised baseline's 0.719). This is reported as a genuine negative ablation result, not adjusted or hidden - masked-reconstruction pretraining, as currently configured (30% mask ratio, LSTM encoder, `imu`-only data), did not help this downstream task, and the trajectory-level evaluation independently confirms the windowed-MSE finding rather than contradicting it. That agreement between two very different evaluation methods (normalized per-window loss vs. real chained-trajectory error) is itself a useful validity check - if they'd disagreed, it would suggest a bug in one of the two measurement pipelines.

**What this figure is built to eventually become**: per `AI_RECOVERY_EXECUTION_PLAN.md` §12 Goal 4 Category B, once a Transformer SSL arm exists, it becomes the fourth line here, replacing or joining the current LSTM-vs-LSTM comparison.

---

## Training curves - `train_supervised.png`, `train_ssl_pretrain.png`, `train_ssl_finetune.png`

These three answer *how* each model got to the state evaluated above, and are the direct evidence behind every number quoted in the trajectory figures.

### `train_supervised.png` - Supervised LSTM Training (imu)

**What it shows**: train loss (black, solid) falling steadily from 0.69 to 0.23 over 6 epochs; validation loss (slate gray, dotted) staying essentially flat around 0.72-0.77 the whole time, never improving past epoch 1 (magenta marker + "best (epoch 1)" annotation). Early stopping triggered at epoch 6 (5 epochs with no improvement past epoch 1, the configured patience).

**What it means**: this is a **textbook overfitting curve** - the model keeps finding ways to reduce training error while gaining nothing on data it hasn't seen, meaning epochs 2 onward are pure memorization, not learning. This confirms (independently, with early stopping now actually implemented and enforced, rather than just observed after the fact) a pattern that first appeared in `AI_RECOVERY_EXECUTION_PLAN.md` §6.2 and recurred through §10: **this architecture (LSTM hidden=64, 2 layers) on this data (`imu_data.csv`, 14 features) converges to its ceiling almost immediately.** More epochs, without a change to the model, data, or objective, does not help - which is exactly why the SSL pretraining and (planned) Transformer/feature-extension work exist as separate tracks rather than "just train longer."

**What it made us do**: this is the concrete, repeated evidence that motivated building the SSL pretraining track in the first place (Goal 1 item 2) - if a plain supervised LSTM plateaus this fast, the natural next question is whether a better-initialized encoder (via self-supervised pretraining on the much larger unlabeled corpus) can push past that ceiling. The answer, per the next two figures, turned out to be "not with this configuration" - itself a valuable, honestly-reported finding.

### `train_ssl_pretrain.png` - SSL Pretraining (Masked Reconstruction)

**What it shows**: the *pretext task's own* training curve - reconstructing randomly-masked timesteps from the unmasked rest of the window. Train loss falls smoothly from 0.184 to 0.137; validation loss falls in step alongside it, from 0.181 to a best of 0.157 at epoch 13, essentially tracking train loss the whole way (no overfitting gap opens up) before early stopping at epoch 15.

**What it means**: as a pretext task in isolation, this **worked correctly** - the encoder is learning genuinely generalizable structure about how the 14 IMU-derived features co-vary and evolve over a 10-step window (not just memorizing training examples), since validation loss tracks training loss closely throughout. This is the expected, healthy shape for a masked-reconstruction objective, and it is the first time in the project a training curve shows *no* overfitting at all - worth noting precisely because the very next figure shows what happens once this well-trained encoder is repurposed for a different task.

### `train_ssl_finetune.png` - SSL-Pretrained LSTM Fine-Tuning (imu)

**What it shows**: the encoder from the previous figure, transferred into a fresh `DeadReckoningLSTM` and fine-tuned on the actual Δposition regression task. Train loss falls from 0.68 to 0.25 over 6 epochs (a similar shape to the from-scratch run); validation loss again never improves past epoch 1 (0.810), then rises.

**What it means, and why it matters**: **epoch 1's val loss here (0.810) is worse than epoch 1's val loss for the from-scratch supervised model (0.719)** in `train_supervised.png`. Pretraining a good encoder on one objective (reconstructing masked IMU values) did not translate into a better starting point for a different objective (predicting position deltas) - the two tasks apparently want different internal representations more than they share one. This is a genuine, informative negative result for the thesis's self-supervised-learning claim: it demonstrates that *just adding SSL pretraining* is not automatically beneficial, and motivates a concrete, specific next step rather than a vague one - trying a contrastive objective (which more directly encourages representations useful for distinguishing/predicting motion, per the SimCLR-style ablation already planned in `AI_RECOVERY_EXECUTION_PLAN.md` §12 Goal 1 item 2), a different mask ratio, or pretraining on a larger/more diverse corpus (once real recorded flights exist via the Godot `FlightRecorder`, §Goal 1 item 3).

### `comparison_val_loss.png` - Best Validation Loss by Model

**What it shows**: a simple, direct bar comparison - Supervised LSTM at 0.719 (magenta) vs. SSL-Pretrained LSTM at 0.810 (cyan) - the two numbers from the annotations above, side by side.

**What it means**: this is the figure to cite for the headline claim "does self-supervised pretraining help here" - and, as designed, the honest answer visible at a glance is "no, not yet, not with this configuration." It's deliberately a two-bar comparison for now; once the Transformer SSL arm and the classical-baseline windowed-equivalent exist (Goal 2's full ablation matrix), this becomes a wider bar chart with the same visual language.

---

## Summary: the narrative arc of this folder

1. Early stopping was implemented for the first time (`AI_RECOVERY_EXECUTION_PLAN.md` §17.1) so training could actually stop where it should, rather than reporting the last epoch of a fixed, arbitrary epoch count.
2. Running the from-scratch supervised model *with* early stopping (`train_supervised.png`) confirmed, this time under proper methodology, that this architecture/data combination has a very low ceiling - it stops improving almost immediately.
3. That finding motivated running the SSL pretraining track for real (not a 1-epoch smoke test) - `train_ssl_pretrain.png` shows the pretext task itself works well.
4. But fine-tuning from that pretrained encoder (`train_ssl_finetune.png`, `comparison_val_loss.png`) shows it does **not** help the actual downstream task - an honest negative result.
5. To understand what any of these normalized-loss numbers mean in practice, `evaluate_trajectory.py` was built to reconstruct a real trajectory by chaining predictions, and compare against two classical, non-learned baselines built from the *same* real data.
6. The result (`trajectory_3d.png`, `trajectory_xy.png`, `trajectory_xz.png`, `error_over_time.png`) is the strongest evidence yet in the project that the learned approach, even in its current, admittedly-plateaued form, meaningfully outperforms classical alternatives on the task the thesis is actually about - recovering a usable position estimate when GPS is gone.
