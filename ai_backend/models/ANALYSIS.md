# `ai_backend/models/` - Methods, Code Walkthrough, and Design Rationale

**Purpose of this document**: thesis-ready explanation of every model architecture in this folder - what it is, how it works, why it was built this way, what changed and why, and where each one sits in the project's fail-safe hierarchy (`AI_RECOVERY_EXECUTION_PLAN.md` §12 Goal 3: RL Policy → SSL/LSTM → Classic EKF → Kinematic Breadcrumb, "degrade toward simplicity").

---

## `dead_reckoning_model.py` - `DeadReckoningLSTM` (the core learned model)

**What it is**: a 2-layer LSTM, hidden size 64, taking a 10-timestep window of 14 sensor-derived features and predicting a single 3-vector - the position delta (Δlat/Δlon/Δalt, or Δx/Δy/Δz depending on data source) over that window.

```python
self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, output_size))
```
Only the LSTM's final timestep hidden state (`out[:, -1, :]`) feeds the regression head - the model reads the whole 10-step window but summarizes it into one hidden vector before predicting, standard for sequence-to-one regression.

**Why an LSTM, and why this small**: the input is a short (10-step, ~40ms-scale) window of correlated inertial signals - exactly the regime recurrent architectures are designed for (order-sensitive, modest sequence length, no need for the long-range attention a Transformer buys you). `hidden_size=64` was not tuned aggressively; it is a reasonable, unremarkable starting size, and `AI_RECOVERY_EXECUTION_PLAN.md` §17 documents that this architecture, on the currently-available data, converges (plateaus on validation loss) within one epoch regardless of size - meaning the current bottleneck is data volume/diversity, not model capacity, which is precisely why the SSL and Transformer-ablation tracks exist rather than "just make the LSTM bigger."

**Why 14 input features, and why this number changed**: `INPUT_SIZE = 14` must match `dataset_parser.FEATURE_COLS` exactly - this constant used to be duplicated across 4+ files (the original audit's top documentation-drift finding); it is now a single source of truth imported everywhere (`ssl_pretext.py` imports it directly rather than redefining it, see below).

**Production checkpoint update (post Track A, following the `cpu_5seeds` ablation extension)**: `dr_lstm_imu_norm.pth` is no longer the original from-scratch training run. `runs/ablation/cpu_5seeds/ANALYSIS.md`'s training-dynamics finding showed that ablation seed 4's LSTM never triggered early stopping across its full 30-epoch budget (`history_imu_norm_ablation_seed4.json`: `best_epoch=30`) and already had, independently, both the best windowed validation loss (0.6846) and the best chained-trajectory error (24.78 m) of all 5 seeds in that run -- two agreeing metrics, not a single lucky chained-error roll. `ai_backend/continue_training_best_lstm.py` resumed exactly that checkpoint (same deterministic data/scaler pipeline, so numerically consistent with the weights it already had) and trained it further to check whether it had genuinely converged or was only cut off by the epoch budget.

**The honest result: it had already converged.** Continued training immediately got worse (val loss rose from 0.6846 to 0.7150 at the very next epoch and never recovered, early-stopping after 5 non-improving epochs, `runs/continue_training_best_lstm.log`) -- epoch 30 was, in fact, at or extremely close to this run's true optimum, not artificially truncated. This is a case where the natural hypothesis ("still improving at the epoch ceiling, so give it more room") was directly tested and did not hold -- recorded here rather than quietly dropped, the same honesty standard this document applies to every other finding.

That said, seed 4's already-converged checkpoint still legitimately beat the prior production model on a fresh, apples-to-apples chained-trajectory evaluation (24.78 m vs. 26.34 m, both measured in the same run of `continue_training_best_lstm.py` against the same held-out segment) -- a real improvement, just not one this specific continuation step produced. It was promoted: the outgoing checkpoint was preserved first under tag `imu_norm_pre_seed4_continuation` (`dr_lstm_imu_norm_pre_seed4_continuation.pth` + its scalers), not deleted, before `dr_lstm_imu_norm.pth`/`scaler_X_imu_norm.pkl`/`scaler_y_imu_norm.pkl`/`history_imu_norm.json` were overwritten with seed 4's weights. This is a deliberate, one-off exception to how the production checkpoint was originally produced, made because a specific later seed happened to train better under the exact same procedure -- not a change to Track A's formal 5-seed ablation comparison, whose own conclusions (Transformer statistically the best architecture, `runs/ablation/cpu_5seeds/ANALYSIS.md`) are unaffected and stand as reported. Anything in this project loading `dr_lstm_imu_norm.pth` by that filename (`recovery_orchestrator.py`, `live_mission_demo.py`, `evaluate_trajectory.py`'s default tag) now gets this improved checkpoint automatically, with no code changes needed anywhere else.

## `dead_reckoning_model_uncertainty.py` - `DeadReckoningLSTMUncertainty` (Track A, Phase A2)

**What it is**: the same LSTM backbone as `DeadReckoningLSTM` (2-layer, hidden size 64), with the
regression head widened to output 6 values instead of 3, split into a mean `mu` and a
log-variance `log_var`, each shape `(batch, 3)`. `log_var` is clamped to `[-10, 10]` before use,
so `exp(log_var)` stays numerically sane while the head is still poorly initialized early in
training. Trained with `gaussian_nll_loss()`, the standard heteroscedastic-regression objective
(Kendall & Gal 2017): `0.5 * (log_var + (target - mu)^2 / exp(log_var))`, averaged over the batch.
This is a new model class, not a modification of `DeadReckoningLSTM` itself, since the live
`recovery_orchestrator.py` loads the production model by that exact class shape and this arm is
additive and offline-only.

No uncertainty-quantification code (MC-dropout, Bayesian layers, Gaussian NLL, or any
heteroscedastic-regression pattern) exists anywhere in the user's MSc coursework, checked
directly across all Deep Learning and Computer Vision course material, so this was designed
fresh rather than ported. The classical `DeadReckoningEKF` already exposes exactly this kind of
uncertainty for free, via its propagated covariance (`position_uncertainty`, see below); this
model is the learned counterpart, and the two are compared directly.

**A real finding from training, not glossed over**: early stopping tracks validation NLL, and
the best NLL landed at epoch 1 (`train_uncertainty.py`, `history_uncertainty_imu_norm.json`).
But validation MSE on `mu` alone kept *improving* for several more epochs after that (0.7052 at
epoch 1, down to 0.6839 by epoch 6) even as NLL got worse (-0.0875 at epoch 1, rising to 0.94 by
epoch 6). This is not a bug, it's the expected signature of joint mean/variance learning: NLL and
plain MSE optimize for different things. As training continues, the model's `mu` keeps getting
slightly more accurate, but its `log_var` shrinks faster than warranted by that accuracy, so it
becomes increasingly overconfident, and NLL (which penalizes overconfidence directly) correctly
flags that as *worse*, even though point accuracy alone looks better. Restoring the epoch-1
weights is the right call for a model whose whole purpose is calibrated uncertainty, not just the
lowest possible position error, but it does mean this arm's chained-trajectory accuracy is
slightly behind the plain LSTM's: 29.44 m final error / 18.64 m mean (`error_lstm_uncertainty` in
`runs/trajectory_eval_imu_norm.json`) versus the plain LSTM's 26.34 m / 16.53 m on the same
held-out segment. That gap is attributable to using less-trained (epoch-1) weights, not to the
architecture change itself.

**Does the learned uncertainty actually track real error?** `evaluate_trajectory.py`'s
`uncertainty_calibration()` answers this directly: it takes the model's predicted per-axis std,
reduces it to a single per-step magnitude the same way `euclidean_error()` reduces position error,
and reports the Pearson correlation against the actual chained-trajectory error on the same
held-out 12.5s segment, plus a 5-bin table (sorted by predicted uncertainty; the two columns
should track each other bin-for-bin in a well-calibrated model). The result: **r = 0.203**, weakly
positive, not zero. The binned table shows why it's weak: mean actual error across the 5 bins is
15.14, 20.10, 18.12, 17.92, 21.93 m, i.e. the general direction (lowest-confidence bin has the
lowest error, highest-confidence bin has the highest) holds at the extremes but the middle bins
are noisy, not a clean monotonic staircase.

Run the same calibration check on the EKF's native covariance over the identical segment and the
comparison is stark: **r = 0.999**, with a cleanly monotonic bin table (13.17 -> 39.51 -> 66.04 ->
92.66 -> 119.20 m). This is not evidence the EKF is simply "better at uncertainty" in a deep
sense, the comparison is between two fundamentally different kinds of uncertainty. The EKF's
covariance is a deterministic function of its known process model and fixed barometric-correction
schedule (`baro_every=24` steps in `evaluate_trajectory.py::reconstruct_filter`); it grows and
resets on a predictable clock that tracks the position error's own growth-and-reset pattern almost
by construction, independent of what the IMU data actually contains that step. The LSTM head, by
contrast, has to infer its confidence purely from the 10-step IMU window's content, a genuinely
harder, data-driven inference problem, one working against the single-continuous-flight,
low-diversity training data already flagged as a risk for this whole track (see the roadmap's
data-sufficiency note). A weak-but-real r=0.203 under those conditions is a plausible, honest
outcome: the head learned *something* about its own uncertainty, not nothing, but nowhere near a
usable confidence signal yet. Concrete next steps this points to, should this arm be revisited:
more diverse training flights (the same fix the data-sufficiency note already recommends for the
Transformer arm), or separating the mean and variance training into two phases (train `mu` to
convergence first, then freeze it and fit `log_var` alone) to avoid the overconfidence-race
dynamic documented above.

**Verification**: `python ai_backend/models/dead_reckoning_model_uncertainty.py` runs a smoke
test asserting `gaussian_nll_loss` correctly penalizes an overconfident-but-wrong prediction more
than a calibrated one on a fixed synthetic error, before the model was ever trusted on real data.

**A further honest finding, from retraining this model three times this session** (once
originally, once at batch=256/GPU during the infrastructure detour above, once more restoring
batch=64 afterward): chained-trajectory error came out **29.44 m, 31.69 m, and 38.54 m**
respectively, and calibration **r = 0.203, 0.212, and 0.119**, all restoring epoch-1 weights each
time, all on the same data and same architecture. This model's chained-trajectory quality is
noticeably more variable run-to-run than the plain LSTM's or Transformer's, plausibly because
always restoring epoch-1 weights means the saved model is inherently less converged than a model
that trained for longer, and less-converged weights are more sensitive to random initialization
and shuffling. The weak-positive calibration finding (r in the 0.12-0.21 range) holds directionally
across all three runs, not zero, but the exact chained-error number should not be read as a fixed,
precisely-reproducible property of this architecture the way the Transformer's ~12-18 m range or
the LSTM's ~26 m appear to be. This is exactly the kind of variance Phase A5's multi-seed
ablation matrix exists to quantify honestly rather than let a single lucky or unlucky run stand
in for the whole picture.

## `ssl_pretext.py` - Self-Supervised Pretraining (`MaskedLSTMAutoencoder`)

**What it is, structurally**: an encoder (`nn.LSTM(14, 64, 2, dropout=0.2)`) - **architecturally byte-for-byte identical** to `DeadReckoningLSTM.lstm` - paired with a lightweight per-timestep linear decoder (`nn.Linear(64, 14)`) that reconstructs the full 14-feature vector at every timestep. The identical-architecture choice is deliberate and load-bearing: it's what makes `transfer_encoder_weights()` a plain `load_state_dict()` copy with no key remapping (`ssl_pretext.py:91-96`) - decoupling the encoder architecture from the supervised model's architecture would have required a translation layer for no benefit.

**The pretext task**: `random_mask()` zeroes a random 30% (`mask_ratio`, a documented, sweepable hyperparameter) of *whole timesteps* per sample - not individual features - and the model is trained to reconstruct exactly those timesteps from the surrounding unmasked context (`masked_reconstruction_loss()` computes MSE only at masked positions, standard MAE/BERT-style "impute what's missing" objective). Whole-timestep masking (rather than per-feature masking) forces the encoder to use *temporal* context (what came before/after) rather than *cross-feature* correlation at the same instant as an easy shortcut - a deliberate choice to make the pretext task actually exercise the LSTM's sequential modeling, not just feature imputation.

**Why self-supervised pretraining at all**: this is the thesis's core methodological claim (per `AI_RECOVERY_EXECUTION_PLAN.md`'s title - "...using Self-Supervised Learning"). The motivating hypothesis: the unlabeled IMU corpus is large and label generation (Δposition, which itself is only as good as the underlying data) is comparatively expensive/noisy, so an encoder pretrained on the abundant unlabeled reconstruction task, then fine-tuned on the smaller supervised objective, should in principle generalize better than training from scratch on the supervised task alone.

**Two-stage training, sharing one early-stopping implementation**: `pretrain_ssl()` (stage 1, masked reconstruction) and `finetune_from_ssl()` (stage 2, supervised Δposition regression after `transfer_encoder_weights()`) both delegate to a single `_run_early_stopped()` helper (`ssl_pretext.py:99-141`) - deep-copies the best `state_dict` whenever validation loss improves, restores it at the end, and stops when `patience` epochs pass with no improvement (subject to `min_epochs`). Factoring this out once, rather than duplicating the loop per stage (or per file, as `train.py` originally did before this consolidation), is what makes the two stages' results directly comparable - identical stopping discipline, not two subtly different training loops that happen to produce two numbers.

**What the honest result was** (see `runs/figures_single_drone/ANALYSIS.md` for the full figure-level narrative): stage 1 (pretraining) trained cleanly, with validation loss tracking training loss the whole way (best val 0.157, epoch 13) - no overfitting, a well-behaved pretext task. Stage 2 (fine-tuning) reached best validation loss 0.810 at epoch 1, **worse** than the from-scratch supervised baseline's 0.719. **This is reported as a genuine negative result, not adjusted or hidden**, per this project's established norm (`AI_RECOVERY_EXECUTION_PLAN.md` §7/§10: "act as if there is a problem, not a solution; don't justify wrong data" - the same honesty standard applied here to a disappointing-but-real ablation outcome). The concrete, motivated next steps this finding points to: try a contrastive (SimCLR-style) objective instead of/alongside masked reconstruction (2025 IMU-SSL literature reports these as complementary, not interchangeable - see plan §Research Grounding), try a different mask ratio via the already-built sweep axis, or pretrain on a larger/more diverse corpus once the Godot `FlightRecorder` accumulates real recorded flights.

**CLI / persisted artifacts**: `python ai_backend/models/ssl_pretext.py --sources imu --ssl-epochs 15 --finetune-epochs 30` saves `ssl_pretrain_{tag}.pth`, `dr_lstm_{tag}.pth`, both scalers, and two `history_*.json` files (consumed by `visualize_training.py`, see `ai_backend/ANALYSIS.md`).

## `dr_transformer.py` / `ssl_pretext_transformer.py` - `DeadReckoningTransformer` (Track A, Phase A3)

**What it is**: hyperparameters (`d_model=64`, `nhead=4`, 4 layers, `dropout=0.1`, pre-norm) are
taken directly from the confirmed working `TinyViT` reference in the DL/CV coursework
(`D:\ΠΜΣ\Υπολογιστική όραση\Εργασία\project-2\ex3_cifar.py`), with `d_model=64` deliberately
matched to the LSTM's `hidden_size=64` for a fair comparison. A 10-timestep IMU window has no 2-D
patch structure, so `PatchEmbed`'s `Conv2d` is replaced by a per-timestep `Linear` projection
(`TransformerTrunk.embed`), the temporal analogue. A learnable `cls_token` (same trunc-normal
init as TinyViT) is prepended, a learnable positional embedding added, and the cls token's final
representation after 4 encoder layers feeds the same `Linear(64,32)+ReLU+Linear(32,3)` head the
LSTM uses. Built with a hand-written encoder layer (`_EncoderLayerWithAttn`) rather than
`nn.TransformerEncoderLayer`, specifically so attention weights are retrievable for the
interpretability analysis below without depending on private internals of the stock module.

**Real chained-trajectory result, not just windowed loss**: from-scratch supervised training
(`train_transformer.py`, with time-series augmentation, see below) converged to best validation
MSE **0.3545** at epoch 8 of 13 before early stopping (`history_transformer_imu_norm.json`),
roughly half the plain LSTM's windowed val loss (~0.72-0.75). Critically, this advantage
**survives the honest chained-trajectory check**, not just the windowed metric: on the same
held-out 12.5s real segment `evaluate_trajectory.py` already uses for every other model, the
Transformer reaches **12.46 m final error / 9.75 m mean error**, clearly beating the LSTM's
26.34 m / 16.53 m (`runs/trajectory_eval_imu_norm.json`), and, on this run, beating every other
model in the comparison including GBT (see the classical-ML section below). This matters because
this project has already seen the opposite pattern (a metric that looks good in isolation but
fails once errors compound, exactly what happened to the EKF's "physically correct" but
132 m-drifting chained result above); here the windowed-MSE improvement genuinely holds up under
compounding. (This model was retrained once, at the same batch_size=64 configuration, after an
infrastructure detour into GPU training is documented separately below; the retrained run's
numbers, quoted here, are what the currently-saved `dr_transformer_imu_norm.pth` actually
produces, and came out modestly better than the first run's 18.33 m / 11.61 m, consistent with
ordinary run-to-run variance rather than any methodology change.)

**Data-diversity mitigation, and whether it mattered**: given the data-sufficiency finding
(`imu_data.csv` is one continuous 37.8-minute flight, not diverse sessions), training used both
dropout (0.1) and `augment_batch()` (Gaussian jitter, magnitude scaling, mild probabilistic time
warp, the time-series analogues of the CV coursework's `RandomHorizontalFlip`/`RandomRotation`/
`ColorJitter`). Whether augmentation specifically, versus the architecture alone, drove the
result is not yet isolated (an `--no-augment` run for direct comparison is a natural follow-up,
not yet run as of this writing), flagged honestly rather than claimed.

**Attention interpretability, run on real held-out data, not just described**: `evaluate_trajectory
.reconstruct_transformer(..., collect_attention=True)` extracts every layer's attention weights
over 500 real windows from the held-out segment; averaging the cls token's attention over heads
and windows produces a genuinely informative, non-obvious finding (`runs/transformer_attention_
summary.json`): **only layer 0 shows real positional selectivity.** Its attention over the 10
input timesteps is U-shaped, highest at the window's earliest step (0.249) and most recent step
(0.115), lowest in the middle (0.003 at timestep 4), i.e. the model's first layer pays the most
attention to the oldest and newest information in the window and comparatively little to the
middle. **Layers 1 through 3, by contrast, are essentially uniform** (~0.082-0.091 at every
timestep, indistinguishable from 1/11 within rounding), showing no measurable temporal
selectivity at all. This is an honest, non-obvious limitation worth stating plainly: the
Transformer's real advantage over the LSTM does not appear to come from deep, multi-layer
attention-based temporal reasoning, since three of its four layers show no learned selectivity.
It's more consistent with the model doing most of its real work through the feedforward
sublayers and residual accumulation rather than attention pattern-matching, an interpretation
reinforced by the classical-ML finding below: a non-sequential tree ensemble (GBT, which cannot
model temporal order at all) reaches comparably strong chained accuracy to this Transformer,
suggesting the task's real difficulty may lie more in nonlinear feature combination within a
window than in genuine multi-step sequence reasoning across it. (Re-extracted on the currently-
saved model after the retrain above; the same U-shaped layer-0 / uniform-elsewhere pattern
reproduced essentially unchanged, reinforcing that this is a property of the architecture and
task rather than a one-off artifact of a single trained instance.)

**Verification**: `python ai_backend/models/dr_transformer.py` runs a smoke test asserting output
shape, attention-weight shape and softmax-normalization (rows sum to 1), and that `augment_batch`
actually changes its input, before any of this was trusted on real data.

**SSL variant**: `ssl_pretext_transformer.py` mirrors `ssl_pretext.py`'s two-stage structure
exactly, a `TransformerTrunk`-based `MaskedTransformerAutoencoder` pretrained via masked-timestep
reconstruction, then its trunk transferred into a fresh `DeadReckoningTransformer` for supervised
fine-tuning, via a plain `trunk.load_state_dict()` (the `TransformerTrunk` class is factored out
specifically so this transfer needs no key remapping, the same relationship `MaskedLSTMAutoencoder
.encoder` has to `DeadReckoningLSTM.lstm`). Framed as a hypothesis test given the LSTM's own SSL
arm was an honest negative result, not a promised win.

**Result: another honest negative, consistent across both architectures now.** Pretraining
converged cleanly (best val 0.0225 at epoch 14 of 15, no overfitting, the same well-behaved
pattern as the LSTM's own pretraining stage), but fine-tuning from it reached best validation
0.4642 (epoch 6), worse than that same batch=256/GPU run's from-scratch Transformer at 0.4378.
Chained-trajectory error: 17.91 m final / 11.28 m mean, in the same general range as the
from-scratch Transformer's validated batch=64 result (12.46 m / 9.75 m) without being a precise
apples-to-apples comparison, given the batch-size difference noted below. SSL pretraining not transferring
usefully to the downstream regression task is now a finding that holds for both the LSTM and the
Transformer, strengthening the case that the issue is with the pretext task or pretraining data
diversity (see the roadmap's data-sufficiency note), not something specific to either
architecture. **One caveat kept explicit rather than glossed over**: this arm was trained at
`batch_size=256` on GPU (see the Infrastructure note above), not the `batch_size=64` standard
every other model in this comparison uses, because completing it was what surfaced the
`random_mask()` performance bug in the first place, and a second full batch=64 CPU run
(pretraining alone would run to several thousand seconds) was not repeated given the effort
already invested and the chained result already looking unremarkable-in-a-good-way (in the same
range as the trustworthy from-scratch arm, not an outlier). Its chained number is reported as
informative, not as fully apples-to-apples with the batch=64 rows in the table above.

## `classical_baselines.py` / `anomaly_gate.py` - Classical ML and Anomaly Detection (Track A, Phase A4)

**Why classical ML at all**: the ablation matrix, before this phase, only spanned classical-DSP
(EKF) and deep learning (LSTM), nothing represented plain supervised machine learning despite
that being one of the three things this extension work was explicitly scoped to cover
(deep learning / ML / RL). `RandomForestRegressor` and `MultiOutputRegressor(HistGradientBoosting
Regressor(...))` are trained on the identical scaled train/val split as every deep model, with
each 10x14 window flattened to a 140-column row (`classical_baselines.py::_flatten`).
`HistGradientBoostingRegressor` (sklearn's histogram-based GBM, the modern fast variant) was used
in place of plain `GradientBoostingRegressor` specifically because the latter's exact-split,
single-threaded algorithm is impractical at this data's ~436K-row scale, an explicit, explained
substitution, not a silent deviation from "GBT."

**A genuinely striking real result, on both the windowed and chained-trajectory metrics**:
`GridSearchCV` (on a 20K-row subsample, refit on the full training set) picked `max_depth=20,
n_estimators=100` for the forest and `max_depth=10, max_iter=100` for the boosted trees.
Validation MSE: RandomForest **0.6665**, GBT **0.3546**, both trained in a fraction of any deep
model's time (707.7 s and 48.3 s respectively, versus the LSTM's ~170 s or the Transformer's
~3300 s). On the same held-out chained-trajectory segment as every other model
(`runs/trajectory_eval_imu_norm.json`): **RandomForest 27.24 m final / 17.65 m mean** (essentially
tied with, marginally worse than, the LSTM's 26.34 m / 16.53 m, despite RF's better windowed MSE)
versus **GBT 18.06 m final / 11.66 m mean** (close to, though after the Transformer's retrain
(see above) now modestly behind, the Transformer's 12.46 m / 9.75 m, and clearly beating the
LSTM regardless). Two honest, distinct lessons sit side by side here: (1) GBT, a purely
feature-based, non-sequential model with zero notion of timestep order, reaches real-world
chained accuracy in the same league as the best sequence/attention model tried so far,
reinforcing the Transformer section's finding that this task's difficulty may lie more in
per-window nonlinear feature combination than deep temporal reasoning; (2) RF's windowed-MSE
improvement over the LSTM did **not** reliably translate to a chained-trajectory improvement, the
same "a metric that looks good in isolation can still fail once errors compound" caution this
project already learned from the EKF, now observed again in a different model family. Both
results are single-run (n=1); Phase A5's multi-seed matrix is what turns "GBT looks very
competitive" into a statistically supportable claim rather than a striking anecdote.

**Anomaly detection, ported pattern**: the ML course's Netflix-recommender notebook fits
`sklearn.ensemble.IsolationForest` (`contamination=0.01`) alongside an independent PCA-
reconstruction-distance threshold (99th percentile) on TF-IDF vectors, comparing two unsupervised
detectors rather than trusting one. `IsolationForestGate`/`PCAReconstructionGate` apply the same
dual-detector pattern to flattened, scaled IMU windows. Fit on the full training set and applied
to the held-out validation set (108,951 windows): IsolationForest flagged 1,027, PCA-
reconstruction flagged 1,414, **the two agree on only 108 of them (Jaccard index 0.046)**,
confirming they catch substantially different kinds of "unusual," exactly the reason to run both
rather than one (`runs/anomaly_gate_imu.json`). Applied specifically to the same 3000-sample
held-out segment used for chained-trajectory evaluation and correlated with the LSTM's actual
per-step error (`runs/anomaly_gate_heldout_correlation.json`): windows IsolationForest flagged
(53 of 2990) have a mean LSTM error of 19.91 m versus 16.47 m for unflagged windows (about 21%
higher); PCA-reconstruction-flagged windows (154 of 2990) show a smaller but still positive gap,
17.82 m versus 16.46 m (about 8% higher). Both gates carry real, if modest, signal about where the
LSTM is less trustworthy, with IsolationForest's signal noticeably stronger on this segment. As
scoped for this phase, this is offline analysis only, live wiring into `recovery_orchestrator.py`'s
layer selection remains an explicit stretch goal, gated on this kind of correlation evidence
existing first, which it now honestly does, if only modestly.

## `ekf_baseline.py` - `DeadReckoningEKF` (classical, zero-learning baseline)

**What it is**: a textbook 6-state (`[px,py,pz,vx,vy,vz]`) Kalman filter - strapdown inertial navigation with a piecewise-white-noise-acceleration process model (`predict()`) and a linear barometric-altitude measurement update (`update_baro()`). No neural network, no training - pure classical estimation theory (Bar-Shalom's *Estimation with Applications to Tracking and Navigation*, ch. 6, cited directly in the code comments for the process-noise discretization formula).

**Why it exists - two roles simultaneously, by design**:
1. **The non-ML floor of Goal 2's benchmark matrix** (Pure Inertial < EKF < Supervised LSTM < SSL-LSTM < ...). Without a real, correctly-implemented filter baseline, any claim that "the learned model beats classical methods" is unfalsifiable - this is what makes that comparison legitimate rather than a strawman.
2. **The tertiary layer of Goal 3's fail-safe hierarchy** (RL → SSL/LSTM → **EKF** → breadcrumb). Because it's built as a clean, reusable, importable `class` (not a one-off comparison script), the same object that generates benchmark numbers is deployable as an actual fallback estimator - no rewrite needed to go from "evaluation artifact" to "production fallback."

**Design details worth noting**:
- `predict()` takes `accel_body` + a `rotation_matrix` (world-from-body) and integrates specific-force acceleration exactly the way `drone_body.gd::_send_telemetry()` computes the simulated accelerometer reading in reverse (comment at `ekf_baseline.py:17-22` states this explicitly) - the same physical convention is used to generate the synthetic sensor data and to invert it, which is what makes the EKF's model internally consistent with the simulation rather than an approximation of a different physical setup.
- **Only altitude is corrected** (`update_baro`), by design - this project currently has no simulated GPS-independent horizontal correction source (no magnetometer-derived heading fusion, no visual odometry), so the EKF is honestly a "vertical-only-corrected" filter. This is exactly why `trajectory_xy.png` (see `runs/figures_single_drone/ANALYSIS.md`) shows the EKF drifting ~90m horizontally while `trajectory_xz.png` shows its altitude staying reasonably close to ground truth - the figure-level result is a direct, expected consequence of this design, not a bug.
- `position_uncertainty` exposes the filter's own covariance diagonal as a 1-sigma per-axis uncertainty - satisfies Goal 1 item 5's "expose confidence, not just a point estimate" requirement natively (a Kalman filter gets this for free; the LSTM/Transformer would need a second output head trained with Gaussian NLL loss to get the same thing, noted as still-pending future work).
- **Verified, not just implemented**: the module's `__main__` smoke test runs two filters side by side (with/without periodic baro correction) on synthetic level-flight data and asserts the baro-corrected filter's vertical uncertainty is lower - this passed (5.20 m → 0.26 m), which is the first concrete evidence in the project that the EKF's correction step behaves as Kalman-filter theory predicts, before it was ever trusted to appear in a benchmark figure.

## `rl_path_recovery.py` - `SwarmRecoveryEnv` / PPO (status: original prototype, not yet upgraded)

**What it is, currently**: a `gymnasium.Env` where a drone's 2D lat/lon position is the whole state (`observation_space` = 6 raw coordinate values: own position + nearest signal zone + nearest relay drone), the action is a continuous 2D steering vector, and the reward is simply negative distance to the nearer of the two targets (`reward = -min_dist * 100.0`, +100 bonus on reaching one). Trained with Stable-Baselines3 PPO for 10,000 timesteps.

**Why this file is flagged, not just described**: this is a **pre-existing, un-upgraded prototype** - it predates this session's audit and has not yet been touched as part of the SSL/EKF/breadcrumb work. Two things distinguish it clearly from the rest of this folder: (1) it consumes **raw simulated GPS coordinates** directly, not the reconstructed position (+ uncertainty) that `AI_RECOVERY_EXECUTION_PLAN.md` §12 Goal 3 specifies as the intended RL state (SSL/LSTM output feeding the policy, not ground-truth coordinates) - meaning it currently cannot function as the hierarchy's primary layer in any GPS-loss scenario, since it assumes exactly the signal that's missing; (2) the original audit already flagged it as **undertrained** (10K timesteps reaching ~28% success vs. a 100%-success rule-based baseline). It remains in the codebase, correctly described here as-is, as the known starting point for Phase 3 (`AI_RECOVERY_EXECUTION_PLAN.md` §Master Execution Roadmap Phase 3 / §12 priority order) rather than mischaracterized as already integrated into the fail-safe stack - that integration (state = reconstructed position + uncertainty, target = return-to-home point, scaled to 500K-1M timesteps) is still-pending work, sequenced explicitly after Goals 1/2/3's single-drone deliverables per the user's own prioritization ("lets focus on single drone now... once all those are done and verified then maybe we will do IoT things such as swarm").

---

## Infrastructure note: batch size, GPU training, and why the defaults stay small

Partway through Track A, an AMD GPU (Radeon RX 9070) was set up via `device_utils.py` and
`torch-directml`, following the working pattern in the user's own DL coursework
(`D:\ΠΜΣ\Βαθιά Μάθηση\Εργασίες εξαμήνου\2η Άσκηση\25118.py`). This produced a real, useful
finding worth keeping visible rather than quietly reverting: at `batch_size=64` (the value every
model in this project has used), GPU dispatch overhead makes DirectML slower than CPU (measured
1692 vs. 2068 rows/sec); GPU only wins once batch size grows past roughly 256 (3572 rows/sec),
reaching a clear 2x advantage by 4096.

A direct experiment followed naturally: does training at the larger, GPU-favorable batch size
also produce a *better* model, not just a faster training run? On windowed validation loss,
yes, decisively: the LSTM reached 0.6455 at batch=256/GPU over 30 epochs (still improving at the
last epoch) versus roughly 0.72-0.78 at batch=64 over a comparable budget. Retraining the
uncertainty model, the from-scratch Transformer, and completing the SSL-Transformer arm for the
first time all followed at batch=256/GPU on this basis.

**But the chained-trajectory metric, the one this whole project has repeatedly established as
the one that actually matters (see the EKF's "physically correct but 132m-drifting" result, and
the classical-ML section's RandomForest finding above), told a different story once actually
checked:**

| Model | Windowed val loss | Chained final error |
|---|---|---|
| LSTM, batch=64 (original) | ~0.72-0.75 | **26.34 m** |
| LSTM, batch=256/GPU | 0.6455 (better) | **37.18 m (41% worse)** |
| Transformer, batch=64 (original) | 0.3696 | **18.33 m** |
| Transformer, batch=256/GPU | 0.4378 (worse) | **29.60 m (61% worse)** |

A larger batch improved the isolated, windowed metric while making real, compounding-error
accuracy substantially worse, for both architectures. This is a plausible instance of a
well-documented deep learning phenomenon (large-batch SGD tending toward sharper, less
noise-robust minima that generalize worse), observed here concretely on a task where
generalization failure shows up specifically as errors that compound badly once chained, not
just a larger average per-window residual. It is also the starkest version yet, in this
project's history, of "an isolated metric looking better is not evidence of real improvement,"
worth remembering the next time a windowed number alone looks like good news.

**Resolution**: every model in this folder keeps `batch_size=64` as its default, the config with
actual chained-accuracy evidence behind it, not the one that merely trains faster or scores
better on the easier proxy metric. `batch_size` remains an explicit parameter on every
`train_model()`/`pretrain_ssl*()` function for anyone who wants to explore the tradeoff further
(the crossover point between "GPU is faster" and "GPU is worse for real accuracy" was not
mapped in detail, only the two endpoints tested here). The one durable, unqualified win from this
detour: `random_mask()` in both `ssl_pretext.py` and `ssl_pretext_transformer.py` had a genuine
performance bug (a per-sample Python loop measured at 156ms/call at batch=256 on DirectML,
disproportionately responsible for the SSL-Transformer's first pretraining pass taking ~7467s),
now fixed with a vectorized topk-based implementation (0.9ms/call, 173x faster), verified to
produce identical masking statistics, and kept regardless of which batch size or device ends up
used for future training.

**A further finding, from running the full Phase A5 ablation matrix**: the gap between CPU and
DirectML at `batch_size=64` is much larger in a real training loop than the raw-throughput
micro-benchmark above suggested. That benchmark measured 1692 vs. 2068 rows/sec, roughly a 1.2x
gap; the same LSTM training loop, run end to end on both devices, measured **~28s/epoch on CPU
versus ~249s/epoch on DirectML, a ~9x gap**. The difference is not epoch count (more epochs cost
proportionally more time on either device, it does not change the per-epoch ratio) -- it is batch
size, compounded by two things the isolated micro-benchmark did not capture. First, DirectML is a
translation layer on top of DirectX12, not a first-class backend like CUDA, so its fixed
per-operation dispatch overhead is higher, and at batch=64 there is not enough work per dispatched
op to amortize it. Second, the LSTM's recurrence makes this worse: each 10-step window is 10
sequential, dependent operations, so even a batch of 64 samples turns into many small sequential
kernel dispatches rather than one large parallel one, the pattern that punishes per-op overhead
hardest. Since batch=64 is not changing (it is the config with actual chained-accuracy evidence,
per the Resolution above), `device_utils.py::get_device()` was changed so DirectML is opt-in
(`prefer_directml=True`) rather than the default -- CPU is simply the correct default for this
project's fixed training regime, not a fallback.

---

## How the four pieces compose (current state, honestly)

| Layer | File | Status |
|---|---|---|
| 1. RL Policy | `rl_path_recovery.py` | Prototype only; not yet wired to reconstructed position; undertrained |
| 2. SSL/LSTM | `ssl_pretext.py` + `dead_reckoning_model.py` | Fully implemented, trained, evaluated end-to-end; SSL pretraining currently underperforms from-scratch supervised (honest negative result) |
| 3. Classic EKF | `ekf_baseline.py` | Fully implemented, verified via smoke test, integrated into `evaluate_trajectory.py`'s benchmark |
| 4. Breadcrumb (emergency floor) | `simulation/scripts/breadcrumb_recovery.gd` | Implemented on the Godot side (outside this folder - by design, so it survives total Python/UDP failure), see `simulation/README.md` |

Layers 2 and 3 are the two that currently have real, evaluated, trustworthy numbers behind them (`runs/figures_single_drone/`). Layer 1 is the clearly-flagged next-step. Layer 4 is deliberately isolated from everything in this folder, on purpose.

---

## Testing this file's own proposed follow-ups directly

Five concrete ideas floated above (the Transformer's seed-to-seed reliability, the uncertainty head's weak calibration, the LSTM's early-stopping patience, whether augmentation is actually helping the Transformer) were tested directly rather than left as unexamined proposals, all reusing artifacts that already existed on disk or a single retraining run, no new flight data required.

**`ai_backend/evaluate_transformer_ensemble.py`** -- does ensembling the Transformer's 5 `cpu_5seeds` checkpoints mitigate its reliability problem (§ above, 71% coefficient of variation, the least consistent model in the ablation matrix)? Averages all 5 seeds' per-step delta prediction at each chained-trajectory step before chaining (not a post-hoc average of 5 independently-drifted trajectories). Result: **3.92 m final / 5.88 m mean error**, against a per-seed range of [3.22, 15.04] m and the ablation study's own reported random-single-seed expectation of 9.10 m. Roughly halves the error a randomly-deployed single seed would be expected to produce, using checkpoints that already existed -- a real, immediately usable mitigation, not a fix for the underlying variance.

**`ai_backend/train_uncertainty_two_phase.py`** -- tests this file's own uncertainty-calibration section's proposed fix directly: freeze `mu` after it converges on plain MSE (phase 1), then fit `log_var` alone on `gaussian_nll_loss` (phase 2). `DeadReckoningLSTMUncertainty.fc[2]` is one shared `nn.Linear(32, 6)` (mu = rows 0:3, log_var = rows 3:6), not two separate heads, so phase 2 freezes the trunk via `requires_grad=False` and zero-masks the mu rows' gradient every step rather than freezing a whole layer. **A real bug surfaced and got fixed while building this**: the first run showed mu's val MSE drift from 0.7080 to 1.0314 one epoch into phase 2 despite the masking -- traced to Adam's `weight_decay` re-adding a nonzero `weight_decay * param` term to the already-zeroed gradient *inside* its own `step()`, undoing the mask; fixed by setting `weight_decay=0` on phase 2's optimizer. Rerun confirmed the freeze held exactly (mu's val MSE identical, 0.7141, across every phase-2 epoch). **Result: another honest negative**, the same pattern as this file's SSL and gyro-reconciliation findings. Calibration barely moved (r=0.199 vs. the original 0.203, inside the 0.119-0.212 range already observed across 3 retrains) and chained error came out worse (38.49 m vs. 29.44 m, still inside that model's already-documented run-to-run range). `log_var` collapsed toward a narrow, still-overconfident range (0.03-0.07 scaled-units across all 5 calibration bins) rather than learning genuine input-dependent variation -- the same overconfidence-race dynamic this file already described, just relocated from epoch 1 into phase 2 instead of being cut short by it. Evaluated via `ai_backend/evaluate_uncertainty_twophase.py`.

**`train_transformer.py --shuffle-order`** (+ `ai_backend/evaluate_transformer_shuffled.py`) -- does the Transformer's real advantage over GBT (order-blind) actually depend on chronological order, or just on having a consistent, learnable structure at all? A single fixed (not per-window-random) permutation of the 10 timestep slots, applied identically to every training and validation window, trained fresh with otherwise-identical hyperparameters. Windowed val loss came out close to the real Transformer's (0.3622 vs. 0.3545). Chained-trajectory result, evaluated with the same fixed permutation applied to the held-out segment: **9.78 m final / 7.24 m mean**, nominally *better* than the real, order-preserving Transformer (12.46 m / 9.75 m), and still clearly ahead of GBT (18.06 m). Consistent with this file's attention-interpretability finding (only layer 0 shows real positional selectivity). **Caveat stated plainly, not glossed over**: this is one shuffled run against one order-preserving run, and the Transformer's own seed-to-seed variance (3.22-15.04 m from random init alone, nothing to do with shuffling) is large enough that a single run landing at 9.78 m is fully consistent with ordinary seed noise. A real answer needs the same multi-seed treatment already applied to every other arm in this study -- not yet run, named here as the honest next step rather than overclaimed from n=1.

**That confirmation was run, and the caveat was right.** `ai_backend/train_transformer_multiseed_confirmation.py` trained the same 5 seeds with `--shuffle-order`: 8.28 / 7.36 / 4.94 / 20.67 / 8.80 m final (seed 3 a clear outlier) -- mean **10.01 m, 95% CI [2.39, 17.64] m**, against the real Transformer's 9.10 m mean, CI [2.61, 15.60] m. Heavily overlapping intervals: statistically indistinguishable, neither better nor worse. The single-run "nominally better" result was ordinary seed noise, not a real effect.

**`ai_backend/train_lstm_patience_experiment.py`** -- does raising the LSTM's early-stopping patience from 5 to 10 close the gap seed 4's own training curve revealed (Part 4h above)? Retrained the same 5 `cpu_5seeds` seeds with `patience=10`, `set_global_seed(seed)` called identically so each seed's initial weights match. Aggregate result: **32.24 m mean final error**, *worse* than the original patience=5 run's 29.67 m -- no clean win. Seed-by-seed: one seed (1) improved substantially (29.48 m -> 26.69 m), two got worse despite similar or better windowed loss (another windowed-vs-chained divergence instance), two were roughly unchanged. **A more important, unplanned finding**: seed 4's rerun stopped at epoch 3 (val 0.7200), nothing like its own famous original (never early-stopped, smooth improvement to val 0.6846 by epoch 30) -- and patience cannot affect epochs 1-3's actual training numbers, only the stop decision, so this divergence cannot come from the patience change itself. Rerunning seed 4 alone in a fresh process (`ai_backend/train_lstm_seed4_patience10_isolated.py`, nothing else executed first) reproduced the sequential run's epoch 1/2/3 values bit-for-bit identical, ruling out same-process state leakage cleanly. The remaining explanation: `set_global_seed()` reliably reproduces results *within* one environment (confirmed directly here) but is not a guarantee of bit-identical CPU floating-point results *across* different sessions -- a documented PyTorch limitation (thread-count-dependent BLAS reduction order), not a bug in this codebase. The original "seed 4 never plateaus" finding was real when it happened, but this environment cannot reproduce that exact trajectory to test the patience hypothesis against it specifically -- a genuine, honestly-reported limit on how far seed-based reproducibility can be trusted across sessions.

**`train_transformer.py --no-augment`** -- does removing time-series augmentation change the Transformer's real result? A flag that already existed but had never actually been run. **A second real bug was caught before it caused damage**: `--no-augment` had no distinct tag suffix (unlike `--shuffle-order`), so running it as originally written would have silently overwritten `dr_transformer_imu_norm.pth`, the actual production checkpoint. Caught within seconds of launch (confirmed via file timestamps the production checkpoint was never touched), fixed by building a composite tag suffix from the CLI flags, relaunched safely. Windowed val loss without augmentation: 0.2212, far better than 0.3545 with it. Chained-trajectory result: **6.47 m final / 4.23 m mean** -- nearly half the with-augmentation Transformer's 12.46 m / 9.75 m, and the best single-run Transformer result anywhere in this project, better than every one of the 5 `cpu_5seeds` seeds individually. **The same caveat as the shuffle finding applies with equal force**: one run against one run, and the Transformer's 71% seed CV means this isn't distinguishable from luck without the same multi-seed treatment. Augmentation was included in the original design specifically to mitigate the data-diversity risk (never isolated until now) -- this is the first direct evidence it may be net-harmful for this architecture on this data, real and actionable, but not yet confirmed.

**Confirmed, and this time the exciting number did not survive.** The same 5-seed treatment: 9.51 / 10.32 / 10.50 / 15.32 / 5.76 m final -- mean **10.28 m, 95% CI [6.05, 14.52] m**, against the real Transformer's 9.10 m mean, CI [2.61, 15.60] m. Heavily overlapping: removing augmentation is not a statistically defensible improvement, its mean is nominally slightly worse than the real Transformer's, the opposite of what the lucky single run (6.47 m) suggested. One secondary finding survives: the no-augment variant is meaningfully more consistent seed-to-seed (CV ~41%) than the real Transformer's 71%, though not enough to move the mean. Net across every intervention tried in this file (ensembling, two-phase uncertainty training, order-shuffling, removing augmentation): only ensembling produced a real, defensible improvement, and only as a mitigation, not a fix. More diverse training data remains the one genuine, un-run candidate fix.
