# Final Results Analysis - What's the Best Model, and What's the Best Number

This document answers one question across the whole project: after every training run, every
architecture, every honest negative result - what is the final result, which model is genuinely
"best," and what number should get cited if only one is allowed? It cross-references the existing
per-track `ANALYSIS.md` files (ablation matrix, models/) rather than duplicating their full detail;
see those for the complete statistical walkthrough behind any number below. Every number here is
re-read from its own source file, not re-derived - see the end of each section for where.

## 1. Scope and method

Every comparison in this project that matters is scored on **chained-trajectory reconstruction
error**: reconstruct a real, held-out 12.5-second flight segment step by step (each prediction feeds
the next window, errors compound exactly as they would in a live GPS-loss episode), then measure the
Euclidean distance between the reconstructed and true final position (`final`, metres) and the mean
distance across the whole segment (`mean`, metres). This is deliberately **not** the windowed
validation loss (MSE on isolated, non-chained windows) that training itself optimizes - this
project found repeatedly, across nearly every model tried, that a better windowed loss does not
reliably predict a better chained result (GBT beats the LSTM on chained error despite worse windowed
loss; batch=256 reaches better windowed loss but *worse* chained accuracy than batch=64; the
Random Forest's windowed MSE looked competitive with the LSTM's but chained accuracy did not follow).
Windowed loss is what the optimizer sees during training; chained error is what actually happens to
a drone that loses GPS for more than one tick. Every number below is chained error unless stated
otherwise.

## 2. The formal comparison - 5-seed ablation matrix

`runs/ablation/cpu_5seeds/` (5 seeds, the final and authoritative run - supersedes the earlier
3-seed `cpu_run1/`, kept only as the earlier-evidence baseline the 5-seed analysis diffs against).
Mean final chained error ± 95% t-distribution confidence interval, `n=5` per stochastic arm:

| Model | Final err (m) | 95% CI | Relative variability* |
|---|---|---|---|
| **Transformer** | **9.10** | [2.61, 15.60] | 71% - least reliable |
| SSL-Transformer | 13.99 | [8.83, 19.16] | 37% |
| GBT | 18.56 | [16.87, 20.24] | 9% |
| RandomForest | 26.42 | [25.76, 27.08] | **2% - most reliable** |
| LSTM | 29.68 | [24.82, 34.53] | 16% |
| SSL-LSTM | 34.57 | [31.50, 37.64] | 9% |
| EKF (deterministic, classical filter) | 132.47 | n/a | n/a |
| Pure Inertial (deterministic, no correction) | 143.00 | n/a | n/a |

*Relative variability = half the 95% CI width divided by the mean, this project's own established
convention for "how much does a single random seed's result swing" (not the raw sample
coefficient-of-variation, which would give different, smaller numbers) - used here for direct
comparability with every other document in this project.

**Statistically clean (non-overlapping-CI) wins**: Transformer beats LSTM, SSL-LSTM, RandomForest,
and GBT outright. SSL-Transformer beats LSTM and SSL-LSTM. GBT beats RandomForest, LSTM, and
SSL-LSTM. RandomForest beats SSL-LSTM. **Still tied even at 5 seeds**: Transformer vs.
SSL-Transformer, SSL-Transformer vs. GBT, RandomForest vs. LSTM.

Source: `runs/ablation/cpu_5seeds/results.json` (re-read directly for this document), full narrative
in `runs/ablation/cpu_5seeds/ANALYSIS.md` and `AI_RECOVERY_EXECUTION_PLAN.md` §19.6.

## 3. Four answers to "best," because it's four different questions

**Best raw number in the entire project - Transformer ensemble: 3.92 m final / 5.88 m mean.**
`ai_backend/evaluate_transformer_ensemble.py` averages all 5 `cpu_5seeds` Transformer checkpoints'
per-step delta prediction *before* chaining (not a post-hoc average of 5 independently-drifted
trajectories). Free - no retraining, built entirely from checkpoints that already existed. Does not
beat the single luckiest seed (3.22 m) - no method can know in advance which untrained seed will be
lucky - but lands close to it and roughly halves the error a randomly-deployed single seed would be
expected to produce. Not itself deployed in the live hierarchy.

**Best statistically-defensible single architecture - Transformer, 9.10 m mean, CI [2.61, 15.60].**
Cleanly beats 4 of the other 5 arms (table above). The catch: it is also, by a wide margin, the
*least* reliable model in the matrix (71% relative variability) - a single freshly-trained instance
is a real gamble, not a guaranteed 9.10 m.

**Best deployed/production result - LSTM seed 4, 24.78 m.** `ai_backend/models/dr_lstm_imu_norm.pth`
is not the ablation matrix's LSTM arm mean (29.68 m) - it is a specific ablation seed
(`continue_training_best_lstm.py` found it never early-stopped, had both the best windowed loss and
the best chained error of the 5 LSTM seeds, and confirmed via continued training that it had
genuinely converged rather than being cut off early) deliberately promoted to production after
beating the prior production checkpoint (26.34 m) on a fresh, apples-to-apples evaluation. Chosen
over the Transformer for the live hierarchy specifically *because* of its low-variance LSTM/
RandomForest-family lineage, not because it's the best point estimate anywhere in the project - the
Transformer's higher ceiling came with a per-run risk this project's own numbers argue against
gambling the live system on.

**Most reliable single number - RandomForest, 26.42 m, 2% relative variability.** If the criterion
is "which number can I trust to reproduce next time I train this," not "which is lowest," this is
the answer - an order of magnitude more consistent than the Transformer, competitive with the
LSTM's own accuracy.

Sources: `ai_backend/evaluate_transformer_ensemble.py`'s own output;
`ai_backend/models/ANALYSIS.md`, `STATUS.md`, `TRAINING_ANALYSIS.md` Part 5 (production checkpoint,
24.78 m); `AI_RECOVERY_EXECUTION_PLAN.md` §19.7 (seed-4 promotion narrative).

## 4. This session's two newest tracks

### RL wiring (`AI_RECOVERY_EXECUTION_PLAN.md` §19.11)

**Why**: the fail-safe hierarchy's seek-bias term (the pull toward the current goal once GPS is
lost) was a hand-coded proportional controller, structurally blind to how much its own position
estimate should be trusted - full-strength correction whether the estimate was fresh or badly
drifted. Named as the one unstarted gap ("RL still does not consume the orchestrator's output") in
every status document up to this session.

**Method**: `RecoveryPolicyEnv` (replacing an old, disconnected toy environment) tracks a hidden true
position separately from an agent-visible, drift-corrupted believed position - forcing the policy to
learn *when to discount its own observation*, something a fixed-gain formula structurally cannot do.
Drift calibrated from this project's own real chained-trajectory numbers (LSTM 26.34 m / EKF 132.5 m
over the standard held-out segment), domain-randomized per episode. Three environment-calibration
bugs and one real production bug (EKF confidence going stale on LSTM-active ticks) were found and
fixed before the real training run. Trained via PPO at 10K timesteps - not the 500K-1M originally
scoped, kept at the smaller budget so any result is attributable to the redesign, not more compute.

**Result**: raw arrival success rate came in *lower* than the hand-coded formula's (2.0% vs. 9.0%),
but RL learned to command corrections roughly 6-14x smaller on average - a real, large-sample-
verified behavioral difference (n>1500 steps per confidence bin), not RL simply mimicking the
formula at a lower gain. A controlled acceptance test (holding position/goal/velocity fixed, varying
only injected confidence) confirms the intended mechanism is genuinely present: `‖seek‖` shrinks
from 0.116 to 0.093 m/s as uncertainty rises from 0.3 m to 9.0 m.

**Honest outcome**: the mechanism is proven in isolation; it is not yet dominant in naturalistic
rollouts, where confidence, distance-to-goal, and elapsed time all grow together and the stronger
correlated effect masks the weaker (but real) causal one. Wired into the live hierarchy as an
opt-in replacement for the hand-coded formula, off by default.

### PX4 data diversity (`AI_RECOVERY_EXECUTION_PLAN.md` §19.12)

**Why**: the one genuine, un-run candidate fix for the Transformer's 71%-variability problem, named
across multiple documents and never executed - but `imu_data.csv` (the Transformer's actual
production training source) can't be diversified without new Godot recordings, which aren't
available in this environment. The standalone PX4 corpus could be grown without Godot at all.

**Method**: re-pulled a Mission-mode-filtered batch (6 → 46 logs total) - finding and fixing a real
bug along the way (`flight_modes` is a list of PX4 integer nav_state codes, not the string this
project's own docs had assumed; the correct id-to-name mapping was taken verbatim from PX4's own
`flight_review` tool source). Then tested a physically-motivated synthetic-diversity augmentation -
frame-consistent heading rotation and state-matched segment splicing, every value actually measured
by a real autopilot at some point, a different category from the statistical jitter/scale/warp noise
already tested and found not to survive confirmation (§5 below).

**Result**: the larger real corpus alone took the standalone `sources=px4` training arm's val loss
from ~1.5 ("worse than predicting the mean") to 0.2768 single-run / **0.2741 3-seed mean**
[0.2657, 0.2824] - confirms the original result was a data-quantity/diversity problem, not
structural. The synthetic-diversity augmentation on top of that gave a 3-seed mean of **0.2587**
[0.2182, 0.2991] - a consistent-direction, ~5.6% better point estimate, with 2 of 3 augmented seeds
beating every baseline seed outright.

**Honest outcome**: the 95% CIs still overlap at n=3 - by this project's own statistical bar, not
yet a proven effect. But unlike the shuffle/no-augment checks below (where more seeds made the
effect vanish entirely, converging back to baseline), this trend is consistent in direction, not
noise centered on zero. Reported as promising, not rounded up to a finding. Full data, logs, and
figures: `datasets/px4_raw/SOURCE.md`, `runs/px4_synthetic_diversity/`.

## 5. What did not work, briefly

For completeness - each of these was tested directly, not assumed, and each is a real, honest
negative result (full detail in the pointers given, not re-derived here):

- **SSL pretraining**, both LSTM and Transformer families - converges cleanly on its own terms but
  fine-tuning from it underperforms training from scratch. `ai_backend/models/ANALYSIS.md`.
- **Order-shuffled Transformer input** - a single-run result (9.78 m) that looked better than normal
  order; multi-seed confirmation (mean 10.01 m, CI [2.39, 17.64]) showed it's statistically
  indistinguishable from the real Transformer (9.10 m, CI [2.61, 15.60]). `AI_RECOVERY_EXECUTION_PLAN.md`
  §19.10.
- **No-augment Transformer** - a single-run result (6.47 m) that looked like nearly half the error;
  multi-seed confirmation (mean 10.28 m, CI [6.05, 14.52]) showed it's also statistically
  indistinguishable, and nominally slightly worse on the mean. Same section.
- **Two-phase mu/log_var uncertainty-head training** - the data-available fix for the uncertainty
  head's weak calibration (r=0.203); tested directly, calibration barely moved (r=0.199) and chained
  error came out worse. Same section.
- **Raising LSTM early-stopping patience (5→10)** - no clean win across the 5 ablation seeds
  (aggregate mean got worse); one seed improved substantially, most didn't. Same section.

## 6. Bottom line

The model actually running in production today is the **LSTM seed-4 checkpoint, 24.78 m** -
deliberately not the best point estimate in the project, chosen for reliability over ceiling. The
best-available-but-undeployed option is the **Transformer ensemble, 3.92 m** - already built, costs
nothing further to compute, and would close nearly all of that 21-metre gap if deployed. What would
actually need to happen for that: the ensemble already runs at inference time by loading and
averaging 5 existing checkpoints, so wiring it into `recovery_orchestrator.py` as a selectable
architecture (alongside the existing single-checkpoint model picker) is a bounded, well-scoped
engineering task, not a research question - the research question (does ensembling genuinely help)
is already answered. That, plus closing the two still-open reliability questions on the project's
actual production training data (the Transformer's own seed variance on `imu_data.csv`, the
uncertainty head's calibration - both blocked on the same thing, new diverse Godot recordings), are
the natural next steps for Track A, whenever this project returns to it rather than swarm/multi-drone
(Track B), which is next up per the user's own prioritization.
