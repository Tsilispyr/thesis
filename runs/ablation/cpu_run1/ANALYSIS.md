# Ablation Matrix Analysis (run: cpu_run1, Track A Phase A5)

Master synthesis for the first fully completed multi-seed ablation matrix: 3 seeds x 6 stochastic
models (LSTM, SSL-LSTM, Transformer, SSL-Transformer, RandomForest, GBT) + 2 deterministic
baselines (EKF, pure inertial), all on the identical held-out chained-trajectory segment
`evaluate_trajectory.py` uses throughout this project. Raw numbers: `results.json`. Full
epoch-by-epoch training record for every seed of every model: `full_console_log.txt`. This file
is the interpretation layer the project's convention calls for (see `ai_backend/models/
ANALYSIS.md`) -- not a repeat of the numbers, an explanation of what they mean.

## The full ranked table

| Model | Final err [m] | 95% CI | Mean err [m] | 95% CI |
|---|---|---|---|---|
| EKF | 132.47 | n/a (deterministic) | 66.12 | n/a (deterministic) |
| Pure Inertial | 143.00 | n/a (deterministic) | 71.39 | n/a (deterministic) |
| **Transformer** | **8.18** | [0.00, 21.96] | 7.28 | [4.00, 10.56] |
| SSL-Transformer | 14.72 | [3.51, 25.94] | 9.14 | [3.02, 15.26] |
| GBT | 18.73 | [14.04, 23.43] | 12.06 | [9.92, 14.20] |
| RandomForest | 26.68 | [25.32, 28.04] | 17.33 | [16.54, 18.13] |
| LSTM | 31.71 | [23.23, 40.19] | 20.01 | [16.28, 23.75] |
| SSL-LSTM | 35.43 | [27.84, 43.03] | 22.03 | [17.48, 26.58] |

Every learned model still beats both classical-physics baselines by a wide margin (EKF 132.47 m,
pure inertial 143.00 m) -- that finding, established earlier in the project, is unchanged and not
revisited further here.

## Does the Transformer actually win, or does it just have the best point estimate?

Point estimate: yes, clearly -- 8.18 m is the best number in the table by a wide margin. But this
project's own established standard (`ai_backend/models/ANALYSIS.md`'s repeated "an isolated
metric looking better is not evidence of real improvement" theme) means the point estimate alone
isn't the answer; the CI is.

Checking which pairs of models have **non-overlapping** 95% CIs (the only pairs where "wins" is
actually a defensible claim at this sample size, not just a ranking of noisy means):

- **RandomForest vs. GBT**: separated (GBT's [14.04, 23.43] sits entirely below RandomForest's
  [25.32, 28.04]). GBT is a real, defensible win over RandomForest.
- **RandomForest vs. Transformer**: separated (Transformer's upper bound 21.96 < RandomForest's
  lower bound 25.32). A real win, by a 3.36 m gap between interval edges.
- **Everything else overlaps.** Transformer vs. SSL-Transformer, Transformer vs. GBT,
  Transformer vs. LSTM, LSTM vs. SSL-LSTM, RandomForest vs. SSL-Transformer -- none of these
  pairs have non-overlapping CIs. Transformer's own CI upper bound (21.96) sits barely above
  LSTM's CI lower bound (23.23), a gap of only 1.27 m: at 3 seeds, "the Transformer beats the
  LSTM" is the right bet, but it is not yet a statistically clean separation the way "GBT beats
  RandomForest" is.

**Honest headline**: the Transformer has the best point estimate and is very likely the best
architecture here, but the only *cleanly proven* pairwise result in this whole table is that both
GBT and the Transformer beat RandomForest. Ranking the rest by point estimate is reasonable
(it's the best evidence available), but claiming statistical certainty for e.g. "Transformer
beats LSTM" or "GBT beats Transformer" would overstate what 3 seeds can support.

## A finding the point estimates alone hide: which models are actually *reliable*

CI width tells a very different story from CI position, and it's worth surfacing on its own,
normalized by each model's own mean (coefficient of variation = margin / mean) since an 8 m-wide
interval means something very different for a model averaging 8 m than one averaging 32 m:

| Model | Mean final err | CI margin | Relative variability (margin/mean) |
|---|---|---|---|
| RandomForest | 26.68 m | 1.36 m | **5%** |
| GBT | 18.73 m | 4.70 m | 25% |
| LSTM | 31.71 m | 8.48 m | 27% |
| SSL-LSTM | 35.43 m | 7.60 m | 21% |
| SSL-Transformer | 14.72 m | 11.22 m | 76% |
| **Transformer** | 8.18 m | 13.78 m | **168%** |

(Transformer's raw t-interval actually dips below zero -- mean 8.18 minus margin 13.78 is
negative -- clipped to 0.00 in the table above since a distance error can't be negative, but the
clipping itself is informative: it means the *true* computed spread is even wider than the
already-large visible interval.)

The Transformer wins on average and is the least *consistent* model in the entire matrix, by a
wide margin. Its three seeds landed at 14.17 m, 3.22 m, and 7.15 m final error -- more than 4x
spread between the best and worst seed of the exact same architecture and training procedure.
RandomForest sits at the opposite extreme: its three seeds (27.27 m, 26.19 m, 26.59 m) are
almost indistinguishable from each other. This is a plausible, explainable pattern, not a
coincidence: RandomForest is itself an internal ensemble of ~100 bootstrap-sampled trees, so a
single training run already averages over a lot of the randomness a neural net's one SGD
trajectory does not; the Transformer, the highest-capacity model in the comparison, has the most
room to fit a different set of incidental quirks in the single 37.8-minute training flight per
initialization. This directly reinforces the data-sufficiency risk flagged before any of Track A
was built (`ai_backend/models/ANALYSIS.md`'s data-sufficiency note, originally in the roadmap):
low training-data diversity doesn't just cap how good these models can get, it specifically
inflates seed-to-seed variance for the models with the most capacity to overfit that limited
diversity differently each time.

**Practical read for the thesis**: "the Transformer is the best architecture, but treat any
single Transformer training run's exact number with real skepticism -- rerun it and expect
meaningful swings" is a more honest and more useful statement than "the Transformer gets 8.18 m."
RandomForest's number, by contrast, can be trusted almost as reported.

## Why does the Transformer win at all? Connecting back to the architecture analysis

`ai_backend/models/ANALYSIS.md`'s attention-interpretability section (Phase A3) already
established, on a single trained instance, that only the Transformer's first encoder layer shows
real temporal selectivity (a U-shaped attention pattern favoring the window's oldest and newest
steps); layers 1-3 are statistically indistinguishable from uniform attention. That finding
already suggested the Transformer's advantage over the LSTM isn't really about deep multi-step
attention-based reasoning, and offered GBT's comparably strong single-run result as supporting
evidence, since GBT has no sequential modeling capability at all.

This multi-seed run **reinforces that reading rather than contradicting it**: GBT (18.73 m) and
the Transformer (8.18 m) are the two models whose CIs overlap the most with each other relative
to their own width, and neither cleanly separates from the other statistically (see above). If
the Transformer's advantage came primarily from genuine sequence-order reasoning that a
feature-flattening tree ensemble structurally cannot do, GBT should have been left further
behind than this. Instead, both a token-order-aware Transformer and an order-blind tree ensemble
land in a broadly similar performance band, well ahead of both LSTM variants and RandomForest.
The most likely explanation, consistent with the earlier single-run attention finding: this
task's real difficulty is dominated by nonlinear combination of the 14 features *within* a
window (which both GBT and the Transformer's feedforward/residual sublayers can do well), more
than by reasoning about temporal order *across* the window (where only the Transformer's largely
inert attention layers, and the LSTM's recurrence, would have an edge -- and the LSTM, despite
having a pure sequential inductive bias, is the *worst* deep model in this table).

The uncertainty-calibration finding (Phase A2, `DeadReckoningLSTMUncertainty`, r = 0.119-0.212
across three retrainings) doesn't bear directly on *why* the Transformer wins -- no
uncertainty-head variant of the Transformer was built -- but it does fit a wider pattern visible
in this table: every LSTM-family arm evaluated across this project (plain LSTM, SSL-LSTM here;
the uncertainty variant elsewhere) has shown higher run-to-run instability than the
non-recurrent alternatives (GBT, RandomForest). The uncertainty model's chained error across
three separate retrainings (29.44 m, 31.69 m, 38.54 m) sits in the same noisy neighborhood as
this run's plain-LSTM CI ([23.23, 40.19]). That's suggestive of an architecture-level property
(the LSTM family is comparatively sensitive to initialization/data-order on this dataset), not
three unrelated coincidences, though confirming that formally would need the uncertainty model
added as a seventh arm in a future ablation run.

## SSL pretraining: now a clean negative result on both architectures, at matched settings

For the first time this session, SSL-LSTM and SSL-Transformer are compared against their
from-scratch counterparts at *identical* settings (batch=64, 3 seeds each, same data) -- earlier
single-run SSL-Transformer numbers were explicitly caveated as batch=256-trained and not
apples-to-apples. With that caveat removed:

- SSL-LSTM (35.43 m) is worse than from-scratch LSTM (31.71 m).
- SSL-Transformer (14.72 m) is worse than from-scratch Transformer (8.18 m).

Both CIs overlap their from-scratch counterparts, so neither is a statistically airtight loss,
but the point estimate now points the same direction for both architectures under matched
conditions, which it did not before. This strengthens (does not yet prove) the standing
hypothesis in `ai_backend/models/ANALYSIS.md`: the pretext task or the single-flight pretraining
data's lack of diversity is the limiting factor, not something specific to either architecture.

## Why did this run's LSTM number (31.71 m) differ from the earlier single-run number (26.34 m)?

Addressed in full when it first came up mid-run: 26.34 m sits inside this run's 3-seed CI
([23.23, 40.19]), so it was a favorable seed on an unseeded run, not a better model. It is
recorded here for completeness because it's the same phenomenon as the variability finding
above, just for the model where it was first noticed.

## What remains genuinely uncertain at n=3

- **Every CI in this table is wide relative to what a thesis claim would ideally rest on.** 3
  seeds is enough to see that real variance exists and roughly how large it is, not enough to
  pin down a precise mean for any stochastic model, especially the Transformer and
  SSL-Transformer, whose margins are the largest in the table.
- **The clean, statistically defensible results from this run are narrower than the headline
  ranking**: only "GBT beats RandomForest" and "Transformer beats RandomForest" survive a
  non-overlapping-CI standard. Every other ordering in the table (including "Transformer is the
  best model overall") is the right bet on current evidence, not a proven result.
- **The variability finding is itself only measured at n=3.** Whether the Transformer's
  observed 4x seed-to-seed spread is a stable property of this architecture/dataset combination,
  or itself partly a small-sample artifact, would need more seeds to know confidently -- though
  given RandomForest's near-zero spread came from the same n=3 protocol, the contrast between
  them is unlikely to be pure sampling noise.
- **The root cause diagnosis (single continuous 37.8-minute flight limits both accuracy and
  reliability) is inferred, not directly tested.** The concrete next step this whole run points
  to, more than additional seeds: evaluate whether the Transformer's variance shrinks once
  trained on multiple diverse flight sessions, which would confirm the diagnosis rather than
  merely being consistent with it.
