# Ablation Matrix Analysis (run: cpu_5seeds, Track A Phase A5 extension)

This run extends `cpu_run1` (3 seeds) to 5 seeds -- seeds 0-2 are the exact same cached
checkpoints `cpu_run1` used (reused via `ablation_matrix.py`'s resume logic, not retrained), seeds
3 and 4 are new. This document focuses on what changed between the two, not a repeat of
`cpu_run1/ANALYSIS.md`'s full walkthrough -- read that one first for the baseline findings
(median-seed selection rule, CV normalization method, the windowed-vs-chained framing) this one
builds on directly.

## The full ranked table, 3 seeds vs 5 seeds

| Model | 3-seed final [m] | 3-seed 95% CI | 5-seed final [m] | 5-seed 95% CI | CI width, 3s -> 5s |
|---|---|---|---|---|---|
| **Transformer** | 8.18 | [0.00, 21.96] | **9.10** | **[2.61, 15.60]** | 21.96 -> 12.99 |
| SSL-Transformer | 14.72 | [3.51, 25.94] | 13.99 | [8.83, 19.16] | 22.43 -> 10.33 |
| GBT | 18.73 | [14.04, 23.43] | 18.56 | [16.87, 20.24] | 9.39 -> 3.37 |
| RandomForest | 26.68 | [25.32, 28.04] | 26.42 | [25.76, 27.08] | 2.72 -> 1.32 |
| LSTM | 31.71 | [23.23, 40.19] | 29.68 | [24.82, 34.53] | 16.96 -> 9.71 |
| SSL-LSTM | 35.43 | [27.84, 43.03] | 34.57 | [31.50, 37.64] | 15.19 -> 6.14 |

Every point estimate moved only slightly (the ranking order is unchanged), but every CI got
substantially tighter -- expected, since 2 more seeds is meaningful additional evidence at this
sample size. The Transformer's CI in particular no longer needs clipping at 0: at 3 seeds the raw
t-interval went negative (clipped, meaning the *true* computed spread was even wider than shown);
at 5 seeds the lower bound is a real, physically meaningful 2.61 m.

## The headline change: several rankings are now statistically clean that weren't before

`cpu_run1/ANALYSIS.md` was explicit that only two pairwise comparisons had non-overlapping CIs at
3 seeds -- "GBT beats RandomForest" and "Transformer beats RandomForest" -- and that the
Transformer's own apparent lead over LSTM was not yet proven (CI gap of only 1.27 m between
interval edges). Re-checking all pairs at 5 seeds:

**Newly clean (were overlapping at 3 seeds, now separated):**
- **Transformer beats LSTM** -- 3-seed gap was 1.27 m (barely not overlapping the wrong way,
  effectively a coin flip); 5-seed gap is 9.22 m (15.60 vs 24.82), a real separation now.
- **Transformer beats GBT** -- 3-seed CIs overlapped (14.04-21.96 shared range); 5-seed CIs do
  not (15.60 vs 16.87).
- **Transformer beats SSL-LSTM**, **SSL-Transformer beats LSTM**, **SSL-Transformer beats
  SSL-LSTM**, **GBT beats LSTM**, **GBT beats SSL-LSTM**, **RandomForest beats SSL-LSTM** -- all
  newly clean.

**Still not statistically separated at 5 seeds** (the honest remaining uncertainty):
- Transformer vs. SSL-Transformer (its own SSL variant)
- SSL-Transformer vs. GBT
- RandomForest vs. LSTM

**Practical upshot**: at 3 seeds, the only defensible claim was "GBT and Transformer beat
RandomForest." At 5 seeds, "the Transformer is genuinely the best architecture in this
comparison" is now a *statistically supported* claim, not just the best point estimate -- it
cleanly beats 4 of the other 5 arms (LSTM, SSL-LSTM, RandomForest, GBT), and is only
statistically tied with its own SSL-pretrained variant. This is a meaningfully stronger result
than `cpu_run1` could support, and is exactly the kind of improvement more seeds is supposed to
buy.

## This also revises part of the earlier interpretation, not just confirms it

`cpu_run1/ANALYSIS.md` used the Transformer-vs-GBT statistical tie (at 3 seeds) to argue the
Transformer's advantage over the LSTM probably comes from nonlinear feature combination *within*
a window (which GBT, an order-blind tree ensemble, can also do), not genuine multi-step temporal
reasoning *across* the window (which only the Transformer's attention or the LSTM's recurrence
could provide, and the attention-interpretability finding showed was mostly inert past layer 0).

That argument is weaker now. The Transformer *cleanly* beats GBT at 5 seeds. Since GBT
structurally cannot use temporal order at all, a clean win over it is at least suggestive evidence
that the Transformer's feedforward/residual processing of an order-*aware* token sequence is
doing something a purely order-*blind* model cannot fully replicate -- even if the attention
mechanism itself is still mostly inert (that finding, from `ai_backend/models/ANALYSIS.md`'s
single-instance attention extraction, has not been re-run at 5 seeds and is not contradicted here,
just no longer the whole explanation). The more accurate statement now: *something* about
processing the window as an explicit, order-tagged sequence of tokens helps, even though it is not
obviously the attention layers doing it. Pinning down whether it's the positional embedding,
the residual path, or something else would need a targeted ablation (e.g. an order-shuffled input
Transformer) not yet run.

## A new finding this run surfaced: LSTM's seed-to-seed variance has a visible training-dynamics story

`train_lstm.png` (this run's version) shows something `cpu_run1`'s 3-seed version couldn't: seeds
0-2 all early-stopped within 6-8 epochs, while seeds 3 and 4 kept improving much longer -- seed 3
stopped at epoch 10, and **seed 4 never triggered early stopping at all, still improving at epoch
30** (`ai_backend/models/history_imu_norm_ablation_seed4.json`). The per-seed numbers line up with
this cleanly at the extremes:

| Seed | Best epoch | Best windowed val loss | Chained final error |
|---|---|---|---|
| 4 | 30 (still improving) | **0.6846 (best)** | **24.78 m (best)** |
| 3 | 5 | 0.6880 | 28.46 m |
| 0 | 3 | 0.7122 | 30.01 m |
| 2 | 1 | 0.7320 | 35.64 m |
| 1 | 1 | 0.7475 | 29.48 m |

Not a perfect monotonic relationship (seed 1's worse windowed loss than seed 2's still produced a
better chained result), but the two extremes are unambiguous: the one seed that trained the
longest without plateauing produced both the best windowed loss and the best chained-trajectory
result. This suggests the LSTM's early-stopping trigger (patience=5) is, for some
initializations, cutting training off before it's actually converged, rather than correctly
detecting overfitting -- a plausible, concrete lever for improving the LSTM arm specifically
(e.g. a longer patience) that the 3-seed run gave no visibility into at all.

**This pattern does not hold for the Transformer.** Checking the same relationship there
(`history_transformer_imu_norm_ablation_seed{0-4}.json`): seed 3 has a mid-pack windowed val loss
(0.3550, better than seed 2's 0.4049) but the *worst* chained error of all 5 seeds (15.04 m,
clearly worse than seed 2's 7.15 m). Windowed loss and chained error move together reasonably well
within the LSTM's own seed variation, but decouple for the Transformer -- a second, independent
piece of evidence (alongside the SSL-Transformer's own windowed-vs-chained ranking reversal
already documented in `cpu_run1/ANALYSIS.md`) that the Transformer arm's real-world quality is
harder to predict from its training curve alone than the LSTM's is.

## Does the reliability finding (Transformer least consistent, RandomForest most consistent) still hold?

Yes, directionally, though the gap narrowed a little as raw evidence accumulated:

| Model | 3-seed CV (margin/mean) | 5-seed CV |
|---|---|---|
| RandomForest | 5% | **2%** (most stable, even more so) |
| SSL-LSTM | 21% | 9% |
| GBT | 25% | 9% |
| LSTM | 27% | 16% |
| SSL-Transformer | 76% | 37% |
| **Transformer** | 168% | **71%** (still by far the least stable) |

Every model's relative variability dropped as CIs tightened (expected -- this is partly just the
CI-narrowing effect of a larger sample, not necessarily the *underlying* seed-to-seed spread
shrinking). But the *ranking* of which models are reliable vs. erratic is unchanged: RandomForest
remains the most trustworthy single number in the table, the Transformer remains the least. The
Transformer's raw seed values (3.22, 5.94, 7.15, 14.17, 15.04 m) still span a 4.7x range between
best and worst seed, essentially the same spread `cpu_run1` already showed with 3 of these 5
values -- the two new seeds (3.22->stayed the min, 15.04 is a new max) *widened* the observed raw
range slightly, even as the statistical margin (which accounts for sample size) shrank. Both
things are true at once and worth holding in mind together: more seeds bought real statistical
confidence in the ranking, but did not demonstrate that the Transformer has become more
predictable run-to-run -- if anything, the raw spread grew a little.

## SSL pretraining: unchanged conclusion, slightly less overlap

SSL-LSTM (34.57 m) still underperforms from-scratch LSTM (29.68 m) point-estimate-wise, and their
CIs are now only barely overlapping (31.50 vs. 34.53, a 3.03 m shared range, down from a much
wider overlap at 3 seeds) -- close to becoming a clean result with perhaps one more seed.
SSL-Transformer (13.99 m) still underperforms from-scratch Transformer (9.10 m) and remains
clearly overlapping. Both arms' directional finding is unchanged and, if anything, slightly
reinforced: this is now 5 seeds of consistent evidence, not 3, for "SSL pretraining does not help
either architecture on this data," strengthening rather than weakening the standing hypothesis
that the limiting factor is the pretraining data's lack of diversity (one continuous flight), not
a fluke of either architecture's first few seeds.

## What remains genuinely uncertain even at 5 seeds

- The three pairs that stayed statistically tied (Transformer vs. SSL-Transformer, SSL-Transformer
  vs. GBT, RandomForest vs. LSTM) would need more than 5 seeds to resolve, if they are resolvable
  at all with this data's inherent variance ceiling.
- The Transformer's own reliability problem is not solved by adding seeds -- more seeds narrows
  the *confidence interval on the mean*, it does not make any individual future training run more
  predictable. Anyone deploying this arm should still expect meaningful run-to-run swings.
- The LSTM early-stopping/patience hypothesis above is inferred from 5 data points at the
  extremes, not tested directly -- the natural next step is a direct comparison (same 5 seeds,
  patience raised from 5 to e.g. 10) to see whether it systematically closes the LSTM's gap with
  the Transformer, or whether seed 4 was simply a favorable initialization that happened to keep
  improving.
- All of Track A's original data-diversity caveat still applies unchanged: this is still one
  continuous 37.8-minute flight. Nothing in this 5-seed extension tests whether more diverse
  training data would reduce the Transformer's variance, the way the roadmap originally
  hypothesized -- that remains the concrete, un-run next step it was before.
