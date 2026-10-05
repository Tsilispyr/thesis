# AI_Recovery - Progress Checklist

*A working overview for tracking and presenting where this subsystem stands. Unlike
[`STATUS.md`](STATUS.md) (the technical dashboard) or [`AI_RECOVERY_EXECUTION_PLAN.md`](AI_RECOVERY_EXECUTION_PLAN.md)
(the full bug-by-bug narrative), this file is meant to be read top to bottom as a
progress story: the problem, the idea, what exists, and what's left. The plan for
"what's left" will change as work progresses, treat it as direction, not a contract.*

---

## 1. What We're Trying to Solve

A swarm of small UAVs (drones) coordinating with ground vehicles (SD-UAV networks)
periodically loses GPS or communication mid-mission, for example due to urban
canyons, jamming, terrain shadowing, or simple range limits. When that happens, a
drone has no reliable way to know where it is or how to get back to safety.

The thesis asks: **can we reconstruct a drone's position purely from what it can
already sense on board (accelerometer, gyroscope, magnetometer, barometer, motor
output), well enough to fly it home without GPS?** And once we can estimate
position, can we act on that estimate, actually recover the drone, not just report
a number in an offline notebook?

This subsystem, `AI_Recovery`, started as the self-contained answer to that question
for a single drone, deliberately saving the multi-drone swarm side of the thesis for
after the single-drone pipeline was fully verified end to end. That verification is
now done, and the swarm side (Track B: does a group of drones stay connected back to
a base station once individual drones start drifting under GPS loss?) has since been
built and is described in §3.5 below. Wiring an actual N-drone swarm into the live
Godot scene over a real UDP multi-node protocol remains the one piece not yet done.

---

## 2. The Idea

Two ideas, stacked:

1. **Reframe "connectivity prediction" as dead reckoning.** Instead of trying to
   predict *when* GPS will drop, reconstruct *where the drone actually is* once it
   already has, using only onboard sensor data, a sensor-fusion extension of
   classical dead reckoning, not a signal-outage forecaster.
2. **Degrade toward simplicity.** No single estimator is trusted blindly. A
   fail-safe hierarchy falls back one level at a time as trust drops, aerospace-style:

   ```text
   Mission autopilot (true GPS)
         │  GPS lost
         ▼
   LSTM dead reckoning  →  Classical EKF  →  Rule-based vector-to-target  →  Breadcrumb (zero-AI)
   (learned, best case)     (no learning,      (last resort while UDP        (pure kinematic replay,
                              always available)   is still alive)              survives total link loss)
   ```

   Each layer only runs when the one above it is unavailable or untrusted. The
   bottom layer (breadcrumb) has zero dependency on Python or the network at all,
   it has to survive the AI stack failing completely, not just GPS.

A reinforcement-learning (RL) policy is meant to eventually sit *above* the LSTM,
consuming its reconstructed position + confidence to plan smarter recovery paths,
that layer exists today only as an untrained prototype (see §4).

---

## 3. What's Done So Far

### 3.1 Data & data integrity

- [x] **Leak-free training pipeline**, strict temporal train/val split, source-boundary-aware
      windowing, scaler fit on training data only. Three separate real bugs were found and
      fixed here (window shuffling, scaler-fit-before-split, unit-scale contamination between
      combined sources), each one had been quietly inflating results before being caught.
- [x] **`verify_unit_consistency()` guard**, a standing, permanent check that blocks combining
      any two data sources whose per-column scale disagrees by more than 3x unless explicitly
      overridden. Not a one-off fix; a mechanism that prevents the same class of bug recurring.
- [x] **Retired the synthetic `nav` dataset**, direct inspection showed it implies physically
      impossible motion (~1000-2200 m/s implied by position deltas vs. 7-30 m/s in its own speed
      column). Dropped as a trustworthy source rather than patched or explained away.
- [x] **Real PX4 flight-log ingestion**, now 46 genuine hardware flight logs (6 original + 40 more,
      `flight_modes=Mission`-filtered for real translational dynamics) pulled from the public
      PX4 Flight Review database, parsed and feature-aligned. Not yet combinable with the main
      `imu` dataset (gyroscope units don't reconcile yet, the consistency guard correctly blocks
      it) -- but the larger corpus fixed the standalone px4 arm's own data-diversity problem (val
      loss 1.5 -> 0.2768), and a physically-motivated synthetic-diversity augmentation on top of it
      showed a promising, not-yet-statistically-proven further improvement (0.2587 vs. 0.2741,
      3-seed means, CIs overlap). See `datasets/px4_raw/SOURCE.md`.

### 3.2 Models

- [x] **`DeadReckoningLSTM`**, the production dead-reckoning model (14 features, 2-layer LSTM,
      hidden size 64). Trained cleanly, real chained-trajectory error 26.3 m over a 12.5s
      held-out real-flight segment.
- [x] **Self-supervised pretraining (`MaskedLSTMAutoencoder`)**, implemented and evaluated
      honestly: pretraining converges well on its own task, but fine-tuning from it (0.81 val
      loss) came out worse than training from scratch (0.72). Reported as a genuine negative
      result, not hidden or reframed.
- [x] **Classical EKF baseline**, a proper Kalman filter with no learned component, used both
      as a comparison baseline and as a live fallback layer. Verified against a synthetic test
      case with known theoretical uncertainty bounds.
- [x] **Real chained-trajectory evaluation**, not just windowed MSE; actual sequential
      reconstruction over a real flight segment. LSTM (26.3 m) dramatically beats the classical
      EKF (132.5 m) and pure inertial integration (143.0 m).

### 3.3 The live, hands-on demo (most recent work)

- [x] **`RecoveryOrchestrator`**, a single, dependency-free, headlessly-testable Python class
      that actually runs the LSTM → EKF → rule-based decision hierarchy live, instead of it
      existing only as separate offline pieces.
- [x] **Seek-bias**, once GPS is lost, dead reckoning alone just drifts; added a
      distance-proportional, capped pull toward the current goal so the drone actually
      tries to get somewhere, not just estimate passively.
- [x] **Target-then-RTL hybrid**, after 8 seconds of continuous GPS loss without closing the
      distance to the mission target, the drone gives up on the mission and heads home instead,
      like a real autopilot's return-to-launch behavior.
- [x] **Godot Mission Control panel + Decision Terminal**, enter a mission target, cut/restore
      GPS on demand, and watch the live decision layer and its reasoning in real time.
- [x] **Standalone Python demo (`live_mission_demo.py`)**, the same live hierarchy, no Godot
      required at all, with a headless self-test mode for verification without a GUI.
- [x] **3D trajectory visualizer**, live reference/predicted/executed path rendering in Godot,
      matching the color scheme used in the offline analysis figures.
- [x] **Realistic Godot telemetry**, fixed simulated sensor data that was previously fake
      (random-noise accelerometer, wrongly-derived gyroscope, no motor/baro/mag channels at
      all), a prerequisite for any of the above to mean anything.

### 3.4 Deeper estimation rigor (Track A, mining MSc coursework for reusable technique)

- [x] **Uncertainty head for the LSTM (Gaussian NLL)**, a second output (mean + log-variance)
      trained with the standard heteroscedastic-regression loss. Real but weak calibration
      (r=0.203 between predicted uncertainty and actual chained error, vs. the EKF's native
      r=0.999); reported honestly rather than oversold. Improving this is being actively
      attempted (see §4).
- [x] **Transformer dead-reckoning arm + SSL-Transformer variant**, `d_model=64` matched to the
      LSTM's hidden size, hand-written attention layers for interpretability. Statistically the
      best architecture in the ablation matrix on chained-trajectory error, but by far the least
      consistent seed-to-seed (71% coefficient of variation vs. RandomForest's 2%), an
      unresolved reliability trade-off, not an oversight.
- [x] **Classical ML bench (RandomForest, GBT) + anomaly-detection gate**, answers the project's
      own "deep learning / ML / RL" framing. GBT reaches competitive accuracy in ~48 seconds
      despite having zero sequential modeling capability at all. The anomaly gate
      (IsolationForest + PCA-reconstruction) carries real, modest signal about where the LSTM is
      less trustworthy, but is offline analysis only, not yet wired into live layer selection.
- [x] **Formal multi-seed ablation matrix**, two complete runs (`cpu_run1`, 3 seeds;
      `cpu_5seeds`, 5 seeds), all 6 stochastic architectures plus 2 deterministic baselines,
      t-distribution 95% confidence intervals, full per-run written analysis.
- [x] **Production LSTM checkpoint update**, the ablation matrix's own byproduct data surfaced a
      seed that never triggered early stopping; testing it directly (not just assuming it would
      help) found a real, evidence-backed improvement (24.78 m vs. 26.34 m chained error),
      promoted safely with the outgoing checkpoint backed up, not deleted.
- [x] **Live model-switching in the hands-on demo**, compare all 6 trained architectures against
      each other live, in Godot or the standalone Python tool, not just the original LSTM-vs-EKF
      pair.
- [x] **imu/px4 gyroscope unit mismatch, investigated exhaustively**, a systematic sweep of every
      plausible unit conversion found none reconciles the two sources: a genuine dynamics
      difference between the simulated flight and real PX4 hardware, not a units bug. `px4` stays
      evaluation-only, permanently, this line item is answered, not merely closed.
- [x] **Transformer ensemble across its 5 already-trained seeds**, tested as a no-retraining
      mitigation for the reliability problem: averaging all 5 `cpu_5seeds` checkpoints' per-step
      prediction before chaining gives 3.92 m final / 5.88 m mean error, close to the single
      luckiest seed (3.22 m) and roughly half the error a randomly-deployed single seed would be
      expected to produce (9.10 m, the ablation study's own reported mean). Does not fix the
      underlying variance, but is a real, immediately usable mitigation.
- [x] **Two-phase mean/variance training for the uncertainty head, tested directly**, freeze `mu`
      after it converges on plain MSE, fit `log_var` alone afterward. Honest negative result:
      calibration barely moved (r=0.199 vs. the original 0.203) and chained error came out worse
      (38.49 m vs. 29.44 m, still within that model's already-observed run-to-run range).
      log_var collapsed toward a narrow, still-overconfident range instead of learning real
      input-dependent variation, the same overconfidence-race dynamic relocated into phase 2.
- [x] **Order-shuffled-input Transformer, tested directly, then confirmed with 5 seeds.** A
      single-run result (9.78 m final, nominally better than the real Transformer's 12.46 m)
      looked suggestive but was honestly flagged as inconclusive on n=1. Confirmed with the same
      5 seeds `cpu_5seeds` uses: mean **10.01 m, 95% CI [2.39, 17.64] m**, heavily overlapping the
      real Transformer's 9.10 m, CI [2.61, 15.60] m -- statistically indistinguishable, neither
      better nor worse. The single-run number was ordinary seed noise, exactly as the caveat
      anticipated.
- [x] **LSTM early-stopping patience raised from 5 to 10, tested directly.** No clean win, the
      aggregate mean final error across the same 5 seeds got *worse* (32.24 m vs. 29.67 m), though
      one seed improved substantially. A more important, unplanned finding surfaced while
      investigating seed 4 specifically: `set_global_seed()` reliably reproduces results within
      one environment (confirmed via an isolated-process control) but not bit-for-bit across
      sessions on CPU, a real, honestly-reported limit on how far seed-based reproducibility
      claims can be trusted, discovered rather than assumed away.
- [x] **Transformer without time-series augmentation, tested directly, then confirmed with 5
      seeds.** A single run reached 6.47 m final, nearly half the with-augmentation result, the
      best single-run Transformer number in the project. Caught and fixed a second real bug before
      it caused damage (the flag had no distinct save tag and would have overwritten the
      production Transformer checkpoint). Confirmed with the same 5 seeds: mean **10.28 m, 95% CI
      [6.05, 14.52] m**, heavily overlapping the real Transformer's 9.10 m, CI [2.61, 15.60] m --
      not a real improvement, nominally slightly worse on the mean. The exciting single-run number
      was seed luck. One secondary finding survives: the no-augment variant is meaningfully more
      consistent seed-to-seed (CV ~41%) than the real Transformer's 71%, just not enough to move
      the mean.

### 3.5 Swarm connectivity prediction (Track B, once single-drone rigor was solid)

A different question from everything above: not "where is one drone," but "does the
group stay connected back to a base station once individual drones start drifting
under GPS loss." Mirrors the three-method structure of the thesis author's own MSc
Graph & Network Analysis coursework, reusing this project's already-validated
GPS-loss drift model rather than inventing a new one.

- [x] **Synthetic multi-drone scenario generator.** No multi-drone dataset exists
      anywhere in this project, so one was built from first principles: 6-10 drones
      + a fixed base station per scenario, each drone's *believed* position drifting
      from its *true* position exactly like the RL environment's own model. Labels
      come from a Monte-Carlo rollout of true future positions, so predicting them
      from believed positions alone is a genuine, non-trivial task. Caught and fixed
      a real calibration bug before trusting the data: an initial guess produced a
      93% degenerate fragmentation rate, an empirical sweep landed on parameters
      giving a stable ~50/50 split.
- [x] **Three prediction methods, three seeds each, compared honestly.** Handcrafted
      graph features (degree stats, clustering, centrality, ...) into an MLP scored
      best: test Macro-F1 **0.841**, 95% CI [0.831, 0.852]. Node2Vec embeddings
      scored 0.780, CI [0.629, 0.932]. Graph neural networks (GCN/GIN) scored 0.722
      and 0.789 respectively, not the winner, plausible rather than a bug given how
      small these graphs are (7-11 nodes), a regime where engineered features
      routinely beat GNNs that need more data to earn their extra capacity.
- [x] **A live, running multi-drone demo, not just an offline dataset.** Mirrors the
      single-drone `live_mission_demo.py`'s own pattern: 8 drones + base station,
      live position drift, a real-time connectivity graph, and the trained model
      scoring fragmentation risk every tick, an actual inference loop, not a canned
      animation. A seeded, scripted test confirms the risk gauge and fragmentation
      state genuinely respond to sustained GPS loss (risk 0.227 -> 0.506, network
      fragments), with screenshots captured for later use.

---

## 4. What Remains, and How We Plan to Tackle It

*(Ordered roughly by priority today. This ordering and the specific approach for
each item is expected to change, some of these will turn out to be harder or
easier than they look, and priorities may get reshuffled by whatever we learn
along the way.)*

- [x] **Wire the RL policy into the live hierarchy.** Done. State redesigned to consume the
      orchestrator's own reconstructed position + EKF confidence instead of raw GPS. Timestep
      budget kept at 10K, not scaled to 500K-1M (unnecessary for this, per explicit call). Honest
      result: raw arrival success came in lower than the hand-coded formula's (2.0% vs. 9.0%), but
      a real, isolated confidence-shrinking behavior was confirmed in a controlled test -- the
      mechanism works, it's just not yet dominant in naturalistic rollouts at this budget. Opt-in,
      off by default.
- [ ] **Improve the uncertainty head's weak calibration** (r=0.203, r=0.199 after a tested
      two-phase attempt, essentially unchanged).
      The data-available option (separating mean/variance training into two phases) was tried
      directly and did not help, see §3.4. The remaining, more expensive option is more diverse
      training flights (the same limitation affecting the Transformer's reliability below) via
      `flight_recorder.py`, not yet available in bulk.
- [ ] **Investigate the cross-session reproducibility gap found while patience-testing.**
      `set_global_seed()` reliably reproduces results within one environment but not bit-for-bit
      across sessions on CPU (confirmed directly: an isolated-process rerun of one seed matched a
      sequential rerun exactly, but neither matched the original run from an earlier session).
      Likely thread-count-dependent CPU floating-point behavior, a known PyTorch limitation, not a
      bug in this codebase, but worth understanding since it limits how far any single seed's
      reported trajectory can be trusted as reproducible later.
- [x] **More diverse training data -- the analogous experiment, run on the standalone px4 corpus.**
      `imu_data.csv` itself (the Transformer's actual production training source) still can't be
      diversified without new Godot recordings, so its own 71%-CV reliability problem remains
      untested by this specifically. What was tested, on the separate px4 source: a larger,
      Mission-filtered batch (6 -> 46 logs) fixed that corpus's own diversity problem (val loss
      1.5 -> 0.2768); a physically-motivated synthetic-diversity augmentation (rotation +
      state-matched segment splicing) on top of that gave a consistent-direction, ~5.6% better
      3-seed mean (0.2587 vs. 0.2741) -- promising, but CIs still overlap at n=3, not yet proven.
- [ ] **Motor/barometer/magnetometer feature extension.**
      Godot now emits these fields correctly, but no bulk dataset has been recorded from them yet.
      Plan: use `flight_recorder.py` to accumulate a real dataset once the current priorities free
      up time, then extend `FEATURE_COLS` and retrain.
- [ ] **Tune the live hierarchy's constants** (`SEEK_KP`, `SEEK_MAX_CONTRIB`, `RTL_TIMEOUT_S`,
      `RTL_SKIP_IF_CLOSE_M`).
      Current values are reasoned first guesses, not tuned against any success-rate metric. Plan:
      once a repeatable mission-success measurement exists (manual today), sweep these
      systematically.
- [ ] **The two originally-specified formal Godot test protocols** (a synthetic-sequence unit
      test for bounded return-to-origin error; an integration test disabling UDP mid-flight).
      The Mission Control panel and breadcrumb fail-safe have already been run and confirmed
      working inside the Godot editor by the user directly (three real bugs were found and fixed
      this way), but the two literally-specified automated tests themselves were never built in
      that exact form, real interactive use substituted for them.
- [x] **Swarm connectivity prediction.** Done, see §3.5: synthetic scenario generator, three
      methods compared with proper multi-seed statistics, a live running demo with real screenshots.
- [ ] **Godot-integrated multi-drone swarm over a real UDP multi-node protocol.**
      §3.5's live demo is a self-contained Python simulation; wiring an actual N-drone swarm into
      the live Godot scene/UDP bridge is a separate, not-yet-started step beyond it.

---

## 5. Bottom Line

The single-drone pipeline works end to end and is honestly evaluated, including its
negative results (SSL fine-tuning, RL's raw success rate). The newest and most
demo-able piece is the live, hands-on fail-safe hierarchy, something you can
actually watch happen in real time, in Godot or in a plain Python window, rather
than only read about in a plot. Track A extended that same honesty to five more
phases (an uncertainty head, a Transformer arm, a classical-ML bench, a formal
multi-seed ablation matrix, a production-model update), and confirmed the
matrix's own headline result with real statistics rather than a single lucky
run. RL is now wired into that same live hierarchy, consuming reconstructed
position + confidence instead of raw GPS -- a real, confirmed mechanism, honestly
reported as not yet dominant at its current training budget. A larger, more
diverse PX4 corpus and a synthetic-diversity augmentation experiment on it gave a
promising, not-yet-proven result. Swarm connectivity prediction (Track B), the item
deliberately saved for last, is now delivered too: a synthetic multi-drone dataset,
three methods compared with proper multi-seed statistics (handcrafted graph
features winning outright), and a live, running demo with real screenshots. What's
left is the two remaining reliability questions on the actual single-drone
production training data (the Transformer's seed-to-seed variance on
`imu_data.csv`, the uncertainty head's weak calibration), both blocked on the same
thing, new diverse Godot recordings, plus wiring an actual Godot-integrated
multi-drone swarm over a real UDP protocol, which stays a separate, later step
beyond Track B's own live Python demo.
