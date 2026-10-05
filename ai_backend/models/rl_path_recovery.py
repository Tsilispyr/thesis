"""RecoveryPolicyEnv -- the Gymnasium environment the seek-navigation RL
policy trains in (Track A/Phase-3 wiring: RL sits above the LSTM/EKF
dead-reckoning hierarchy, consuming its reconstructed position + confidence
to decide how to navigate, per AI_RECOVERY_EXECUTION_PLAN.md sec 12 Goal 3 /
recovery_orchestrator.py's own docstring). Single-drone scope; a swarm/
multi-node version is Track B, deliberately deferred, not started here.

This env replaces `_apply_seek_bias()` in recovery_orchestrator.py -- the
LSTM/EKF estimators are untouched and still answer "where am I probably";
this env's policy only answers "given that estimate and how much to trust
it, how should I move toward the goal." The core mechanism that makes this
different from the fixed proportional controller it replaces: reward is
graded on a hidden `true_pos`, while the policy only ever observes a
corrupted `believed_pos` -- this is what forces it to learn *when to
discount its own observation*, something no memoryless function of
(believed_pos, goal_pos) alone (a fixed Kp formula) can ever do.

Usage (training driver lives in ai_backend/train_rl_recovery.py):
  from models.rl_path_recovery import RecoveryPolicyEnv
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Mirrors recovery_orchestrator.py's constants of the same name/purpose --
# duplicated rather than imported to avoid a circular import (that module
# will import this one to load a trained policy). Keep these two files'
# values in sync by hand if either changes.
RTL_TIMEOUT_S = 8.0             # seconds -- normalizes the elapsed-time observation
RTL_SKIP_IF_CLOSE_M = 3.0       # metres -- arrival/success radius, reused as-is
SEEK_MAX_CONTRIB = 1.5          # m/s -- action norm-cap, matches the formula's own cap

DT = 0.1                        # seconds/tick, matches the project's real 10Hz telemetry rate
# Exactly RTL_TIMEOUT_S worth of steps, not an approximation of it: this is
# the real system's own hard boundary on how long continuous blind
# single-goal seeking is allowed to run before recovery_orchestrator.py
# gives up and switches to RTL (see RTL_TIMEOUT_S above) -- the env should
# model that exact bounded window, not a loosely-similar one. Originally
# 300 (30s), then 150 (15s, "close to" the boundary but still ~2x over it)
# -- both found miscalibrated by running the hand-coded formula through
# eval_metrics.py/one-off diagnostics and watching believed_pos's estimation
# error (a random walk, see DRIFT_SIGMA_* below) grow past the goal
# distance well before the episode ended, at which point the formula
# doesn't just get noisy -- direction = goal - believed_pos is dominated by
# -est_error, so it commits confidently to a wrong heading and drives the
# true position monotonically AWAY from the goal at full speed for
# whatever episode time remains. Letting the episode run substantially
# longer than RTL_TIMEOUT_S just spends more steps inside that
# already-diverged regime, which the real system would never actually
# allow to happen.
MAX_EPISODE_STEPS = int(RTL_TIMEOUT_S / DT)

# Per-0.1s-step, per-axis random-walk sigma range for the simulated
# estimation-error drift, derived from this project's own real
# chained-trajectory numbers over the standard 12.5s/125-step held-out
# segment (not arbitrary constants). 26.34m/132.5m are themselves *vector*
# Euclidean errors (evaluate_trajectory.py's euclidean_error() is
# np.linalg.norm(pred-gt, axis=1) over 3 position columns), not per-axis
# values -- so deriving a per-axis step sigma needs an extra /sqrt(3): for
# a 3-axis isotropic random walk after N iid per-axis steps of std
# sigma_step, the per-axis std is sigma_step*sqrt(N) and the *vector*
# magnitude's RMS is sqrt(3) times that (sum of 3 independent
# squared-Gaussian components). Omitting the /sqrt(3) here first (i.e.
# solving sigma_step = final_error/sqrt(125) and using that as a per-axis
# sigma) silently inflated the simulated vector-magnitude drift by ~1.73x
# -- found by diagnosing why the hand-coded formula's success rate stayed
# low and flat across the whole drift_sigma range even after fixing the
# episode-length/goal-distance calibration (see MAX_EPISODE_STEPS and
# reset() below): with the inflated sigma, per-axis error already reached
# a magnitude comparable to a typical 5-25m goal distance within ~1s of
# simulated flight for BOTH the good and bad ends of the range, swamping
# the direction signal almost immediately regardless of estimator quality
# -- not a plausible read of "LSTM drifts less than EKF."
#   LSTM-like (good case):  26.34 / sqrt(125) / sqrt(3) = 1.36 m/step
#   EKF-like  (bad case):  132.5  / sqrt(125) / sqrt(3) = 6.84 m/step
# Sampled log-uniform per episode (not fixed) since _apply_seek_bias today
# runs identically whether the raw estimate came from LSTM or EKF -- this
# is what a robust policy needs to see across training.
DRIFT_SIGMA_LOW = 26.34 / np.sqrt(125) / np.sqrt(3)
DRIFT_SIGMA_HIGH = 132.5 / np.sqrt(125) / np.sqrt(3)

# Reward-shaping weights -- first-pass, reasoned values in the same spirit
# as SEEK_KP/SEEK_MAX_CONTRIB ("reasoned first guesses, not yet tuned
# against a success-rate metric"), flagged as such, expected to need at
# least one iteration once real training curves exist.
W_PROGRESS = 1.0                # reward per metre of *true* progress toward goal
TIME_PENALTY = 0.05             # per-tick cost, discourages stalling
W_EFFORT = 0.01                 # small action-smoothness regularizer
LAMBDA_CONFIDENCE = 0.5         # cost per (unit action norm x normalized confidence) --
                                 # the confidence-aware term: an auxiliary, dense proxy
                                 # for "large corrections under high uncertainty are risky"
ARRIVAL_BONUS = 50.0

# Normalization divisors -- keep every observation component roughly O(1),
# scales taken from the real system's own units (SEEK_MAX_CONTRIB's m/s
# scale, a representative ~100m mid-mission distance, EKF's own documented
# 5.20 m/axis bench-test uncertainty).
POS_NORM = 100.0
UNC_NORM = 10.0
VEL_NORM = 3.0


class RecoveryPolicyEnv(gym.Env):
    """Single-drone seek-navigation environment. Observation (11,):
      [0:3]  (goal_pos - believed_pos) / POS_NORM        -- relative vector to goal
      [3]    ||goal_pos - believed_pos|| / POS_NORM       -- radial distance (redundant w/ 0:3, speeds convergence)
      [4:7]  confidence_obs / UNC_NORM                    -- EKF-style per-axis 1-sigma uncertainty estimate
      [7:10] v_raw / VEL_NORM                              -- the raw pre-seek dead-reckoning velocity
      [10]   elapsed_loss_time / RTL_TIMEOUT_S              -- temporal context, clipped

    Action (3,) in [-1,1]: mapped to a seek velocity via the same isotropic
    norm-cap the real formula uses, so a badly-trained policy can never
    command more correction than today's hand-coded version could.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(11,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

        # Episode state, set in reset()
        self.true_pos = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.est_error = np.zeros(3)
        self.v_raw = np.zeros(3)
        self.drift_sigma = DRIFT_SIGMA_LOW
        self.step_count = 0

    @property
    def believed_pos(self) -> np.ndarray:
        return self.true_pos + self.est_error

    def _confidence_obs(self) -> np.ndarray:
        # Analytic expected-growth curve of a random walk, not the literal
        # realized est_error -- matches how the real EKF's covariance
        # actually behaves (a deterministic function of its process model,
        # not of the noise realized that episode; this is exactly why its
        # own calibration is r=0.999 self-consistency, not 1.0 oracle
        # knowledge of the true error). Isotropic in this v1 (no baro-style
        # per-axis correction modeled yet -- flagged as a later refinement,
        # not required for the core truth/belief mechanism to work).
        growth = self.drift_sigma * np.sqrt(max(self.step_count, 1))
        return np.full(3, growth)

    def _obs(self) -> np.ndarray:
        rel = (self.goal_pos - self.believed_pos) / POS_NORM
        dist = np.array([np.linalg.norm(self.goal_pos - self.believed_pos) / POS_NORM])
        conf = self._confidence_obs() / UNC_NORM
        v = self.v_raw / VEL_NORM
        t = np.array([min(self.step_count * DT / RTL_TIMEOUT_S, 3.0)])
        obs = np.concatenate([rel, dist, conf, v, t]).astype(np.float32)
        return np.clip(obs, -20.0, 20.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.true_pos = np.zeros(3)
        # Goal sampled at a random direction, distance 5-15m -- calibrated
        # against this env's own real speed cap (SEEK_MAX_CONTRIB=1.5 m/s,
        # matching recovery_orchestrator.py's constant exactly) and
        # MAX_EPISODE_STEPS==RTL_TIMEOUT_S/DT (8s): at the cap, that window
        # covers at most 12m of straight-line travel, so anything much
        # beyond that would be geometrically unreachable within one episode
        # regardless of how good the seek policy is, let alone under
        # believed-position drift. Went through three ranges (10-150m,
        # 10-50m, 5-25m) before landing here, each found miscalibrated by
        # running the hand-coded formula through eval_metrics.py /
        # one-off diagnostics and observing either near-zero success or (in
        # the 5-25m case) a flat, sigma-independent success rate once
        # MAX_EPISODE_STEPS was fixed to match RTL_TIMEOUT_S -- see that
        # constant's own comment for the real mechanism (belief drift
        # overtaking the goal-direction signal, not just distance).
        direction = self._rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-6)
        dist = self._rng.uniform(5.0, 15.0)
        self.goal_pos = direction * dist

        self.est_error = np.zeros(3)
        # Log-uniform per episode -- domain randomization across the
        # LSTM-like/EKF-like drift range documented above.
        self.drift_sigma = float(np.exp(self._rng.uniform(
            np.log(DRIFT_SIGMA_LOW), np.log(DRIFT_SIGMA_HIGH))))
        # Small-magnitude noise, representing the real, documented behavior
        # of raw LSTM/EKF dead-reckoning velocity once GPS is lost (settles
        # near-static, not zero) -- see recovery_orchestrator.py's own
        # SEEK_* comment on why a seek bias is needed at all.
        self.v_raw = self._rng.normal(0.0, 0.2, size=3)
        self.step_count = 0

        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        norm = np.linalg.norm(action)
        seek_rl = action / max(norm, 1.0) * SEEK_MAX_CONTRIB

        true_dist_prev = float(np.linalg.norm(self.goal_pos - self.true_pos))

        v_total = self.v_raw + seek_rl
        self.true_pos = self.true_pos + v_total * DT
        # v_raw itself drifts slowly, independent of the agent's action --
        # it's the (untouched) LSTM/EKF estimate, not something the policy
        # controls directly.
        self.v_raw = self.v_raw + self._rng.normal(0.0, 0.05, size=3)
        self.v_raw = np.clip(self.v_raw, -2.0, 2.0)

        self.est_error = self.est_error + self._rng.normal(0.0, self.drift_sigma, size=3)
        self.step_count += 1

        true_dist_now = float(np.linalg.norm(self.goal_pos - self.true_pos))
        confidence_norm = float(np.mean(self._confidence_obs())) / UNC_NORM

        reward = W_PROGRESS * (true_dist_prev - true_dist_now)
        reward -= TIME_PENALTY
        reward -= W_EFFORT * float(np.dot(seek_rl, seek_rl))
        reward -= LAMBDA_CONFIDENCE * float(np.linalg.norm(seek_rl)) * confidence_norm

        terminated = False
        truncated = False
        if true_dist_now < RTL_SKIP_IF_CLOSE_M:
            reward += ARRIVAL_BONUS
            terminated = True
        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        info = {"true_dist": true_dist_now, "seek_norm": float(np.linalg.norm(seek_rl))}
        return self._obs(), reward, terminated, truncated, info


if __name__ == '__main__':
    print("=== RecoveryPolicyEnv smoke test (no training) ===\n")
    env = RecoveryPolicyEnv(seed=0)

    obs, info = env.reset(seed=0)
    assert obs.shape == (11,), f"expected obs shape (11,), got {obs.shape}"
    assert np.all(np.isfinite(obs)), "reset() produced non-finite observation"
    print(f"reset() obs: {obs}")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (11,)
    assert np.isfinite(reward), f"non-finite reward: {reward}"
    assert np.all(np.isfinite(obs2)), "step() produced non-finite observation"
    print(f"step() obs: {obs2}  reward: {reward:.4f}  info: {info}")

    # A zero action (no seek contribution) should still integrate v_raw
    # forward without crashing, and should never terminate/truncate on the
    # very first step from a 5-15m starting distance.
    assert not terminated and not truncated, "should not terminate/truncate on step 1"

    # Full random-policy episode -- confirms the episode actually ends
    # (terminates on arrival or truncates at MAX_EPISODE_STEPS), never
    # produces non-finite values along the way.
    env.reset(seed=1)
    for i in range(MAX_EPISODE_STEPS + 5):
        a = env.action_space.sample()
        o, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(o)) and np.isfinite(r), f"non-finite output at step {i}"
        if term or trunc:
            print(f"Episode ended at step {i+1}: terminated={term} truncated={trunc} "
                  f"true_dist={info['true_dist']:.2f}")
            break
    assert term or trunc, "episode never ended within MAX_EPISODE_STEPS + 5"

    print("\nAll RecoveryPolicyEnv smoke tests passed.")
