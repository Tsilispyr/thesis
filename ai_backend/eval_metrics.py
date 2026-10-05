"""Evaluates the hand-coded seek formula and the learned RL seek policy on
the SAME environment (RecoveryPolicyEnv) and the SAME success criterion
(RTL_SKIP_IF_CLOSE_M, imported from recovery_orchestrator.py, not
re-hardcoded) -- previously these were two independent, disconnected toy
simulations (a [0,10]x[0,10] grid with a fixed target here, a degree-scale
lat/lon grid in the old SwarmRecoveryEnv), neither of which ever touched the
real system's actual `_apply_seek_bias` formula. This is the first genuine
apples-to-apples comparison this project has had for this piece.

Both methods are run against the SAME per-episode seeds (paired comparison,
same goal placement/drift-sigma/initial jitter for both), not just
independently-sampled aggregate statistics.

Usage:
  python ai_backend/eval_metrics.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rl_path_recovery import RecoveryPolicyEnv, RTL_SKIP_IF_CLOSE_M, SEEK_MAX_CONTRIB
from api.recovery_orchestrator import _hand_coded_seek
from plot_style import style_axes, GRID_KW, COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE

N_EPISODES = 100
N_BINS = 5
_RL_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ppo_recovery_seek.zip')
_FIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'rl_seek_comparison.png')


def _run_episodes(get_seek, n_episodes: int = N_EPISODES, seed_offset: int = 0) -> list:
    """Runs n_episodes through RecoveryPolicyEnv, calling
    get_seek(v, believed_pos, goal_pos, confidence) -> seek_contribution
    (shape (3,), already safety-capped by the caller if needed) at each
    step. Routes the seek contribution back through the env's own
    action-space convention (env.step expects an action in [-1,1]^3 mapped
    via the same isotropic norm-cap get_seek's output already respects),
    so both methods experience identical true_pos/believed_pos/reward
    dynamics -- the env is the single source of truth for the simulation,
    neither method reimplements it."""
    results = []
    for ep in range(n_episodes):
        env = RecoveryPolicyEnv(seed=seed_offset + ep)
        env.reset(seed=seed_offset + ep)
        steps = 0
        success = False
        seek_norms, confidences = [], []
        while True:
            steps += 1
            v, believed, goal = env.v_raw, env.believed_pos, env.goal_pos
            confidence = env._confidence_obs()
            seek = get_seek(v, believed, goal, confidence)
            norm = float(np.linalg.norm(seek))
            if norm > SEEK_MAX_CONTRIB:
                seek = seek / norm * SEEK_MAX_CONTRIB
                norm = SEEK_MAX_CONTRIB
            action = seek / SEEK_MAX_CONTRIB   # round-trips exactly through env.step()'s own cap
            _, _, terminated, truncated, info = env.step(action)
            seek_norms.append(norm)
            confidences.append(float(np.mean(confidence)))
            if terminated:
                success = True
                break
            if truncated:
                break
        results.append({
            'success': success, 'steps': steps,
            'mean_confidence': float(np.mean(confidences)),
            'mean_seek_norm': float(np.mean(seek_norms)),
            # Raw per-step samples, not just the episode-level means above --
            # needed for confidence_binned_table's per-STEP binning, since
            # the reward's confidence-awareness operates on the instant a
            # correction is issued, not on an episode's overall average
            # drift severity (two very different things: episode-level
            # binning mostly separates low-drift_sigma episodes from
            # high-drift_sigma ones, since confidence grows ~monotonically
            # within an episode and most episodes run to the same
            # truncation length).
            'step_confidences': confidences,
            'step_seek_norms': seek_norms,
        })
    return results


def _hand_coded_get_seek(v, believed_pos, goal_pos, confidence):
    # The real formula ignores confidence entirely -- structurally, by
    # construction, which is exactly the property under test.
    return _hand_coded_seek(v, believed_pos, goal_pos) - v


def run_rule_based_eval(n_episodes: int = N_EPISODES) -> tuple:
    """Evaluates the real hand-coded `_apply_seek_bias` formula (imported
    from recovery_orchestrator.py, not reimplemented) inside
    RecoveryPolicyEnv. Returns (success_rate, avg_steps, per_episode_results)."""
    print("Evaluating hand-coded seek formula (recovery_orchestrator._hand_coded_seek)...")
    results = _run_episodes(_hand_coded_get_seek, n_episodes=n_episodes, seed_offset=0)
    successes = sum(r['success'] for r in results)
    avg_steps = float(np.mean([r['steps'] for r in results]))
    print(f"Hand-coded formula: Success Rate: {successes / n_episodes * 100:.1f}% | "
          f"Avg Steps: {avg_steps:.1f}")
    return successes / n_episodes, avg_steps, results


def run_rl_eval(n_episodes: int = N_EPISODES) -> tuple:
    """Loads the trained PPO seek policy (ppo_recovery_seek.zip) and
    evaluates it on the SAME RecoveryPolicyEnv, same episode seeds as
    run_rule_based_eval(). Returns (success_rate, avg_steps,
    per_episode_results) or (None, None, None) if no checkpoint exists yet."""
    print("Evaluating RL seek policy...")
    if not os.path.exists(_RL_POLICY_PATH):
        print(f"RL seek policy not found at {_RL_POLICY_PATH}. "
              f"Run train_rl_recovery.py first.")
        return None, None, None

    from stable_baselines3 import PPO
    model = PPO.load(_RL_POLICY_PATH)

    def _rl_get_seek(v, believed_pos, goal_pos, confidence):
        from models.rl_path_recovery import POS_NORM, UNC_NORM, VEL_NORM, RTL_TIMEOUT_S
        rel = (goal_pos - believed_pos) / POS_NORM
        dist = np.array([np.linalg.norm(goal_pos - believed_pos) / POS_NORM])
        conf = confidence / UNC_NORM
        v_norm = v / VEL_NORM
        t = np.array([0.0])   # matches _run_episodes' fresh-env-per-episode framing
        obs = np.clip(np.concatenate([rel, dist, conf, v_norm, t]).astype(np.float32), -20.0, 20.0)
        action, _ = model.predict(obs, deterministic=True)
        norm = np.linalg.norm(action)
        return action / max(norm, 1.0) * SEEK_MAX_CONTRIB

    results = _run_episodes(_rl_get_seek, n_episodes=n_episodes, seed_offset=0)
    successes = sum(r['success'] for r in results)
    avg_steps = float(np.mean([r['steps'] for r in results]))
    print(f"RL seek policy: Success Rate: {successes / n_episodes * 100:.1f}% | "
          f"Avg Steps: {avg_steps:.1f}")
    return successes / n_episodes, avg_steps, results


def confidence_binned_table(hand_coded_results: list, rl_results: list, n_bins: int = N_BINS) -> list:
    """Bins individual STEPS (pooled across all episodes) by their
    instantaneous confidence value (same 5-bin-by-quantile convention
    already used by evaluate_trajectory.py's uncertainty_calibration()) and
    reports mean ||seek|| per bin for both methods side by side -- the
    concrete, checkable evidence for whether RL learned confidence-sensitive
    behavior (shrinking correction as uncertainty grows) that the
    hand-coded formula structurally cannot exhibit (its ||seek|| depends
    only on distance, never on confidence, by construction).

    Binning at the step level, not the episode level, matters here: the
    reward's confidence-awareness operates on the instant a correction is
    issued (LAMBDA_CONFIDENCE * ||seek|| * confidence_norm, see
    rl_path_recovery.py), not on an episode's overall average drift
    severity. Since confidence grows ~monotonically within an episode
    (sigma * sqrt(steps_since_loss)) and most episodes run to the same
    truncation length, binning by *episode*-mean confidence mostly just
    separates low-drift_sigma episodes from high-drift_sigma ones -- a much
    weaker test than asking whether the SAME policy backs off its own
    correction as ITS OWN confidence degrades over the course of a single
    episode."""
    hc_conf = np.concatenate([r['step_confidences'] for r in hand_coded_results])
    hc_seek = np.concatenate([r['step_seek_norms'] for r in hand_coded_results])
    rl_conf = (np.concatenate([r['step_confidences'] for r in rl_results])
               if rl_results is not None else None)
    rl_seek = (np.concatenate([r['step_seek_norms'] for r in rl_results])
               if rl_results is not None else None)

    order = np.argsort(hc_conf)
    bin_edges = np.array_split(order, n_bins)

    table = []
    for b in bin_edges:
        hc_bin_seek = float(np.mean(hc_seek[b]))
        if rl_results is not None:
            # RL steps aren't index-aligned with hand-coded steps (different
            # trajectories through the same env), so bin RL's own steps by
            # its own confidence quantiles rather than reusing hc's indices.
            rl_order = np.argsort(rl_conf)
            rl_bin = np.array_split(rl_order, n_bins)[len(table)]
            rl_bin_seek = float(np.mean(rl_seek[rl_bin]))
            rl_bin_conf = float(rl_conf[rl_bin].mean())
        else:
            rl_bin_seek, rl_bin_conf = float('nan'), float('nan')
        table.append({
            'n': int(len(b)),
            'mean_confidence': float(hc_conf[b].mean()),
            'rl_mean_confidence': rl_bin_conf,
            'hand_coded_mean_seek': hc_bin_seek,
            'rl_mean_seek': rl_bin_seek,
        })
    return table


def _plot_comparison(table: list, out_path: str = _FIG_PATH):
    if not any(np.isfinite(row['rl_mean_seek']) for row in table):
        return   # nothing to plot without an RL run
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(table))
    width = 0.35
    hc_vals = [row['hand_coded_mean_seek'] for row in table]
    rl_vals = [row['rl_mean_seek'] for row in table]
    ax.bar(x - width / 2, hc_vals, width, label='Hand-coded formula', color=COLOR_ACCENT_ORANGE)
    ax.bar(x + width / 2, rl_vals, width, label='RL seek policy', color=COLOR_ACCENT_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"bin {i}\n({row['mean_confidence']:.2f}m)" for i, row in enumerate(table)])
    style_axes(ax, title='Mean seek magnitude by confidence bin',
               xlabel='Confidence bin (mean 1-sigma position uncertainty, m)',
               ylabel='Mean ||seek|| (m/s)')
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    print("--- SD-UAV Path Recovery Evaluation ---")
    _, _, hc_results = run_rule_based_eval()
    _, _, rl_results = run_rl_eval()

    if rl_results is not None:
        table = confidence_binned_table(hc_results, rl_results)
        print("\nConfidence-binned mean ||seek|| (hand-coded vs. RL):")
        print(f"{'bin':>4} {'n':>4} {'mean_conf(m)':>13} {'hand_coded(m/s)':>17} {'rl(m/s)':>10}")
        for i, row in enumerate(table):
            print(f"{i:>4} {row['n']:>4} {row['mean_confidence']:>13.3f} "
                  f"{row['hand_coded_mean_seek']:>17.3f} {row['rl_mean_seek']:>10.3f}")
        _plot_comparison(table)
    else:
        print("\n(Skipping confidence-binned comparison and plot -- no trained RL policy yet.)")
