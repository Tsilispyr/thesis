"""Trains the seek-navigation RL policy (PPO) that replaces
recovery_orchestrator.py's hand-coded `_apply_seek_bias()` formula, on
`RecoveryPolicyEnv` (models/rl_path_recovery.py). Mirrors this project's
other training drivers' shape (train.py/train_transformer.py): CLI args,
tag_suffix='' so an experimental run never overwrites the default/production
checkpoint (the exact bug class this project hit and fixed twice already
this session, see train_transformer.py's --no-augment tag-suffix fix).

Budget: 10K timesteps by default -- the same total budget the original
prototype used, so any improvement measured is attributable to the
redesigned environment/reward, not to more training compute. Scale up via
--total-timesteps only if a 10K run shows no learning signal at all.

Usage:
  python ai_backend/train_rl_recovery.py --total-timesteps 10000
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from models.rl_path_recovery import RecoveryPolicyEnv

_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_rl_model(total_timesteps: int = 10000, n_envs: int = 4, seed: int = 0,
                    eval_freq: int = 2000, checkpoint_freq: int = 5000,
                    tag_suffix: str = ''):
    """Returns (model, history). Saves ppo_recovery_seek{tag_suffix}.zip +
    history_rl_seek{tag_suffix}.json under ai_backend/models/, matching this
    project's history_{tag}.json naming convention (SB3's own .zip format
    for the checkpoint itself, not a raw state_dict)."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    tag = f"seek{tag_suffix}"

    print(f"\nStarting RL Recovery-Policy Training | total_timesteps={total_timesteps} "
          f"n_envs={n_envs} seed={seed} tag={tag}")

    train_env = make_vec_env(lambda: RecoveryPolicyEnv(), n_envs=n_envs, seed=seed)
    eval_env = make_vec_env(lambda: RecoveryPolicyEnv(), n_envs=1, seed=seed + 1000)

    ckpt_dir = os.path.join(_MODELS_DIR, f'rl_checkpoints_{tag}')
    best_dir = os.path.join(_MODELS_DIR, f'rl_best_{tag}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    eval_callback = EvalCallback(eval_env, best_model_save_path=best_dir,
                                  eval_freq=max(eval_freq // n_envs, 1),
                                  n_eval_episodes=20, deterministic=True, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=max(checkpoint_freq // n_envs, 1),
                                              save_path=ckpt_dir,
                                              name_prefix=f'ppo_recovery_{tag}')

    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps,
                callback=[eval_callback, checkpoint_callback])
    elapsed = time.time() - t0

    out_path = os.path.join(_MODELS_DIR, f'ppo_recovery_{tag}.zip')
    model.save(out_path)
    # model.num_timesteps is the actual steps trained, which SB3 rounds up
    # to the nearest multiple of n_steps*n_envs -- can exceed the requested
    # total_timesteps, so report the real count, not the requested one.
    actual_steps = model.num_timesteps
    print(f"Training done in {elapsed:.1f}s, {actual_steps} actual steps "
          f"({actual_steps / max(elapsed, 1e-6):.0f} steps/s). Model saved -> {out_path}")

    # EvalCallback logs to eval_env's internal Monitor; pull the running
    # eval history straight from its own recorded arrays rather than
    # re-deriving it, so this history file reflects exactly what the
    # callback actually measured during training.
    # evaluations_results is a list of one array-of-episode-rewards per eval
    # round (not a single 2D array) in this SB3 version -- average each
    # round's episodes separately rather than assuming a fixed array shape.
    eval_mean_reward = [float(np.mean(round_rewards))
                         for round_rewards in eval_callback.evaluations_results]
    history = {
        'tag': tag,
        'requested_total_timesteps': total_timesteps,
        'actual_total_timesteps': actual_steps,
        'n_envs': n_envs,
        'seed': seed,
        'elapsed_s': elapsed,
        'eval_timesteps': [int(x) for x in eval_callback.evaluations_timesteps],
        'eval_mean_reward': eval_mean_reward,
        'best_mean_reward': float(eval_callback.best_mean_reward),
    }
    hist_path = os.path.join(_MODELS_DIR, f'history_rl_{tag}.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved -> {hist_path}")

    return model, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-timesteps', type=int, default=10000,
                     help='PPO training budget (default 10K, matching the original prototype -- '
                          'see this script\'s docstring for why not the roadmap\'s 500K-1M).')
    ap.add_argument('--n-envs', type=int, default=4,
                     help='SB3 vectorized envs -- a real CPU-parallel speedup for this '
                          'lightweight custom env, unrelated to the DirectML-is-slower finding '
                          '(that was about a different, GPU-vs-CPU workload).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-freq', type=int, default=2000)
    ap.add_argument('--checkpoint-freq', type=int, default=5000)
    ap.add_argument('--tag-suffix', type=str, default='',
                     help='Appends to the saved tag so an experimental run never overwrites '
                          'the default checkpoint.')
    args = ap.parse_args()

    train_rl_model(total_timesteps=args.total_timesteps, n_envs=args.n_envs, seed=args.seed,
                    eval_freq=args.eval_freq, checkpoint_freq=args.checkpoint_freq,
                    tag_suffix=args.tag_suffix)
