"""Stable-Baselines3's PPO trains PlugRL env clients through the bridge.

    python ppo.py --num-envs 16 --port 8800 --action-dim 6 \
        --total-timesteps 409600 --seed 0 --log-dir runs/sb3-seed0

Start the env clients against --port: one env each, one-step chunks, for
example `plugrl-run-env-client mujoco-v1 --server-port 8800 --num-procs 16
--num-envs 1 --runner.replan-steps 1`.

By default PPO runs with SB3's defaults except `n_steps` (256 per env, so
that a rollout is 16 x 256 = 4,096 frames), behind VecNormalize as SB3's
MuJoCo examples are. `--preset pendulum` uses rl-baselines3-zoo's published
PPO settings for Pendulum-v1 instead (gamma 0.9, gae_lambda 0.95, 10
epochs, lr 1e-3, clip 0.2, gSDE every 4 steps, no normalisation). VecMonitor
writes every episode's raw return and length to <log-dir>/monitor.csv.
"""

from __future__ import annotations

import argparse
import pathlib

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from plugrl_bridges.sb3 import PlugRLVecEnv

# rl-baselines3-zoo hyperparams/ppo.yml, Pendulum-v1 (n_envs and n_steps
# are this script's: 16 x 256 per rollout, where rl-zoo uses 4 x 1,024).
PRESETS = {
    "default": dict(normalize=True, ppo={}),
    "pendulum": dict(
        normalize=False,
        ppo=dict(
            gamma=0.9,
            gae_lambda=0.95,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=1e-3,
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
        ),
    ),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--action-low", type=float, default=-1.0)
    p.add_argument("--action-high", type=float, default=1.0)
    p.add_argument("--state-keys", nargs="+", default=["obs"])
    p.add_argument("--total-timesteps", type=int, default=409_600)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--preset", choices=sorted(PRESETS), default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/sb3")
    a = p.parse_args()

    preset = PRESETS[a.preset]
    log_dir = pathlib.Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = PlugRLVecEnv(
        num_envs=a.num_envs,
        action_space=gym.spaces.Box(
            a.action_low, a.action_high, (a.action_dim,), np.float32
        ),
        port=a.port,
        host=a.host,
        state_keys=tuple(a.state_keys),
    )
    env = VecMonitor(env, filename=str(log_dir / "monitor.csv"))
    if preset["normalize"]:
        env = VecNormalize(env, gamma=preset["ppo"].get("gamma", 0.99))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=a.n_steps,
        seed=a.seed,
        tensorboard_log=str(log_dir),
        device="cpu",
        verbose=1,
        **preset["ppo"],
    )
    model.learn(total_timesteps=a.total_timesteps)
    model.save(log_dir / "model")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
