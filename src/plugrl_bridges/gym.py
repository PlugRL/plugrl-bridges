"""A gymnasium VectorEnv whose environments are PlugRL env clients.

Any trainer that takes a `gymnasium.vector.VectorEnv` - CleanRL's scripts,
TorchRL, a loop of your own - can train environments that run in other
processes, on other machines or in other languages, through the PlugRL
protocol:

    env = PlugRLVectorEnv(num_envs=16, port=8800,
                          action_space=gym.spaces.Box(-1, 1, (6,), np.float32))
    obs, _ = env.reset()
    obs, rewards, terminated, truncated, infos = env.step(actions)

Construction starts a BridgeServer and blocks until `num_envs` environments
have connected and asked for their first action; the observation space is
read off what they sent. The clients run one-step chunks (`action_horizon`
1), and reset their own environments when an episode ends.

Autoreset is gymnasium's SAME_STEP mode: the step that ends an env's
episode returns that env's reset observation, and the terminal one is in
`infos["final_obs"]`, masked by `infos["_final_obs"]`. `reset()` cannot end
the episodes in progress - the protocol has no way for a server to ask - so
it returns the observations the next step starts from.

Episode statistics come with it, in `infos["episode"]` ({"r", "l", "t"},
masked by `infos["_episode"]`), counted from the raw rewards. Do not wrap it
in gymnasium's `wrappers.vector.RecordEpisodeStatistics`: as of gymnasium
1.3 that wrapper counts as if autoreset were NEXT_STEP, and here it would
drop the first reward of every episode.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from plugrl_bridges.observations import EpisodeStats, stack_states
from plugrl_bridges.server import BridgeServer


class PlugRLVectorEnv(gym.vector.VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        *,
        num_envs: int,
        action_space: gym.spaces.Box,
        port: int = 8800,
        host: str = "0.0.0.0",
        state_keys: tuple[str, ...] = ("obs",),
        timeout: float = 600.0,
        episode_log: str | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.state_keys = list(state_keys)
        self.single_action_space = action_space
        self.action_space = batch_space(action_space, self.num_envs)
        self.server = BridgeServer(
            host=host,
            port=port,
            num_envs=self.num_envs,
            action_horizon=1,
            action_dim=int(np.prod(action_space.shape)),
            timeout=timeout,
            episode_log=episode_log,
        )
        self.server.start()
        self._next = stack_states(self.server.gather(), self.state_keys)
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self._next.shape[1],), np.float32
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self._stats = EpisodeStats(self.num_envs)
        self._started = np.full(self.num_envs, time.perf_counter())

    def reset(self, *, seed: Any = None, options: dict | None = None):
        # The clients seed and reset their own environments.
        return self._next.copy(), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 1, -1)
        results = self.server.step(actions)
        obs = stack_states([r.next_obs for r in results], self.state_keys)
        rewards = np.asarray([r.reward for r in results], dtype=np.float64)
        terminated = np.asarray([r.terminated for r in results], dtype=np.bool_)
        truncated = np.asarray([r.truncated for r in results], dtype=np.bool_)
        done = terminated | truncated
        self._stats.add(rewards)
        infos: dict[str, Any] = {}
        if done.any():
            now = time.perf_counter()
            infos["episode"] = {
                "r": np.where(done, self._stats.returns, 0.0),
                "l": np.where(done, self._stats.lengths, 0),
                "t": np.where(done, np.round(now - self._started, 6), 0.0),
            }
            infos["_episode"] = done.copy()
            self._started[done] = now
            self._stats.end(done)
            final_obs = np.full(self.num_envs, None, dtype=object)
            for i in np.flatnonzero(done):
                final_obs[i] = stack_states([results[i].feedback_obs], self.state_keys)[
                    0
                ]
            infos["final_obs"] = final_obs
            infos["_final_obs"] = done.copy()
            infos["final_info"] = {}
            infos["_final_info"] = done.copy()
        self._next = obs
        return obs, rewards, terminated, truncated, infos

    def close_extras(self, **kwargs: Any) -> None:
        self.server.close()


class NormalizeObservation(gym.vector.VectorWrapper):
    """gymnasium's vector NormalizeObservation (and an optional clip), for SAME_STEP.

    gymnasium's own vector observation wrappers refuse SAME_STEP autoreset,
    because they would leave `infos["final_obs"]` untransformed. This one
    normalizes the observations and the final ones with the same running
    mean and variance, updated from each batch of observations as
    gymnasium's is.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        epsilon: float = 1e-8,
        clip: float | None = None,
    ) -> None:
        super().__init__(env)
        from gymnasium.wrappers.utils import RunningMeanStd

        self.obs_rms = RunningMeanStd(
            shape=env.single_observation_space.shape, dtype=np.float64
        )
        self.epsilon = epsilon
        self.clip = clip
        self.update_running_mean = True

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        out = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)

    def reset(self, *, seed: Any = None, options: dict | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        if self.update_running_mean:
            self.obs_rms.update(obs)
        return self._normalize(obs), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if self.update_running_mean:
            self.obs_rms.update(obs)
        if "final_obs" in infos:
            final = infos["final_obs"].copy()
            for i in np.flatnonzero(infos["_final_obs"]):
                final[i] = self._normalize(final[i][None])[0]
            infos = {**infos, "final_obs": final}
        return self._normalize(obs), rewards, terminated, truncated, infos
