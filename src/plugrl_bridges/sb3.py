"""A Stable-Baselines3 VecEnv whose environments are PlugRL env clients.

    env = PlugRLVecEnv(num_envs=16, port=8800,
                       action_space=gym.spaces.Box(-1, 1, (6,), np.float32))
    PPO("MlpPolicy", env).learn(409_600)

Construction starts a BridgeServer and blocks until `num_envs` environments
have connected and asked for their first action; the observation space is
read off what they sent. The clients run one-step chunks and reset their own
environments.

What SB3 expects of an episode end is there: `terminal_observation` and
`TimeLimit.truncated` in that env's info, and `episode` (return, length,
time) as SB3's Monitor would put it, counted by the bridge, so SB3 logs
`rollout/ep_rew_mean` with no wrapper. `reset()` cannot end the episodes in
progress - the protocol has no way for a server to ask - so it returns the
observations the next step starts from.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from plugrl_bridges.observations import EpisodeStats, stack_states
from plugrl_bridges.server import BridgeServer


class PlugRLVecEnv(VecEnv):
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
        self.state_keys = list(state_keys)
        self.server = BridgeServer(
            host=host,
            port=port,
            num_envs=num_envs,
            action_horizon=1,
            action_dim=int(np.prod(action_space.shape)),
            timeout=timeout,
            episode_log=episode_log,
        )
        self.server.start()
        self._next = stack_states(self.server.gather(), self.state_keys)
        observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self._next.shape[1],), np.float32
        )
        super().__init__(num_envs, observation_space, action_space)
        self._stats = EpisodeStats(num_envs)
        self._started = time.time()
        self._actions: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        # The clients seed and reset their own environments.
        return self._next.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, 1, -1
        )

    def step_wait(self):
        results = self.server.step(self._actions)
        obs = stack_states([r.next_obs for r in results], self.state_keys)
        rewards = np.asarray([r.reward for r in results], dtype=np.float32)
        terminated = np.asarray([r.terminated for r in results], dtype=np.bool_)
        truncated = np.asarray([r.truncated for r in results], dtype=np.bool_)
        dones = terminated | truncated
        self._stats.add(rewards)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = stack_states(
                [results[i].feedback_obs], self.state_keys
            )[0]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
            infos[i]["episode"] = {
                "r": float(self._stats.returns[i]),
                "l": int(self._stats.lengths[i]),
                "t": round(time.time() - self._started, 6),
            }
        self._stats.end(dones)
        self._next = obs
        return obs, rewards, dones, infos

    def close(self) -> None:
        self.server.close()

    # Per-env attributes and methods have no meaning here: the envs are
    # other processes, reached only through the protocol.
    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        return None

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        raise NotImplementedError(
            f"{method_name}: a PlugRL env client's environment is not reachable "
            "from the trainer except through the protocol"
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._get_indices(indices))
