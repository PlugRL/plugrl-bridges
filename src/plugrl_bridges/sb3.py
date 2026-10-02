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

The observation is the clients' `state_keys`, flattened into float32, or,
with `image_key`, that one image as sent: a uint8 [H, W, C] camera frame
gives a uint8 Box that SB3's CnnPolicy takes as an image. A Discrete action
space works too: actions cross as one float per env.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from plugrl_bridges.observations import (
    EpisodeStats,
    observation_space_of,
    stack_images,
    stack_states,
)
from plugrl_bridges.server import BridgeServer


class PlugRLVecEnv(VecEnv):
    def __init__(
        self,
        *,
        num_envs: int,
        action_space: gym.Space,
        port: int = 8800,
        host: str = "0.0.0.0",
        state_keys: tuple[str, ...] = ("obs",),
        image_key: str | None = None,
        timeout: float = 600.0,
        episode_log: str | None = None,
    ) -> None:
        self.state_keys = list(state_keys)
        self.image_key = image_key
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
        self._next = self._observe(self.server.gather())
        super().__init__(num_envs, observation_space_of(self._next), action_space)
        self._stats = EpisodeStats(num_envs)
        self._started = time.time()
        self._actions: np.ndarray | None = None

    def _observe(self, rows: list[dict]) -> np.ndarray:
        if self.image_key is not None:
            return stack_images(rows, self.image_key)
        return stack_states(rows, self.state_keys)

    def reset(self) -> np.ndarray:
        # The clients seed and reset their own environments.
        return self._next.copy()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32).reshape(
            self.num_envs, 1, -1
        )

    def step_wait(self):
        results = self.server.step(self._actions)
        obs = self._observe([r.next_obs for r in results])
        rewards = np.asarray([r.reward for r in results], dtype=np.float32)
        terminated = np.asarray([r.terminated for r in results], dtype=np.bool_)
        truncated = np.asarray([r.truncated for r in results], dtype=np.bool_)
        dones = terminated | truncated
        self._stats.add(rewards)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = self._observe([results[i].feedback_obs])[
                0
            ]
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
