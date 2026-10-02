"""The RLinf side of the bridge: an RLinf environment whose environments are PlugRL env clients.

RLinf's environment worker constructs an environment class and calls
`reset()` once, then `chunk_step(actions)` for every step (with
`auto_reset: True`). `PlugRLEnv` implements that contract on top of a
BridgeServer. Every slot is an environment that a PlugRL env client runs,
anywhere that can reach this port, in any language that speaks the protocol.

What RLinf gets matches its own simulators under auto-reset:
- the observation after the step, which for an env whose episode ended is
  the reset observation the client sent with its next infer;
- `infos["final_observation"]`, where those envs carry the terminal
  observation from the feedback, with `_final_observation` as the mask;
- `infos["final_info"]["episode"]`: the return, the length and the mean
  reward of each episode that ended;
- `infos["episode"]`: the same, for the episodes in progress.

Observations become `{"states": float32 [N, D]}`: the `state_keys` of the
client's `states`, flattened and concatenated in that order. With
`image_key`, that image is added as `main_images` (uint8 [N, H, W, C]), and
the text as `task_descriptions`.

Configuration, under `env.train` in an RLinf config:

    env_type: plugrl
    port: 8800             # this process listens on port + seed_offset
    host: 0.0.0.0
    state_keys: [obs]
    image_key: null
    action_dim: 6
    action_low: -1.0       # the action space RLinf samples from
    action_high: 1.0
    timeout: 600           # seconds to wait for every client, per step
    auto_reset: True       # required: a PlugRL client resets itself
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from plugrl_rlinf.server import BridgeServer, SlotResult

try:
    import gymnasium as gym

    _Base = gym.Env
except ImportError:  # the bridge itself does not need it; RLinf's wrappers do
    gym = None
    _Base = object


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


class PlugRLEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any = None,
    ) -> None:
        if not bool(_get(cfg, "auto_reset", True)):
            raise ValueError(
                "env_type plugrl needs auto_reset: True. A PlugRL env client resets "
                "its own environments; the protocol has no way for a server to ask."
            )
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.state_keys = list(_get(cfg, "state_keys", ["obs"]))
        self.image_key = _get(cfg, "image_key", None)
        self.action_dim = int(_get(cfg, "action_dim"))
        self.action_horizon = int(_get(cfg, "action_horizon", 1))
        low = np.full(self.action_dim, float(_get(cfg, "action_low", -1.0)), np.float32)
        high = np.full(
            self.action_dim, float(_get(cfg, "action_high", 1.0)), np.float32
        )
        if gym is not None:
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._low, self._high = low, high
        self.port = int(_get(cfg, "port", 8800)) + self.seed_offset
        self.server = BridgeServer(
            host=str(_get(cfg, "host", "0.0.0.0")),
            port=self.port,
            num_envs=self.num_envs,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            timeout=float(_get(cfg, "timeout", 600.0)),
        )
        self.server.start()
        self._device = torch.device("cpu")
        self._is_start = True
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        self._returns = torch.zeros(self.num_envs, dtype=torch.float32)

    # ------------------------------------------------- the attributes RLinf reads

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    # ------------------------------------------------------- observations

    def _wrap(self, rows: list[dict]) -> dict[str, Any]:
        parts = []
        for row in rows:
            states = row["states"]
            missing = [k for k in self.state_keys if k not in states]
            if missing:
                raise KeyError(
                    f"a client sent states {sorted(states)}; state_keys asks for {missing}"
                )
            parts.append(
                np.concatenate(
                    [
                        np.asarray(states[k], dtype=np.float32).reshape(-1)
                        for k in self.state_keys
                    ]
                )
            )
        obs: dict[str, Any] = {"states": torch.from_numpy(np.stack(parts))}
        if self.image_key:
            obs["main_images"] = torch.from_numpy(
                np.stack([np.asarray(row["images"][self.image_key]) for row in rows])
            )
            obs["task_descriptions"] = [row["text"] for row in rows]
        return obs

    def _episode(self) -> dict[str, torch.Tensor]:
        length = self._elapsed_steps.clone()
        ret = self._returns.clone()
        return {
            "return": ret,
            "episode_len": length,
            "reward": ret / torch.clamp(length.float(), min=1.0),
        }

    # ---------------------------------------------------------- the contract

    def reset(self, *, seed: Any = None, options: dict | None = None):
        if options and options.get("env_idx") is not None:
            raise NotImplementedError(
                "a PlugRL env client resets its own environments; resetting "
                "chosen ones from the trainer is not in the protocol"
            )
        rows = self.server.gather()
        self._elapsed_steps.zero_()
        self._returns.zero_()
        self._is_start = True
        return self._wrap(rows), {}

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().to("cpu", torch.float32).numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        if chunk_actions.ndim != 3:
            raise ValueError(
                "chunk_actions must be [num_envs, chunk_steps, action_dim], "
                f"got {chunk_actions.shape}"
            )
        results: list[SlotResult] = self.server.step(chunk_actions)
        n, h = self.num_envs, chunk_actions.shape[1]

        rewards = torch.zeros((n, h), dtype=torch.float32)
        terminations = torch.zeros((n, h), dtype=torch.bool)
        truncations = torch.zeros((n, h), dtype=torch.bool)
        # The client sums the chunk's rewards (SPEC.md section 5.4), so the
        # sum goes on the chunk's last step: RLinf's chunk-level reward.
        rewards[:, -1] = torch.tensor([r.reward for r in results])
        terminations[:, -1] = torch.tensor([r.terminated for r in results])
        truncations[:, -1] = torch.tensor([r.truncated for r in results])
        done = terminations[:, -1] | truncations[:, -1]

        self._elapsed_steps += h
        self._returns += rewards[:, -1]
        obs = self._wrap([r.next_obs for r in results])
        infos: dict[str, Any] = {"episode": self._episode()}
        if done.any():
            final_obs = self._wrap(
                [
                    r.feedback_obs if d else r.next_obs
                    for r, d in zip(results, done.tolist())
                ]
            )
            infos["final_observation"] = final_obs
            infos["final_info"] = {"episode": self._episode()}
            infos["_final_observation"] = done.clone()
            infos["_final_info"] = done.clone()
            infos["_elapsed_steps"] = done.clone()
            self._elapsed_steps[done] = 0
            self._returns[done] = 0.0
        self._is_start = False
        return [obs], rewards, terminations, truncations, [infos]

    def step(self, actions):
        """One step of a one-step chunk, for callers that do not chunk."""
        actions = torch.as_tensor(actions, dtype=torch.float32).reshape(
            self.num_envs, 1, -1
        )
        obs_list, rewards, terms, truncs, infos_list = self.chunk_step(actions)
        return obs_list[-1], rewards[:, -1], terms[:, -1], truncs[:, -1], infos_list[-1]

    def update_reset_state_ids(self) -> None:
        return None

    def sample_action_space(self) -> torch.Tensor:
        return torch.from_numpy(
            np.random.uniform(
                self._low, self._high, (self.num_envs, self.action_dim)
            ).astype(np.float32)
        )

    def close(self) -> None:
        self.server.close()
