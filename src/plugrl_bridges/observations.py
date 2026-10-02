"""PlugRL observations as the flat state vectors trainers expect."""

from __future__ import annotations

import numpy as np


def stack_states(rows: list[dict], state_keys: list[str]) -> np.ndarray:
    """float32 [n, D]: each row's `state_keys`, flattened and concatenated in order."""
    parts = []
    for row in rows:
        states = row["states"]
        missing = [k for k in state_keys if k not in states]
        if missing:
            raise KeyError(
                f"a client sent states {sorted(states)}; state_keys asks for {missing}"
            )
        parts.append(
            np.concatenate(
                [
                    np.asarray(states[k], dtype=np.float32).reshape(-1)
                    for k in state_keys
                ]
            )
        )
    return np.stack(parts)


def stack_images(rows: list[dict], key: str) -> np.ndarray:
    """[n, ...]: each row's image `key`, in the dtype the client sent (uint8 for a camera)."""
    for row in rows:
        if key not in row["images"]:
            raise KeyError(
                f"a client sent images {sorted(row['images'])}; image_key asks for {key!r}"
            )
    return np.stack([np.asarray(row["images"][key]) for row in rows])


def observation_space_of(batch: np.ndarray):
    """One env's observation space, read off a batch the clients sent.

    A uint8 batch is a camera's: Box(0, 255) in its own shape, which SB3 and
    gymnasium take as an image. Anything else is a state vector: an unbounded
    float32 Box.
    """
    import gymnasium as gym

    if batch.dtype == np.uint8:
        return gym.spaces.Box(0, 255, batch.shape[1:], np.uint8)
    return gym.spaces.Box(-np.inf, np.inf, batch.shape[1:], np.float32)


class EpisodeStats:
    """Each env's return and length so far: the bridge counts them itself, so
    that a trainer's episode logging does not depend on what a client sends."""

    def __init__(self, n: int) -> None:
        self.returns = np.zeros(n, dtype=np.float64)
        self.lengths = np.zeros(n, dtype=np.int64)

    def add(self, rewards: np.ndarray, steps: int = 1) -> None:
        self.returns += rewards
        self.lengths += steps

    def end(self, done: np.ndarray) -> None:
        self.returns[done] = 0.0
        self.lengths[done] = 0
