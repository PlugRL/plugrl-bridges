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
