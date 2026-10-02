"""Trainers that were not written for PlugRL train environments that PlugRL env clients run.

One protocol server, `BridgeServer`, steps every connected env client in
lockstep; an adapter per trainer puts it behind that trainer's own
environment interface:

- `plugrl_bridges.rlinf.PlugRLEnv`: RLinf's environment contract (env_type plugrl).
- `plugrl_bridges.gym.PlugRLVectorEnv`: a gymnasium VectorEnv (CleanRL, TorchRL, ...).
- `plugrl_bridges.sb3.PlugRLVecEnv`: a Stable-Baselines3 VecEnv.

The adapters import their trainer's package; the server needs none of them.
"""

from plugrl_bridges.server import BridgeServer, ClientLost, SlotResult

__all__ = ["BridgeServer", "ClientLost", "SlotResult"]
__version__ = "0.1.0"
