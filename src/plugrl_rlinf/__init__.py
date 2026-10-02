"""RLinf trains environments that PlugRL env clients run, through the PlugRL protocol."""

from plugrl_rlinf.server import BridgeServer, ClientLost, SlotResult

__all__ = ["BridgeServer", "ClientLost", "SlotResult", "PlugRLEnv"]
__version__ = "0.1.0"


def __getattr__(name: str):
    # PlugRLEnv needs torch; the protocol server does not.
    if name == "PlugRLEnv":
        from plugrl_rlinf.env import PlugRLEnv

        return PlugRLEnv
    raise AttributeError(name)
