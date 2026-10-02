"""The PlugRL side of the bridge: a protocol server that RLinf steps in lockstep.

A PlugRL training server answers each client's `infer` as soon as it can.
RLinf cannot: its environment worker steps all of its environments together,
one `chunk_step` at a time. So this server holds every infer until RLinf
asks for a step, answers all of them with that step's actions, and returns
once every client has fed back and asked again.

    server = BridgeServer(port=8800, num_envs=8, action_horizon=1, action_dim=6)
    server.start()
    first = server.gather()            # each env's first observation
    result = server.step(actions)      # actions: [num_envs, horizon, action_dim]
    server.close()                     # ends the run with plugrl-server-stop

Clients are assigned slots in the order their envs first appear, and a
connection keeps its slots for the whole run. It is a lockstep server: every
infer and every feedback on a connection must name all of its envs, which a
client running chunks of one step (`action_horizon` 1) always does. SPEC.md
section 4.3 lets a client send a different env set. This server cannot
accept that, because RLinf needs every environment's outcome for every step.
It closes such a connection for a resync and says why.

It never invents data. A connection that closes mid-run fails the next
`step` or `gather` with ClientLost, and the run stops there.

With `episode_log` (or PLUGRL_BRIDGES_EPISODE_LOG set), it writes one CSV
line per episode that ends, counted from the rewards the clients sent: the
same record whichever trainer steps it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import threading
from typing import Any

import numpy as np
import websockets.asyncio.server as ws_server
import websockets.exceptions as ws_exceptions

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.reuse import REUSE_FEEDBACK_OBS, ObservationCache, ReuseError
from plugrl_protocol.websocket_protocol import (
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
    MessageType,
)


class ProtocolError(ValueError):
    """A client's message breaks SPEC.md; its connection closes for a resync."""


class Full(ProtocolError):
    """Every slot is taken. The client is told to stop, not to resync.

    A resync means "reconnect at once", and SPEC.md section 7.2 says a client
    does. Told that while the bridge is full, it reconnects, is refused again,
    and loops: on guangzhao, 48 leftover client processes made 2 million
    connections in ten minutes and slowed RLinf to a crawl.
    """


# websockets logs every connection at INFO; one line each is noise in RLinf's
# log, and a client stuck reconnecting turns it into hundreds of MB.
_ws_logger = logging.getLogger("plugrl_bridges.websockets")
_ws_logger.setLevel(logging.WARNING)


class ClientLost(RuntimeError):
    """A client's connection closed, so its environments cannot be stepped."""


@dataclasses.dataclass
class SlotResult:
    """One environment's outcome of one step."""

    reward: float
    terminated: bool
    truncated: bool
    info: dict
    feedback_obs: dict  # the observation the step ended on: terminal if done
    next_obs: dict  # the observation the next step starts from: reset if done


EPISODE_LOG_ENV = "PLUGRL_BRIDGES_EPISODE_LOG"


class EpisodeLog:
    """`step,slot,episode,return,length`, one line per episode that ends.

    `step` is the server's step on which it ended, counting from 1. A step is
    one action chunk, so with `action_horizon` 1 steps and lengths are env
    steps. `episode` counts each slot's episodes from 1. The return is the
    sum of the rewards the client fed back, before any trainer touches them.
    """

    HEADER = "step,slot,episode,return,length\n"

    def __init__(self, path: str | os.PathLike, num_envs: int) -> None:
        self._file = open(path, "w", encoding="utf-8", newline="")
        self._file.write(self.HEADER)
        self._file.flush()
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)
        self._episodes = np.zeros(num_envs, dtype=np.int64)
        self._step = 0

    def record(self, results: list[SlotResult]) -> None:
        self._step += 1
        ended = False
        for slot, r in enumerate(results):
            self._returns[slot] += r.reward
            self._lengths[slot] += 1
            if r.terminated or r.truncated:
                self._episodes[slot] += 1
                self._file.write(
                    f"{self._step},{slot},{self._episodes[slot]},"
                    f"{self._returns[slot]:.6f},{self._lengths[slot]}\n"
                )
                self._returns[slot] = 0.0
                self._lengths[slot] = 0
                ended = True
        if ended:
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def _int_vector(value, what: str) -> np.ndarray:
    if not (
        isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in "iu"
    ):
        raise ProtocolError(f"{what} is not a one-dimensional integer array")
    return value.astype(np.int64)


def split_rows(observation: Any, n: int, what: str) -> list[dict]:
    """An observation batched to `n`, as `n` observations of one row each."""
    if not isinstance(observation, dict) or set(observation) != {
        "images",
        "states",
        "text",
    }:
        raise ProtocolError(f"{what} is not a map of images, states and text")
    rows: list[dict] = [{"images": {}, "states": {}, "text": None} for _ in range(n)]
    for group in ("images", "states"):
        arrays = observation[group] or {}
        if not isinstance(arrays, dict):
            raise ProtocolError(f"{what}: {group} is not a map")
        for name, array in arrays.items():
            if not (
                isinstance(array, np.ndarray) and array.ndim >= 1 and len(array) == n
            ):
                raise ProtocolError(f"{what}: {group}[{name!r}] is not batched to {n}")
            for i in range(n):
                rows[i][group][name] = array[i]
    text = observation["text"]
    for i in range(n):
        if isinstance(text, np.ndarray) and text.ndim >= 1:
            rows[i]["text"] = str(text[i])
        elif isinstance(text, (list, tuple)) and len(text) == n:
            rows[i]["text"] = str(text[i])
        elif isinstance(text, str):
            rows[i]["text"] = text
        else:
            rows[i]["text"] = ""
    return rows


def _info_entries(info: dict) -> int | None:
    """SPEC.md section 5.4: the leading dimension of the first ndarray at the
    top level or one level into a nested map; None if there is none."""
    for value in info.values():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, np.ndarray) and nested.ndim >= 1:
                    return int(nested.shape[0])
    return None


def _split_info(info: Any, n: int) -> list[dict]:
    """A feedback's info, one entry per env, as plugrl-server splits it.

    Arrays and sequences of length `n` are split by env; anything else, such
    as a scalar or a string, is the same for every env. An info whose entry
    count (section 5.4) is not `n`, or that has none while `n` > 1, cannot be
    split, and closes the connection for a resync.
    """
    if not isinstance(info, dict):
        raise ProtocolError("info is not a map")
    if not info:
        return [{} for _ in range(n)]
    entries = _info_entries(info)
    if (entries is None and n > 1) or (entries is not None and entries != n):
        raise ProtocolError(f"info cannot be split into {n} entries")

    def take(value, i):
        if isinstance(value, dict):
            return {k: take(v, i) for k, v in value.items()}
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == n:
            return value[i]
        if isinstance(value, (list, tuple)) and len(value) == n:
            return value[i]
        return value

    return [take(info, i) for i in range(n)]


class _Connection:
    def __init__(self, ws, number: int) -> None:
        self.ws = ws
        self.number = number
        self.envs: list[int] | None = None  # its env indices, in slot order
        self.slots: list[int] = []
        self.cache = ObservationCache()
        self.infer: list[dict] | None = None  # the held infer, one row per env
        self.feedback: list[SlotResult] | None = None
        self.action: asyncio.Future | None = None
        self.closed: str | None = None


class BridgeServer:
    """Serves the PlugRL protocol to env clients; stepped by RLinf from its own thread."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8800,
        num_envs: int,
        action_horizon: int,
        action_dim: int,
        timeout: float = 600.0,
        metadata: dict | None = None,
        episode_log: str | os.PathLike | None = None,
    ) -> None:
        self.host, self.port = host, port
        episode_log = episode_log or os.environ.get(EPISODE_LOG_ENV) or None
        self._episode_log = (
            EpisodeLog(episode_log, num_envs) if episode_log is not None else None
        )
        self.num_envs = num_envs
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.timeout = timeout
        self.metadata = {
            "protocol_version": 1,
            "server": "plugrl-bridges",
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "num_envs": num_envs,
            "lockstep": True,
            "features": [REUSE_FEEDBACK_OBS],
            **(metadata or {}),
        }
        self._connections: list[_Connection] = []
        self._slots_taken = 0
        self._numbered = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._changed: asyncio.Condition | None = None
        self._stopping = False
        self._ready = threading.Event()
        self._started_error: BaseException | None = None
        self._server = None

    # ------------------------------------------------------------ RLinf's side

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="plugrl-bridge"
        )
        self._thread.start()
        self._ready.wait()
        if self._started_error is not None:
            raise self._started_error

    def gather(self) -> list[dict]:
        """Block until every env has asked for its first action; their observations."""
        return self._call(self._gather())

    def step(self, actions: np.ndarray) -> list[SlotResult]:
        """Send each env its actions, [num_envs, horizon, action_dim]; one result per env."""
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, self.action_horizon, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f"actions must be {expected}, got {actions.shape}")
        results = self._call(self._step(actions))
        if self._episode_log is not None:
            self._episode_log.record(results)
        return results

    def close(self) -> None:
        """End the run: every client gets plugrl-server-stop."""
        if self._episode_log is not None:
            self._episode_log.close()
            self._episode_log = None
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close(), self._loop).result(
                timeout=30
            )
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=30)
            self._loop = None

    @property
    def bound_port(self) -> int:
        return self._server.sockets[0].getsockname()[1]

    # ------------------------------------------------------------- the loop

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._changed = asyncio.Condition()

        async def open_server():
            # websockets builds its server object inside the running loop.
            return await ws_server.serve(
                self._handle,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                logger=_ws_logger,
            )

        try:
            self._server = self._loop.run_until_complete(open_server())
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            self._started_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        self._server.close()
        self._loop.run_until_complete(self._server.wait_closed())
        self._loop.close()

    def _call(self, coro):
        if self._loop is None:
            raise RuntimeError("the bridge server is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    async def _wait_for(self, predicate, what: str) -> None:
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait_for(predicate), self.timeout)
            except TimeoutError:
                raise TimeoutError(f"no {what} within {self.timeout:.0f} s") from None

    def _lost(self) -> list[_Connection]:
        # Only a connection holding slots can be lost. One refused before it
        # took any (a client too many, say) costs RLinf nothing.
        return [c for c in self._connections if c.closed is not None and c.envs]

    def _raise_if_lost(self) -> None:
        lost = self._lost()
        if lost:
            c = lost[0]
            raise ClientLost(
                f"the connection holding slots {c.slots} closed ({c.closed}); "
                "RLinf needs every environment for every step, so the run stops"
            )

    async def _gather(self) -> list[dict]:
        def ready():
            return bool(self._lost()) or (
                self._slots_taken == self.num_envs
                and all(c.infer is not None for c in self._connections if c.envs)
            )

        await self._wait_for(ready, f"{self.num_envs} environments")
        self._raise_if_lost()
        return self._rows_by_slot(lambda c: c.infer)

    async def _step(self, actions: np.ndarray) -> list[SlotResult]:
        self._raise_if_lost()
        live = [c for c in self._connections if c.envs]
        if self._slots_taken != self.num_envs or any(c.infer is None for c in live):
            raise RuntimeError("step() before every environment asked for an action")
        for c in live:
            chunk = actions[c.slots].swapaxes(0, 1)  # time-major, [H, n, da]
            c.infer, c.feedback = None, None
            c.action.set_result(
                {
                    "message_type": str(MessageType.ACTION),
                    "data": {
                        "action": np.ascontiguousarray(chunk),
                        "env_ids": np.asarray(c.envs, dtype=np.int64),
                    },
                }
            )

        def done():
            return bool(self._lost()) or all(
                c.feedback is not None and c.infer is not None for c in live
            )

        await self._wait_for(done, "feedback and next infer from every client")
        self._raise_if_lost()
        results: list[SlotResult | None] = [None] * self.num_envs
        for c in live:
            for slot, result, nxt in zip(c.slots, c.feedback, c.infer):
                result.next_obs = nxt
                results[slot] = result
        return results  # type: ignore[return-value]

    def _rows_by_slot(self, rows_of) -> list[dict]:
        out: list[dict | None] = [None] * self.num_envs
        for c in self._connections:
            if c.envs:
                for slot, row in zip(c.slots, rows_of(c)):
                    out[slot] = row
        return out  # type: ignore[return-value]

    async def _close(self) -> None:
        self._stopping = True
        for c in list(self._connections):
            if c.closed is None:
                try:
                    await c.ws.close(1001, SERVER_STOP_REASON)
                except Exception:  # noqa: BLE001 - it is going anyway
                    pass
            # A handler holding an infer waits on its action, not on the
            # socket, so it would not see the close. The run is over.
            if c.action is not None and not c.action.done():
                c.action.cancel()

    # ---------------------------------------------------------- a connection

    async def _handle(self, ws) -> None:
        packer = msgpack_numpy.Packer()
        self._numbered += 1
        c = _Connection(ws, self._numbered)
        self._connections.append(c)
        try:
            await ws.send(
                packer.pack(
                    {"message_type": str(MessageType.METADATA), "data": self.metadata}
                )
            )
            while True:
                infer = await self._recv(ws, MessageType.INFER)
                c.infer = self._check_infer(c, infer)
                c.action = asyncio.get_running_loop().create_future()
                await self._notify()
                action = await c.action
                await ws.send(packer.pack(action))
                feedback = await self._recv(ws, MessageType.FEEDBACK)
                c.feedback = self._check_feedback(c, feedback)
                await self._notify()
        except Full as exc:
            c.closed = f"refused: {exc}"
            await ws.close(1001, SERVER_STOP_REASON)
        except ProtocolError as exc:
            c.closed = f"protocol error: {exc}"
            await ws.close(1001, SERVER_RESYNC_REASON)
        except ws_exceptions.ConnectionClosed as exc:
            c.closed = "stopped" if self._stopping else f"closed by the client ({exc})"
        finally:
            if c.closed is None:
                c.closed = "closed"
            if not c.envs:
                # It never held a slot, so nothing waits on it.
                self._connections.remove(c)
            await self._notify()

    async def _recv(self, ws, expected: MessageType) -> dict:
        raw = await ws.recv()
        if not isinstance(raw, bytes):
            raise ProtocolError("a text frame")
        try:
            message = msgpack_numpy.unpackb(raw)
        except Exception as exc:  # noqa: BLE001 - any undecodable frame
            raise ProtocolError(f"undecodable frame: {exc}") from exc
        if not isinstance(message, dict) or message.get("message_type") != str(
            expected
        ):
            got = (
                message.get("message_type")
                if isinstance(message, dict)
                else type(message).__name__
            )
            raise ProtocolError(f"expected {expected}, got {got!r}")
        for key in ("data", "env_indices", "step_ids"):
            if key not in message:
                raise ProtocolError(f"{expected} has no {key!r}")
        return message

    def _check_infer(self, c: _Connection, message: dict) -> list[dict]:
        envs = _int_vector(message["env_indices"], "env_indices").tolist()
        if len(set(envs)) != len(envs) or not envs:
            raise ProtocolError("env_indices are empty or repeated")
        try:
            observation = c.cache.complete(
                message["env_indices"], message["data"], message.get("reuse")
            )
        except ReuseError as exc:
            raise ProtocolError(str(exc)) from exc
        rows = split_rows(observation, len(envs), "the infer observation")
        if c.envs is None:
            free = self.num_envs - self._slots_taken
            if len(envs) > free:
                raise Full(
                    f"this bridge steps {self.num_envs} environments and has "
                    f"{free} left; this connection asked for {len(envs)}"
                )
            c.envs = envs
            c.slots = list(range(self._slots_taken, self._slots_taken + len(envs)))
            self._slots_taken += len(envs)
            # One line per connection, so a run's log says where its
            # environments were (RLinf collects the worker's stdout).
            peer = getattr(c.ws, "remote_address", None)
            print(
                f"plugrl-bridges: slots {c.slots[0]}-{c.slots[-1]} <- {peer}",
                flush=True,
            )
            return rows
        if envs != c.envs:
            raise ProtocolError(
                "this server steps every environment in lockstep, so each infer "
                f"must name all of this connection's envs, {c.envs}, in order; "
                f"got {envs}"
            )
        return rows

    def _check_feedback(self, c: _Connection, message: dict) -> list[SlotResult]:
        envs = _int_vector(message["env_indices"], "env_indices").tolist()
        if envs != c.envs:
            raise ProtocolError(
                "this server steps every environment in lockstep, so each feedback "
                f"must name all of this connection's envs, {c.envs}, in order; "
                f"got {envs}"
            )
        data = message["data"]
        if not isinstance(data, dict) or set(data) != {
            "obs", "rewards", "terminated", "truncated", "info",
        }:  # fmt: skip
            raise ProtocolError("feedback data does not have exactly its five keys")
        n = len(envs)
        arrays = {}
        for key in ("rewards", "terminated", "truncated"):
            value = data[key]
            if not (isinstance(value, np.ndarray) and value.shape == (n,)):
                raise ProtocolError(f"{key} is not an array of length {n}")
            arrays[key] = value
        rows = split_rows(data["obs"], n, "the feedback observation")
        infos = _split_info(data["info"], n)
        c.cache.on_feedback(
            message["env_indices"],
            data["obs"],
            arrays["terminated"],
            arrays["truncated"],
        )
        return [
            SlotResult(
                reward=float(arrays["rewards"][i]),
                terminated=bool(arrays["terminated"][i]),
                truncated=bool(arrays["truncated"][i]),
                info=infos[i],
                feedback_obs=rows[i],
                next_obs={},
            )
            for i in range(n)
        ]
