"""PlugRLEnv, driven as RLinf's environment worker drives an environment.

Each fake client below runs scripted environments over the real protocol,
in a thread, and records what it receives. The tests call reset() and
chunk_step() as RLinf does, and check what comes back against the script.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import torch
import websockets.exceptions
import websockets.sync.client

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import SERVER_RESYNC_REASON, SERVER_STOP_REASON
from plugrl_bridges import ClientLost
from plugrl_bridges.rlinf import PlugRLEnv


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _obs(values: list[float], text: str = "go") -> dict:
    """States `obs` of [value, 2 * value] per env."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "images": {},
        "states": {"obs": np.stack([v, 2 * v], axis=1)},
        "text": np.asarray([text] * len(values)),
    }


class FakeClient(threading.Thread):
    """`n` envs. Env i's observation at step t is 100 * base + 10 * i + t.

    Its reward is the sum of the action it got. Env i ends its episode after
    `episode[i]` steps: the terminal observation is -1 - i, and the reset
    observation starts again at t = 0. With `lockstep_break`, the second
    feedback names only env 0.
    """

    def __init__(
        self, port, n, *, base=0, episode=None, reuse=False, lockstep_break=False,
        slow=0.0,
    ):  # fmt: skip
        super().__init__(daemon=True)
        self.port, self.n, self.base = port, n, base
        self.episode = episode or [10**9] * n
        self.reuse = reuse
        self.lockstep_break = lockstep_break
        self.slow = slow  # seconds to wait before each feedback
        self.actions: list[np.ndarray] = []
        self.metadata: dict | None = None
        self.close_reason: str | None = None

    def value(self, i: int, t: int) -> float:
        return 100.0 * self.base + 10.0 * i + t

    def run(self) -> None:
        packer = msgpack_numpy.Packer()
        t = [0] * self.n
        held = False  # whether the server holds this step's observations
        with websockets.sync.client.connect(
            f"ws://127.0.0.1:{self.port}", compression=None, max_size=None
        ) as ws:
            self.metadata = msgpack_numpy.unpackb(ws.recv())["data"]
            step = 0
            try:
                while True:
                    infer = {
                        "message_type": "infer",
                        "env_indices": np.arange(self.n),
                        "step_ids": np.asarray(t),
                    }
                    reusable = [held and t[i] > 0 for i in range(self.n)]
                    if self.reuse and any(reusable):
                        infer["reuse"] = np.asarray(reusable)
                        infer["data"] = _obs(
                            [
                                self.value(i, t[i])
                                for i in range(self.n)
                                if not reusable[i]
                            ]
                        )
                    else:
                        infer["data"] = _obs(
                            [self.value(i, t[i]) for i in range(self.n)]
                        )
                    ws.send(packer.pack(infer))
                    reply = msgpack_numpy.unpackb(ws.recv())
                    action = reply["data"]["action"]  # [H, n, da]
                    assert reply["data"]["env_ids"].tolist() == list(range(self.n))
                    self.actions.append(action)
                    rewards = action.sum(axis=(0, 2)).astype(np.float32)
                    time.sleep(self.slow)
                    ended_all = [t[i] + 1 == self.episode[i] for i in range(self.n)]
                    nxt = [
                        -1.0 - i if ended_all[i] else self.value(i, t[i] + 1)
                        for i in range(self.n)
                    ]
                    envs, ended = np.arange(self.n), ended_all
                    step += 1
                    if self.lockstep_break and step == 2:
                        envs = envs[:1]
                        nxt, rewards, ended = nxt[:1], rewards[:1], ended_all[:1]
                    ws.send(
                        packer.pack(
                            {
                                "message_type": "feedback",
                                "env_indices": envs,
                                "step_ids": np.asarray(t)[: len(envs)],
                                "data": {
                                    "obs": _obs(nxt),
                                    "rewards": rewards,
                                    "terminated": np.asarray(ended),
                                    "truncated": np.zeros(len(envs), bool),
                                    "info": {},
                                },
                            }
                        )
                    )
                    t = [0 if ended_all[i] else t[i] + 1 for i in range(self.n)]
                    held = True
            except websockets.exceptions.ConnectionClosed as closed:
                self.close_reason = closed.rcvd.reason if closed.rcvd else ""


def _env(port, num_envs, **cfg):
    return PlugRLEnv(
        {"port": port, "host": "127.0.0.1", "state_keys": ["obs"], "action_dim": 2,
         "timeout": 20, **cfg},
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
    )  # fmt: skip


@pytest.fixture
def port():
    return _free_port()


def test_reset_waits_for_every_env_and_slots_follow_arrival(port):
    env = _env(port, 3)
    first = FakeClient(port, 2, base=1)
    first.start()
    while env.server._slots_taken < 2:
        time.sleep(0.01)
    second = FakeClient(port, 1, base=2)
    second.start()
    try:
        obs, _ = env.reset()
        assert obs["states"].dtype == torch.float32
        assert obs["states"].tolist() == [[100, 200], [110, 220], [200, 400]]
        assert first.metadata["lockstep"] is True
        assert first.metadata["action_horizon"] == 1
        assert "reuse-feedback-obs" in first.metadata["features"]
    finally:
        env.close()
    first.join(5)
    second.join(5)
    assert first.close_reason == second.close_reason == SERVER_STOP_REASON


def test_chunk_step_sends_each_slot_its_actions_time_major(port):
    env = _env(port, 3)
    a, b = FakeClient(port, 2, base=1), FakeClient(port, 1, base=2)
    a.start()
    while env.server._slots_taken < 2:
        time.sleep(0.01)
    b.start()
    try:
        env.reset()
        actions = torch.tensor(
            [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        )  # [N, H=1, d]
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(actions)
        assert a.actions[0].shape == (1, 2, 2)  # [H, n, da]
        assert a.actions[0][0].tolist() == [[1, 2], [3, 4]]
        assert b.actions[0][0].tolist() == [[5, 6]]
        assert rewards.tolist() == [[3.0], [7.0], [11.0]]  # each client's sum
        assert not terms.any() and not truncs.any()
        assert obs_list[-1]["states"][:, 0].tolist() == [101, 111, 201]
        assert "final_observation" not in infos_list[-1]
        assert infos_list[-1]["episode"]["episode_len"].tolist() == [1, 1, 1]
    finally:
        env.close()


def test_an_ended_episode_reports_its_terminal_observation_and_statistics(port):
    env = _env(port, 2)
    client = FakeClient(port, 2, episode=[2, 3])
    client.start()
    try:
        env.reset()
        ones = torch.ones((2, 1, 2))
        env.chunk_step(ones)  # rewards 2, 2
        obs_list, rewards, terms, _, infos_list = env.chunk_step(ones * 2)  # 4, 4
        infos = infos_list[-1]
        assert terms[:, -1].tolist() == [True, False]
        # Env 0 ended: its terminal observation, then its reset one.
        assert infos["_final_observation"].tolist() == [True, False]
        assert infos["final_observation"]["states"][0, 0].item() == -1.0
        assert obs_list[-1]["states"][0, 0].item() == 0.0  # t = 0 again
        assert obs_list[-1]["states"][1, 0].item() == 12.0  # env 1 at t = 2
        episode = infos["final_info"]["episode"]
        assert episode["return"][0].item() == 6.0
        assert episode["episode_len"][0].item() == 2
        # The next step starts env 0's new episode from zero.
        _, _, _, _, infos_list = env.chunk_step(ones)
        assert infos_list[-1]["episode"]["episode_len"].tolist() == [1, 3]
    finally:
        env.close()


def test_the_episode_log_counts_each_episode_from_the_clients_rewards(port, tmp_path):
    log = tmp_path / "episodes.csv"
    env = _env(port, 2, episode_log=str(log))
    client = FakeClient(port, 2, episode=[2, 3])
    client.start()
    try:
        env.reset()
        for k in range(1, 7):  # each env's reward on step k is 2k
            env.chunk_step(torch.full((2, 1, 2), float(k)))
    finally:
        env.close()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,slot,episode,return,length"
    rows = [
        (int(s), int(slot), int(e), float(r), int(n))
        for s, slot, e, r, n in (line.split(",") for line in lines[1:])
    ]
    assert rows == [
        (2, 0, 1, 2 + 4, 2),
        (3, 1, 1, 2 + 4 + 6, 3),
        (4, 0, 2, 6 + 8, 2),
        (6, 0, 3, 10 + 12, 2),
        (6, 1, 2, 8 + 10 + 12, 3),
    ]


def test_the_episode_log_path_can_come_from_the_environment(
    port, tmp_path, monkeypatch
):
    from plugrl_bridges.server import EPISODE_LOG_ENV

    log = tmp_path / "from-env.csv"
    monkeypatch.setenv(EPISODE_LOG_ENV, str(log))
    env = _env(port, 1)
    client = FakeClient(port, 1, episode=[1])
    client.start()
    try:
        env.reset()
        env.chunk_step(torch.ones((1, 1, 2)))
    finally:
        env.close()
    assert log.read_text(encoding="utf-8").splitlines()[1:] == ["1,0,1,2.000000,1"]


def test_a_client_that_reuses_observations_gets_the_same_observations(port):
    env = _env(port, 2)
    client = FakeClient(port, 2, episode=[2, 5], reuse=True)
    client.start()
    try:
        env.reset()
        seen = [
            env.chunk_step(torch.zeros((2, 1, 2)))[0][-1]["states"][:, 0].tolist()
            for _ in range(3)
        ]
        assert seen == [[1, 11], [0, 12], [1, 13]]
    finally:
        env.close()


def test_a_client_breaking_lockstep_is_closed_and_the_run_stops(port):
    env = _env(port, 2)
    client = FakeClient(port, 2, lockstep_break=True)
    client.start()
    try:
        env.reset()
        env.chunk_step(torch.zeros((2, 1, 2)))
        with pytest.raises(ClientLost, match="lockstep"):
            env.chunk_step(torch.zeros((2, 1, 2)))
    finally:
        env.close()
    client.join(5)
    assert client.close_reason == SERVER_RESYNC_REASON


def test_a_client_asking_for_more_envs_than_are_left_is_refused(port):
    env = _env(port, 2)
    greedy = FakeClient(port, 3)
    greedy.start()
    greedy.join(10)
    try:
        # Stop, not resync: a resync would have it reconnect at once, forever.
        assert greedy.close_reason == SERVER_STOP_REASON
        assert env.server._slots_taken == 0
        fits = FakeClient(port, 2)
        fits.start()
        obs, _ = env.reset()
        assert obs["states"].shape == (2, 2)
    finally:
        env.close()


def test_a_client_too_many_mid_run_is_refused_and_the_run_goes_on(port):
    """Found on guangzhao: a client left over from an earlier run connected
    once every slot was taken, and its refusal stopped the run."""
    env = _env(port, 2)
    client = FakeClient(port, 2, slow=1.0)  # so the extra arrives mid-step
    client.start()
    extra = FakeClient(port, 1)
    try:
        env.reset()
        threading.Timer(0.3, extra.start).start()
        obs_list, *_ = env.chunk_step(torch.zeros((2, 1, 2)))
        assert obs_list[-1]["states"][:, 0].tolist() == [1, 11]
        extra.join(10)
        assert extra.close_reason == SERVER_STOP_REASON
        obs_list, *_ = env.chunk_step(torch.zeros((2, 1, 2)))
        assert obs_list[-1]["states"][:, 0].tolist() == [2, 12]
    finally:
        env.close()


def test_only_a_connection_holding_slots_counts_as_lost(port):
    """The rule behind the test above, without its timing: a connection that
    was refused, or closed, before it took a slot stops nothing."""
    from plugrl_bridges.server import BridgeServer, _Connection

    server = BridgeServer(port=port, num_envs=2, action_horizon=1, action_dim=2)
    refused, holding = _Connection(None, 1), _Connection(None, 2)
    refused.closed = "protocol error: this bridge steps 2 environments and has 0 left"
    holding.envs, holding.slots = [0, 1], [0, 1]
    server._connections = [refused, holding]
    assert server._lost() == []
    holding.closed = "closed by the client"
    assert server._lost() == [holding]


def test_auto_reset_is_required(port):
    with pytest.raises(ValueError, match="auto_reset"):
        _env(port, 1, auto_reset=False)


def test_info_splits_as_plugrl_server_splits_it():
    """Found by the real env client: its info has leaves that are not per env."""
    from plugrl_bridges.server import ProtocolError, _split_info

    info = {
        "episode": {"r": np.asarray([1.0, 2.0]), "l": np.asarray([3, 4])},
        "_episode": np.asarray([True, False]),
        "task": "pick",  # the same for every env
    }
    first, second = _split_info(info, 2)
    assert first["episode"]["r"] == 1.0 and second["episode"]["l"] == 4
    assert first["task"] == second["task"] == "pick"
    # SPEC.md section 5.4: with no array to split by, two envs cannot share one.
    with pytest.raises(ProtocolError, match="cannot be split"):
        _split_info({"task": "pick"}, 2)
    assert _split_info({"task": "pick"}, 1) == [{"task": "pick"}]
