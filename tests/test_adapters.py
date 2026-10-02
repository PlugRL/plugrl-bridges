"""The gymnasium and Stable-Baselines3 adapters, against fake clients that speak the real protocol."""

from __future__ import annotations

import socket

import gymnasium as gym
import numpy as np
import pytest
from test_bridge import FakeClient

from plugrl_protocol.websocket_protocol import SERVER_STOP_REASON


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


ACTIONS = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)


@pytest.fixture
def port():
    return _free_port()


def _connect_then(port, n, make, **client_kwargs):
    """Start a fake client, then the adapter, which blocks until it connects."""
    client = FakeClient(port, n, **client_kwargs)
    # The adapter's server must be listening first; FakeClient retries are
    # not built in, so start it just after the server comes up.
    import threading

    holder = {}

    def build():
        holder["env"] = make()

    t = threading.Thread(target=build)
    t.start()
    import time

    for _ in range(200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    client.start()
    t.join(20)
    return holder["env"], client


def test_gym_vector_env_reads_its_spaces_off_the_clients_and_steps(port):
    from plugrl_bridges.gym import PlugRLVectorEnv

    env, client = _connect_then(
        port, 2,
        lambda: PlugRLVectorEnv(num_envs=2, action_space=ACTIONS, port=port, host="127.0.0.1", timeout=20),
        episode=[2, 3],
    )  # fmt: skip
    try:
        assert env.single_observation_space.shape == (2,)
        assert env.observation_space.shape == (2, 2)
        assert env.action_space.shape == (2, 2)
        obs, _ = env.reset()
        assert obs[:, 0].tolist() == [0, 10]
        obs, rewards, term, trunc, infos = env.step(np.ones((2, 2)))
        assert rewards.tolist() == [2.0, 2.0] and "final_obs" not in infos
        obs, rewards, term, trunc, infos = env.step(np.ones((2, 2)))
        # Env 0 ended: SAME_STEP autoreset, the terminal observation in infos.
        assert term.tolist() == [True, False]
        assert infos["_final_obs"].tolist() == [True, False]
        assert infos["final_obs"][0].tolist() == [-1.0, -2.0]
        assert infos["final_obs"][1] is None
        assert obs[:, 0].tolist() == [0, 12]
        # Its own episode statistics, from the raw rewards: 2 + 2 over 2 steps.
        assert infos["_episode"].tolist() == [True, False]
        assert infos["episode"]["r"][0] == 4.0 and infos["episode"]["l"][0] == 2
        # The next episode counts from its first step (gymnasium's own vector
        # RecordEpisodeStatistics would drop it under SAME_STEP).
        for _ in range(2):
            obs, rewards, term, trunc, infos = env.step(np.ones((2, 2)))
        assert infos["_episode"].tolist() == [True, False]
        assert infos["episode"]["r"][0] == 4.0 and infos["episode"]["l"][0] == 2
    finally:
        env.close()
    client.join(5)
    assert client.close_reason == SERVER_STOP_REASON


def test_normalize_observation_normalizes_final_obs_with_the_same_statistics(port):
    """gymnasium's vector observation wrappers refuse SAME_STEP; ours handles it."""
    from plugrl_bridges.gym import NormalizeObservation, PlugRLVectorEnv

    env, client = _connect_then(
        port, 2,
        lambda: PlugRLVectorEnv(num_envs=2, action_space=ACTIONS, port=port, host="127.0.0.1", timeout=20),
        episode=[2, 3],
    )  # fmt: skip
    wrapped = NormalizeObservation(env)
    try:
        wrapped.reset()
        wrapped.step(np.ones((2, 2)))
        rms = wrapped.obs_rms
        mean, var = rms.mean.copy(), rms.var.copy()
        obs, _, term, _, infos = wrapped.step(np.ones((2, 2)))
        assert term.tolist() == [True, False]
        # The same update that normalized obs normalized the final one.
        mean, var = wrapped.obs_rms.mean, wrapped.obs_rms.var
        expected = (np.asarray([-1.0, -2.0]) - mean) / np.sqrt(var + 1e-8)
        np.testing.assert_allclose(infos["final_obs"][0], expected, rtol=1e-5)
        assert infos["final_obs"][1] is None
    finally:
        wrapped.close()


def test_sb3_vec_env_reports_episodes_as_sb3_expects(port):
    from plugrl_bridges.sb3 import PlugRLVecEnv

    env, client = _connect_then(
        port, 2,
        lambda: PlugRLVecEnv(num_envs=2, action_space=ACTIONS, port=port, host="127.0.0.1", timeout=20),
        episode=[2, 3],
    )  # fmt: skip
    try:
        assert env.observation_space.shape == (2,)
        assert env.reset()[:, 0].tolist() == [0, 10]
        env.step(np.ones((2, 2)))
        obs, rewards, dones, infos = env.step(np.ones((2, 2)))
        assert dones.tolist() == [True, False]
        assert infos[0]["terminal_observation"].tolist() == [-1.0, -2.0]
        assert infos[0]["TimeLimit.truncated"] is False  # it terminated
        assert infos[0]["episode"]["r"] == 4.0 and infos[0]["episode"]["l"] == 2
        assert "episode" not in infos[1]
        assert obs[:, 0].tolist() == [0, 12]
    finally:
        env.close()


def test_sb3_ppo_learns_on_it_without_complaint(port):
    """Not whether it learns (that is an experiment), but that SB3 runs on it."""
    from stable_baselines3 import PPO

    from plugrl_bridges.sb3 import PlugRLVecEnv

    env, client = _connect_then(
        port, 2,
        lambda: PlugRLVecEnv(num_envs=2, action_space=ACTIONS, port=port, host="127.0.0.1", timeout=20),
        episode=[5, 7],
    )  # fmt: skip
    try:
        model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1, verbose=0)
        model.learn(total_timesteps=64)
        assert model.num_timesteps >= 64
        assert len(model.ep_info_buffer) > 0  # SB3 saw the episodes end
    finally:
        env.close()
