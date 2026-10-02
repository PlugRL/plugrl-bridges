"""Drive real plugrl-env-client processes through PlugRLEnv, as RLinf would, and check every step.

Two env clients run SPEC.md section 8.1's probe environment: one with 3
envs, one with 2, so the bridge holds 5 slots. Env i of a client counts its
steps in `states["t"]`, keeps the action it applied last in `states["a"]`,
pays 1 per step and ends after 3 + 2 * (i % 3) steps. Every action this
script sends encodes the step and the slot, so after each chunk_step it can
check, for every slot:
- the client applied exactly the action sent to that slot;
- the step count advanced by one, or the episode ended at its length;
- an ended episode's terminal observation is in final_observation, and the
  observation returned is the reset one;
- the reward is 1, and the episode statistics are right.

    python drive_probe.py --client-python PATH/TO/env-client/python [--steps 40]
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time

import torch

from plugrl_bridges.rlinf import PlugRLEnv

LENGTHS = [
    3,
    5,
    7,
    3,
    5,
]  # slot -> episode length: client A envs 0-2, client B envs 0-1


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def client(python: str, port: int, n: int) -> subprocess.Popen:
    return subprocess.Popen(
        [python, "-m", "plugrl_env_client.cli", "probe-v1",
         "--server-host", "127.0.0.1", "--server-port", str(port),
         "--num-envs", str(n), "--num-episodes", "1000000",
         "--runner.replan-steps", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--client-python", required=True)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()

    port = free_port()
    env = PlugRLEnv(
        {"port": port, "host": "127.0.0.1", "state_keys": ["t", "a"], "action_dim": 3,
         "timeout": 120},
        num_envs=5, seed_offset=0, total_num_processes=1,
    )  # fmt: skip
    a = client(args.client_python, port, 3)
    while env.server._slots_taken < 3:  # so that client A holds slots 0-2
        time.sleep(0.05)
    b = client(args.client_python, port, 2)
    failures: list[str] = []
    ended = 0
    try:
        obs, _ = env.reset()
        t = obs["states"][:, 0].clone()
        assert t.tolist() == [0] * 5, t
        for step in range(args.steps):
            actions = torch.tensor(
                [
                    [[1000.0 * step + 10 * slot + d for d in range(3)]]
                    for slot in range(5)
                ]
            )
            obs_list, rewards, terms, truncs, infos_list = env.chunk_step(actions)
            obs, infos = obs_list[-1]["states"], infos_list[-1]
            for slot in range(5):
                sent = actions[slot, 0].tolist()
                done = bool(terms[slot, -1] or truncs[slot, -1])
                if rewards[slot, -1].item() != 1.0:
                    failures.append(
                        f"step {step} slot {slot}: reward {rewards[slot, -1]}"
                    )
                if done:
                    ended += 1
                    final = infos["final_observation"]["states"][slot]
                    if final[1:].tolist() != sent:
                        failures.append(
                            f"step {step} slot {slot}: terminal a {final[1:].tolist()} != {sent}"
                        )
                    if final[0].item() != LENGTHS[slot]:
                        failures.append(
                            f"step {step} slot {slot}: ended at t={final[0]}, length {LENGTHS[slot]}"
                        )
                    if obs[slot].tolist() != [0, 0, 0, 0]:
                        failures.append(
                            f"step {step} slot {slot}: after the end, obs {obs[slot].tolist()} is not a reset"
                        )
                    ep = infos["final_info"]["episode"]
                    if (ep["episode_len"][slot].item(), ep["return"][slot].item()) != (
                        LENGTHS[slot],
                        LENGTHS[slot],
                    ):
                        failures.append(
                            f"step {step} slot {slot}: episode stats {ep['episode_len'][slot]}, {ep['return'][slot]}"
                        )
                else:
                    if obs[slot, 1:].tolist() != sent:
                        failures.append(
                            f"step {step} slot {slot}: applied {obs[slot, 1:].tolist()}, sent {sent}"
                        )
                    if obs[slot, 0].item() != t[slot].item() + 1:
                        failures.append(
                            f"step {step} slot {slot}: t {obs[slot, 0]} after {t[slot]}"
                        )
            t = obs[:, 0].clone()
    finally:
        env.close()
        codes = []
        for proc in (a, b):
            try:
                codes.append(proc.wait(timeout=60))
            except subprocess.TimeoutExpired:
                proc.kill()  # still reconnecting after a resync
                codes.append("killed")
    reused = sum(c.cache is not None for c in env.server._connections)
    print(
        f"{args.steps} steps x 5 slots, {ended} episodes ended, client exit codes {codes}"
    )
    for f in failures[:20]:
        print("FAIL", f)
    print("no failures" if not failures else f"{len(failures)} failures")
    del reused
    return 0 if not failures and codes == [0, 0] else 1


if __name__ == "__main__":
    sys.exit(main())
