"""Smoke test: a trainer's example trains two real env clients on the probe env, briefly.

    python smoke_probe.py --trainer cleanrl|sb3 --client-python PATH/TO/env-client/python

Not a learning test: it checks that the trainer runs end to end on the
bridge with real clients, logs their episodes, and that the clients exit
when it closes.
"""

from __future__ import annotations

import argparse
import pathlib
import socket
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trainer_command(trainer: str, port: int, log_dir: str) -> list[str]:
    common = ["--port", str(port), "--host", "127.0.0.1", "--action-dim", "3"]
    if trainer == "cleanrl":
        return [sys.executable, str(HERE / "cleanrl" / "ppo_continuous_action.py"),
                "--num-envs", "2", "--num-steps", "8", "--num-minibatches", "2",
                "--update-epochs", "1", "--total-timesteps", "64", "--no-cuda",
                "--state-keys", "t", "a", *common]  # fmt: skip
    return [sys.executable, str(HERE / "sb3" / "ppo.py"), "--num-envs", "2",
            "--n-steps", "16", "--total-timesteps", "64", "--state-keys", "t", "a",
            "--log-dir", log_dir, *common]  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer", choices=("cleanrl", "sb3"), required=True)
    p.add_argument("--client-python", required=True)
    a = p.parse_args()
    port = free_port()
    log_dir = tempfile.mkdtemp(prefix="plugrl-smoke-")
    trainer = subprocess.Popen(
        trainer_command(a.trainer, port, log_dir),
        cwd=log_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    client = subprocess.Popen(
        [a.client_python, "-m", "plugrl_env_client.cli", "probe-v1",
         "--server-host", "127.0.0.1", "--server-port", str(port),
         "--num-envs", "2", "--num-episodes", "1000000", "--runner.replan-steps", "1"],
        cwd=log_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    out, _ = trainer.communicate(timeout=300)
    try:
        code = client.wait(timeout=60)
    except subprocess.TimeoutExpired:
        client.kill()
        code = "killed"
    if a.trainer == "cleanrl":
        episodes = sum("episodic_return" in line for line in out.splitlines())
    else:
        monitor = pathlib.Path(log_dir) / "monitor.csv"
        episodes = (
            max(0, len(monitor.read_text().splitlines()) - 2) if monitor.exists() else 0
        )
    print(
        f"{a.trainer}: trainer exit {trainer.returncode}, client exit {code}, episodes logged {episodes}"
    )
    if trainer.returncode != 0:
        print("\n".join(out.splitlines()[-8:]))
    return 0 if trainer.returncode == 0 and code == 0 and episodes else 1


if __name__ == "__main__":
    sys.exit(main())
