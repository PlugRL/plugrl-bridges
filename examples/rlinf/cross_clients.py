"""The env clients of a cross-machine run, on the laptop: 16 MuJoCo env client
processes connecting to the bridge on guangzhao over Tailscale.

    python cross_clients.py --client-python PY --pythonpath P --run-dir REMOTE_DIR --seed S --out DIR

It starts them, then polls the run on guangzhao over ssh every 10 s. When
the run's run.txt says RUN_DONE, it stops the clients: RLinf ends its env
worker without closing the bridge, so they would otherwise keep trying to
reconnect.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--client-python", required=True)
    p.add_argument("--pythonpath", required=True)
    p.add_argument("--host", default="100.75.226.89")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--remote", default="guangzhao-memeseeks")
    p.add_argument("--run-dir", required=True, help="the run's OUT_DIR on guangzhao")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "VIRTUAL_ENV")
    }
    env["PYTHONPATH"] = a.pythonpath
    env["OMP_NUM_THREADS"] = "1"
    log = open(out / "clients.log", "w", encoding="utf-8")
    clients = subprocess.Popen(
        [a.client_python, "-m", "plugrl_env_client.cli", "mujoco-v1",
         "--server-host", a.host, "--server-port", str(a.port),
         "--num-procs", "16", "--num-envs", "1", "--num-episodes", "1000000",
         "--runner.replan-steps", "1", "--runner.seed", str(a.seed),
         "--exp-name", "rlinf-cross-clients"],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=out,
    )  # fmt: skip
    print(f"clients pid {clients.pid} started {time.strftime('%T')}", flush=True)
    try:
        while clients.poll() is None:
            time.sleep(10)
            # -n and BatchMode: without them, ssh started from here waited on
            # stdin, and the pilot's clients outlived their run.
            try:
                done = subprocess.run(
                    ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                     a.remote, f"grep -q RUN_DONE {a.run_dir}/run.txt"],
                    capture_output=True, env=env, timeout=30, stdin=subprocess.DEVNULL,
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                continue
            if done.returncode == 0:
                print(f"run done {time.strftime('%T')}; stopping clients", flush=True)
                break
    finally:
        # The whole tree: the launcher and its 16 spawned workers.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(clients.pid)], capture_output=True
        )
        clients.wait()
        log.close()
    print(f"clients exit {clients.returncode} {time.strftime('%T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
