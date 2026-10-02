#!/usr/bin/env bash
# RLinf trains HalfCheetah through the bridge, with RLinf on guangzhao.
#
#   bash run_guangzhao.sh OUT_DIR SEED CLIENTS [MAX_EPOCHS]
#
# CLIENTS=local starts 16 plugrl-env-client processes here, outside RLinf's
# venv and Ray cluster. CLIENTS=remote starts none: the clients connect from
# another machine (cross_clients.py on the laptop). SEED seeds RLinf's actor
# and, for local clients, the env clients. MAX_EPOCHS overrides the config's 100.
set -uo pipefail
OUT="$1"
SEED="$2"
CLIENTS="$3"
MAX_EPOCHS="${4:-}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
RLINF="$BASE/rlinf-src"
CLIENT_PY="$BASE/plugrl-env-client/.venv/bin/python"
CLIENT_PATH="$BASE/e46/client-src:$BASE/e46/protocol-src"
PORT=8800
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"
# A client left over from an earlier run would take a slot of this one. Its
# worker processes are multiprocessing.spawn children: match the interpreter.
if pgrep -u "$USER" -f "[p]lugrl-env-client/.venv/bin/python" >/dev/null; then
  echo "env clients are still running here; stop them first" >&2
  exit 1
fi
date '+start %F %T' | tee "$OUT/run.txt"
echo "seed $SEED clients $CLIENTS max_epochs ${MAX_EPOCHS:-config}" | tee -a "$OUT/run.txt"

overrides="runner.logger.log_path=$OUT/rlinf-logs actor.seed=$SEED"
[ -n "$MAX_EPOCHS" ] && overrides="$overrides runner.max_epochs=$MAX_EPOCHS"
# What run_embodiment.sh sets, without its fixed argument list.
(cd "$RLINF" && source .venv/bin/activate && \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$RLINF" \
  exec setsid python examples/embodiment/train_embodied_agent.py \
    --config-path "$RLINF/examples/embodiment/config/" \
    --config-name plugrl_halfcheetah_ppo_mlp $overrides \
  > "$OUT/rlinf.log" 2>&1) &
RLINF_PID=$!

CLIENT_PID=""
if [ "$CLIENTS" = local ]; then
  # One env per process: 16 processes, 16 connections, in their own process
  # group so that stopping them stops all of them.
  (cd "$OUT" && OMP_NUM_THREADS=1 PYTHONPATH="$CLIENT_PATH" exec setsid "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" --num-procs 16 --num-envs 1 \
      --num-episodes 1000000 --runner.replan-steps 1 --runner.seed "$SEED" \
      --exp-name rlinf-clients > "$OUT/clients.log" 2>&1) &
  CLIENT_PID=$!
fi

wait "$RLINF_PID"
echo "rlinf rc=$? at $(date '+%T')" | tee -a "$OUT/run.txt"
grep -E "TimeoutError|ClientLost|Traceback" "$OUT/rlinf.log" | head -3 | tee -a "$OUT/run.txt"
if [ -n "$CLIENT_PID" ]; then
  # Ray ends the env worker without closing the bridge, so the clients see a
  # dropped connection, not plugrl-server-stop, and keep trying to reconnect.
  sleep 5
  kill -- "-$CLIENT_PID" 2>/dev/null || kill "$CLIENT_PID" 2>/dev/null
  wait "$CLIENT_PID"; echo "clients stopped" | tee -a "$OUT/run.txt"
fi
cd "$RLINF" && .venv/bin/ray stop --force >/dev/null 2>&1
date '+end   %F %T' | tee -a "$OUT/run.txt"
echo RUN_DONE | tee -a "$OUT/run.txt"
