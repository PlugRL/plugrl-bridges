#!/usr/bin/env bash
# One bridged training run on guangzhao: a trainer that was not written for
# PlugRL, and 16 env clients on the same machine, one env each.
#
#   bash run_bridged.sh TRAINER ENV OUT_DIR SEED [STEPS]
#
# TRAINER  rlinf | sb3 | cleanrl
# ENV      halfcheetah  plugrl-env-client's mujoco-v1 (Python), 16 processes
#          pendulum     plugrl-server E44's pendulum_client (C++, no
#                       third-party libraries), 16 processes
# STEPS    total environment steps; defaults: 409,600 (halfcheetah), 102,400
#          (pendulum), E43's and E44's budgets
#
# Paths are guangzhao's, set below. The clients start in a process group of
# their own and are stopped with it, whatever the trainer does at its end.
# Whatever the trainer, the bridge writes every episode's return to
# OUT_DIR/episodes.csv.
set -uo pipefail
TRAINER="$1"
ENV="$2"
OUT="$3"
SEED="$4"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
BRIDGES="$BASE/plugrl-bridges"
RLINF="$BASE/rlinf-src"
TRAIN_PY="$BASE/bridges-venv/bin/python"      # sb3, cleanrl: torch (CPU), SB3, tyro
CLIENT_PY="$BASE/plugrl-env-client/.venv/bin/python"
CLIENT_PATH="$BASE/e46/client-src:$BASE/e46/protocol-src"
CPP_CLIENT="$BASE/e44-build/pendulum_client"
PORT="${PORT:-8810}"
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
export PLUGRL_BRIDGES_EPISODE_LOG="$OUT/episodes.csv"
mkdir -p "$OUT"

case "$ENV" in
  halfcheetah) STEPS="${5:-409600}"; ACTION_DIM=6; LOW=-1; HIGH=1 ;;
  pendulum)    STEPS="${5:-102400}"; ACTION_DIM=1; LOW=-2; HIGH=2 ;;
  *) echo "ENV must be halfcheetah or pendulum" >&2; exit 2 ;;
esac
EPOCHS=$((STEPS / 4096))   # 16 envs x 256 steps a rollout, for every trainer

if pgrep -u "$USER" -f "[p]lugrl-env-client/.venv/bin/python|[p]endulum_client " >/dev/null; then
  echo "env clients are still running here; stop them first" >&2
  exit 1
fi
{ date '+start %F %T'; echo "trainer $TRAINER env $ENV seed $SEED steps $STEPS port $PORT"; } | tee "$OUT/run.txt"

case "$TRAINER" in
  rlinf)
    cfg="plugrl_${ENV}_ppo_mlp"
    (cd "$RLINF" && source .venv/bin/activate && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="$RLINF" \
      exec setsid python examples/embodiment/train_embodied_agent.py \
        --config-path "$RLINF/examples/embodiment/config/" --config-name "$cfg" \
        runner.logger.log_path="$OUT/trainer-logs" actor.seed="$SEED" runner.max_epochs="$EPOCHS" \
        env.train.port="$PORT" env.train.episode_log="$PLUGRL_BRIDGES_EPISODE_LOG" \
      > "$OUT/trainer.log" 2>&1) & ;;
  sb3)
    preset=default; [ "$ENV" = pendulum ] && preset=pendulum
    (cd "$OUT" && exec setsid "$TRAIN_PY" "$BRIDGES/examples/sb3/ppo.py" --num-envs 16 --port "$PORT" \
        --action-dim "$ACTION_DIM" --action-low "$LOW" --action-high "$HIGH" \
        --total-timesteps "$STEPS" --preset "$preset" --seed "$SEED" --log-dir "$OUT/trainer-logs" \
      > "$OUT/trainer.log" 2>&1) & ;;
  cleanrl)
    (cd "$OUT" && exec setsid "$TRAIN_PY" "$BRIDGES/examples/cleanrl/ppo_continuous_action.py" \
        --num-envs 16 --num-steps 256 --total-timesteps "$STEPS" --port "$PORT" \
        --action-dim "$ACTION_DIM" --action-low "$LOW" --action-high "$HIGH" --seed "$SEED" --no-cuda \
      > "$OUT/trainer.log" 2>&1) & ;;
  *) echo "TRAINER must be rlinf, sb3 or cleanrl" >&2; exit 2 ;;
esac
TRAINER_PID=$!

if [ "$ENV" = halfcheetah ]; then
  (cd "$OUT" && PYTHONPATH="$CLIENT_PATH" exec setsid "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" --num-procs 16 --num-envs 1 \
      --num-episodes 1000000 --runner.replan-steps 1 --runner.seed "$SEED" \
      --exp-name clients > "$OUT/clients.log" 2>&1) &
else
  # 16 C++ processes, seeds SEED*100+1 ... +16, one process group.
  (cd "$OUT" && exec setsid bash -c "for i in \$(seq 16); do '$CPP_CLIENT' 127.0.0.1 $PORT \$(( $SEED * 100 + i )) > cpp-client-\$i.log 2>&1 & done; wait") &
fi
CLIENT_PID=$!

wait "$TRAINER_PID"
echo "trainer rc=$? at $(date '+%T')" | tee -a "$OUT/run.txt"
grep -E "Traceback|ClientLost|TimeoutError" "$OUT/trainer.log" | head -3 | tee -a "$OUT/run.txt"
sleep 5
kill -- "-$CLIENT_PID" 2>/dev/null
wait "$CLIENT_PID" 2>/dev/null
echo "clients stopped" | tee -a "$OUT/run.txt"
[ "$TRAINER" = rlinf ] && (cd "$RLINF" && .venv/bin/ray stop --force >/dev/null 2>&1)
{ date '+end   %F %T'; echo RUN_DONE; } | tee -a "$OUT/run.txt"
