#!/usr/bin/env bash
# Kill env-client worker processes left over from earlier runs: this user's
# multiprocessing.spawn children that hold a socket to the bridge's port.
set -uo pipefail
PORT="${1:-8800}"
pids=$(ss -tanp "( dport = :$PORT )" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
killed=0
for pid in $pids; do
  [ "$(stat -c %U /proc/$pid 2>/dev/null)" = "$USER" ] || continue
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'multiprocessing.spawn\|plugrl_env_client'; then
    kill "$pid" && killed=$((killed + 1))
  fi
done
sleep 2
echo "killed $killed; sockets to $PORT left: $(ss -tan "( dport = :$PORT )" | grep -c ESTAB)"
