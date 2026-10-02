#!/usr/bin/env bash
# Point an RLinf checkout at plugrl-bridges: install it into RLinf's venv,
# patch env_type plugrl in (or move an older patch that imported
# plugrl_rlinf over to plugrl_bridges.rlinf), copy the example configs, and
# check that RLinf resolves the environment class.
#
#   bash install_guangzhao.sh [RLINF_DIR] [BRIDGES_DIR]
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
BASE="$HOME/zuogou/plugrl"
RLINF="${1:-$BASE/rlinf-src}"
BRIDGES="${2:-$BASE/plugrl-bridges}"
PY="$RLINF/.venv/bin/python"
F="$RLINF/rlinf/envs/__init__.py"

if grep -q "plugrl_bridges.rlinf" "$F"; then
  echo "already patched for plugrl-bridges"
elif grep -q "from plugrl_rlinf.env import PlugRLEnv" "$F"; then
  sed -i 's/from plugrl_rlinf.env import PlugRLEnv/from plugrl_bridges.rlinf import PlugRLEnv/' "$F"
  echo "moved the plugrl_rlinf patch over to plugrl_bridges.rlinf"
else
  (cd "$RLINF" && patch -p1 < "$BRIDGES/patches/rlinf-plugrl-env-type.patch")
fi
uv pip install --python "$PY" -e "$BRIDGES"
cp "$BRIDGES"/examples/rlinf/config/plugrl_*_ppo_mlp.yaml "$RLINF/examples/embodiment/config/"
cp "$BRIDGES"/examples/rlinf/config/env/plugrl_*.yaml "$RLINF/examples/embodiment/config/env/"
# RLinf is not installed as a package; its launcher puts the checkout on
# PYTHONPATH, so the check does too.
(cd "$RLINF" && PYTHONPATH="$RLINF" "$PY" - <<'EOF'
from rlinf.envs import SupportedEnvType, get_env_cls
print("env cls", get_env_cls(SupportedEnvType("plugrl").value))
EOF
)
