#!/usr/bin/env bash
# Launch the 2048 web demo: the trained Stochastic Gumbel AlphaZero agent plays
# 2048 live in the browser (VSCode port-forward friendly).
#
# Picks the data dir from the experiment config, the same way as run.sh:
#   CONFIG_DIR=configs/vtransform bash scripts/play_web.sh
# Env overrides: CKPT, PORT, SIMS, DEVICE, PY.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

# Resolve CONFIG_DIR like run.sh, then source paths.cfg for DATA_DIR and
# env_paths.cfg(.local) for PY (the conda env that has torch).
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[play_web.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }
source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"

CKPT="${CKPT:-$DATA_DIR/models/latest.pt}"   # active TorchScript mirror for this config
PORT="${PORT:-8868}"
SIMS="${SIMS:-128}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$CKPT" ]]; then
    echo "[play_web.sh] no checkpoint at $CKPT — the server will fall back to an"
    echo "              untrained stub policy. Train first: CONFIG_DIR=$CONFIG_DIR bash scripts/run.sh"
fi

cd "$ROOT/python"
exec "$PY" play_web.py --ckpt "$CKPT" --port "$PORT" --sims "$SIMS" --device "$DEVICE" "$@"
