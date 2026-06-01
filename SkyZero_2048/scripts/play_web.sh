#!/usr/bin/env bash
# Launch the 2048 web demo: the trained Stochastic Gumbel AlphaZero agent plays
# 2048 live in the browser (VSCode port-forward friendly).
#
# Env overrides: CKPT, PORT, SIMS, DEVICE, PY.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

# PY comes from env_paths.cfg(.local) — the conda env that has torch.
source "$SCRIPT_DIR/env_paths.cfg"

CKPT="${CKPT:-$ROOT/data2048_nbt/models/latest.pt}"   # active TorchScript mirror
PORT="${PORT:-8858}"
SIMS="${SIMS:-128}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$CKPT" ]]; then
    echo "[play_web.sh] no checkpoint at $CKPT — the server will fall back to an"
    echo "              untrained stub policy. Train first: CONFIG_DIR=configs/baseline bash scripts/run.sh"
fi

cd "$ROOT/python"
exec "$PY" play_web.py --ckpt "$CKPT" --port "$PORT" --sims "$SIMS" --device "$DEVICE" "$@"
