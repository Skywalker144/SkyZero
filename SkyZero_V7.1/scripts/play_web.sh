#!/usr/bin/env bash
# Launch the HTTP front-end for human-vs-AI play (VSCode port-forward friendly).
#
# Env overrides: MODEL, PLAY_BIN, PLAY_CFG, DATA_DIR, PYTHON, HOST, PORT.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[play_web.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }

source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"

MODEL="${MODEL:-$DATA_DIR/models/latest.pt}"
PLAY_BIN="${PLAY_BIN:-$ROOT/cpp/build/gomoku_play}"
PLAY_CFG="${PLAY_CFG:-$CONFIG_DIR/play.cfg}"
PYTHON="${PYTHON:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

[[ -f "$MODEL" ]]    || { echo "no model at $MODEL"; exit 1; }
[[ -f "$PLAY_CFG" ]] || { echo "no config at $PLAY_CFG"; exit 1; }
[[ -x "$PLAY_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_play"; exit 1; }

exec "$PYTHON" "$ROOT/python/play_web.py" \
    --model "$MODEL" --bin "$PLAY_BIN" --config "$PLAY_CFG" \
    --host "$HOST" --port "$PORT" "$@"
