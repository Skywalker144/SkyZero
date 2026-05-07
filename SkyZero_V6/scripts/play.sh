#!/usr/bin/env bash
# Launch interactive human-vs-AI Gomoku. Auto-loads the latest exported model.
#
# Env overrides: MODEL, PLAY_BIN, PLAY_CFG, DATA_DIR.
# Extra args are forwarded to the binary (e.g. --num-simulations 1600).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
source "$SCRIPT_DIR/paths.cfg"

MODEL="${MODEL:-$DATA_DIR/models/latest.pt}"
PLAY_BIN="${PLAY_BIN:-$ROOT/cpp/build/gomoku_play}"
PLAY_CFG="${PLAY_CFG:-$SCRIPT_DIR/play.cfg}"

[[ -f "$MODEL" ]]    || { echo "no model at $MODEL"; exit 1; }
[[ -f "$PLAY_CFG" ]] || { echo "no config at $PLAY_CFG"; exit 1; }
[[ -x "$PLAY_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_play"; exit 1; }

exec "$PLAY_BIN" --model "$MODEL" --config "$PLAY_CFG" "$@"
