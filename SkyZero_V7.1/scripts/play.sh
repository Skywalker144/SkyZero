#!/usr/bin/env bash
# Launch interactive human-vs-AI Gomoku. Auto-loads the latest exported model.
#
# Env overrides: MODEL, PLAY_BIN, PLAY_CFG, DATA_DIR.
# Extra args are forwarded to the binary (e.g. --num-simulations 1600).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[play.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }

source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"

MODEL="${MODEL:-$DATA_DIR/models/latest.pt}"
PLAY_BIN="${PLAY_BIN:-$ROOT/cpp/build/gomoku_play}"
PLAY_CFG="${PLAY_CFG:-$CONFIG_DIR/play.cfg}"
RUN_CFG="${RUN_CFG:-$CONFIG_DIR/run.cfg}"

[[ -f "$MODEL" ]]    || { echo "no model at $MODEL"; exit 1; }
[[ -f "$PLAY_CFG" ]] || { echo "no config at $PLAY_CFG"; exit 1; }
[[ -f "$RUN_CFG" ]]  || { echo "no run config at $RUN_CFG"; exit 1; }
[[ -x "$PLAY_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_play"; exit 1; }

# Source MAIN_BOARD_SIZE / MAIN_RULE from run.cfg (with .local overlay).
set -a
# shellcheck disable=SC1090
source "$RUN_CFG"
[[ -f "$RUN_CFG.local" ]] && source "$RUN_CFG.local"
set +a

exec "$PLAY_BIN" --model "$MODEL" --config "$PLAY_CFG" \
    --board-size "$MAIN_BOARD_SIZE" --rule "$MAIN_RULE" "$@"
