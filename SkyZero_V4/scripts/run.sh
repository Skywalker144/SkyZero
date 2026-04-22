#!/usr/bin/env bash
# Main orchestration loop: selfplay -> shuffle -> train -> export, repeat.
#
# Usage: bash scripts/run.sh [max_iters]
#   If max_iters is omitted the loop runs until Ctrl+C.
set -euo pipefail

trap 'echo "[run.sh] interrupted by signal; stopping."; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

# Load config: export every assigned variable
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run.cfg"
set +a

DATA_DIR="${DATA_DIR:-$ROOT/data}"
export DATA_DIR

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,checkpoints,logs}

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

# Resume iter from data/checkpoints/state.json if present
iter=0
if [[ -f "$DATA_DIR/checkpoints/state.json" ]]; then
    iter=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['iter'])" \
        "$DATA_DIR/checkpoints/state.json")
    iter=$((iter + 1))
fi

# First-time init: need a TorchScript model for C++ to load
if [[ ! -f "$DATA_DIR/models/latest.pt" ]]; then
    echo "[run.sh] bootstrapping random-init model"
    ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" )
fi

max_iters="${1:-}"

OVERLAP_SHUFFLE="${OVERLAP_SHUFFLE:-0}"

while true; do
    echo ""
    echo "=================================================================="
    echo "[run.sh] === iter $iter ==="
    date

    # (1) compute games for this iter
    GAMES=$( cd "$ROOT/python" && "$PY" compute_games.py --data-dir "$DATA_DIR" )

    if [[ "$OVERLAP_SHUFFLE" == "1" && "$iter" -gt 0 ]]; then
        # Pipelined mode: shuffle (CPU) runs concurrently with selfplay (GPU).
        # The shuffle sees data collected up through the previous iter; the
        # training step that follows therefore trains on a 1-iter-lagged
        # window. Selfplay's new iter-N files are written but ignored by this
        # shuffle (shuffle.py snapshots the file list once at start).
        echo "[run.sh] shuffle (bg) || selfplay (fg)"
        bash "$SCRIPT_DIR/shuffle.sh" &
        SHUFFLE_PID=$!
        bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"
        wait "$SHUFFLE_PID"
    else
        # (2) selfplay (C++)
        bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"
        # (3) shuffle
        bash "$SCRIPT_DIR/shuffle.sh"
    fi

    # (4) train
    bash "$SCRIPT_DIR/train.sh" "$iter"

    # (5) export TorchScript
    bash "$SCRIPT_DIR/export.sh" "$iter"

    # (5b) post-export diagnostic: empty-board MCTS rootValue probe
    "$ROOT/cpp/build/mcts_probe" \
        --model "$DATA_DIR/models/latest.pt" \
        --config "$SCRIPT_DIR/run.cfg" \
        || echo "[run.sh] mcts_probe failed (non-fatal)"

    # (6) plot loss curve
    ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[run.sh] reached max_iters=$max_iters; stopping."
        break
    fi
done
