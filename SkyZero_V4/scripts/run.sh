#!/usr/bin/env bash
# Main orchestration loop: selfplay -> shuffle -> train -> export, repeat.
#
# Usage: bash scripts/run.sh [max_iters]
#   If max_iters is omitted the loop runs until Ctrl+C.
set -euo pipefail

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

while true; do
    echo ""
    echo "=================================================================="
    echo "[run.sh] === iter $iter ==="
    date

    # (1) compute games for this iter
    GAMES=$( cd "$ROOT/python" && "$PY" compute_games.py --data-dir "$DATA_DIR" )

    # (2) selfplay (C++)
    bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"

    # (3) shuffle
    bash "$SCRIPT_DIR/shuffle.sh"

    # (3a) gate
    if ! ( cd "$ROOT/python" && "$PY" wait_for_data.py --data-dir "$DATA_DIR" ); then
        echo "[run.sh] not enough shuffled data yet; skipping train this iter"
    else
        # (4) train
        bash "$SCRIPT_DIR/train.sh" "$iter"

        # (5) export TorchScript
        bash "$SCRIPT_DIR/export.sh" "$iter"

        # (6) plot loss curve
        ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )
    fi

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[run.sh] reached max_iters=$max_iters; stopping."
        break
    fi
done
