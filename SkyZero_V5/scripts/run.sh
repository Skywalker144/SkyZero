#!/usr/bin/env bash
# Main orchestration loop: selfplay -> shuffle -> train -> export, repeat.
#
# Usage: bash scripts/run.sh [max_iters]
#   If max_iters is omitted the loop runs until Ctrl+C.
set -euo pipefail

DAEMON_PID=""
trap 'trap - INT TERM; echo "[run.sh] interrupted; stopping."; [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

# Load config: export every assigned variable
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run.cfg"
# Server-local overrides (not in git — survives pull)
if [[ -f "$SCRIPT_DIR/run.cfg.local" ]]; then
    source "$SCRIPT_DIR/run.cfg.local"
fi
set +a

source "$SCRIPT_DIR/paths.cfg"
export DATA_DIR

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,checkpoints,logs}

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

# Auto-detect GPU count. Main loop runs on GPU 0; if >1 GPUs, spare GPUs
# (1..N-1) are owned by run_selfplay_daemon.sh, started in the background below.
if [[ -z "${GPU_NUM:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NUM=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        GPU_NUM=1
    fi
fi
GPU_NUM="${GPU_NUM:-1}"
export GPU_NUM
echo "[run.sh] detected GPU_NUM=$GPU_NUM"

# Keep C++ binaries in sync with run.cfg / sources. cmake parses MAX_BOARD_SIZE
# from run.cfg via CONFIGURE_DEPENDS and bakes it in as -DSKYZERO_MAX_BOARD_SIZE.
# This is a no-op (~<1s) when nothing has changed.
cmake --build "$ROOT/cpp/build" -j

# Multi-GPU: launch the selfplay daemon on GPUs 1..GPU_NUM-1 in the background.
if [[ "$GPU_NUM" -gt 1 ]]; then
    echo "[run.sh] starting selfplay daemon on GPUs 1..$((GPU_NUM-1))"
    bash "$SCRIPT_DIR/run_selfplay_daemon.sh" &
    DAEMON_PID=$!
fi

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

    # Cumulative selfplay totals so far (through previous iter)
    awk 'NR>1 {g+=$2; r+=$3} END {printf "[run.sh] cumulative so far: games=%d samples=%d\n", g+0, r+0}' \
        "$DATA_DIR/logs/last_run.tsv" 2>/dev/null \
        || echo "[run.sh] cumulative so far: games=0 samples=0"

    # (1) compute games for this iter
    GAMES=$( cd "$ROOT/python" && "$PY" compute_games.py --data-dir "$DATA_DIR" )

    # SHUFFLE_RC=0 success, =2 skipped (N < MIN_ROWS), other = failure.
    SHUFFLE_RC=0
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
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        # (2) selfplay (C++)
        bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"
        # (3) shuffle
        bash "$SCRIPT_DIR/shuffle.sh" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "[run.sh] shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "[run.sh] shuffle failed with code $SHUFFLE_RC"
        exit "$SHUFFLE_RC"
    else
        # (4) train
        bash "$SCRIPT_DIR/train.sh" "$iter"

        # (5) export TorchScript
        bash "$SCRIPT_DIR/export.sh" "$iter"

        # (5b) post-export diagnostic: empty-board MCTS rootValue probe
        "$ROOT/cpp/build/mcts_probe" \
            --model "$DATA_DIR/models/latest.pt" \
            --config "$SCRIPT_DIR/run.cfg" \
            --iter "$iter" \
            --log "$DATA_DIR/logs/probe.tsv" \
            || echo "[run.sh] mcts_probe failed (non-fatal)"

        # (6) plot loss curve
        ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )
    fi

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[run.sh] reached max_iters=$max_iters; stopping."
        break
    fi
done

if [[ -n "$DAEMON_PID" ]]; then
    echo "[run.sh] stopping selfplay daemon (pid=$DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
fi
