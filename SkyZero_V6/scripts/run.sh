#!/usr/bin/env bash
# Main orchestration loop: selfplay -> shuffle -> {train+export per slot} ->
# promote active slot, repeat.
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

# Refresh sidecar JSON consumed by run_selfplay_daemon.sh / selfplay_main --daemon
# so the multi-GPU daemon picks up per-iter NUM_SIMULATIONS / CHEAP_SEARCH_VISITS
# warmup values. Daemon polls mtime; on change it exits rc=99 and the watchdog
# restarts it with the new values. selfplay.sh (main GPU) reads warmup separately.
write_warmup_sidecar(){
    ( cd "$ROOT/python" && "$PY" warmup.py write-sidecar --data-dir "$DATA_DIR" )
}

# Validate slot config (assert first activate=0, strictly increasing, lengths
# match across MODEL_SLOTS / MODEL_BLOCKS / MODEL_CHANNELS / MODEL_ACTIVATE_SAMPLES).
"$PY" "$ROOT/python/slots.py" validate
mapfile -t SLOT_NAMES < <( "$PY" "$ROOT/python/slots.py" list-slots )
echo "[run.sh] slots: ${SLOT_NAMES[*]}"

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

# Cold-start: write the warmup sidecar before launching the daemon so the
# daemon's startup read picks up the right (resume- or first-iter-)values
# instead of falling back to cfg defaults.
write_warmup_sidecar

# Multi-GPU: launch the selfplay daemon on GPUs 1..GPU_NUM-1 in the background.
if [[ "$GPU_NUM" -gt 1 ]]; then
    echo "[run.sh] starting selfplay daemon on GPUs 1..$((GPU_NUM-1))"
    bash "$SCRIPT_DIR/run_selfplay_daemon.sh" &
    DAEMON_PID=$!
fi

# Resume iter from data/checkpoints/state.json if present. State is written by
# this script at the end of each fully-completed iter (after every slot has
# been trained, exported, and the active slot promoted), so resume always
# picks up at a clean iter boundary.
iter=0
if [[ -f "$DATA_DIR/checkpoints/state.json" ]]; then
    iter=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['iter'])" \
        "$DATA_DIR/checkpoints/state.json")
    iter=$((iter + 1))
fi

# First-time init: random-init every slot and seed models/latest.pt.
if [[ ! -f "$DATA_DIR/models/latest.pt" ]]; then
    echo "[run.sh] bootstrapping random-init models for all slots"
    ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" )
fi

max_iters="${1:-}"

OVERLAP_SHUFFLE="${OVERLAP_SHUFFLE:-0}"

while true; do
    echo ""
    echo "=================================================================="
    echo "[run.sh] === iter $iter ==="
    date

    # Cumulative selfplay samples so far (main + daemon, includes pruned history).
    # Read via pool_rows.cumulative_produced — canonical, matches the values
    # warmup / compute_games / shuffle all see. (awk-summing last_run.tsv would
    # be main-GPU-only and miss the pruned history.)
    CUM_SAMPLES=$( cd "$ROOT/python" && "$PY" pool_rows.py produced --data-dir "$DATA_DIR" 2>/dev/null || echo 0 )
    echo "[run.sh] cumulative samples so far: ${CUM_SAMPLES} (main + daemon, includes pruned history)"
    ACTIVE_SLOT=$( "$PY" "$ROOT/python/slots.py" active --data-dir "$DATA_DIR" )
    echo "[run.sh] active selfplay slot for iter $iter: $ACTIVE_SLOT"

    # Refresh the warmup sidecar at iter start so the daemon (if running) sees
    # the current-iter values when it next polls.
    write_warmup_sidecar

    # (1) compute games for this iter
    TRAIN_STEPS_PER_EPOCH=$( cd "$ROOT/python" && "$PY" warmup.py train-steps --data-dir "$DATA_DIR" )
    export TRAIN_STEPS_PER_EPOCH
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
        # OVERLAP_SHUFFLE=1: bg shuffle sees iter K-1's selfplay output (snapshot
        # at process start), so its pruning is tagged with iter K-1.
        bash "$SCRIPT_DIR/shuffle.sh" "$((iter - 1))" &
        SHUFFLE_PID=$!
        bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        # (2) selfplay (C++)
        bash "$SCRIPT_DIR/selfplay.sh" "$iter" "$GAMES"
        # (3) shuffle
        bash "$SCRIPT_DIR/shuffle.sh" "$iter" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "[run.sh] shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "[run.sh] shuffle failed with code $SHUFFLE_RC"
        exit "$SHUFFLE_RC"
    else
        # (4) train + (5) export each slot sequentially on MAIN_GPU.
        for slot in "${SLOT_NAMES[@]}"; do
            echo "[run.sh] --- iter $iter slot $slot: train ---"
            bash "$SCRIPT_DIR/train.sh" "$iter" "$slot"
            echo "[run.sh] --- iter $iter slot $slot: export ---"
            bash "$SCRIPT_DIR/export.sh" "$iter" "$slot"
        done

        # (6) Promote the now-active slot to data/models/latest.pt for the next
        # selfplay round. Active slot is recomputed against the cumulative
        # samples observed at this iter's *start* (selfplay this iter has
        # already finished with the previous active slot).
        PROMOTED=$( "$PY" "$ROOT/python/slots.py" promote \
            --data-dir "$DATA_DIR" --iter "$iter" )
        echo "[run.sh] promoted slot for next iter: $PROMOTED"

        # (7) post-export diagnostic: empty-board MCTS rootValue probe on the
        # active slot's freshly-promoted model.
        "$ROOT/cpp/build/mcts_probe" \
            --model "$DATA_DIR/models/latest.pt" \
            --config "$SCRIPT_DIR/run.cfg" \
            --iter "$iter" \
            --log "$DATA_DIR/logs/probe.tsv" \
            || echo "[run.sh] mcts_probe failed (non-fatal)"

        # (8) plot loss curves (overlays all slots — see view_loss.py)
        ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )

        # (8.5) snapshot total selfplay-pool rows for next iter's
        # compute_games dead-reckoning (Plan C). Writes one row to
        # data/logs/pool_rows.tsv. Kept in sync with state.json below — the
        # two together mark a clean iter boundary on resume.
        ( cd "$ROOT/python" && "$PY" pool_rows.py snapshot --data-dir "$DATA_DIR" --iter "$iter" )

        # (9) global state.json — written only after every slot completed,
        # so resume always picks up at a clean iter boundary.
        "$PY" -c "
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'checkpoints' / 'state.json'
p.write_text(json.dumps({'iter': int(sys.argv[2])}, indent=2))
" "$DATA_DIR" "$iter"
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
