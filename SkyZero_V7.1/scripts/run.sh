#!/usr/bin/env bash
# Main orchestration loop for multi-network training.
#
# Each iter:
#   1. Compute ACTIVE_NETWORK from cumulative selfplay samples + SELFPLAY_SCHEDULE.
#   2. Mirror data/nets/$ACTIVE_NETWORK/{latest.pt,latest.meta.json} -> data/models/
#      so C++ selfplay (main + daemon) keeps reading a stable path.
#   3. selfplay  -> shuffle  -> bucket (decide train_steps)
#   4. For each network in NETWORKS: train -> export (all train on the SAME
#      shuffled data this iter).
#   5. Re-mirror active network's freshly-exported weights -> data/models/.
#   6. mcts_probe + view_loss + log a row to data/logs/schedule.tsv.
#
# Usage: bash scripts/run.sh [max_iters]
#   max_iters (CLI arg): stop after this many iters.
#   MAX_TIME_SECONDS (run.cfg): stop after this many seconds of loop time.
#   If neither is set, runs until Ctrl+C.
set -euo pipefail

DAEMON_PID=""
trap 'trap - INT TERM; echo "[run.sh] interrupted; stopping."; [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

# Pick experiment config. Default = configs/baseline; override with:
#   CONFIG_DIR=configs/nsim_64 bash scripts/run.sh
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[run.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }
export CONFIG_DIR  # children (internal/*.sh) read this

# Load experiment config: run.cfg + paths.cfg (set -a exports to subprocesses).
set -a
# shellcheck disable=SC1091
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
set +a

# Load machine env (LIBTORCH/NVCC/PY). env_paths.cfg has its own .local hook.
set -a
source "$SCRIPT_DIR/env_paths.cfg"
set +a
export DATA_DIR

if [[ -z "${NETWORKS:-}" || -z "${SELFPLAY_SCHEDULE:-}" ]]; then
    echo "[run.sh] NETWORKS and SELFPLAY_SCHEDULE must be set in run.cfg" >&2
    exit 1
fi

# Networks list (supports comma and/or whitespace separators).
read -ra NET_ARR <<< "$(echo "$NETWORKS" | tr ',' ' ')"
if [[ "${#NET_ARR[@]}" -eq 0 ]]; then
    echo "[run.sh] NETWORKS parsed to empty list" >&2
    exit 1
fi
FIRST_NET="${NET_ARR[0]}"

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,logs}
for net in "${NET_ARR[@]}"; do
    mkdir -p "$DATA_DIR/nets/$net"
done

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

# Auto-detect GPU count. Main loop runs on GPU 0; if >1 GPUs, spare GPUs
# (1..N-1) are owned by internal/selfplay_daemon.sh, started in the background below.
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

# Resume iter — read EVERY network's state.json. Training within an iter is
# serial in NET_ARR order, so a mid-iter Ctrl+C can leave nets earlier in
# NET_ARR at iter=N while later ones are still at N-1. Reading only FIRST_NET
# would set resume = N+1 and permanently skip iter N for the lagging nets
# (their train.tsv gets a gap and the bucket consumption for that iter is
# wasted). Take max(per-net iter); the catch-up block below trains any
# lagging nets at iter=max_iter before the main loop resumes.
NET_ITERS=()
max_iter=-1
for net in "${NET_ARR[@]}"; do
    v=-1
    if [[ -f "$DATA_DIR/nets/$net/state.json" ]]; then
        v=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('iter', -1))" \
            "$DATA_DIR/nets/$net/state.json")
    fi
    NET_ITERS+=("$v")
    [[ "$v" -gt "$max_iter" ]] && max_iter="$v"
done
iter=$((max_iter + 1))
[[ "$iter" -lt 0 ]] && iter=0

# Bootstrap any missing per-network random-init artifacts.
for net in "${NET_ARR[@]}"; do
    if [[ ! -f "$DATA_DIR/nets/$net/latest.pt" ]]; then
        echo "[run.sh] bootstrapping random-init model for $net"
        ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" --network "$net" )
    fi
done

mirror_active_to_models() {
    local active="$1"
    local src_pt="$DATA_DIR/nets/$active/latest.pt"
    local src_meta="$DATA_DIR/nets/$active/latest.meta.json"
    local dst_pt="$DATA_DIR/models/latest.pt"
    local dst_meta="$DATA_DIR/models/latest.meta.json"
    cp "$src_pt" "${dst_pt}.tmp" && mv "${dst_pt}.tmp" "$dst_pt"
    if [[ -f "$src_meta" ]]; then
        cp "$src_meta" "${dst_meta}.tmp" && mv "${dst_meta}.tmp" "$dst_meta"
    fi
}

# Now that all networks have init artifacts, populate the canonical mirror
# at data/models/ so selfplay (main + daemon) has something to load before
# we even enter the loop.
INITIAL_ACTIVE=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" 2>/dev/null )
INITIAL_ACTIVE="${INITIAL_ACTIVE:-$FIRST_NET}"
mirror_active_to_models "$INITIAL_ACTIVE"

# --- Catch-up for mid-iter Ctrl+C ---
# Any net with iter < max_iter was interrupted before training that iter.
# Train + export it at iter=max_iter on the still-intact shuffled/current/
# data (the bucket consumption for that iter was already deducted in
# bucket.json, so do NOT re-run bucket.py / selfplay / shuffle here).
# Steps come from a lead net's train.tsv row at iter=max_iter (matches what
# the interrupted iter actually trained), with a 1-epoch fallback.
if [[ "$max_iter" -ge 0 ]]; then
    LAGGING=()
    LEAD_NET=""
    for i in "${!NET_ARR[@]}"; do
        if [[ "${NET_ITERS[$i]}" -lt "$max_iter" ]]; then
            LAGGING+=("${NET_ARR[$i]}")
        elif [[ -z "$LEAD_NET" ]]; then
            LEAD_NET="${NET_ARR[$i]}"
        fi
    done
    if [[ "${#LAGGING[@]}" -gt 0 ]]; then
        if compgen -G "$DATA_DIR/shuffled/current/*" > /dev/null; then
            CATCHUP_STEPS=""
            if [[ -n "$LEAD_NET" && -f "$DATA_DIR/nets/$LEAD_NET/train.tsv" ]]; then
                CATCHUP_STEPS=$(awk -v it="$max_iter" 'BEGIN{FS="\t"} NR>1 && $1==it {print $2; exit}' \
                    "$DATA_DIR/nets/$LEAD_NET/train.tsv")
            fi
            if [[ -z "$CATCHUP_STEPS" || "$CATCHUP_STEPS" -le 0 ]]; then
                CATCHUP_STEPS=$(( ${TRAIN_SAMPLES_PER_EPOCH:-100000} / ${BATCH_SIZE:-128} ))
                echo "[run.sh] catch-up: lead steps unavailable, falling back to 1 epoch ($CATCHUP_STEPS steps)"
            fi
            echo "[run.sh] resume catch-up: lagging=(${LAGGING[*]}) at iter=$max_iter steps=$CATCHUP_STEPS (using existing shuffled/current/)"
            for net in "${LAGGING[@]}"; do
                echo "[run.sh] catch-up: training $net at iter=$max_iter"
                TRAIN_STEPS_PER_EPOCH="$CATCHUP_STEPS" \
                    bash "$SCRIPT_DIR/internal/train.sh" "$max_iter" "$net"
                bash "$SCRIPT_DIR/internal/export.sh" "$max_iter" "$net"
            done
            # Re-mirror in case the active network was among the lagging.
            mirror_active_to_models "$INITIAL_ACTIVE"
        else
            echo "[run.sh] WARNING: lagging networks (${LAGGING[*]}) behind max_iter=$max_iter but shuffled/current/ is empty; cannot catch up — those iters stay as gaps in train.tsv"
        fi
    fi
fi

# Multi-GPU: launch the selfplay daemon on GPUs 1..GPU_NUM-1 in the background.
# Daemon hot-reloads $DATA_DIR/models/latest.pt — the mirror above keeps that
# path always pointing at the current active network's TorchScript.
if [[ "$GPU_NUM" -gt 1 ]]; then
    echo "[run.sh] starting selfplay daemon on GPUs 1..$((GPU_NUM-1))"
    bash "$SCRIPT_DIR/internal/selfplay_daemon.sh" &
    DAEMON_PID=$!
fi

max_iters="${1:-}"
OVERLAP_SHUFFLE="${OVERLAP_SHUFFLE:-0}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-0}"

# Initialize schedule log header once.
SCHEDULE_LOG="$DATA_DIR/logs/schedule.tsv"
if [[ ! -f "$SCHEDULE_LOG" ]]; then
    printf "iter\tcum_samples\tactive_network\n" > "$SCHEDULE_LOG"
fi

# Wall-clock budget: capture start right before entering the loop so cmake
# rebuild and model bootstrap above are excluded.
LOOP_START_SECONDS=$SECONDS

while true; do
    echo ""
    echo "=================================================================="
    echo "[run.sh] === iter $iter ==="
    date

    # Cumulative selfplay totals so far (through previous iter).
    # selfplay.tsv schema: producer iter_or_version games rows ...
    awk 'NR>1 {g+=$3; r+=$4} END {printf "[run.sh] cumulative so far: games=%d samples=%d (main+daemon)\n", g+0, r+0}' \
        "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null \
        || echo "[run.sh] cumulative so far: games=0 samples=0"

    # Decide active network for this iter and refresh the mirror.
    ACTIVE_NETWORK=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" )
    export ACTIVE_NETWORK
    mirror_active_to_models "$ACTIVE_NETWORK"

    # Log schedule decision (one row per iter, including no-train iters).
    CUM_SAMPLES=$(awk 'NR>1 {r+=$4} END {printf "%d", r+0}' "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null || echo 0)
    printf "%d\t%d\t%s\n" "$iter" "$CUM_SAMPLES" "$ACTIVE_NETWORK" >> "$SCHEDULE_LOG"
    echo "[run.sh] iter $iter active=$ACTIVE_NETWORK cum_samples=$CUM_SAMPLES"

    # (1) selfplay [+ overlapped shuffle] — GAMES_PER_ITER is fixed in run.cfg
    GAMES="${GAMES_PER_ITER:-2000}"
    # SHUFFLE_RC=0 success, =2 skipped (N < MIN_ROWS), other = failure.
    SHUFFLE_RC=0
    if [[ "$OVERLAP_SHUFFLE" == "1" && "$iter" -gt 0 ]]; then
        # Pipelined mode: shuffle (CPU) runs concurrently with selfplay (GPU).
        # The shuffle sees data collected up through the previous iter; the
        # training step that follows therefore trains on a 1-iter-lagged
        # window. Selfplay's new iter-N files are written but ignored by this
        # shuffle (shuffle.py snapshots the file list once at start).
        echo "[run.sh] shuffle (bg) || selfplay (fg)"
        bash "$SCRIPT_DIR/internal/shuffle.sh" &
        SHUFFLE_PID=$!
        bash "$SCRIPT_DIR/internal/selfplay.sh" "$iter" "$GAMES"
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        bash "$SCRIPT_DIR/internal/selfplay.sh" "$iter" "$GAMES"
        bash "$SCRIPT_DIR/internal/shuffle.sh" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "[run.sh] shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "[run.sh] shuffle failed with code $SHUFFLE_RC"
        exit "$SHUFFLE_RC"
    else
        # (2) Token bucket: fill from disk row delta, decide train_steps.
        TRAIN_STEPS=$( cd "$ROOT/python" && "$PY" bucket.py --data-dir "$DATA_DIR" )
        if [[ "$TRAIN_STEPS" -le 0 ]]; then
            echo "[run.sh] bucket below epoch threshold; skipping train+export this iter"
        else
            # (3) train each network on the same shuffled pool (serial).
            for net in "${NET_ARR[@]}"; do
                echo "[run.sh] ---- training $net (steps=$TRAIN_STEPS) ----"
                TRAIN_STEPS_PER_EPOCH="$TRAIN_STEPS" \
                    bash "$SCRIPT_DIR/internal/train.sh" "$iter" "$net"
            done

            # (4) export TorchScript for each network.
            for net in "${NET_ARR[@]}"; do
                echo "[run.sh] ---- exporting $net ----"
                bash "$SCRIPT_DIR/internal/export.sh" "$iter" "$net"
            done

            # Refresh the active mirror with the freshly-exported weights so
            # the next selfplay iter (and the probe below) see the new model.
            mirror_active_to_models "$ACTIVE_NETWORK"

            # (4b) post-export diagnostic: empty-board MCTS rootValue probe
            # on the currently-active network (read via the canonical mirror).
            CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}" "$ROOT/cpp/build/mcts_probe" \
                --model "$DATA_DIR/models/latest.pt" \
                --config "$CONFIG_DIR/run.cfg" \
                --iter "$iter" \
                --log "$DATA_DIR/logs/probe.tsv" \
                || echo "[run.sh] mcts_probe failed (non-fatal)"

            # (5) plot loss curve (combined plot for all networks)
            ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )
        fi
    fi

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[run.sh] reached max_iters=$max_iters; stopping."
        break
    fi
    if [[ "$MAX_TIME_SECONDS" -gt 0 ]]; then
        elapsed=$((SECONDS - LOOP_START_SECONDS))
        if [[ "$elapsed" -ge "$MAX_TIME_SECONDS" ]]; then
            echo "[run.sh] reached MAX_TIME_SECONDS=$MAX_TIME_SECONDS (elapsed=${elapsed}s); stopping."
            break
        fi
    fi
done

if [[ -n "$DAEMON_PID" ]]; then
    echo "[run.sh] stopping selfplay daemon (pid=$DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
fi
