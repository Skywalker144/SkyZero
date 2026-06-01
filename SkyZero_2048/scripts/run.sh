#!/usr/bin/env bash
# Main orchestration loop for 2048 Stochastic Gumbel AlphaZero (V7.1-style).
#
# Each iter:
#   1. ACTIVE_NETWORK = schedule.py (cumulative selfplay rows vs SELFPLAY_SCHEDULE)
#   2. mirror nets/$ACTIVE/latest.pt -> models/latest.pt (stable path for C++ selfplay)
#   3. selfplay.sh  [|| shuffle.sh overlap]  -> shuffle.sh
#   4. bucket.py -> train_steps (0 = skip; below MIN_ROWS shuffle exits 2 = skip)
#   5. for each NETWORK: train.sh -> export.sh   (all train on the same shuffle)
#   6. re-mirror active; mcts_probe_2048 (if built); periodic eval.sh; view_loss.py
#   7. log a row to logs/schedule.tsv
#
# Usage: CONFIG_DIR=configs/baseline bash scripts/run.sh [max_iters]
#   max_iters (CLI arg) or ITERS (run.cfg) caps iters; MAX_TIME_SECONDS caps time.
set -euo pipefail

DAEMON_PID=""
trap 'trap - INT TERM; echo "[run.sh] interrupted; stopping."; [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[run.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }
export CONFIG_DIR

# Experiment config (run.cfg + paths.cfg) + machine paths (env_paths.cfg).
set -a
# shellcheck disable=SC1091
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"
set +a
export DATA_DIR

if [[ -z "${NETWORKS:-}" || -z "${SELFPLAY_SCHEDULE:-}" ]]; then
    echo "[run.sh] NETWORKS and SELFPLAY_SCHEDULE must be set in run.cfg" >&2
    exit 1
fi
read -ra NET_ARR <<< "$(echo "$NETWORKS" | tr ',' ' ')"
[[ "${#NET_ARR[@]}" -gt 0 ]] || { echo "[run.sh] NETWORKS empty" >&2; exit 1; }
FIRST_NET="${NET_ARR[0]}"

PY="${PY:-python}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay2048_par}"
PROBE_BIN="$ROOT/cpp/build/mcts_probe_2048"

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,logs}
for net in "${NET_ARR[@]}"; do mkdir -p "$DATA_DIR/nets/$net"; done

# Auto-detect GPU count. Main loop on GPU 0; spare GPUs -> selfplay daemon.
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

# Keep C++ binaries in sync (configure-if-needed + incremental build; ~0 when nothing changed).
bash "$SCRIPT_DIR/build.sh"

mirror_active_to_models() {
    local active="$1"
    local src_pt="$DATA_DIR/nets/$active/latest.pt"
    local src_meta="$DATA_DIR/nets/$active/latest.meta.json"
    cp "$src_pt" "$DATA_DIR/models/latest.pt.tmp" && mv "$DATA_DIR/models/latest.pt.tmp" "$DATA_DIR/models/latest.pt"
    [[ -f "$src_meta" ]] && { cp "$src_meta" "$DATA_DIR/models/latest.meta.json.tmp" && mv "$DATA_DIR/models/latest.meta.json.tmp" "$DATA_DIR/models/latest.meta.json"; }
}

# Resume iter = max over per-net state.json (mid-iter Ctrl+C can leave nets at
# different iters; the catch-up block below trains laggards at max_iter).
max_iter=-1
NET_ITERS=()
for net in "${NET_ARR[@]}"; do
    v=-1
    if [[ -f "$DATA_DIR/nets/$net/state.json" ]]; then
        v=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('iter',-1))" "$DATA_DIR/nets/$net/state.json")
    fi
    NET_ITERS+=("$v")
    [[ "$v" -gt "$max_iter" ]] && max_iter="$v"
done
iter=$((max_iter + 1)); [[ "$iter" -lt 0 ]] && iter=0

# Bootstrap missing per-net random-init artifacts.
for net in "${NET_ARR[@]}"; do
    if [[ ! -f "$DATA_DIR/nets/$net/latest.pt" ]]; then
        echo "[run.sh] bootstrapping random-init model for $net"
        ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" --network "$net" )
    fi
done

INITIAL_ACTIVE=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" 2>/dev/null )
INITIAL_ACTIVE="${INITIAL_ACTIVE:-$FIRST_NET}"
mirror_active_to_models "$INITIAL_ACTIVE"

# Catch-up: any net with iter < max_iter was interrupted before training it.
# Train+export it at max_iter on the still-intact shuffled/current/ (bucket was
# already deducted, so don't re-run selfplay/shuffle/bucket).
if [[ "$max_iter" -ge 0 ]]; then
    LAGGING=(); LEAD_NET=""
    for i in "${!NET_ARR[@]}"; do
        if [[ "${NET_ITERS[$i]}" -lt "$max_iter" ]]; then LAGGING+=("${NET_ARR[$i]}")
        elif [[ -z "$LEAD_NET" ]]; then LEAD_NET="${NET_ARR[$i]}"; fi
    done
    if [[ "${#LAGGING[@]}" -gt 0 ]] && compgen -G "$DATA_DIR/shuffled/current/*" > /dev/null; then
        CATCHUP_STEPS=""
        if [[ -n "$LEAD_NET" && -f "$DATA_DIR/nets/$LEAD_NET/train.tsv" ]]; then
            CATCHUP_STEPS=$(awk -v it="$max_iter" 'BEGIN{FS="\t"} NR>1 && $1==it {print $2; exit}' "$DATA_DIR/nets/$LEAD_NET/train.tsv")
        fi
        [[ -z "$CATCHUP_STEPS" || "$CATCHUP_STEPS" -le 0 ]] && CATCHUP_STEPS=$(( ${TRAIN_SAMPLES_PER_EPOCH:-256000} / ${BATCH_SIZE:-128} ))
        echo "[run.sh] resume catch-up: lagging=(${LAGGING[*]}) at iter=$max_iter steps=$CATCHUP_STEPS"
        for net in "${LAGGING[@]}"; do
            TRAIN_STEPS_PER_EPOCH="$CATCHUP_STEPS" bash "$SCRIPT_DIR/internal/train.sh" "$max_iter" "$net"
            bash "$SCRIPT_DIR/internal/export.sh" "$max_iter" "$net"
        done
        mirror_active_to_models "$INITIAL_ACTIVE"
    fi
fi

# Multi-GPU: launch the selfplay daemon on spare GPUs (hot-reloads models/latest.pt).
if [[ "$GPU_NUM" -gt 1 ]]; then
    echo "[run.sh] starting selfplay daemon on spare GPUs"
    bash "$SCRIPT_DIR/internal/selfplay_daemon.sh" &
    DAEMON_PID=$!
fi

max_iters="${1:-${ITERS:-}}"
OVERLAP_SHUFFLE="${OVERLAP_SHUFFLE:-0}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-0}"
EVAL_EVERY_ITERS="${EVAL_EVERY_ITERS:-5}"

SCHEDULE_LOG="$DATA_DIR/logs/schedule.tsv"
[[ -f "$SCHEDULE_LOG" ]] || printf "iter\tcum_samples\tactive_network\n" > "$SCHEDULE_LOG"

LOOP_START_SECONDS=$SECONDS

while true; do
    echo ""
    echo "=================================================================="
    echo "[run.sh] === iter $iter ==="
    date

    # games from col3 only for new-schema rows (producer main/daemon); old
    # loop_cpp rows had a timestamp in col3, so summing those inflates games.
    awk -F'\t' 'NR>1 {if($1=="main"||$1=="daemon") g+=$3; r+=$4} END {printf "[run.sh] cumulative so far: games=%d rows=%d (main+daemon)\n", g+0, r+0}' \
        "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null || echo "[run.sh] cumulative so far: games=0 rows=0"

    ACTIVE_NETWORK=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" )
    export ACTIVE_NETWORK
    mirror_active_to_models "$ACTIVE_NETWORK"

    CUM_SAMPLES=$(awk 'NR>1 {r+=$4} END {printf "%d", r+0}' "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null || echo 0)
    printf "%d\t%d\t%s\n" "$iter" "$CUM_SAMPLES" "$ACTIVE_NETWORK" >> "$SCHEDULE_LOG"

    GAMES="${GAMES_PER_ITER:-800}"
    SHUFFLE_RC=0
    if [[ "$OVERLAP_SHUFFLE" == "1" && "$iter" -gt 0 ]]; then
        echo "[run.sh] shuffle (bg) || selfplay (fg)"
        bash "$SCRIPT_DIR/internal/shuffle.sh" & SHUFFLE_PID=$!
        bash "$SCRIPT_DIR/internal/selfplay.sh" "$iter" "$GAMES"
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        bash "$SCRIPT_DIR/internal/selfplay.sh" "$iter" "$GAMES"
        bash "$SCRIPT_DIR/internal/shuffle.sh" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "[run.sh] shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "[run.sh] shuffle failed with code $SHUFFLE_RC"; exit "$SHUFFLE_RC"
    else
        TRAIN_STEPS=$( cd "$ROOT/python" && "$PY" bucket.py --data-dir "$DATA_DIR" )
        if [[ "$TRAIN_STEPS" -le 0 ]]; then
            echo "[run.sh] bucket below epoch threshold; skipping train+export this iter"
        else
            for net in "${NET_ARR[@]}"; do
                echo "[run.sh] ---- training $net (steps=$TRAIN_STEPS) ----"
                TRAIN_STEPS_PER_EPOCH="$TRAIN_STEPS" bash "$SCRIPT_DIR/internal/train.sh" "$iter" "$net"
            done
            for net in "${NET_ARR[@]}"; do
                echo "[run.sh] ---- exporting $net ----"
                bash "$SCRIPT_DIR/internal/export.sh" "$iter" "$net"
            done
            mirror_active_to_models "$ACTIVE_NETWORK"

            # Post-export diagnostic: empty-board MCTS root-value probe (if built).
            if [[ -x "$PROBE_BIN" ]]; then
                CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}" "$PROBE_BIN" \
                    --model "$DATA_DIR/models/latest.pt" \
                    --config "$CONFIG_DIR/run.cfg" --iter "$iter" \
                    --log "$DATA_DIR/logs/probe.tsv" \
                    || echo "[run.sh] mcts_probe_2048 failed (non-fatal)"
            fi

            # Periodic single-agent eval of the active network.
            if (( EVAL_EVERY_ITERS > 0 )) && (( (iter + 1) % EVAL_EVERY_ITERS == 0 )); then
                bash "$SCRIPT_DIR/internal/eval.sh" "$iter" "$ACTIVE_NETWORK" \
                    || echo "[run.sh] eval failed (non-fatal)"
            fi

            ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" >/dev/null 2>&1 ) || true
        fi
    fi

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[run.sh] reached max_iters=$max_iters; stopping."; break
    fi
    if [[ "$MAX_TIME_SECONDS" -gt 0 ]]; then
        elapsed=$((SECONDS - LOOP_START_SECONDS))
        if [[ "$elapsed" -ge "$MAX_TIME_SECONDS" ]]; then
            echo "[run.sh] reached MAX_TIME_SECONDS=$MAX_TIME_SECONDS (elapsed=${elapsed}s); stopping."; break
        fi
    fi
done

if [[ -n "$DAEMON_PID" ]]; then
    echo "[run.sh] stopping selfplay daemon (pid=$DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
fi
