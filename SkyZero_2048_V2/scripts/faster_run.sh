#!/usr/bin/env bash
# Async (continuous-selfplay) orchestration loop for 2048 — the throughput
# alternative to scripts/run.sh. run.sh is a strict per-iteration BARRIER: it
# runs a BOUNDED self-play batch and waits for the LAST (longest) game before
# training, so the GPU/CPU idle through 2048's long-game tail (slots drain, the
# inference batch shrinks to a few stragglers, sps collapses). faster_run.sh
# removes that barrier with the canonical async-AlphaZero producer/consumer split:
#
#   PRODUCER : one selfplay2048_par --daemon per GPU (incl. MAIN_GPU). It never
#              hits a games barrier — a finished game slot refills from an
#              unbounded source, so the inference batch stays full and no long
#              game ever stalls the others. Hot-reloads models/latest.pt on
#              export. A long game just finishes whenever it finishes, still
#              contributing its (high-tile) data — nothing is dropped.
#   CONSUMER : this loop. It polls the disk for freshly produced rows and, once
#              enough have accumulated, runs shuffle -> train -> export -> mirror.
#              export bumps models/latest.pt mtime -> daemons hot-reload the new
#              weights. The GPU is never idle: the producer keeps generating while
#              the (tiny 4x4 net) trainer time-shares the card.
#
# SINGLE GPU: producer daemon + trainer share GPU 0. The net is small enough
# (GPU ~20% during inference) that they time-slice fine; you trade the tail's
# near-zero-sps idle for a busy shared card. The real knob is CPU: leave a few
# cores for the trainer (lower WORKERS_PER_GPU a touch) so the MCTS workers and
# the DataLoader don't starve each other.
#
# Usage: CONFIG_DIR=configs/baseline bash scripts/faster_run.sh [max_iters]
#   max_iters (CLI arg) or ITERS (run.cfg) caps trained iters; MAX_TIME_SECONDS caps time.
#
# Reuses run.cfg (same knobs as run.sh). Cadence = the gate's TARGET_REPLAY_RATIO
# as the replay multiplier: the consumer trains ~TARGET_REPLAY_RATIO epochs' worth
# per produced epoch's worth of rows. Async-only knobs (all optional):
#   ASYNC_ROWS_PER_CYCLE   new rows that must accumulate before a train cycle
#                          (default = TRAIN_SAMPLES_PER_EPOCH / TARGET_REPLAY_RATIO)
#   ASYNC_POLL_SECONDS     poll interval while waiting for data (default 30)
#   ASYNC_SELFPLAY_GPUS    comma list of GPUs to run daemons on (default: all 0..GPU_NUM-1)
#
# State (separate from run.sh so the two paths never clobber each other):
#   <DATA_DIR>/async_bucket.state  (text: "cum_rows bucket_level" — token bucket)
#   <DATA_DIR>/logs/daemon_gpu<N>.log
set -euo pipefail

DAEMON_PIDS=()
CYCLE_CHILD=""          # pid of the in-flight train/export subshell (empty when idle)

# Recursively SIGTERM a process and all its descendants (subshell -> train.sh -> python).
kill_tree() {
    local pid="$1" sig="${2:-TERM}" k
    for k in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$k" "$sig"; done
    kill -"$sig" "$pid" 2>/dev/null || true
}
stop_daemons() {
    [[ "${#DAEMON_PIDS[@]}" -eq 0 ]] && return 0
    echo "[faster_run] stopping ${#DAEMON_PIDS[@]} self-play daemon(s) (SIGINT -> drain+flush)"
    for pid in "${DAEMON_PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${DAEMON_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    DAEMON_PIDS=()
}
on_interrupt() {
    trap - INT TERM
    echo "[faster_run] interrupted; stopping."
    [[ -n "$CYCLE_CHILD" ]] && kill_tree "$CYCLE_CHILD" TERM   # abort any in-flight train/export
    stop_daemons
    exit 130
}
trap on_interrupt INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[faster_run] no config dir at $CONFIG_DIR" >&2; exit 1; }
export CONFIG_DIR

# Experiment config (run.cfg + paths.cfg) + machine paths (env_paths.cfg). set -a
# so every knob is exported into the daemon / internal/*.sh env.
set -a
# shellcheck disable=SC1091
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"
set +a
export DATA_DIR

if [[ -z "${NETWORKS:-}" || -z "${SELFPLAY_SCHEDULE:-}" ]]; then
    echo "[faster_run] NETWORKS and SELFPLAY_SCHEDULE must be set in run.cfg" >&2
    exit 1
fi
read -ra NET_ARR <<< "$(echo "$NETWORKS" | tr ',' ' ')"
[[ "${#NET_ARR[@]}" -gt 0 ]] || { echo "[faster_run] NETWORKS empty" >&2; exit 1; }
FIRST_NET="${NET_ARR[0]}"

PY="${PY:-python}"
BUILD_DIR="${BUILD_DIR:-$DATA_DIR/build}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$BUILD_DIR/selfplay2048_par}"
PROBE_BIN="${PROBE_BIN:-$BUILD_DIR/mcts_probe_2048}"

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,logs}
for net in "${NET_ARR[@]}"; do mkdir -p "$DATA_DIR/nets/$net"; done

# Auto-detect GPU count (same as run.sh).
if [[ -z "${GPU_NUM:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NUM=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        GPU_NUM=1
    fi
fi
GPU_NUM="${GPU_NUM:-1}"
export GPU_NUM
MAIN_GPU="${MAIN_GPU:-0}"
echo "[faster_run] detected GPU_NUM=$GPU_NUM main_gpu=$MAIN_GPU"

# Keep C++ binaries in sync (configure-if-needed + incremental build into $DATA_DIR/build).
bash "$SCRIPT_DIR/build.sh"
[[ -x "$SELFPLAY_BIN" ]] || { echo "[faster_run] binary not found: $SELFPLAY_BIN (build: bash scripts/build.sh --target selfplay2048_par)" >&2; exit 1; }

mirror_active_to_models() {
    local active="$1"
    local src_pt="$DATA_DIR/nets/$active/latest.pt"
    local src_meta="$DATA_DIR/nets/$active/latest.meta.json"
    cp "$src_pt" "$DATA_DIR/models/latest.pt.tmp" && mv "$DATA_DIR/models/latest.pt.tmp" "$DATA_DIR/models/latest.pt"
    [[ -f "$src_meta" ]] && { cp "$src_meta" "$DATA_DIR/models/latest.meta.json.tmp" && mv "$DATA_DIR/models/latest.meta.json.tmp" "$DATA_DIR/models/latest.meta.json"; }
}

# --- resume iter = max over per-net state.json (same as run.sh) ---
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
        echo "[faster_run] bootstrapping random-init model for $net"
        ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" --network "$net" )
    fi
done

INITIAL_ACTIVE=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" 2>/dev/null )
INITIAL_ACTIVE="${INITIAL_ACTIVE:-$FIRST_NET}"
mirror_active_to_models "$INITIAL_ACTIVE"
LAST_ACTIVE="$INITIAL_ACTIVE"

# Catch-up: any net interrupted before training at max_iter (same as run.sh).
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
        echo "[faster_run] resume catch-up: lagging=(${LAGGING[*]}) at iter=$max_iter steps=$CATCHUP_STEPS"
        for net in "${LAGGING[@]}"; do
            TRAIN_STEPS_PER_EPOCH="$CATCHUP_STEPS" bash "$SCRIPT_DIR/internal/train.sh" "$max_iter" "$net"
            bash "$SCRIPT_DIR/internal/export.sh" "$max_iter" "$net"
        done
        mirror_active_to_models "$INITIAL_ACTIVE"
    fi
fi

# --- launch one self-play daemon per GPU (producer) -------------------------
# Shared search/MCTS/value/stochastic tuning args (identical to the bounded
# selfplay.sh path — single source of truth so the two producers can't drift).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/internal/selfplay_tuning_args.sh"

if [[ -z "${ASYNC_SELFPLAY_GPUS:-}" ]]; then
    gpus=""
    for ((i=0; i<GPU_NUM; i++)); do [[ -n "$gpus" ]] && gpus+=","; gpus+="$i"; done
    ASYNC_SELFPLAY_GPUS="$gpus"
fi
SIMS_WARMUP_CMD="cd $ROOT/python && $PY warmup.py num-simulations --data-dir $DATA_DIR"
POLL_MS="${DAEMON_RELOAD_POLL_MS:-2000}"
NSIM0=$( cd "$ROOT/python" && "$PY" warmup.py num-simulations --data-dir "$DATA_DIR" 2>/dev/null )
NSIM0="${NSIM0:-${SIMS:-64}}"
echo "[faster_run] starting self-play daemon(s) on GPUs: $ASYNC_SELFPLAY_GPUS (initial sims=$NSIM0)"
IFS=',' read -ra DG <<< "$ASYNC_SELFPLAY_GPUS"
for gpu in "${DG[@]}"; do
    [[ -z "$gpu" ]] && continue
    CUDA_VISIBLE_DEVICES="$gpu" "$SELFPLAY_BIN" --daemon \
        --model "$DATA_DIR/models/latest.pt" \
        --out "$DATA_DIR/selfplay" \
        --log-dir "$DATA_DIR/logs" \
        --model-watch-poll-ms "$POLL_MS" \
        --sims-warmup-cmd "$SIMS_WARMUP_CMD" \
        --sims "$NSIM0" \
        "${SP_TUNING_ARGS[@]}" \
        --progress-secs "${PROGRESS_SECS:-30}" \
        --stats-games "${STATS_GAMES:-1000}" \
        --noise 1 \
        > "$DATA_DIR/logs/daemon_gpu${gpu}.log" 2>&1 &
    DAEMON_PIDS+=($!)
    echo "[faster_run]   gpu=$gpu pid=${DAEMON_PIDS[-1]} -> logs/daemon_gpu${gpu}.log"
done

# --- async knobs (cadence reuses the gate's TARGET_REPLAY_RATIO) ---
SPE="${TRAIN_SAMPLES_PER_EPOCH:-256000}"
RATIO="${TARGET_REPLAY_RATIO:-6}"
ASYNC_ROWS_PER_CYCLE="${ASYNC_ROWS_PER_CYCLE:-$(awk -v s="$SPE" -v r="$RATIO" 'BEGIN{printf "%d", s/r}')}"
ASYNC_POLL_SECONDS="${ASYNC_POLL_SECONDS:-30}"
max_iters="${1:-${ITERS:-}}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-0}"
EVAL_EVERY_ITERS="${EVAL_EVERY_ITERS:-0}"
BUCKET_STATE="$DATA_DIR/async_bucket.state"

SCHEDULE_LOG="$DATA_DIR/logs/schedule.tsv"
[[ -f "$SCHEDULE_LOG" ]] || printf "iter\tcum_samples\tactive_network\n" > "$SCHEDULE_LOG"

echo "[faster_run] consumer: rows_per_cycle=$ASYNC_ROWS_PER_CYCLE replay_ratio=$RATIO poll=${ASYNC_POLL_SECONDS}s resume_iter=$iter"

# Token bucket (replay-ratio cadence): bucket += new_rows * RATIO; spend one
# TRAIN_SAMPLES_PER_EPOCH per trained epoch. Prints train_steps; persists
# "cum_rows bucket_level" to BUCKET_STATE. Capped at MAX_TRAIN_BUCKET_SIZE so a
# long idle gap can't bank unbounded epochs.
compute_train_steps() {
    local new_rows="$1"
    local maxb="${MAX_TRAIN_BUCKET_SIZE:-$((2 * SPE))}"
    awk -v nr="$new_rows" -v spe="$SPE" -v bs="${BATCH_SIZE:-128}" -v ratio="$RATIO" \
        -v maxb="$maxb" -v sf="$BUCKET_STATE" '
    BEGIN{
        cum=0; bucket=spe;                       # init: one epoch banked
        if ((getline line < sf) > 0) { split(line, a, " "); cum=a[1]; bucket=a[2] }
        close(sf)
        cum += nr
        bucket += nr * ratio
        cap = (maxb > spe ? maxb : spe)
        if (bucket > cap) bucket = cap
        thr = 0.99 * spe
        epochs = 0
        while (bucket >= thr) { bucket -= spe; epochs++ }
        steps = epochs * int(spe / bs)
        printf "%d %.6f\n", cum, bucket > sf
        print steps
    }'
}

LOOP_START_SECONDS=$SECONDS
last_cycle_ts=0   # unix seconds; only files newer than this count as "new data"

while true; do
    # Time cap (checked every pass so a slow data rate still honors it).
    if [[ "$MAX_TIME_SECONDS" -gt 0 ]]; then
        elapsed=$((SECONDS - LOOP_START_SECONDS))
        if [[ "$elapsed" -ge "$MAX_TIME_SECONDS" ]]; then
            echo "[faster_run] reached MAX_TIME_SECONDS=$MAX_TIME_SECONDS (elapsed=${elapsed}s); stopping."; break
        fi
    fi

    # Liveness: if every daemon died, there is no producer -> bail loudly.
    alive=0
    for pid in "${DAEMON_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive+1)); done
    if [[ "$alive" -eq 0 ]]; then
        echo "[faster_run] all self-play daemons have exited; stopping. (see logs/daemon_gpu*.log)" >&2
        break
    fi

    new_rows=$( cd "$ROOT/python" && "$PY" async_rows.py --data-dir "$DATA_DIR" --since "$last_cycle_ts" )
    new_rows="${new_rows:-0}"
    if [[ "$new_rows" -lt "$ASYNC_ROWS_PER_CYCLE" ]]; then
        sleep "$ASYNC_POLL_SECONDS"
        continue
    fi

    echo ""
    echo "=================================================================="
    echo "[faster_run] === iter $iter ===  (new_rows=$new_rows, daemons_alive=$alive)"
    date

    # Capture the cutoff BEFORE shuffle so rows produced during this cycle are
    # counted in the NEXT cycle, never dropped and never double-counted.
    cycle_ts=$( date +%s.%N )

    ACTIVE_NETWORK=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" )
    export ACTIVE_NETWORK
    # Switch the served net immediately on a schedule change (daemon hot-reloads);
    # otherwise leave models/latest.pt alone to avoid spurious reloads.
    if [[ "$ACTIVE_NETWORK" != "$LAST_ACTIVE" ]]; then
        echo "[faster_run] active network $LAST_ACTIVE -> $ACTIVE_NETWORK (mirror + hot-reload)"
        mirror_active_to_models "$ACTIVE_NETWORK"
        LAST_ACTIVE="$ACTIVE_NETWORK"
    fi

    CUM_SAMPLES=$(awk 'NR>1 {r+=$4} END {printf "%d", r+0}' "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null || echo 0)
    printf "%d\t%d\t%s\n" "$iter" "$CUM_SAMPLES" "$ACTIVE_NETWORK" >> "$SCHEDULE_LOG"

    # Shuffle the on-disk window (idempotent; reads disk truth, deletes OOW).
    SHUFFLE_RC=0
    bash "$SCRIPT_DIR/internal/shuffle.sh" || SHUFFLE_RC=$?
    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "[faster_run] shuffle skipped (window < MIN_ROWS); waiting for more data"
        last_cycle_ts="$cycle_ts"   # don't re-trigger on the same warmup rows
        sleep "$ASYNC_POLL_SECONDS"
        continue
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "[faster_run] shuffle failed with code $SHUFFLE_RC" >&2; stop_daemons; exit "$SHUFFLE_RC"
    fi

    TRAIN_STEPS=$(compute_train_steps "$new_rows")
    last_cycle_ts="$cycle_ts"

    if [[ "$TRAIN_STEPS" -le 0 ]]; then
        echo "[faster_run] token bucket below epoch threshold; no train this cycle"
        continue
    fi

    # Run train+export in a background subshell and wait on it, so a SIGINT (or
    # the MAX_TIME cap, checked at loop top) interrupts mid-training instead of
    # being deferred until the foreground train.py returns. on_interrupt() kills
    # this subtree (subshell -> train.sh -> python) so nothing leaks on the GPU.
    (
        for net in "${NET_ARR[@]}"; do
            echo "[faster_run] ---- training $net (steps=$TRAIN_STEPS) ----"
            TRAIN_STEPS_PER_EPOCH="$TRAIN_STEPS" bash "$SCRIPT_DIR/internal/train.sh" "$iter" "$net"
        done
        for net in "${NET_ARR[@]}"; do
            echo "[faster_run] ---- exporting $net ----"
            bash "$SCRIPT_DIR/internal/export.sh" "$iter" "$net"
        done
    ) &
    CYCLE_CHILD=$!
    cyc_rc=0
    wait "$CYCLE_CHILD" || cyc_rc=$?
    CYCLE_CHILD=""
    if [[ "$cyc_rc" -ne 0 ]]; then
        echo "[faster_run] train/export failed (rc=$cyc_rc); stopping." >&2
        stop_daemons; exit "$cyc_rc"
    fi
    # New weights -> bump models/latest.pt -> daemons hot-reload to this version.
    mirror_active_to_models "$ACTIVE_NETWORK"
    LAST_ACTIVE="$ACTIVE_NETWORK"

    if [[ -x "$PROBE_BIN" ]]; then
        CUDA_VISIBLE_DEVICES="$MAIN_GPU" "$PROBE_BIN" \
            --model "$DATA_DIR/models/latest.pt" \
            --config "$CONFIG_DIR/run.cfg" --iter "$iter" \
            --log "$DATA_DIR/logs/probe.tsv" \
            || echo "[faster_run] mcts_probe_2048 failed (non-fatal)"
    fi
    # eval is a P4 item (no internal/eval.sh in V2 yet); run it only if present.
    if (( EVAL_EVERY_ITERS > 0 )) && (( (iter + 1) % EVAL_EVERY_ITERS == 0 )) \
            && [[ -f "$SCRIPT_DIR/internal/eval.sh" ]]; then
        bash "$SCRIPT_DIR/internal/eval.sh" "$iter" "$ACTIVE_NETWORK" \
            || echo "[faster_run] eval failed (non-fatal)"
    fi
    ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" >/dev/null 2>&1 ) || true

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "[faster_run] reached max_iters=$max_iters; stopping."; break
    fi
done

stop_daemons
echo "[faster_run] done."
