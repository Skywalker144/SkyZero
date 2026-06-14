#!/usr/bin/env bash
# Main orchestration loop for multi-network training.
#
# Each iter:
#   1. Compute ACTIVE_NETWORK from cumulative selfplay samples + SELFPLAY_SCHEDULE.
#   2. Mirror data/nets/$ACTIVE_NETWORK/{latest.pt,latest.meta.json} -> data/models/
#      so C++ selfplay (batch + daemon) keeps reading a stable path.
#   3. Selfplay order (order model): compute_selfplay_target.py --order
#      advances the cumulative target (TRAIN_SAMPLES_PER_EPOCH /
#      TARGET_REPLAY_RATIO rows per iter) and converts the remaining deficit
#      into a game count (recent rows/game from selfplay.tsv); the train
#      cards split it evenly and run bounded selfplay batches in parallel
#      (selfplay_main --max-games via internal/selfplay.sh, one per card),
#      each settling its own selfplay.tsv row on exit. 0 games when the
#      spare-card daemon already produced ahead (those cards run the
#      persistent daemon continuously, settling every DAEMON_SETTLE_SECONDS).
#      Then shuffle [optionally overlapped with the batch phase].
#   4. Train (FIXED steps = TRAIN_SAMPLES_PER_EPOCH / BATCH_SIZE): networks run
#      in parallel on their LPT-assigned GPUs, active network first on its
#      card; each net exports right after it trains, and the active export
#      re-mirrors latest.pt immediately (the daemon hot-reloads it).
#   5. mcts_probe + view_loss + log a row to data/logs/schedule.tsv.
#
# Usage: bash scripts/run.sh [max_iters]
#   max_iters (CLI arg): stop after this many iters.
#   MAX_TIME_SECONDS (run.cfg): stop after this many seconds of loop time.
#   If neither is set, runs until Ctrl+C.
set -euo pipefail

DAEMON_PID=""
# EXIT backstop for the persistent selfplay daemon. A `set -e` failure or an
# explicit abort `exit` (shuffle/train failure paths below) goes through
# neither the INT/TERM trap nor the normal shutdown, which would orphan the
# daemon (+ its inference servers) onto the spare GPUs — and unlike per-card
# selfplay the daemon has no startup reap, so the next run would stack a second
# one. Idempotent (kills an already-dead pid harmlessly). The INT/TERM handler
# disables it (trap - ... EXIT) since it kills the daemon itself.
trap '[[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null || true' EXIT
# `trap "" INT TERM` (ignore) before `kill 0`: we are in our own process
# group, so the group kill would otherwise SIGTERM this shell mid-trap and we
# would die with rc=143 before reaching `exit 130`.
trap 'trap "" INT TERM; trap - EXIT; persist_runtime 2>/dev/null || true; echo "$(_tag Run) interrupted; stopping."; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

# Shared color/centered logging tags ($(_tag Run) etc).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/internal/log_common.sh"

# Cumulative wall-clock that survives Ctrl+C / resume. Persisted as plain
# seconds in data/logs/runtime_seconds.txt; each run adds its own session
# time on top of the stored total. Defined here (before any trap can fire)
# but the file lives under $DATA_DIR, populated once that's known below.
RUNTIME_FILE=""
PREV_RUNTIME=0
LOOP_START_SECONDS=0
persist_runtime() {
    [[ -n "$RUNTIME_FILE" ]] || return 0
    local total=$(( PREV_RUNTIME + SECONDS - LOOP_START_SECONDS ))
    printf '%d' "$total" > "$RUNTIME_FILE.tmp" && mv "$RUNTIME_FILE.tmp" "$RUNTIME_FILE"
}
fmt_hms() {  # seconds -> "12h03m41s"
    local s=$1
    printf '%dh%02dm%02ds' $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 ))
}

# Pick experiment config. Default = configs/baseline; override with:
#   CONFIG_DIR=configs/nsim_64 bash scripts/run.sh
CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "$(_tag Run) no config dir at $CONFIG_DIR" >&2; exit 1; }
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

# Per-experiment build tree (see build.sh): keeps this run's binaries isolated
# from other experiments' (different MAX_BOARD_SIZE) in their own build dirs.
BUILD_DIR="${BUILD_DIR:-$DATA_DIR/build}"

if [[ -z "${NETWORKS:-}" || -z "${SELFPLAY_SCHEDULE:-}" ]]; then
    echo "$(_tag Run) NETWORKS and SELFPLAY_SCHEDULE must be set in run.cfg" >&2
    exit 1
fi

# Networks list (supports comma and/or whitespace separators).
read -ra NET_ARR <<< "$(echo "$NETWORKS" | tr ',' ' ')"
if [[ "${#NET_ARR[@]}" -eq 0 ]]; then
    echo "$(_tag Run) NETWORKS parsed to empty list" >&2
    exit 1
fi
FIRST_NET="${NET_ARR[0]}"

mkdir -p "$DATA_DIR"/{models,selfplay,shuffled/current,logs}
for net in "${NET_ARR[@]}"; do
    mkdir -p "$DATA_DIR/nets/$net"
done

PY=${PY:-python}

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
echo "$(_tag Run) detected GPU_NUM=$GPU_NUM"

# Keep this experiment's C++ binaries in sync with run.cfg / sources. build.sh
# owns the first-time configure (libtorch/nvcc/arch + MAX_BOARD_SIZE from
# run.cfg) and the reconfigure-on-CONFIG_DIR-change guard, building into this
# experiment's BUILD_DIR. No-op (~<1s) when nothing changed. CONFIG_DIR is
# already exported above, so build.sh resolves the same BUILD_DIR we do.
bash "$SCRIPT_DIR/build.sh"

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
        echo "$(_tag Run) bootstrapping random-init model for $net"
        ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" --network "$net" )
    fi
done

mirror_active_to_models() {
    local active="$1"
    local src_pt="$DATA_DIR/nets/$active/latest.pt"
    local src_meta="$DATA_DIR/nets/$active/latest.meta.json"
    local dst_pt="$DATA_DIR/models/latest.pt"
    local dst_meta="$DATA_DIR/models/latest.meta.json"
    # Meta first, pt last: the daemon triggers on the pt mtime and then reads
    # the meta, so the meta must already be current when the trigger lands.
    if [[ -f "$src_meta" ]]; then
        cp "$src_meta" "${dst_meta}.tmp" && mv "${dst_meta}.tmp" "$dst_meta"
    fi
    cp "$src_pt" "${dst_pt}.tmp" && mv "${dst_pt}.tmp" "$dst_pt"
}

# Print a card's network list with the active network moved to the front:
# only its weights drive selfplay, so training it first shortens the window
# where cards already back in selfplay produce on stale weights (design §2.5).
order_card_nets() {
    local active="$1"; shift
    local head="" tail="" n
    for n in "$@"; do
        if [[ "$n" == "$active" ]]; then head="$n"; else tail+=" $n"; fi
    done
    echo "$head$tail"
}

# The persistent daemon (internal/selfplay_daemon.sh, owns the spare cards) gets
# the same pidfile + startup-reap treatment as per-card selfplay, so a run that
# died without cleanup (SIGKILL — the EXIT trap can't run) doesn't leave it
# orphaned on the spare GPUs and stack a second daemon next startup.
daemon_pidfile() { echo "$DATA_DIR/logs/selfplay_daemon.pid"; }

reap_orphan_daemon() {
    local pidfile pid i
    pidfile="$(daemon_pidfile)"
    [[ -f "$pidfile" ]] || return 0
    pid=$(cat "$pidfile" 2>/dev/null)
    # ps-args check guards against killing an unrelated process that reused the
    # pid after a clean exit (where the daemon is already gone).
    if [[ "$pid" =~ ^[0-9]+$ ]] && \
       [[ "$(ps -p "$pid" -o args= 2>/dev/null)" == *internal/selfplay_daemon.sh* ]]; then
        echo "$(_tag Run) reaping orphan selfplay daemon from a previous run (pid=$pid)"
        kill "$pid" 2>/dev/null || true
        for ((i=0; i<150; i++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "$(_tag Run) orphan daemon did not exit in 30s; SIGKILL"
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pidfile"
}

# Now that all networks have init artifacts, populate the canonical mirror
# at data/models/ so selfplay (main + daemon) has something to load before
# we even enter the loop.
# `|| true`: under `set -e` a schedule.py failure would otherwise kill the
# whole run silently (stderr is discarded here); fall back to FIRST_NET and
# let the in-loop call at the iter top surface the real error.
INITIAL_ACTIVE=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" 2>/dev/null || true )
INITIAL_ACTIVE="${INITIAL_ACTIVE:-$FIRST_NET}"
mirror_active_to_models "$INITIAL_ACTIVE"

# Reap a persistent daemon orphaned by a previous run that didn't exit
# through the clean paths (e.g. run.sh got SIGKILLed). Must happen before any
# training below (catch-up included) and before we start our own daemon —
# an orphan would collide on the cards / stack a second daemon.
reap_orphan_daemon

# --- Catch-up for mid-iter Ctrl+C ---
# Any net with iter < max_iter was interrupted before training that iter.
# Train + export it at iter=max_iter on the still-intact shuffled/current/
# data (the target accounting for that iter was already advanced in
# selfplay_target.json, so do NOT re-run compute_selfplay_target.py / selfplay
# / shuffle here).
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
                echo "$(_tag Run) catch-up: lead steps unavailable, falling back to 1 epoch ($CATCHUP_STEPS steps)"
            fi
            echo "$(_tag Run) resume catch-up: lagging=(${LAGGING[*]}) at iter=$max_iter steps=$CATCHUP_STEPS (using existing shuffled/current/)"
            for net in "${LAGGING[@]}"; do
                echo "$(_tag Run) catch-up: training $net at iter=$max_iter"
                TRAIN_STEPS_PER_EPOCH="$CATCHUP_STEPS" \
                    bash "$SCRIPT_DIR/internal/train.sh" "$max_iter" "$net"
                bash "$SCRIPT_DIR/internal/export.sh" "$max_iter" "$net"
            done
            # Re-mirror in case the active network was among the lagging.
            mirror_active_to_models "$INITIAL_ACTIVE"
        else
            echo "$(_tag Run) WARNING: lagging networks (${LAGGING[*]}) behind max_iter=$max_iter but shuffled/current/ is empty; cannot catch up — those iters stay as gaps in train.tsv"
        fi
    fi
fi

# LPT assignment of networks to GPUs (cost = blocks*channels^2; fixed across
# iters, so compute once). Each GPU 0..GPU_NUM-1 gets a (possibly empty) list
# of networks to train serially; cards with no networks are free for the
# selfplay daemon.
mapfile -t CARD_NETS < <( cd "$ROOT/python" && "$PY" schedule_train_gpus.py --gpus "$GPU_NUM" --networks "$NETWORKS" )
if [[ "${#CARD_NETS[@]}" -ne "$GPU_NUM" ]]; then
    echo "$(_tag Run) schedule_train_gpus.py returned ${#CARD_NETS[@]} lines, expected $GPU_NUM" >&2
    exit 1
fi
TRAIN_GPUS=(); DAEMON_GPUS=()
for g in "${!CARD_NETS[@]}"; do
    if [[ -n "${CARD_NETS[$g]// /}" ]]; then TRAIN_GPUS+=("$g"); else DAEMON_GPUS+=("$g"); fi
done
echo "$(_tag Run) LPT train assignment (GPU_NUM=$GPU_NUM):"
for g in "${!CARD_NETS[@]}"; do
    echo "$(_tag Run)   gpu $g -> ${CARD_NETS[$g]:-<selfplay>}"
done

# Daemon hot-reloads $DATA_DIR/models/latest.pt (the mirror above keeps that
# pointing at the active network's TorchScript). It owns ONLY the cards that
# training doesn't use; when training fills every card (N <= M) it isn't started.
if [[ "${#DAEMON_GPUS[@]}" -gt 0 ]]; then
    SELFPLAY_DAEMON_GPUS=$(IFS=,; echo "${DAEMON_GPUS[*]}")
    export SELFPLAY_DAEMON_GPUS
    echo "$(_tag Run) starting selfplay daemon on GPUs $SELFPLAY_DAEMON_GPUS"
    bash "$SCRIPT_DIR/internal/selfplay_daemon.sh" &
    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$(daemon_pidfile)"
else
    echo "$(_tag Run) all $GPU_NUM GPU(s) used for training (N<=M); no selfplay daemon"
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
# rebuild and model bootstrap above are excluded. PREV_RUNTIME carries the
# accumulated seconds from earlier (interrupted) sessions; the cumulative
# total = PREV_RUNTIME + this session's elapsed.
RUNTIME_FILE="$DATA_DIR/logs/runtime_seconds.txt"
PREV_RUNTIME=$(cat "$RUNTIME_FILE" 2>/dev/null || true)
[[ "$PREV_RUNTIME" =~ ^[0-9]+$ ]] || PREV_RUNTIME=0
LOOP_START_SECONDS=$SECONDS

while true; do
    # Persist + report cumulative wall-clock (survives Ctrl+C / resume).
    persist_runtime
    _total_rt=$(( PREV_RUNTIME + SECONDS - LOOP_START_SECONDS ))
    _session_rt=$(( SECONDS - LOOP_START_SECONDS ))

    echo ""
    echo "=================================================================="
    echo "$(_tag Run) === iter $iter ===   runtime $(fmt_hms "$_total_rt") (session $(fmt_hms "$_session_rt"))"

    # Cumulative selfplay totals so far (through previous iter).
    # selfplay.tsv schema: producer iter_or_version games rows ...
    awk -v tag="$(_tag Run)" '
        function fmt_sci(x,   s, n, m, e) {
            if (x+0 == 0) return "0"
            s = sprintf("%.1e", x)
            n = index(s, "e")
            m = substr(s, 1, n-1)
            e = substr(s, n+1) + 0
            if (substr(m, length(m)-1) == ".0") m = substr(m, 1, length(m)-2)
            return m "e" e
        }
        NR>1 {g+=$3; r+=$4}
        END {printf "%s cumulative so far: games=%s samples=%s (main+daemon)\n", tag, fmt_sci(g+0), fmt_sci(r+0)}
    ' "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null \
        || echo "$(_tag Run) cumulative so far: games=0 samples=0"

    # Decide active network for this iter and refresh the mirror.
    ACTIVE_NETWORK=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" )
    export ACTIVE_NETWORK
    mirror_active_to_models "$ACTIVE_NETWORK"

    # Log schedule decision (one row per iter, including no-train iters).
    CUM_SAMPLES=$(awk 'NR>1 {r+=$4} END {printf "%d", r+0}' "$DATA_DIR/logs/selfplay.tsv" 2>/dev/null || echo 0)
    printf "%d\t%d\t%s\n" "$iter" "$CUM_SAMPLES" "$ACTIVE_NETWORK" >> "$SCHEDULE_LOG"

    # (1) Selfplay order [+ overlapped shuffle]. --order advances target_cum
    # by TRAIN_SAMPLES_PER_EPOCH / TARGET_REPLAY_RATIO, subtracts settled
    # cum_rows (the spare-card daemon keeps producing and counts), and
    # converts the remaining deficit into a game count via recent rows/game
    # (0 when production already ran ahead — ratio floats below target).
    # Train cards split the order evenly and run bounded batches in parallel;
    # each settles its own selfplay.tsv row on exit (SIGTERM included, so an
    # interrupted iter re-orders only the true remainder on resume).
    # SHUFFLE_RC=0 success, =2 skipped (N < MIN_ROWS), other = failure.
    ORDER_GAMES=$( cd "$ROOT/python" && "$PY" compute_selfplay_target.py \
        --data-dir "$DATA_DIR" --iter "$iter" --order )

    SHUFFLE_RC=0
    SHUFFLE_PID=""
    if [[ "$OVERLAP_SHUFFLE" == "1" && "$iter" -gt 0 ]]; then
        # Pipelined: shuffle (CPU) concurrent with the batch phase (GPU).
        # Shuffle sees data through the previous iter; training then uses a
        # 1-iter-lagged window. Rows produced by this iter's batches are
        # ignored by this shuffle (it snapshots the file list at start) and
        # counted next iter.
        echo "$(_tag Run) shuffle (bg) || selfplay batch (fg)"
        bash "$SCRIPT_DIR/internal/shuffle.sh" &
        SHUFFLE_PID=$!
    fi

    BATCH_S=0
    if [[ "$ORDER_GAMES" -gt 0 ]]; then
        N_TRAIN_GPUS="${#TRAIN_GPUS[@]}"
        CARD_GAMES=$(( (ORDER_GAMES + N_TRAIN_GPUS - 1) / N_TRAIN_GPUS ))
        echo "$(_tag Run) selfplay order: $ORDER_GAMES games -> $N_TRAIN_GPUS card(s) x $CARD_GAMES"
        BATCH_T0=$SECONDS
        BATCH_PIDS=()
        for g in "${TRAIN_GPUS[@]}"; do
            MAIN_GPU="$g" bash "$SCRIPT_DIR/internal/selfplay.sh" "$iter" "$CARD_GAMES" &
            BATCH_PIDS+=("$!")
        done
        BATCH_FAIL=0
        for pid in "${BATCH_PIDS[@]}"; do
            wait "$pid" || BATCH_FAIL=1
        done
        BATCH_S=$(( SECONDS - BATCH_T0 ))
        if [[ "$BATCH_FAIL" -ne 0 ]]; then
            echo "$(_tag Run) a selfplay batch failed; aborting." >&2
            if [[ -n "$SHUFFLE_PID" ]]; then
                # shuffle.sh has no trap; TERM the python child too.
                pkill -TERM -P "$SHUFFLE_PID" 2>/dev/null || true
                kill "$SHUFFLE_PID" 2>/dev/null || true
                wait "$SHUFFLE_PID" 2>/dev/null || true
            fi
            exit 1
        fi
    else
        echo "$(_tag Run) selfplay order: 0 games (production ahead)"
    fi
    # Post-batch ledger snapshot -> ratio.tsv (wait_s = batch seconds).
    ( cd "$ROOT/python" && "$PY" compute_selfplay_target.py \
        --data-dir "$DATA_DIR" --iter "$iter" --log-row --batch-seconds "$BATCH_S" )

    if [[ -n "$SHUFFLE_PID" ]]; then
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        bash "$SCRIPT_DIR/internal/shuffle.sh" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "$(_tag Run) shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "$(_tag Run) shuffle failed with code $SHUFFLE_RC"
        exit "$SHUFFLE_RC"
    else
        # (2) Fixed train volume: train_steps = SAMPLES_PER_EPOCH / BATCH_SIZE.
        TRAIN_STEPS=$(( ${TRAIN_SAMPLES_PER_EPOCH:-128000} / ${BATCH_SIZE:-128} ))
        if [[ "$TRAIN_STEPS" -le 0 ]]; then
            echo "$(_tag Run) train_steps=0 (check SAMPLES_PER_EPOCH/BATCH_SIZE); skipping train+export"
        else
            # (3) train + export: LPT-assigned networks run in PARALLEL across
            # GPUs; each GPU trains its list serially (shared shuffled pool,
            # read-only), the ACTIVE network first. Each net exports right
            # after it trains, on the same card; the active export re-mirrors
            # latest.pt immediately, so the persistent daemon hot-reloads the
            # new weights while the slower cards still train. A card that
            # finished its whole list just idles until the stragglers are
            # done (next iter's batch phase re-uses it).
            # Wait for all cards; a failure on any aborts the iter.
            TRAIN_PIDS=()
            for g in "${TRAIN_GPUS[@]}"; do
                (
                    for net in $(order_card_nets "$ACTIVE_NETWORK" ${CARD_NETS[$g]}); do
                        echo "$(_tag Run) ---- training $net on gpu $g (steps=$TRAIN_STEPS) ----"
                        CUDA_VISIBLE_DEVICES="$g" TRAIN_STEPS_PER_EPOCH="$TRAIN_STEPS" \
                            bash "$SCRIPT_DIR/internal/train.sh" "$iter" "$net"
                        echo "$(_tag Run) ---- exporting $net (gpu $g) ----"
                        CUDA_VISIBLE_DEVICES="$g" \
                            bash "$SCRIPT_DIR/internal/export.sh" "$iter" "$net"
                        if [[ "$net" == "$ACTIVE_NETWORK" ]]; then
                            mirror_active_to_models "$ACTIVE_NETWORK"
                            echo "$(_tag Run) active $net mirrored -> latest.pt (daemon hot-reloads)"
                        fi
                    done
                ) &
                TRAIN_PIDS+=("$!")
            done
            TRAIN_FAIL=0
            for pid in "${TRAIN_PIDS[@]}"; do
                wait "$pid" || TRAIN_FAIL=1
            done
            if [[ "$TRAIN_FAIL" -ne 0 ]]; then
                echo "$(_tag Run) a training subprocess failed; aborting." >&2
                exit 1
            fi

            # (4) post-export diagnostic: empty-board MCTS rootValue probe
            # on the currently-active network (read via the canonical mirror).
            CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}" "$BUILD_DIR/mcts_probe_2048" \
                --model "$DATA_DIR/models/latest.pt" \
                --config "$CONFIG_DIR/run.cfg" \
                --iter "$iter" \
                --log "$DATA_DIR/logs/probe.tsv" \
                || echo "$(_tag Run) mcts_probe_2048 failed (non-fatal)"

            # (5) plots. loss.png + probe.png redraw every training iter
            # (tracking train.tsv / probe.tsv); selfplay/ratio redraw only when
            # PLOT_EVERY_SAMPLES new selfplay rows accumulated since the last
            # redraw (<=0 = every training iter). CUM_SAMPLES is the
            # start-of-iter snapshot; one iter of lag is fine for a render
            # cadence gate. Run view_loss.py manually for an up-to-the-minute
            # plot.
            ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot --only loss,probe >/dev/null )
            PLOT_STATE="$DATA_DIR/logs/last_plot_rows.txt"
            LAST_PLOT_ROWS=$(cat "$PLOT_STATE" 2>/dev/null || true)
            [[ "$LAST_PLOT_ROWS" =~ ^[0-9]+$ ]] || LAST_PLOT_ROWS=0
            if [[ "${PLOT_EVERY_SAMPLES:-0}" -le 0 ]] \
               || [[ $(( CUM_SAMPLES - LAST_PLOT_ROWS )) -ge "${PLOT_EVERY_SAMPLES:-0}" ]]; then
                ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot --only selfplay,ratio >/dev/null )
                echo "$CUM_SAMPLES" > "$PLOT_STATE"
            fi
        fi
    fi

    # Snapshot cumulative runtime at the end of each iter so a Ctrl+C in the
    # *next* iter resumes from a fresh-as-possible total.
    persist_runtime

    iter=$((iter + 1))
    if [[ -n "$max_iters" && "$iter" -ge "$max_iters" ]]; then
        echo "$(_tag Run) reached max_iters=$max_iters; stopping."
        break
    fi
    if [[ "$MAX_TIME_SECONDS" -gt 0 ]]; then
        # Budget is cumulative across sessions (PREV_RUNTIME + this session).
        total_rt=$(( PREV_RUNTIME + SECONDS - LOOP_START_SECONDS ))
        if [[ "$total_rt" -ge "$MAX_TIME_SECONDS" ]]; then
            echo "$(_tag Run) reached MAX_TIME_SECONDS=$MAX_TIME_SECONDS (cumulative=${total_rt}s); stopping."
            break
        fi
    fi
done

persist_runtime  # final snapshot on clean exit

if [[ -n "$DAEMON_PID" ]]; then
    echo "$(_tag Run) stopping selfplay daemon (pid=$DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
    rm -f "$(daemon_pidfile)"
fi
