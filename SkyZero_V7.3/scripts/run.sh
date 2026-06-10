#!/usr/bin/env bash
# Main orchestration loop for multi-network training.
#
# Each iter:
#   1. Compute ACTIVE_NETWORK from cumulative selfplay samples + SELFPLAY_SCHEDULE.
#   2. Mirror data/nets/$ACTIVE_NETWORK/{latest.pt,latest.meta.json} -> data/models/
#      so C++ selfplay (main + daemon) keeps reading a stable path.
#   3. Production gate: every card selfplays continuously (per-card daemons on
#      train cards, persistent daemon on the rest; C++ settles stats into
#      selfplay.tsv every DAEMON_SETTLE_SECONDS). compute_selfplay_target
#      --wait advances the cumulative target (TRAIN_SAMPLES_PER_EPOCH /
#      TARGET_REPLAY_RATIO rows per iter) and blocks until settled cum_rows
#      reach it. Then shuffle.
#   4. Train (FIXED steps = TRAIN_SAMPLES_PER_EPOCH / BATCH_SIZE): networks run
#      in parallel on their LPT-assigned GPUs, active network first on its
#      card; each net exports right after it trains, and the active export
#      re-mirrors latest.pt immediately (selfplay cards hot-reload it). A card
#      that finished its whole list runs per-card selfplay until re-drafted.
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
trap 'trap - INT TERM EXIT; persist_runtime 2>/dev/null || true; echo "$(_tag Run) interrupted; stopping."; [[ -n "$DAEMON_PID" ]] && kill "$DAEMON_PID" 2>/dev/null; kill 0 2>/dev/null; exit 130' INT TERM

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
echo "$(_tag Run) detected GPU_NUM=$GPU_NUM"

# Keep C++ binaries in sync with run.cfg / sources. cmake parses MAX_BOARD_SIZE
# from run.cfg via CONFIGURE_DEPENDS and bakes it in as -DSKYZERO_MAX_BOARD_SIZE.
# This is a no-op (~<1s) when nothing has changed.
#
# SKYZERO_CONFIG_DIR is a cmake cache var: only refreshed when we pass -D, not
# when CONFIG_DIR changes between invocations. Detect a switched experiment
# and update the cache so MAX_BOARD_SIZE is re-read from the new run.cfg.
_cache="$ROOT/cpp/build/CMakeCache.txt"
if [[ -f "$_cache" ]]; then
    _cached_cfg=$(sed -n 's|^SKYZERO_CONFIG_DIR:PATH=||p' "$_cache")
    if [[ -n "$_cached_cfg" && "$_cached_cfg" != "$CONFIG_DIR" ]]; then
        echo "$(_tag Run) CONFIG_DIR changed ($_cached_cfg -> $CONFIG_DIR); reconfiguring cmake"
        cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DSKYZERO_CONFIG_DIR="$CONFIG_DIR"
    fi
fi
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
    cp "$src_pt" "${dst_pt}.tmp" && mv "${dst_pt}.tmp" "$dst_pt"
    if [[ -f "$src_meta" ]]; then
        cp "$src_meta" "${dst_meta}.tmp" && mv "${dst_meta}.tmp" "$dst_meta"
    fi
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

# --- Per-card transient selfplay (design §2.5; gate model) ---
# Every train card runs internal/card_selfplay.sh (selfplay_main --daemon +
# watchdog, pinned to that card) whenever it isn't training: from startup,
# and again as soon as it finishes its serial network list. Cards are only
# evicted right before the train phase re-drafts them. SIGTERM is a clean
# shutdown chain (wrapper forwards to selfplay_main, which flushes NPZs and
# settles its selfplay.tsv row), so the production accounting stays intact.
# PID files (wrapper pids) live under $DATA_DIR/logs/ so a crashed run's
# orphans get reaped at next startup.
card_selfplay_pidfile() { echo "$DATA_DIR/logs/card_selfplay.gpu$1.pid"; }

start_card_selfplay() {
    local g="$1"
    echo "$(_tag Run) gpu $g -> per-card selfplay (until re-drafted for training)"
    bash "$SCRIPT_DIR/internal/card_selfplay.sh" "$g" &
    echo $! > "$(card_selfplay_pidfile "$g")"
}

kill_card_selfplay() {
    local g="$1" pidfile pid i
    pidfile="$(card_selfplay_pidfile "$g")"
    [[ -f "$pidfile" ]] || return 0
    pid=$(cat "$pidfile" 2>/dev/null)
    if [[ "$pid" =~ ^[0-9]+$ ]] && \
       [[ "$(ps -p "$pid" -o args= 2>/dev/null)" == *internal/card_selfplay.sh* ]]; then
        echo "$(_tag Run) stopping per-card selfplay on gpu $g (pid=$pid)"
        kill "$pid" 2>/dev/null || true
        # Started inside a train subshell, so it's not our direct child and
        # `wait` won't work — poll until the clean shutdown chain (npz flush +
        # tsv settle) releases the card, then SIGKILL wrapper AND child as a
        # last resort (npz writes are tmp+rename, so no torn file behind).
        for ((i=0; i<150; i++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "$(_tag Run) per-card selfplay on gpu $g did not exit in 30s; SIGKILL"
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pidfile"
}

kill_all_card_selfplay() {
    local f g
    for f in "$DATA_DIR"/logs/card_selfplay.gpu*.pid; do
        [[ -e "$f" ]] || continue
        g="${f##*card_selfplay.gpu}"
        kill_card_selfplay "${g%.pid}"
    done
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
INITIAL_ACTIVE=$( cd "$ROOT/python" && "$PY" schedule.py active --data-dir "$DATA_DIR" 2>/dev/null )
INITIAL_ACTIVE="${INITIAL_ACTIVE:-$FIRST_NET}"
mirror_active_to_models "$INITIAL_ACTIVE"

# Reap selfplay (per-card + persistent daemon) orphaned by a previous run that
# didn't exit through the clean paths (e.g. run.sh got SIGKILLed). Must happen
# before any training below (catch-up included) and before we start our own
# daemon — an orphan would collide on the card / stack a second daemon.
kill_all_card_selfplay
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

# Gate model: every card produces from t=0. Train cards run per-card selfplay
# until the train phase re-drafts them (kill_all_card_selfplay there); they
# come back via start_card_selfplay as each finishes its list.
for g in "${TRAIN_GPUS[@]}"; do
    start_card_selfplay "$g"
done

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

    # (1) Production gate [+ overlapped shuffle]. No per-card orders: all
    # cards selfplay continuously (per-card + persistent daemons, settling
    # into selfplay.tsv every DAEMON_SETTLE_SECONDS); compute_selfplay_target
    # --wait advances target_cum by TRAIN_SAMPLES_PER_EPOCH /
    # TARGET_REPLAY_RATIO and blocks until settled cum_rows reach it
    # (instant when production already ran ahead — ratio floats below target).
    # SHUFFLE_RC=0 success, =2 skipped (N < MIN_ROWS), other = failure.
    SHUFFLE_RC=0
    if [[ "$OVERLAP_SHUFFLE" == "1" && "$iter" -gt 0 ]]; then
        # Pipelined: shuffle (CPU) concurrent with the gate wait (GPU cards
        # keep producing). Shuffle sees data through the previous iter;
        # training then uses a 1-iter-lagged window. Rows produced during the
        # wait are ignored by this shuffle (it snapshots the file list at
        # start) and counted next iter.
        echo "$(_tag Run) shuffle (bg) || production gate (fg)"
        bash "$SCRIPT_DIR/internal/shuffle.sh" &
        SHUFFLE_PID=$!
        ( cd "$ROOT/python" && "$PY" compute_selfplay_target.py --data-dir "$DATA_DIR" --iter "$iter" --wait )
        wait "$SHUFFLE_PID" || SHUFFLE_RC=$?
    else
        ( cd "$ROOT/python" && "$PY" compute_selfplay_target.py --data-dir "$DATA_DIR" --iter "$iter" --wait )
        bash "$SCRIPT_DIR/internal/shuffle.sh" || SHUFFLE_RC=$?
    fi

    if [[ "$SHUFFLE_RC" -eq 2 ]]; then
        echo "$(_tag Run) shuffle skipped (N < MIN_ROWS); skipping train+export this iter"
    elif [[ "$SHUFFLE_RC" -ne 0 ]]; then
        echo "$(_tag Run) shuffle failed with code $SHUFFLE_RC"
        kill_all_card_selfplay
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
            # latest.pt immediately, so cards already back in selfplay (and
            # the persistent daemon) hot-reload the new weights while the
            # slower cards still train. A card that finished its whole list
            # starts per-card selfplay to fill the wait-for-stragglers window.
            # Wait for all cards; a failure on any aborts the iter.
            kill_all_card_selfplay   # training re-drafts every card
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
                            echo "$(_tag Run) active $net mirrored -> latest.pt (selfplay cards hot-reload)"
                        fi
                    done
                    start_card_selfplay "$g"
                ) &
                TRAIN_PIDS+=("$!")
            done
            TRAIN_FAIL=0
            for pid in "${TRAIN_PIDS[@]}"; do
                wait "$pid" || TRAIN_FAIL=1
            done
            if [[ "$TRAIN_FAIL" -ne 0 ]]; then
                echo "$(_tag Run) a training subprocess failed; aborting." >&2
                kill_all_card_selfplay
                exit 1
            fi

            # (4) post-export diagnostic: empty-board MCTS rootValue probe
            # on the currently-active network (read via the canonical mirror).
            CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}" "$ROOT/cpp/build/mcts_probe" \
                --model "$DATA_DIR/models/latest.pt" \
                --config "$CONFIG_DIR/run.cfg" \
                --iter "$iter" \
                --log "$DATA_DIR/logs/probe.tsv" \
                || echo "$(_tag Run) mcts_probe failed (non-fatal)"

            # (5) plot loss curve (combined plot for all networks)
            ( cd "$ROOT/python" && "$PY" view_loss.py --data-dir "$DATA_DIR" --plot >/dev/null )
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

kill_all_card_selfplay
if [[ -n "$DAEMON_PID" ]]; then
    echo "$(_tag Run) stopping selfplay daemon (pid=$DAEMON_PID)"
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
    rm -f "$(daemon_pidfile)"
fi
