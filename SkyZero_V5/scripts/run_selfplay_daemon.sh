#!/usr/bin/env bash
# Persistent multi-GPU selfplay daemon.
#
# Owns cuda:1..GPU_NUM-1 (or whatever SELFPLAY_DAEMON_GPUS lists), continuously
# generates selfplay NPZs into data/selfplay/, and hot-reloads latest.pt every
# time the main train loop (run.sh) exports a new model. Runs forever; restart
# behavior is delegated to the inner watchdog loop.
#
# Usage: GPU_NUM=4 bash scripts/run_selfplay_daemon.sh
#   GPU_NUM<=1 -> exits with a hint (daemon not needed in single-GPU mode).
set -euo pipefail

trap 'trap - INT TERM; echo "[daemon-watchdog] stopping."; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/run.cfg"
if [[ -f "$SCRIPT_DIR/run.cfg.local" ]]; then
    source "$SCRIPT_DIR/run.cfg.local"
fi
set +a

source "$SCRIPT_DIR/paths.cfg"
export DATA_DIR
mkdir -p "$DATA_DIR"/{models,selfplay,logs}

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[daemon] binary not found or not executable: $SELFPLAY_BIN" >&2
    echo "Build it first: (cd cpp && cmake -B build && cmake --build build -j)" >&2
    exit 1
fi

GPU_NUM="${GPU_NUM:-1}"

# Derive SELFPLAY_DAEMON_GPUS from GPU_NUM if user didn't pin it.
if [[ -z "${SELFPLAY_DAEMON_GPUS:-}" ]]; then
    if [[ "$GPU_NUM" -le 1 ]]; then
        echo "[daemon] GPU_NUM<=1, daemon not needed; exiting."
        exit 1
    fi
    gpus=""
    main_gpu="${MAIN_GPU:-0}"
    for ((i=0; i<GPU_NUM; i++)); do
        if [[ "$i" -eq "$main_gpu" ]]; then continue; fi
        [[ -n "$gpus" ]] && gpus+=","
        gpus+="$i"
    done
    SELFPLAY_DAEMON_GPUS="$gpus"
fi

# Daemon's server / worker counts default to the main keys.
DSV="${DAEMON_NUM_INFERENCE_SERVERS:-${NUM_INFERENCE_SERVERS:-2}}"
DWK="${DAEMON_NUM_WORKERS:-${NUM_WORKERS:-32}}"

# Round-robin INFERENCE_SERVER_DEVICES across SELFPLAY_DAEMON_GPUS.
INFERENCE_SERVER_DEVICES=$("$PY" -c "
gpus = '$SELFPLAY_DAEMON_GPUS'.split(',')
gpus = [g for g in gpus if g.strip() != '']
n = $DSV
print(','.join(gpus[i % len(gpus)] for i in range(n)))
")
export INFERENCE_SERVER_DEVICES
export NUM_INFERENCE_SERVERS="$DSV"
export NUM_WORKERS="$DWK"

POLL_MS="${DAEMON_RELOAD_POLL_MS:-2000}"
STATS="$DATA_DIR/logs/daemon_stats.tsv"

echo "[daemon-watchdog] gpus=$SELFPLAY_DAEMON_GPUS servers=$DSV workers=$DWK devices=$INFERENCE_SERVER_DEVICES"
echo "[daemon-watchdog] poll_ms=$POLL_MS stats=$STATS"

while true; do
    "$SELFPLAY_BIN" --daemon \
        --model "$DATA_DIR/models/latest.pt" \
        --output-dir "$DATA_DIR/selfplay" \
        --config "$SCRIPT_DIR/run.cfg" \
        --log-dir "$DATA_DIR/logs" \
        --model-watch-poll-ms "$POLL_MS" \
        --stats-file "$STATS" \
        || rc=$?
    rc=${rc:-0}
    if [[ "$rc" -eq 130 ]]; then
        # SIGINT: user-initiated shutdown.
        echo "[daemon-watchdog] interrupted; exiting."
        exit 0
    fi
    echo "[daemon-watchdog] selfplay_main exited rc=$rc; restarting in 5s"
    unset rc
    sleep 5
done
