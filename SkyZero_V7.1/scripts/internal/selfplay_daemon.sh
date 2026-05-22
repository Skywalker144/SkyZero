#!/usr/bin/env bash
# Persistent multi-GPU selfplay daemon.
#
# Owns cuda:1..GPU_NUM-1 (or whatever SELFPLAY_DAEMON_GPUS lists), continuously
# generates selfplay NPZs into data/selfplay/, and hot-reloads latest.pt every
# time the main train loop (run.sh) exports a new model. Runs forever; restart
# behavior is delegated to the inner watchdog loop.
#
# Usage: GPU_NUM=4 bash scripts/internal/selfplay_daemon.sh
#   GPU_NUM<=1 -> exits with a hint (daemon not needed in single-GPU mode).
set -euo pipefail

trap 'trap - INT TERM; echo "[daemon-watchdog] stopping."; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

set -a
# shellcheck disable=SC1091
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
source "$SCRIPTS_DIR/env_paths.cfg"
set +a

export DATA_DIR
mkdir -p "$DATA_DIR"/{models,selfplay,logs}

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[daemon] binary not found or not executable: $SELFPLAY_BIN" >&2
    echo "Build it first: bash scripts/build.sh" >&2
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

# Resolve per-GPU → absolute counts using SELFPLAY_DAEMON_GPUS as the GPU set.
GPU_COUNT=$("$PY" -c "
gpus = '$SELFPLAY_DAEMON_GPUS'.split(',')
gpus = [g for g in gpus if g.strip() != '']
print(len(gpus))
")
if [[ "$GPU_COUNT" -le 0 ]]; then
    echo "[daemon] no GPUs selected for SELFPLAY_DAEMON_GPUS='$SELFPLAY_DAEMON_GPUS'" >&2
    exit 1
fi
SV_PER_GPU="${INFERENCE_SERVERS_PER_GPU:-2}"
WK_PER_GPU="${WORKERS_PER_GPU:-32}"
DSV=$((SV_PER_GPU * GPU_COUNT))
DWK=$((WK_PER_GPU * GPU_COUNT))

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

# Daemon consults this on startup and on every model reload to pick up the
# current NUM_SIMULATIONS_STAGES warmup value. Mirrors selfplay.sh's
# resolution for the main loop. Empty string disables (cfg fallback wins).
SIMS_WARMUP_CMD="cd $ROOT/python && $PY warmup.py num-simulations --data-dir $DATA_DIR"

echo "[daemon-watchdog] gpus=$SELFPLAY_DAEMON_GPUS servers=$DSV per_gpu=$SV_PER_GPU workers=$DWK devices=$INFERENCE_SERVER_DEVICES"
echo "[daemon-watchdog] poll_ms=$POLL_MS"

while true; do
    "$SELFPLAY_BIN" --daemon \
        --model "$DATA_DIR/models/latest.pt" \
        --output-dir "$DATA_DIR/selfplay" \
        --config "$CONFIG_DIR/run.cfg" \
        --log-dir "$DATA_DIR/logs" \
        --model-watch-poll-ms "$POLL_MS" \
        --sims-warmup-cmd "$SIMS_WARMUP_CMD" \
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
