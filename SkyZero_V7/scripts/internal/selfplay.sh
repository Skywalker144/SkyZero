#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
games="${2:?games required}"

SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"
source "$SCRIPTS_DIR/paths.cfg"
PY="${PY:-python}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[selfplay.sh] binary not found or not executable: $SELFPLAY_BIN" >&2
    echo "Build it first: bash scripts/build.sh" >&2
    exit 1
fi

# Main-loop selfplay always runs on MAIN_GPU only. Multi-GPU spread is the
# job of selfplay_daemon.sh (it owns the spare GPUs and has its own
# INFERENCE_SERVER_DEVICES). When INFERENCE_SERVER_DEVICES is already set
# (e.g. test harness override), respect it.
MAIN_GPU="${MAIN_GPU:-0}"
GPU_NUM="${GPU_NUM:-1}"

# Resolve per-GPU → absolute counts. The main loop owns exactly one GPU.
export NUM_INFERENCE_SERVERS="${INFERENCE_SERVERS_PER_GPU:-2}"
export NUM_WORKERS="${WORKERS_PER_GPU:-32}"

if [[ -z "${INFERENCE_SERVER_DEVICES:-}" ]]; then
    devices=""
    for ((i=0; i<NUM_INFERENCE_SERVERS; i++)); do
        [[ -n "$devices" ]] && devices+=","
        devices+="$MAIN_GPU"
    done
    export INFERENCE_SERVER_DEVICES="$devices"
fi

# Per-iter warmup for NUM_SIMULATIONS. Falls back to cfg's NUM_SIMULATIONS
# when NUM_SIMULATIONS_STAGES has < 2 entries (warmup disabled).
NSIM=$( cd "$ROOT/python" && "$PY" warmup.py num-simulations --data-dir "$DATA_DIR" )

echo "[selfplay.sh] iter=$iter games=$games num_simulations=$NSIM main_gpu=$MAIN_GPU devices=${INFERENCE_SERVER_DEVICES}"
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --output-dir "$DATA_DIR/selfplay" \
    --iter "$iter" \
    --max-games "$games" \
    --num-simulations "$NSIM" \
    --config "$SCRIPTS_DIR/run.cfg" \
    --log-dir "$DATA_DIR/logs"
