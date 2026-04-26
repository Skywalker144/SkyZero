#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
games="${2:?games required}"

SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[selfplay.sh] binary not found or not executable: $SELFPLAY_BIN" >&2
    echo "Build it first: (cd cpp && cmake -B build && cmake --build build -j)" >&2
    exit 1
fi

# Main-loop selfplay always runs on MAIN_GPU only. Multi-GPU spread is the
# job of run_selfplay_daemon.sh (it owns the spare GPUs and has its own
# INFERENCE_SERVER_DEVICES). When INFERENCE_SERVER_DEVICES is already set
# (e.g. test harness override), respect it.
MAIN_GPU="${MAIN_GPU:-0}"
GPU_NUM="${GPU_NUM:-1}"
if [[ -z "${INFERENCE_SERVER_DEVICES:-}" ]]; then
    n="${NUM_INFERENCE_SERVERS:-2}"
    devices=""
    for ((i=0; i<n; i++)); do
        [[ -n "$devices" ]] && devices+=","
        devices+="$MAIN_GPU"
    done
    export INFERENCE_SERVER_DEVICES="$devices"
fi

echo "[selfplay.sh] iter=$iter games=$games main_gpu=$MAIN_GPU devices=${INFERENCE_SERVER_DEVICES}"
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --output-dir "$DATA_DIR/selfplay" \
    --iter "$iter" \
    --max-games "$games" \
    --config "$SCRIPT_DIR/run.cfg" \
    --log-dir "$DATA_DIR/logs"
