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

# Multi-GPU: if GPU_NUM>1 and INFERENCE_SERVER_DEVICES isn't already set
# (in run.cfg or env), round-robin distribute the inference servers across
# GPUs and pass via env. NUM_INFERENCE_SERVERS comes from run.cfg (exported
# by run.sh's `set -a` + source). GPU_NUM=1 (or unset) is a no-op, so the
# binary falls back to "all servers on cuda:0" — identical to single-GPU.
GPU_NUM="${GPU_NUM:-1}"
if [[ -z "${INFERENCE_SERVER_DEVICES:-}" && "$GPU_NUM" -gt 1 ]]; then
    n="${NUM_INFERENCE_SERVERS:-2}"
    devices=""
    for ((i=0; i<n; i++)); do
        [[ -n "$devices" ]] && devices+=","
        devices+="$((i % GPU_NUM))"
    done
    export INFERENCE_SERVER_DEVICES="$devices"
fi

echo "[selfplay.sh] iter=$iter games=$games gpu_num=$GPU_NUM devices=${INFERENCE_SERVER_DEVICES:-<default cuda:0>}"
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --output-dir "$DATA_DIR/selfplay" \
    --iter "$iter" \
    --max-games "$games" \
    --config "$SCRIPT_DIR/run.cfg" \
    --log-dir "$DATA_DIR/logs"
