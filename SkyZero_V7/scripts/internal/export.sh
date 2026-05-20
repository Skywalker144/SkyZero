#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
PY=${PY:-python}
source "$SCRIPTS_DIR/paths.cfg"

export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" export_model.py \
    --data-dir "$DATA_DIR" \
    --iter "$iter" \
    --ckpt "$DATA_DIR/checkpoints/model_latest.pt"
