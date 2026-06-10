#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
network="${2:?network required (e.g. b5c128)}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
PY=${PY:-python}

# run.sh injects CUDA_VISIBLE_DEVICES so each net exports on the card that
# just trained it; absent that (e.g. catch-up), fall back to MAIN_GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${MAIN_GPU:-0}}"

cd "$ROOT/python"
"$PY" export_model.py \
    --data-dir "$DATA_DIR" \
    --network "$network" \
    --iter "$iter"
