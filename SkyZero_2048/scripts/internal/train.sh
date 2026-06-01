#!/usr/bin/env bash
# Train one network on the current shuffle window. env TRAIN_STEPS_PER_EPOCH
# (from bucket.py) = gradient steps this iter. Game-agnostic (same as V7.1).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
network="${2:?network required (e.g. b6c96)}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
PY=${PY:-python}

# Train runs on MAIN_GPU; the daemon owns the other GPUs.
export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" train.py --data-dir "$DATA_DIR" --network "$network" --iter "$iter"
