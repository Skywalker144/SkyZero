#!/usr/bin/env bash
# Export one network's state_dict -> TorchScript (nets/<net>/latest.pt).
# Game-agnostic (same as V7.1).
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

export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" export_model.py --data-dir "$DATA_DIR" --network "$network" --iter "$iter"
