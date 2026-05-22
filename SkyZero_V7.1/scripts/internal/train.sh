#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
network="${2:?network required (e.g. b5c128)}"
PY=${PY:-python}
source "$SCRIPTS_DIR/paths.cfg"

# Train runs on MAIN_GPU. The daemon owns the other GPUs (see
# selfplay_daemon.sh) — without this pin, train.py defaults to cuda:0
# and collides with the daemon when MAIN_GPU != 0.
export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" train.py --data-dir "$DATA_DIR" --network "$network" --iter "$iter"
