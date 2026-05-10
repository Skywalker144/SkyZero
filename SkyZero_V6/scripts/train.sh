#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
slot="${2:?slot required}"
PY=${PY:-python}
source "$SCRIPT_DIR/paths.cfg"

# train.py uses cuda:0 unconditionally. Pin it to MAIN_GPU so multi-GPU runs
# (where the daemon owns the other GPUs) don't collide.
export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" train.py --data-dir "$DATA_DIR" --iter "$iter" --slot "$slot"
