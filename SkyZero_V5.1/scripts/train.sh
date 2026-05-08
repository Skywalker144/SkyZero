#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
PY=${PY:-python}
source "$SCRIPT_DIR/paths.cfg"

cd "$ROOT/python"
"$PY" train.py --data-dir "$DATA_DIR" --iter "$iter"
