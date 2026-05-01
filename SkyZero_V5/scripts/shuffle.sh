#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

PY=${PY:-python}
source "$SCRIPT_DIR/paths.cfg"
SHUFFLE_SHARD_ROWS="${SHUFFLE_SHARD_ROWS:-200000}"

cd "$ROOT/python"
"$PY" shuffle.py --data-dir "$DATA_DIR" --shard-rows "$SHUFFLE_SHARD_ROWS"
