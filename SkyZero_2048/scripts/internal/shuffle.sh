#!/usr/bin/env bash
# Power-law-window shuffle: selfplay/*.npz -> shuffled/current/. Exit 2 = below
# MIN_ROWS (caller skips training this iter). Game-agnostic (same as V7.1).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
PY=${PY:-python}
SHUFFLE_SHARD_ROWS="${SHUFFLE_SHARD_ROWS:-200000}"

cd "$ROOT/python"
"$PY" shuffle.py --data-dir "$DATA_DIR" --shard-rows "$SHUFFLE_SHARD_ROWS"
