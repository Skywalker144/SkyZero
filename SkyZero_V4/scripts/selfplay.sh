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

echo "[selfplay.sh] iter=$iter games=$games"
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --output-dir "$DATA_DIR/selfplay" \
    --iter "$iter" \
    --max-games "$games" \
    --config "$SCRIPT_DIR/run.cfg" \
    --log-dir "$DATA_DIR/logs"
