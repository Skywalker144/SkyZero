#!/usr/bin/env bash
# Wall-clock head-to-head: root-parallel (ParallelMCTS, leaf batching) vs
# tree-parallel (TreeParallelMCTS, shared tree with virtual loss).
#
# Sweeps leaf_batch_size and search_threads; prints a TSV report to stdout
# (and optionally appends to $LOG_FILE when set).
#
# Env overrides: MODEL, BENCH_BIN, BENCH_CFG, SIMS, SEARCHES, WARMUP,
#                ROOT_BATCH, TREE_THREADS, LOG_FILE.
# Extra args are forwarded to the binary.
#
# Example:
#   bash scripts/bench.sh
#   SIMS=800 SEARCHES=32 bash scripts/bench.sh
#   ROOT_BATCH=1,8,32 TREE_THREADS=1,4,16 bash scripts/bench.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[bench.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }

source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"

MODEL="${MODEL:-$DATA_DIR/models/latest.pt}"
BUILD_DIR="${BUILD_DIR:-$DATA_DIR/build}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/mcts_bench}"
BENCH_CFG="${BENCH_CFG:-$CONFIG_DIR/run.cfg}"
SIMS="${SIMS:-400}"
SEARCHES="${SEARCHES:-16}"
WARMUP="${WARMUP:-2}"
ROOT_BATCH="${ROOT_BATCH:-1,4,8,16,32}"
TREE_THREADS="${TREE_THREADS:-1,2,4,8,16,32}"
LOG_FILE="${LOG_FILE:-}"

[[ -f "$MODEL" ]]     || { echo "no model at $MODEL"; exit 1; }
[[ -f "$BENCH_CFG" ]] || { echo "no config at $BENCH_CFG"; exit 1; }
[[ -x "$BENCH_BIN" ]] || { echo "build first: bash scripts/build.sh --target mcts_bench (-> $BUILD_DIR)"; exit 1; }

CMD=(
    "$BENCH_BIN"
    --model "$MODEL"
    --config "$BENCH_CFG"
    --sims "$SIMS"
    --searches "$SEARCHES"
    --warmup "$WARMUP"
    --root-batch "$ROOT_BATCH"
    --tree-threads "$TREE_THREADS"
    "$@"
)

if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    "${CMD[@]}" | tee -a "$LOG_FILE"
else
    "${CMD[@]}"
fi
