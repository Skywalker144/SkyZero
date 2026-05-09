#!/usr/bin/env bash
# Dual-GPU sps bench: spawn 2 selfplay_main processes in parallel (one per GPU),
# matching how run.sh + run_selfplay_daemon.sh wire things in real training
# (independent processes pinned to one GPU each, no cross-process coordination).
#
# Usage:
#   bash scripts/smoke_bench/dual_bench.sh
#   GAMES=300 bash scripts/smoke_bench/dual_bench.sh   # per-GPU game count
#
# Reports per-GPU rows + combined sps = (rows_a + rows_b) / max(wall_a, wall_b).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

GAMES="${GAMES:-300}"
PY="${PY:-python}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"
DATA_DIR="${DATA_DIR:-$ROOT/data_smoke_bench}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[dual_bench] selfplay_main missing at $SELFPLAY_BIN — run scripts/build.sh first" >&2
    exit 1
fi

# Bootstrap shared random-init model (both processes load the same one).
mkdir -p "$DATA_DIR"/{models,checkpoints}
if [[ ! -f "$DATA_DIR/models/latest.pt" ]]; then
    echo "[dual_bench] bootstrapping random-init model"
    ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" )
fi

# Per-GPU output dirs to avoid NPZ filename collisions.
mkdir -p "$DATA_DIR"/{selfplay_a,selfplay_b,logs_a,logs_b}

run_one() {
    local gpu="$1" out="$2" logs="$3" iter="$4" timefile="$5" log="$6"
    INFERENCE_SERVER_DEVICES="$gpu,$gpu" \
        /usr/bin/time -f "%e" -o "$timefile" \
        "$SELFPLAY_BIN" \
            --model "$DATA_DIR/models/latest.pt" \
            --output-dir "$DATA_DIR/$out" \
            --iter "$iter" \
            --max-games "$GAMES" \
            --num-simulations 400 \
            --cheap-search-visits 200 \
            --config "$SCRIPT_DIR/bench.cfg" \
            --log-dir "$DATA_DIR/$logs" > "$log" 2>&1
}

LOG_A="$DATA_DIR/logs_a/dual_bench.log"
LOG_B="$DATA_DIR/logs_b/dual_bench.log"
TIME_A="$DATA_DIR/logs_a/dual_bench.time"
TIME_B="$DATA_DIR/logs_b/dual_bench.time"

echo "[dual_bench] GPU 0 + GPU 1, $GAMES games each, sims=400, USE_VCT=0"
run_one 0 selfplay_a logs_a 100 "$TIME_A" "$LOG_A" &
PID_A=$!
run_one 1 selfplay_b logs_b 200 "$TIME_B" "$LOG_B" &
PID_B=$!

wait "$PID_A"
wait "$PID_B"

# Parse each side from its own last_run.tsv (one row per invocation).
TSV_A="$DATA_DIR/logs_a/last_run.tsv"
TSV_B="$DATA_DIR/logs_b/last_run.tsv"

read GAMES_A ROWS_A SEC_A AVG_A <<<"$(awk -F'\t' '$1==100 {print $2, $3, $4, $7}' "$TSV_A" | tail -1)"
read GAMES_B ROWS_B SEC_B AVG_B <<<"$(awk -F'\t' '$1==200 {print $2, $3, $4, $7}' "$TSV_B" | tail -1)"
WALL_A=$(cat "$TIME_A")
WALL_B=$(cat "$TIME_B")

TOTAL_ROWS=$(( ROWS_A + ROWS_B ))
WALL_MAX=$("$PY" -c "print(max($WALL_A, $WALL_B))")
COMBINED_SPS=$("$PY" -c "print(f'{$TOTAL_ROWS / $WALL_MAX:.1f}')")
PER_GPU_SPS_A=$("$PY" -c "print(f'{$ROWS_A / $WALL_A:.1f}')")
PER_GPU_SPS_B=$("$PY" -c "print(f'{$ROWS_B / $WALL_B:.1f}')")

echo
echo "[dual_bench] === result ==="
printf "  GPU 0:  games=%s rows=%s avg_len=%s wall=%ss  sps=%s\n" "$GAMES_A" "$ROWS_A" "$AVG_A" "$WALL_A" "$PER_GPU_SPS_A"
printf "  GPU 1:  games=%s rows=%s avg_len=%s wall=%ss  sps=%s\n" "$GAMES_B" "$ROWS_B" "$AVG_B" "$WALL_B" "$PER_GPU_SPS_B"
printf "  TOTAL:  rows=%d  wall=%ss  combined sps=%s\n" "$TOTAL_ROWS" "$WALL_MAX" "$COMBINED_SPS"
