#!/usr/bin/env bash
# Single-GPU sps bench for V6 selfplay, matched to V5 4090 baseline params:
# Gumbel, sims=400, no PCR, USE_VCT=0, NUM_WORKERS=64, NUM_INFERENCE_SERVERS=2.
#
# Usage:
#   bash scripts/smoke_bench/bench.sh                 # 200 games, GPU 0
#   GAMES=500 GPU=0 bash scripts/smoke_bench/bench.sh # override game count / gpu
#
# Reports rows / wall_seconds / sps. Bootstraps a random-init b10c128 model
# into <repo>/data_smoke_bench on first run; reuse on subsequent runs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

GAMES="${GAMES:-200}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-64}"
SERVERS="${SERVERS:-2}"
PY="${PY:-python}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"
DATA_DIR="${DATA_DIR:-$ROOT/data_smoke_bench}"

# Derive a per-run cfg from bench.cfg with NUM_WORKERS / NUM_INFERENCE_SERVERS
# overridden so we can sweep parallelism without editing bench.cfg.
RUN_CFG="$DATA_DIR/.bench_run.cfg"
mkdir -p "$DATA_DIR"
sed -e "s|^NUM_WORKERS=.*|NUM_WORKERS=$WORKERS|" \
    -e "s|^NUM_INFERENCE_SERVERS=.*|NUM_INFERENCE_SERVERS=$SERVERS|" \
    "$SCRIPT_DIR/bench.cfg" > "$RUN_CFG"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[bench] selfplay_main missing at $SELFPLAY_BIN — run scripts/build.sh first" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"/{models,selfplay,logs,checkpoints,shuffled/current}

# Bootstrap random-init model if absent. Different boxes will get different
# random weights but game-length distributions average out across $GAMES.
if [[ ! -f "$DATA_DIR/models/latest.pt" ]]; then
    echo "[bench] bootstrapping random-init model"
    ( cd "$ROOT/python" && "$PY" init_model.py --data-dir "$DATA_DIR" )
fi

# Pin all inference servers to the chosen GPU.
DEVS=""
for ((i=0; i<SERVERS; i++)); do
    [[ -n "$DEVS" ]] && DEVS+=","
    DEVS+="$GPU"
done
export INFERENCE_SERVER_DEVICES="$DEVS"

LOG="$DATA_DIR/logs/bench.log"
TIMEFILE="$DATA_DIR/logs/bench.time"

echo "[bench] gpu=$GPU games=$GAMES sims=400 workers=$WORKERS servers=$SERVERS"
/usr/bin/time -f "%e" -o "$TIMEFILE" "$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --output-dir "$DATA_DIR/selfplay" \
    --iter 0 \
    --max-games "$GAMES" \
    --num-simulations 400 \
    --cheap-search-visits 200 \
    --config "$RUN_CFG" \
    --log-dir "$DATA_DIR/logs" > "$LOG" 2>&1

# Parse the just-written iter row from last_run.tsv (the run we kicked off has
# iter=0; selfplay appends one row per invocation).
TSV="$DATA_DIR/logs/last_run.tsv"
read GAMES_DONE ROWS SECONDS_INNER AVG_LEN <<<"$(awk -F'\t' '$1==0 {print $2, $3, $4, $7}' "$TSV" | tail -1)"
WALL=$(cat "$TIMEFILE")
SPS=$("$PY" -c "print(f'{$ROWS / $WALL:.1f}')")

echo
echo "[bench] === result ==="
printf "  games=%d  rows=%d  avg_len=%s  inner_t=%ss  wall=%ss  sps=%s\n" \
    "$GAMES_DONE" "$ROWS" "$AVG_LEN" "$SECONDS_INNER" "$WALL" "$SPS"
echo "  log: $LOG"
