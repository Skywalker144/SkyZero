#!/bin/bash -eu
set -o pipefail

# =============================================================================
# SkyZero V4 — Selfplay-only worker
# Runs selfplay in a loop; intended for secondary machines that contribute
# game data to the main training host via Syncthing (or any file sync tool).
#
# Setup:
#   - This machine: runs selfplay.sh, writes NPZ data to BASEDIR/selfplay/NODE_ID/
#   - Main machine: runs run.sh (full loop); reads selfplay data from ALL subdirs
#                   of BASEDIR/selfplay/ (shuffle.py uses os.walk recursively)
#
# Syncthing sync directions (configure on Syncthing UI):
#   main  → this  :  BASEDIR/models/     (this machine picks up new weights)
#   this  → main  :  BASEDIR/selfplay/NODE_ID/  (main picks up game data)
#
# Usage:
#   bash selfplay.sh                         # defaults + run.cfg
#   NODE_ID=node2 GPU=0 bash selfplay.sh     # explicit node id and GPU
# =============================================================================

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
PROJECTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
SELFPLAY="$PROJECTDIR/cpp/build/selfplay"

# --- Defaults ---
GPU="${GPU:-0}"
BASEDIR="${BASEDIR:-$PROJECTDIR/data}"
NODE_ID="${NODE_ID:-$(hostname)}"   # unique name for this worker machine

BOARD_SIZE="${BOARD_SIZE:-15}"
RENJU="${RENJU:-true}"
OPENINGS="${OPENINGS:-}"
EMPTY_BOARD_PROB="${EMPTY_BOARD_PROB:-0.0}"
ONLINE_OPENINGS="${ONLINE_OPENINGS:-true}"
OPENING_MIN_MOVES="${OPENING_MIN_MOVES:-3}"
OPENING_MAX_MOVES="${OPENING_MAX_MOVES:-10}"
OPENING_BALANCE_POWER="${OPENING_BALANCE_POWER:-4.0}"
OPENING_REJECT_THRESHOLD="${OPENING_REJECT_THRESHOLD:-0.20}"
OPENING_MAX_RETRIES="${OPENING_MAX_RETRIES:-20}"

NUM_SIMULATIONS="${NUM_SIMULATIONS:-256}"
GUMBEL_M="${GUMBEL_M:-16}"
GUMBEL_C_VISIT="${GUMBEL_C_VISIT:-50.0}"
GUMBEL_C_SCALE="${GUMBEL_C_SCALE:-1.0}"
C_PUCT="${C_PUCT:-1.1}"
HALF_LIFE="${HALF_LIFE:-$BOARD_SIZE}"
MOVE_TEMP_INIT="${MOVE_TEMP_INIT:-1.1}"
MOVE_TEMP_FINAL="${MOVE_TEMP_FINAL:-1.0}"
ENABLE_SVB="${ENABLE_SVB:-true}"
SVB_FACTOR="${SVB_FACTOR:-0.35}"

NUM_BLOCKS="${NUM_BLOCKS:-4}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"

NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_SERVERS="${NUM_SERVERS:-1}"
INFERENCE_BATCH="${INFERENCE_BATCH:-256}"
INFERENCE_BATCH_WAIT_US="${INFERENCE_BATCH_WAIT_US:-1500}"
LEAF_BATCH="${LEAF_BATCH:-32}"

MAX_GAMES="${MAX_GAMES:-4000}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-25000}"
MODEL_CHECK_MS="${MODEL_CHECK_MS:-10000}"

POLICY_SURPRISE_WEIGHT="${POLICY_SURPRISE_WEIGHT:-0.5}"
VALUE_SURPRISE_WEIGHT="${VALUE_SURPRISE_WEIGHT:-0.1}"
SOFT_RESIGN_THRESHOLD="${SOFT_RESIGN_THRESHOLD:-0.9}"
SOFT_RESIGN_PROB="${SOFT_RESIGN_PROB:-0.7}"

# Playout Cap Randomization
FULL_SEARCH_PROB="${FULL_SEARCH_PROB:-0.25}"
CHEAP_SIMULATIONS="${CHEAP_SIMULATIONS:-64}"
CHEAP_GUMBEL_M="${CHEAP_GUMBEL_M:-8}"
CHEAP_SAMPLE_WEIGHT="${CHEAP_SAMPLE_WEIGHT:-0.1}"

# Fork Side Positions
FORK_SIDE_PROB="${FORK_SIDE_PROB:-0.04}"
MAX_FORK_QUEUE="${MAX_FORK_QUEUE:-1000}"
FORK_SKIP_FIRST_N="${FORK_SKIP_FIRST_N:-3}"

# Uncertainty-Weighted MCTS Backup
ENABLE_UNCERTAINTY_WEIGHTING="${ENABLE_UNCERTAINTY_WEIGHTING:-true}"
UNCERTAINTY_PRIOR="${UNCERTAINTY_PRIOR:-0.25}"
UNCERTAINTY_EXPONENT="${UNCERTAINTY_EXPONENT:-1.0}"
UNCERTAINTY_MAX_WEIGHT="${UNCERTAINTY_MAX_WEIGHT:-8.0}"

# --- Source config (overrides defaults, but env vars take priority) ---
CFGFILE="${CFGFILE:-$SCRIPTDIR/run.cfg}"
if [ -f "$CFGFILE" ]; then
    source "$CFGFILE"
fi

# Resolve BASEDIR to absolute
mkdir -p "$BASEDIR"
BASEDIR="$(cd "$BASEDIR" && pwd)"

# Output dir is a node-specific subdirectory so files from different machines
# never collide, and Syncthing can sync only this machine's subtree.
OUTPUT_DIR="$BASEDIR/selfplay/$NODE_ID"
mkdir -p "$OUTPUT_DIR"

# Ensure libs (libtorch, libzip, etc.) are on LD_LIBRARY_PATH
CONDA_LIB="${CONDA_PREFIX:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== SkyZero V4 Selfplay Worker ==="
echo "Node: $NODE_ID | GPU: $GPU | Board: ${BOARD_SIZE}x${BOARD_SIZE} | Renju: $RENJU"
echo "Blocks: $NUM_BLOCKS | Channels: $NUM_CHANNELS"
echo "Sims: $NUM_SIMULATIONS | Workers: $NUM_WORKERS | Servers: $NUM_SERVERS"
echo "MaxGames/iter: $MAX_GAMES"
echo "Model dir:  $BASEDIR/models"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check selfplay binary
if [ ! -f "$SELFPLAY" ]; then
    echo "ERROR: selfplay binary not found at $SELFPLAY"
    echo "Build it first: cd cpp/build && cmake ... && make -j"
    exit 1
fi

# Wait for model (the C++ binary also waits, but give an early human-readable msg)
if [ -z "$(find "$BASEDIR/models" -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "No model found in $BASEDIR/models — waiting for main machine to sync one..."
    while [ -z "$(find "$BASEDIR/models" -name '*.pt' -print -quit 2>/dev/null)" ]; do
        sleep 10
    done
    echo "Model found, starting selfplay."
fi

# --- Build selfplay args ---
SELFPLAY_ARGS=(
    --model-dir "$BASEDIR/models"
    --output-dir "$OUTPUT_DIR"
    --max-games "$MAX_GAMES"
    --board-size "$BOARD_SIZE"
    --num-simulations "$NUM_SIMULATIONS"
    --gumbel-m "$GUMBEL_M"
    --gumbel-c-visit "$GUMBEL_C_VISIT"
    --gumbel-c-scale "$GUMBEL_C_SCALE"
    --c-puct "$C_PUCT"
    --half-life "$HALF_LIFE"
    --move-temp-init "$MOVE_TEMP_INIT"
    --move-temp-final "$MOVE_TEMP_FINAL"
    --svb-factor "$SVB_FACTOR"
    --num-workers "$NUM_WORKERS"
    --num-servers "$NUM_SERVERS"
    --inference-batch "$INFERENCE_BATCH"
    --inference-batch-wait-us "$INFERENCE_BATCH_WAIT_US"
    --leaf-batch "$LEAF_BATCH"
    --num-blocks "$NUM_BLOCKS"
    --num-channels "$NUM_CHANNELS"
    --max-rows-per-file "$MAX_ROWS_PER_FILE"
    --model-check-ms "$MODEL_CHECK_MS"
    --policy-surprise-weight "$POLICY_SURPRISE_WEIGHT"
    --value-surprise-weight "$VALUE_SURPRISE_WEIGHT"
    --soft-resign-threshold "$SOFT_RESIGN_THRESHOLD"
    --soft-resign-prob "$SOFT_RESIGN_PROB"
    --full-search-prob "$FULL_SEARCH_PROB"
    --cheap-simulations "$CHEAP_SIMULATIONS"
    --cheap-gumbel-m "$CHEAP_GUMBEL_M"
    --cheap-sample-weight "$CHEAP_SAMPLE_WEIGHT"
    --fork-side-prob "$FORK_SIDE_PROB"
    --max-fork-queue "$MAX_FORK_QUEUE"
    --fork-skip-first-n "$FORK_SKIP_FIRST_N"
    --uncertainty-prior "$UNCERTAINTY_PRIOR"
    --uncertainty-exponent "$UNCERTAINTY_EXPONENT"
    --uncertainty-max-weight "$UNCERTAINTY_MAX_WEIGHT"
)
[[ "$RENJU" == "false" ]] && SELFPLAY_ARGS+=(--no-renju)
[[ "$ENABLE_SVB" == "true" ]] && SELFPLAY_ARGS+=(--enable-svb)
[[ "$ENABLE_UNCERTAINTY_WEIGHTING" == "true" ]] && SELFPLAY_ARGS+=(--enable-uncertainty-weighting)
[[ -n "$OPENINGS" ]] && SELFPLAY_ARGS+=(--openings "$OPENINGS" --empty-board-prob "$EMPTY_BOARD_PROB")
if [[ "$ONLINE_OPENINGS" == "true" ]]; then
    SELFPLAY_ARGS+=(
        --online-openings
        --opening-min-moves "$OPENING_MIN_MOVES"
        --opening-max-moves "$OPENING_MAX_MOVES"
        --opening-balance-power "$OPENING_BALANCE_POWER"
        --opening-reject-threshold "$OPENING_REJECT_THRESHOLD"
        --opening-max-retries "$OPENING_MAX_RETRIES"
    )
fi

# ==========================================================================
# Main loop — selfplay only
# ==========================================================================
ITERATION=0
while true
do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "==================== Selfplay Iteration $ITERATION ===================="
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"

    CUDA_VISIBLE_DEVICES=$GPU "$SELFPLAY" "${SELFPLAY_ARGS[@]}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 130 ]; then
        echo "Interrupted. Exiting."
        exit 0
    fi

    echo "--- Iteration $ITERATION complete at $(date '+%Y-%m-%d %H:%M:%S') ---"
done
