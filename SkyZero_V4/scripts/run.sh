#!/bin/bash -eu
set -o pipefail

# =============================================================================
# SkyZero V4 — Main training loop
# Orchestrates: selfplay (C++) -> shuffle (Python) -> train (Python) -> export (Python)
# =============================================================================

# Configuration
GPU="${GPU:-0}"
BASEDIR="${BASEDIR:-$(dirname "$0")/../data}"
BATCHSIZE="${BATCHSIZE:-128}"
NTHREADS="${NTHREADS:-16}"
MAX_GAMES="${MAX_GAMES:-4000}"
NUM_WORKERS="${NUM_WORKERS:-32}"
NUM_SERVERS="${NUM_SERVERS:-1}"
NUM_SIMULATIONS="${NUM_SIMULATIONS:-32}"
BOARD_SIZE="${BOARD_SIZE:-15}"

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
PROJECTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
PYTHONDIR="$PROJECTDIR/python"
SELFPLAY="$PROJECTDIR/cpp/build/selfplay"

# Resolve BASEDIR to absolute path
BASEDIR="$(cd "$BASEDIR" 2>/dev/null && pwd || mkdir -p "$BASEDIR" && cd "$BASEDIR" && pwd)"

echo "=== SkyZero V4 Training Pipeline ==="
echo "GPU: $GPU"
echo "BASEDIR: $BASEDIR"
echo "BATCHSIZE: $BATCHSIZE"
echo "MAX_GAMES: $MAX_GAMES"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "NUM_SIMULATIONS: $NUM_SIMULATIONS"
echo ""

# Create directories
mkdir -p "$BASEDIR"/selfplay
mkdir -p "$BASEDIR"/models
mkdir -p "$BASEDIR"/shuffleddata
mkdir -p "$BASEDIR"/train/skyzero
mkdir -p "$BASEDIR"/torchmodels_toexport

# Check selfplay binary exists
if [ ! -f "$SELFPLAY" ]; then
    echo "ERROR: selfplay binary not found at $SELFPLAY"
    echo "Build it first: cd cpp/build && cmake .. && make -j"
    exit 1
fi

# Bootstrap: create initial random model if no models exist
if [ -z "$(find "$BASEDIR"/models -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "No model found. Creating initial random model..."
    python "$PYTHONDIR/init_model.py" \
        -output "$BASEDIR/models/random_init.pt" \
        -board-size "$BOARD_SIZE" \
        -num-planes 4 \
        -num-blocks 4 \
        -num-channels 128
    echo "Initial model created."
fi

# Main loop
ITERATION=0
while true
do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "==================== Iteration $ITERATION ===================="
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 1. Selfplay (C++)
    echo "--- Stage 1: Selfplay ---"
    CUDA_VISIBLE_DEVICES=$GPU "$SELFPLAY" \
        --model-dir "$BASEDIR/models" \
        --output-dir "$BASEDIR/selfplay" \
        --max-games "$MAX_GAMES" \
        --board-size "$BOARD_SIZE" \
        --num-simulations "$NUM_SIMULATIONS" \
        --num-workers "$NUM_WORKERS" \
        --num-servers "$NUM_SERVERS" \
        --inference-batch 64 \
        --leaf-batch 8 \
        --num-blocks 4 \
        --num-channels 128 \
        --enable-svb

    # 2. Shuffle (Python)
    echo ""
    echo "--- Stage 2: Shuffle ---"
    cd "$PYTHONDIR"
    bash shuffle.sh "$BASEDIR" "$BASEDIR/tmp" "$NTHREADS" "$BATCHSIZE"
    cd "$PROJECTDIR"

    # 3. Train (Python)
    echo ""
    echo "--- Stage 3: Train ---"
    cd "$PYTHONDIR"
    CUDA_VISIBLE_DEVICES=$GPU bash train.sh "$BASEDIR" "$BATCHSIZE"
    cd "$PROJECTDIR"

    # 4. Export (Python)
    echo ""
    echo "--- Stage 4: Export ---"
    cd "$PYTHONDIR"
    CUDA_VISIBLE_DEVICES=$GPU bash export.sh "$BASEDIR"
    cd "$PROJECTDIR"

    # 5. View loss (optional)
    echo ""
    echo "--- Iteration $ITERATION complete at $(date '+%Y-%m-%d %H:%M:%S') ---"

done
