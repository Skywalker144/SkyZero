#!/bin/bash -eu
set -o pipefail

# =============================================================================
# SkyZero V4 — Main training loop
# Orchestrates: selfplay (C++) -> shuffle (Python) -> train (Python) -> export (Python)
#
# Usage:
#   bash run.sh                          # use defaults + selfplay.cfg
#   GPU=1 MAX_GAMES=8000 bash run.sh     # override via env
# =============================================================================

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
PROJECTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
PYTHONDIR="$PROJECTDIR/python"
SELFPLAY="$PROJECTDIR/cpp/build/selfplay"

# --- Defaults ---
GPU="${GPU:-0}"
BASEDIR="${BASEDIR:-$PROJECTDIR/data}"
BATCHSIZE="${BATCHSIZE:-128}"
NTHREADS="${NTHREADS:-16}"

BOARD_SIZE="${BOARD_SIZE:-15}"
RENJU="${RENJU:-true}"
OPENINGS="${OPENINGS:-}"
EMPTY_BOARD_PROB="${EMPTY_BOARD_PROB:-0.0}"

NUM_SIMULATIONS="${NUM_SIMULATIONS:-32}"
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
INFERENCE_BATCH="${INFERENCE_BATCH:-64}"
INFERENCE_BATCH_WAIT_US="${INFERENCE_BATCH_WAIT_US:-250}"
LEAF_BATCH="${LEAF_BATCH:-8}"

MAX_GAMES="${MAX_GAMES:-4000}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-25000}"
MODEL_CHECK_MS="${MODEL_CHECK_MS:-10000}"

POLICY_SURPRISE_WEIGHT="${POLICY_SURPRISE_WEIGHT:-0.5}"
VALUE_SURPRISE_WEIGHT="${VALUE_SURPRISE_WEIGHT:-0.1}"
SOFT_RESIGN_THRESHOLD="${SOFT_RESIGN_THRESHOLD:-0.9}"
SOFT_RESIGN_PROB="${SOFT_RESIGN_PROB:-0.7}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
USE_FP16="${USE_FP16:-true}"
SWA_SCALE="${SWA_SCALE:-1.0}"
LOOKAHEAD_K="${LOOKAHEAD_K:-6}"
LOOKAHEAD_ALPHA="${LOOKAHEAD_ALPHA:-0.5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-2000000}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

# --- Source config (overrides defaults, but env vars take priority) ---
CFGFILE="${CFGFILE:-$SCRIPTDIR/selfplay.cfg}"
if [ -f "$CFGFILE" ]; then
    source "$CFGFILE"
fi

# Resolve BASEDIR to absolute
mkdir -p "$BASEDIR"
BASEDIR="$(cd "$BASEDIR" && pwd)"

# Ensure libzip from conda is on LD_LIBRARY_PATH
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
export LD_LIBRARY_PATH="${CONDA_BASE}/lib:${LD_LIBRARY_PATH:-}"

echo "=== SkyZero V4 Training Pipeline ==="
echo "GPU: $GPU | Board: ${BOARD_SIZE}x${BOARD_SIZE} | Renju: $RENJU"
echo "Blocks: $NUM_BLOCKS | Channels: $NUM_CHANNELS"
echo "Sims: $NUM_SIMULATIONS | Workers: $NUM_WORKERS | Servers: $NUM_SERVERS"
echo "MaxGames/iter: $MAX_GAMES | BatchSize: $BATCHSIZE"
echo "BASEDIR: $BASEDIR"
echo ""

# Create directories
mkdir -p "$BASEDIR"/{selfplay,models,shuffleddata,train/skyzero,torchmodels_toexport}

# Check selfplay binary
if [ ! -f "$SELFPLAY" ]; then
    echo "ERROR: selfplay binary not found at $SELFPLAY"
    echo "Build it first: cd cpp/build && CONDA_PREFIX=$CONDA_BASE cmake -DCMAKE_PREFIX_PATH=\$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') .. && make -j"
    exit 1
fi

# Bootstrap: create initial random model if needed
if [ -z "$(find "$BASEDIR"/models -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "No model found. Creating initial random model..."
    python "$PYTHONDIR/init_model.py" \
        -output "$BASEDIR/models/random_init.pt" \
        -board-size "$BOARD_SIZE" \
        -num-planes 4 \
        -num-blocks "$NUM_BLOCKS" \
        -num-channels "$NUM_CHANNELS"
    echo "Initial model created."
fi

# --- Build selfplay args ---
SELFPLAY_ARGS=(
    --model-dir "$BASEDIR/models"
    --output-dir "$BASEDIR/selfplay"
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
)
[[ "$RENJU" == "false" ]] && SELFPLAY_ARGS+=(--no-renju)
[[ "$ENABLE_SVB" == "true" ]] && SELFPLAY_ARGS+=(--enable-svb)
[[ -n "$OPENINGS" ]] && SELFPLAY_ARGS+=(--openings "$OPENINGS" --empty-board-prob "$EMPTY_BOARD_PROB")

# --- Build train.py extra args ---
TRAIN_EXTRA_ARGS=()
TRAIN_EXTRA_ARGS+=(-lr "$LR" -weight-decay "$WEIGHT_DECAY")
TRAIN_EXTRA_ARGS+=(-swa-scale "$SWA_SCALE")
TRAIN_EXTRA_ARGS+=(-lookahead-k "$LOOKAHEAD_K" -lookahead-alpha "$LOOKAHEAD_ALPHA")
TRAIN_EXTRA_ARGS+=(-samples-per-epoch "$SAMPLES_PER_EPOCH")
TRAIN_EXTRA_ARGS+=(-max-epochs-this-instance "$MAX_EPOCHS")
TRAIN_EXTRA_ARGS+=(-num-planes 4 -num-blocks "$NUM_BLOCKS" -num-channels "$NUM_CHANNELS")
[[ "$USE_FP16" == "true" ]] && TRAIN_EXTRA_ARGS+=(-use-fp16)

# ==========================================================================
# Main loop
# ==========================================================================
ITERATION=0
while true
do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "==================== Iteration $ITERATION ===================="
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"

    # 1. Selfplay (C++)
    echo ""
    echo "--- Stage 1: Selfplay ---"
    CUDA_VISIBLE_DEVICES=$GPU "$SELFPLAY" "${SELFPLAY_ARGS[@]}"

    # 2. Shuffle (Python)
    echo ""
    echo "--- Stage 2: Shuffle ---"
    cd "$PYTHONDIR"
    bash shuffle.sh "$BASEDIR" "$BASEDIR/tmp" "$NTHREADS" "$BATCHSIZE"

    # 3. Train (Python)
    echo ""
    echo "--- Stage 3: Train ---"
    mkdir -p "$BASEDIR"/train/skyzero
    mkdir -p "$BASEDIR"/torchmodels_toexport
    CUDA_VISIBLE_DEVICES=$GPU python ./train.py \
        -traindir "$BASEDIR"/train/skyzero \
        -datadir "$BASEDIR"/shuffleddata/current/ \
        -exportdir "$BASEDIR"/torchmodels_toexport \
        -exportprefix skyzero \
        -pos-len "$BOARD_SIZE" \
        -batch-size "$BATCHSIZE" \
        "${TRAIN_EXTRA_ARGS[@]}" \
        2>&1 | tee -a "$BASEDIR"/train/skyzero/stdout.txt

    # 4. Export (Python)
    echo ""
    echo "--- Stage 4: Export ---"
    CUDA_VISIBLE_DEVICES=$GPU bash export.sh "$BASEDIR"
    cd "$PROJECTDIR"

    # 5. Plot loss
    python "$PYTHONDIR/view_loss.py" --traindir "$BASEDIR/train/skyzero" --output "$BASEDIR/loss.png" || true

    echo ""
    echo "--- Iteration $ITERATION complete at $(date '+%Y-%m-%d %H:%M:%S') ---"
done
