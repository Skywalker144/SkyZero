#!/bin/bash -eu
set -o pipefail
trap 'echo ""; echo "Pipeline interrupted by user. Exiting."; exit 130' INT TERM

# =============================================================================
# SkyZero V4 — Main training loop
# Orchestrates: selfplay (C++) -> shuffle (Python) -> train (Python) -> export (Python)
#
# Usage:
#   bash run.sh                          # use defaults + run.cfg
#   GPU=1 MAX_GAMES=8000 bash run.sh     # override via env
# =============================================================================

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
PROJECTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
PYTHONDIR="$PROJECTDIR/python"
SELFPLAY="$PROJECTDIR/cpp/build/selfplay"

# --- Defaults ---
GPU="${GPU:-0}"
BASEDIR="${BASEDIR:-$PROJECTDIR/data}"
BATCHSIZE="${BATCHSIZE:-512}"
NTHREADS="${NTHREADS:-16}"

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

MODEL_CONFIG="${MODEL_CONFIG:-b6c96}"

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

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
USE_FP16="${USE_FP16:-true}"
SWA_SCALE="${SWA_SCALE:-1.0}"
LOOKAHEAD_K="${LOOKAHEAD_K:-6}"
LOOKAHEAD_ALPHA="${LOOKAHEAD_ALPHA:-0.5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-1024000}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
VALUE_ERROR_LOSS_WEIGHT="${VALUE_ERROR_LOSS_WEIGHT:-0.05}"

TRAIN_PER_DATA="${TRAIN_PER_DATA:-2.0}"
MIN_GAMES="${MIN_GAMES:-500}"

# --- Source config (overrides defaults, but env vars take priority) ---
CFGFILE="${CFGFILE:-$SCRIPTDIR/run.cfg}"
if [ -f "$CFGFILE" ]; then
    source "$CFGFILE"
fi

# Resolve BASEDIR to absolute
mkdir -p "$BASEDIR"
BASEDIR="$(cd "$BASEDIR" && pwd)"

# Ensure libs (libtorch, libzip, etc.) are on LD_LIBRARY_PATH
# When a conda env is activated, CONDA_PREFIX points to it; otherwise fall back to base.
CONDA_LIB="${CONDA_PREFIX:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== SkyZero V4 Training Pipeline ==="
echo "GPU: $GPU | Board: ${BOARD_SIZE}x${BOARD_SIZE} | Renju: $RENJU"
echo "Model config: $MODEL_CONFIG"
echo "Sims: $NUM_SIMULATIONS | Workers: $NUM_WORKERS | Servers: $NUM_SERVERS"
echo "MaxGames/iter: $MAX_GAMES | MinGames: $MIN_GAMES | TrainPerData: $TRAIN_PER_DATA | BatchSize: $BATCHSIZE"
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
        -model-config "$MODEL_CONFIG"
    echo "Initial model created."
fi

# --- Build selfplay args ---
SELFPLAY_ARGS=(
    --model-dir "$BASEDIR/models"
    --output-dir "$BASEDIR/selfplay"
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

# --- Build train.py extra args ---
TRAIN_EXTRA_ARGS=()
TRAIN_EXTRA_ARGS+=(-lr "$LR" -weight-decay "$WEIGHT_DECAY")
TRAIN_EXTRA_ARGS+=(-swa-scale "$SWA_SCALE")
TRAIN_EXTRA_ARGS+=(-lookahead-k "$LOOKAHEAD_K" -lookahead-alpha "$LOOKAHEAD_ALPHA")
TRAIN_EXTRA_ARGS+=(-samples-per-epoch "$SAMPLES_PER_EPOCH")
TRAIN_EXTRA_ARGS+=(-max-epochs-this-instance "$MAX_EPOCHS")
TRAIN_EXTRA_ARGS+=(-num-planes 4 -model-config "$MODEL_CONFIG")
TRAIN_EXTRA_ARGS+=(-lr-scale-auto)
TRAIN_EXTRA_ARGS+=(-brenorm-target-rmax 3.0 -brenorm-target-dmax 5.0 -brenorm-adjustment-scale 50000000)
TRAIN_EXTRA_ARGS+=(-value-error-loss-weight "$VALUE_ERROR_LOSS_WEIGHT")
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

    # 0. Compute dynamic selfplay game count based on train_per_data ratio
    DYNAMIC_GAMES=$(python "$PYTHONDIR/compute_games.py" \
        --traindir "$BASEDIR/train/skyzero" \
        --selfplay-dir "$BASEDIR/selfplay" \
        --train-per-data "$TRAIN_PER_DATA" \
        --samples-per-epoch "$SAMPLES_PER_EPOCH" \
        --default-games "$MAX_GAMES" \
        --min-games "$MIN_GAMES" \
        --max-games "$MAX_GAMES" \
        || echo "$MAX_GAMES")
    echo "Dynamic selfplay games: $DYNAMIC_GAMES (train_per_data=$TRAIN_PER_DATA)"

    # 1. Selfplay (C++)
    echo ""
    echo "--- Stage 1: Selfplay ---"
    SP_STDOUT="$BASEDIR/selfplay/.last_run_stdout.txt"
    set +e
    CUDA_VISIBLE_DEVICES=$GPU "$SELFPLAY" "${SELFPLAY_ARGS[@]}" --max-games "$DYNAMIC_GAMES" \
        | tee "$SP_STDOUT"
    SP_EXIT=${PIPESTATUS[0]}
    set -e
    if [ $SP_EXIT -ne 0 ]; then
        echo "Selfplay exited with code $SP_EXIT. Stopping pipeline."
        exit $SP_EXIT
    fi

    # Capture last-run stats for compute_games.py's avg_rows_per_game estimate.
    # Parses: "Selfplay complete. Games: G | Total rows written: R"
    STATS_LINE=$(grep "^Selfplay complete" "$SP_STDOUT" | tail -n 1 || true)
    if [[ -n "$STATS_LINE" ]]; then
        GAMES_RUN=$(echo "$STATS_LINE" | sed -n 's/.*Games: \([0-9]\+\).*/\1/p')
        ROWS_RUN=$(echo "$STATS_LINE" | sed -n 's/.*Total rows written: \([0-9]\+\).*/\1/p')
        if [[ -n "$GAMES_RUN" && -n "$ROWS_RUN" && "$GAMES_RUN" -gt 0 ]]; then
            printf '%s\t%s\n' "$GAMES_RUN" "$ROWS_RUN" > "$BASEDIR/selfplay/last_run.tsv"
        fi
    fi
    rm -f "$SP_STDOUT"

    # 1.5. Check if total selfplay data satisfies train_per_data ratio;
    # if not, loop back to selfplay instead of waiting.
    echo ""
    echo "--- Stage 1.5: Check data sufficiency ---"
    if ! python "$PYTHONDIR/wait_for_data.py" \
        --traindir "$BASEDIR/train/skyzero" \
        --selfplay-dir "$BASEDIR/selfplay" \
        --train-per-data "$TRAIN_PER_DATA" \
        --samples-per-epoch "$SAMPLES_PER_EPOCH" \
        --once; then
        echo "Insufficient data, looping back to selfplay..."
        continue
    fi

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
