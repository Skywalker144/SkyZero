#!/usr/bin/env bash
# Pit checkpoint(s) against anchors and/or each other (neighbor chain) and
# append per-game results to data/elo/games.jsonl. Batch mode sweeps all
# checkpoints at a stride and re-plots the Elo curve at the end.
#
# Default mode is neighbor-only: each evaluated checkpoint plays its
# NEIGHBOR_K nearest forward successors. Filling those near-50%-winrate
# pairs gives much better Elo resolution than a fixed anchor that all
# strong models thrash at ~100%. Anchors are optional and only needed for
# cross-run comparability.
#
# Idempotent: re-running tops up only pairs that haven't reached their
# target game count, so cron'ing this after each new checkpoint just
# evaluates the new model against its neighbors.
#
# All parameters live in scripts/elo.cfg. Env vars of the same name
# override the cfg value for quick one-off tweaks.
#
# Usage:
#   scripts/elo.sh                          # BATCH: all model_iter_*.pt at STRIDE
#   scripts/elo.sh latest                   # single: $MODELS_DIR/latest.pt
#   scripts/elo.sh model_iter_000300.pt     # single: $MODELS_DIR/<arg>
#   scripts/elo.sh /abs/path/to/model.pt    # single: absolute path
#
# Multi-network training: set NET=<name> in elo.cfg (e.g. NET=b5c128) to
# evaluate models under $DATA_DIR/nets/<name>/ instead of $DATA_DIR/models/.
# Output then lands in $DATA_DIR/elo/<name>/ so different networks don't
# share games.jsonl.
#
# Multi-GPU: pending pairs are LPT-packed (by remaining games) into one
# match-schedule file per visible GPU; each GPU runs a single gomoku_elo
# tournament process (models loaded once, global game queue), sharded
# output merged at the end. Use ELO_GPUS=0,2,5 to pick a subset of
# physical GPU IDs.
#
# Env overrides (all optional): NUM_GAMES, NEIGHBOR_K, NEIGHBOR_GAMES,
# STRIDE, NUM_SIMULATIONS, ANCHOR_DIR, ANCHORS, ELO_BIN, ELO_CFG,
# DATA_DIR, MODELS_DIR, NET, OUT_FILE, PLOT_FILE, ELO_GPUS.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[elo.sh] no config dir at $CONFIG_DIR" >&2; exit 1; }

source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"
ELO_BIN="${ELO_BIN:-$ROOT/cpp/build/gomoku_elo}"
ELO_CFG="${ELO_CFG:-$CONFIG_DIR/elo.cfg}"

[[ -x "$ELO_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_elo"; exit 1; }
[[ -f "$ELO_CFG" ]] || { echo "no config at $ELO_CFG"; exit 1; }

# --- Read a key from elo.cfg (last occurrence wins, strips comments/quotes).
cfg_get() {
    local key="$1" default="${2:-}"
    local val
    val="$(awk -F= -v k="$key" '
        { sub(/#.*/, "") }                       # strip comments
        { gsub(/^[ \t]+|[ \t\r]+$/, "") }        # trim line
        $0 == "" { next }
        {
            split($0, kv, "=")
            key = kv[1]; gsub(/^[ \t]+|[ \t]+$/, "", key)
            if (key != k) next
            val = substr($0, index($0, "=") + 1)
            gsub(/^[ \t]+|[ \t]+$/, "", val)
            if (val ~ /^".*"$/ || val ~ /^'\''.*'\''$/) val = substr(val, 2, length(val) - 2)
            last = val
        }
        END { print last }
    ' "$ELO_CFG")"
    echo "${val:-$default}"
}

# Parse the trailing iter number from a model path (matches the regex used
# by python/elo.py:parse_iter). Returns 999999999 for paths with no digits
# in the stem, so "latest.pt" sorts as the highest iter.
parse_iter() {
    local stem="${1##*/}"
    stem="${stem%.*}"
    local n
    n="$(grep -oE '[0-9]+' <<< "$stem" | tail -n 1)"
    if [[ -z "$n" ]]; then echo 999999999; else echo "$((10#$n))"; fi
}

# NET picks a multi-network subdir under $DATA_DIR/nets/. Empty = legacy
# single-network mode (read $DATA_DIR/models/). When NET is set, output also
# moves to $DATA_DIR/elo/<NET>/ so different nets don't share games.jsonl.
#
# File naming differs between the two dirs: $DATA_DIR/nets/<net>/ holds BOTH
# train.py's raw state_dict (model_iter_*.pt — not loadable by C++) AND
# export_model.py's TorchScript (scripted_iter_*.pt). C++ gomoku_elo uses
# torch::jit::load, so we must point it at scripted_iter_*. Legacy
# $DATA_DIR/models/ uses model_iter_*.pt (those were already TorchScript).
NET="${NET:-$(cfg_get NET '')}"
if [[ -n "$NET" ]]; then
    MODELS_DIR="${MODELS_DIR:-$DATA_DIR/nets/$NET}"
    MODEL_GLOB="scripted_iter_*.pt"
    ELO_OUT_DIR="$DATA_DIR/elo/$NET"
else
    MODELS_DIR="${MODELS_DIR:-$DATA_DIR/models}"
    MODEL_GLOB="model_iter_*.pt"
    ELO_OUT_DIR="$DATA_DIR/elo"
fi
[[ -d "$MODELS_DIR" ]] || { echo "[elo.sh] no models dir at $MODELS_DIR" >&2; exit 1; }
OUT_FILE="${OUT_FILE:-$ELO_OUT_DIR/games.jsonl}"
PLOT_FILE="${PLOT_FILE:-$ELO_OUT_DIR/elo.png}"
# Plain-text ratings table (one row per model: name, games, Elo, ±se, iter),
# written next to the PNG so each run leaves a readable record of the scores.
TEXT_FILE="${TEXT_FILE:-$ELO_OUT_DIR/elo.txt}"

ANCHOR_DIR="${ANCHOR_DIR:-$(cfg_get ANCHOR_DIR anchors)}"
[[ "$ANCHOR_DIR" = /* ]] || ANCHOR_DIR="$ROOT/$ANCHOR_DIR"
STRIDE="${STRIDE:-$(cfg_get STRIDE 20)}"
NUM_GAMES="${NUM_GAMES:-$(cfg_get NUM_GAMES 40)}"
ANCHORS_CFG="${ANCHORS:-$(cfg_get ANCHORS '')}"
NEIGHBOR_K="${NEIGHBOR_K:-$(cfg_get NEIGHBOR_K 2)}"
NEIGHBOR_GAMES="${NEIGHBOR_GAMES:-$(cfg_get NEIGHBOR_GAMES 30)}"
# Skip checkpoints with iter < START_ITER when building TARGETS / STRIDED.
# Existing games.jsonl entries below this stay (python/elo.py reads them all);
# only NEW match scheduling is filtered.
START_ITER="${START_ITER:-$(cfg_get START_ITER 0)}"

# Returns sorted list of $MODEL_GLOB paths with iter >= START_ITER.
list_models() {
    while read -r f; do
        [[ -z "$f" ]] && continue
        local f_iter
        f_iter="$(parse_iter "$f")"
        if (( f_iter >= START_ITER )); then
            echo "$f"
        fi
    done < <(ls "$MODELS_DIR"/$MODEL_GLOB 2>/dev/null | sort)
}

mkdir -p "$(dirname "$OUT_FILE")"

# --- Resolve anchor list (optional; empty => neighbor-only mode) -------
anchor_list=()
if [[ -n "$ANCHORS_CFG" ]]; then
    # Split on commas and whitespace.
    IFS=', ' read -r -a names <<< "$ANCHORS_CFG"
    for name in "${names[@]}"; do
        [[ -z "$name" ]] && continue
        if [[ "$name" = /* ]]; then
            p="$name"
        else
            p="$ANCHOR_DIR/$name"
        fi
        [[ -f "$p" ]] || { echo "[elo.sh] anchor not found: $p"; exit 1; }
        anchor_list+=("$p")
    done
fi

# --- Build target list + neighbor pairs --------------------------------
# NEIGHBOR_PAIRS entries are "smaller_iter|larger_iter" (canonical), used
# to dedup with count_existing regardless of historical (a, b) ordering.
TARGETS=()
NEIGHBOR_PAIRS=()
if [[ $# -eq 0 ]]; then
    mapfile -t ALL < <(list_models)
    if [[ ${#ALL[@]} -eq 0 ]]; then
        echo "[elo.sh] no checkpoints in $MODELS_DIR/ (START_ITER=$START_ITER)"
        exit 1
    fi
    i=0
    for f in "${ALL[@]}"; do
        if (( i % STRIDE == 0 )); then
            TARGETS+=("$f")
        fi
        i=$((i + 1))
    done
    last="${ALL[-1]}"
    if [[ "${TARGETS[-1]}" != "$last" ]]; then
        TARGETS+=("$last")
    fi
    echo "[elo.sh] batch mode: ${#TARGETS[@]} checkpoints (stride=$STRIDE of ${#ALL[@]}, start_iter=$START_ITER)"

    # Chain pairs: every checkpoint plays its NEIGHBOR_K nearest forward
    # successors. Filenames are zero-padded so string-sort = iter-sort.
    n=${#TARGETS[@]}
    for ((i = 0; i < n; i++)); do
        for ((k = 1; k <= NEIGHBOR_K; k++)); do
            j=$((i + k))
            (( j >= n )) && break
            NEIGHBOR_PAIRS+=("${TARGETS[i]}|${TARGETS[j]}")
        done
    done
else
    target_arg="$1"
    if [[ "$target_arg" = /* ]]; then
        t="$target_arg"
    elif [[ "$target_arg" = *.pt ]]; then
        t="$MODELS_DIR/$target_arg"
    else
        t="$MODELS_DIR/${target_arg}.pt"
    fi
    [[ -f "$t" ]] || { echo "no target model at $t"; exit 1; }
    TARGETS=("$t")

    # Neighbor pairs: pick the K strided checkpoints with iter just below t
    # and just above. Compare by parsed iter number (last digit run in stem),
    # so that non-numeric paths like "latest.pt" sort to the end as the
    # "highest iter".
    if (( NEIGHBOR_K > 0 )); then
        mapfile -t ALL < <(list_models)
        STRIDED=()
        i=0
        for f in "${ALL[@]}"; do
            if (( i % STRIDE == 0 )); then
                STRIDED+=("$f")
            fi
            i=$((i + 1))
        done
        if [[ ${#ALL[@]} -gt 0 ]]; then
            last="${ALL[-1]}"
            if [[ ${#STRIDED[@]} -eq 0 || "${STRIDED[-1]}" != "$last" ]]; then
                STRIDED+=("$last")
            fi
        fi
        # Bucket STRIDED into preds (iter < t) and succs (iter > t). Skip
        # any entry whose absolute path matches t (e.g., user passed a path
        # that is itself a strided checkpoint).
        t_real="$(readlink -f "$t")"
        t_iter="$(parse_iter "$t")"
        preds=()  # ascending iter
        succs=()  # ascending iter
        for s in "${STRIDED[@]}"; do
            [[ "$(readlink -f "$s")" == "$t_real" ]] && continue
            s_iter="$(parse_iter "$s")"
            if (( s_iter < t_iter )); then
                preds+=("$s")
            elif (( s_iter > t_iter )); then
                succs+=("$s")
            fi
        done
        # K nearest predecessors (closest first = end of preds), pair canonical.
        np=${#preds[@]}
        for ((k = 1; k <= NEIGHBOR_K && k <= np; k++)); do
            NEIGHBOR_PAIRS+=("${preds[np-k]}|$t")
        done
        # K nearest successors (closest first = start of succs).
        ns=${#succs[@]}
        for ((k = 0; k < NEIGHBOR_K && k < ns; k++)); do
            NEIGHBOR_PAIRS+=("$t|${succs[k]}")
        done
    fi
fi

sim_flag=()
if [[ -n "${NUM_SIMULATIONS:-}" ]]; then
    sim_flag=(--num-simulations "$NUM_SIMULATIONS")
fi

# --- Count existing games for a pair (sums both orderings) --------------
count_existing() {
    local a="$1" b="$2"
    [[ -f "$OUT_FILE" ]] || { echo 0; return; }
    local n1 n2
    n1="$(grep -cF -- "\"a\":\"$a\",\"b\":\"$b\"" "$OUT_FILE" 2>/dev/null || true)"
    n2="$(grep -cF -- "\"a\":\"$b\",\"b\":\"$a\"" "$OUT_FILE" 2>/dev/null || true)"
    echo "$(( ${n1:-0} + ${n2:-0} ))"
}

# --- GPU pool -----------------------------------------------------------
# Default: every visible GPU runs one worker in parallel. Override with
# ELO_GPUS=0,2,5 (comma-separated physical IDs) to use a subset. Each worker
# is launched under CUDA_VISIBLE_DEVICES=<id>, so the C++ binary's hardcoded
# kCUDA:0 maps to that physical card.
if [[ -n "${ELO_GPUS:-}" ]]; then
    IFS=', ' read -r -a GPU_IDS <<< "$ELO_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=', ' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi &> /dev/null; then
    mapfile -t GPU_IDS < <(nvidia-smi -L | awk -F'[ :]' '/^GPU/ {print $2}')
fi
(( ${#GPU_IDS[@]} == 0 )) && GPU_IDS=(0)
NUM_GPUS=${#GPU_IDS[@]}

# --- Build full work list (a, b, target_games, label) ------------------
WORK=()
for target in "${TARGETS[@]}"; do
    target_abs="$(readlink -f "$target")"
    for anchor in "${anchor_list[@]}"; do
        anchor_abs="$(readlink -f "$anchor")"
        [[ "$anchor_abs" == "$target_abs" ]] && continue
        WORK+=("$target|$anchor|$NUM_GAMES|anchor")
    done
done
for pair in "${NEIGHBOR_PAIRS[@]}"; do
    WORK+=("${pair%|*}|${pair#*|}|$NEIGHBOR_GAMES|neighbor")
done

# Filter to pairs that still need games; record remaining count per pair.
PENDING=()
for w in "${WORK[@]}"; do
    IFS='|' read -r a b target_games label <<< "$w"
    existing="$(count_existing "$a" "$b")"
    if (( existing >= target_games )); then
        echo "[elo.sh] skip $(basename "$a") vs $(basename "$b") ($label): already $existing games"
    else
        PENDING+=("$a|$b|$(( target_games - existing ))|$label")
    fi
done

# --- Schedule summary ---------------------------------------------------
if (( ${#anchor_list[@]} == 0 )); then
    echo "[elo.sh] anchors: none (neighbor-only mode)"
else
    echo "[elo.sh] anchors: $(printf '%s\n' "${anchor_list[@]}" | xargs -n1 basename | paste -sd, -)"
fi
echo "[elo.sh] output=$OUT_FILE  anchor_games/pair=$NUM_GAMES  neighbor_games/pair=$NEIGHBOR_GAMES  K=$NEIGHBOR_K"
echo "[elo.sh] schedule: ${#TARGETS[@]} target(s) x ${#anchor_list[@]} anchor(s) + ${#NEIGHBOR_PAIRS[@]} neighbor pair(s); ${#PENDING[@]} pending across ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

# --- Run sharded tournament workers --------------------------------------
SHARDS=()
SCHEDS=()
# Trap merges any partial shards back into OUT_FILE on exit (Ctrl+C safe).
# gomoku_elo flushes after each game under a mutex, so a worker killed mid-
# game leaves at most one incomplete trailing line; grep filters to lines
# that look like complete JSON objects.
cleanup_shards() {
    for shard in "${SHARDS[@]:-}"; do
        [[ -f "$shard" ]] || continue
        if [[ -s "$shard" ]]; then
            grep -E '^\{.*\}$' "$shard" >> "$OUT_FILE" 2> /dev/null || true
        fi
        rm -f "$shard"
    done
    rm -f "${SCHEDS[@]:-}"
}
trap cleanup_shards EXIT

if (( ${#PENDING[@]} > 0 )); then
    # LPT-pack pairs into one match-schedule file per GPU: walk pairs by
    # remaining games (descending), always assign to the least-loaded GPU.
    # Each GPU then runs a single gomoku_elo tournament process — every
    # distinct model is loaded once and the global game queue keeps
    # NUM_CONCURRENT_GAMES saturated across pair boundaries.
    declare -a BUCKET_GAMES
    for ((g = 0; g < NUM_GPUS; g++)); do
        BUCKET_GAMES[g]=0
        sched="$OUT_FILE.sched.gpu${GPU_IDS[g]}"
        : > "$sched"
        SCHEDS+=("$sched")
    done
    while IFS='|' read -r a b remaining label; do
        [[ -z "$a" ]] && continue
        best=0
        for ((g = 1; g < NUM_GPUS; g++)); do
            (( BUCKET_GAMES[g] < BUCKET_GAMES[best] )) && best=$g
        done
        printf '%s|%s|%d\n' "$a" "$b" "$remaining" >> "${SCHEDS[best]}"
        BUCKET_GAMES[best]=$(( BUCKET_GAMES[best] + remaining ))
    done < <(printf '%s\n' "${PENDING[@]}" | sort -t'|' -k3,3nr)

    PIDS=()
    for ((g = 0; g < NUM_GPUS; g++)); do
        sched="${SCHEDS[g]}"
        [[ -s "$sched" ]] || continue
        gpu="${GPU_IDS[g]}"
        shard="$OUT_FILE.shard.gpu${gpu}.jsonl"
        : > "$shard"
        SHARDS+=("$shard")
        echo "[elo.sh] gpu $gpu: $(wc -l < "$sched") pair(s), ${BUCKET_GAMES[g]} games"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            "$ELO_BIN" \
                --match-schedule "$sched" \
                --config "$ELO_CFG" \
                --output "$shard" \
                "${sim_flag[@]}"
        ) 2>&1 | sed -u "s/^/[gpu $gpu] /" &
        PIDS+=($!)
    done

    fail=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || fail=1
    done
    (( fail )) && echo "[elo.sh] WARNING: one or more workers exited non-zero"
fi

# Merge shards now so the fit below sees this run's games (previously only
# the EXIT trap merged, i.e. one run late). Trap stays off afterwards —
# there is nothing left to clean.
cleanup_shards
trap - EXIT

# --- Regenerate Elo table + curve ---------------------------------------
echo "[elo.sh] updating Elo: $PLOT_FILE (+ $TEXT_FILE)"
python "$ROOT/python/elo.py" --games "$OUT_FILE" --plot "$PLOT_FILE" --out-text "$TEXT_FILE"
