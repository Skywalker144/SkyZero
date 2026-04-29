#!/usr/bin/env bash
# Pit checkpoint(s) against anchors and append per-game results to
# data/eval/games.jsonl. Batch mode sweeps all checkpoints at a stride and
# re-plots the Elo curve at the end.
#
# All parameters (anchor dir/list, stride, num games, MCTS settings) live in
# scripts/eval.cfg. Env vars of the same name override the cfg value for
# quick one-off tweaks.
#
# Usage:
#   scripts/eval.sh                          # BATCH: all model_iter_*.pt at STRIDE
#   scripts/eval.sh latest                   # single: data/models/latest.pt
#   scripts/eval.sh model_iter_000300.pt     # single: data/models/<arg>
#   scripts/eval.sh /abs/path/to/model.pt    # single: absolute path
#
# Env overrides (all optional): NUM_GAMES, STRIDE, NUM_SIMULATIONS,
# ANCHOR_DIR, ANCHORS, EVAL_BIN, EVAL_CFG, DATA_DIR, OUT_FILE, PLOT_FILE.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUT_FILE="${OUT_FILE:-$DATA_DIR/eval/games.jsonl}"
PLOT_FILE="${PLOT_FILE:-$DATA_DIR/eval/elo.png}"
EVAL_BIN="${EVAL_BIN:-$ROOT/cpp/build/gomoku_eval}"
EVAL_CFG="${EVAL_CFG:-$SCRIPT_DIR/eval.cfg}"

[[ -x "$EVAL_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_eval"; exit 1; }
[[ -f "$EVAL_CFG" ]] || { echo "no config at $EVAL_CFG"; exit 1; }

# --- Read a key from eval.cfg (last occurrence wins, strips comments/quotes).
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
    ' "$EVAL_CFG")"
    echo "${val:-$default}"
}

ANCHOR_DIR="${ANCHOR_DIR:-$(cfg_get ANCHOR_DIR anchors)}"
[[ "$ANCHOR_DIR" = /* ]] || ANCHOR_DIR="$ROOT/$ANCHOR_DIR"
STRIDE="${STRIDE:-$(cfg_get STRIDE 20)}"
NUM_GAMES="${NUM_GAMES:-$(cfg_get NUM_GAMES 40)}"
ANCHORS_CFG="${ANCHORS:-$(cfg_get ANCHORS '')}"

mkdir -p "$ANCHOR_DIR" "$(dirname "$OUT_FILE")"

# --- Resolve anchor list -----------------------------------------------
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
        [[ -f "$p" ]] || { echo "[eval.sh] anchor not found: $p"; exit 1; }
        anchor_list+=("$p")
    done
else
    if ! compgen -G "$ANCHOR_DIR/*.pt" > /dev/null; then
        echo "[eval.sh] no .pt files in $ANCHOR_DIR and ANCHORS unset"
        exit 1
    fi
    anchor_list=("$ANCHOR_DIR"/*.pt)
fi

# --- Build target list --------------------------------------------------
TARGETS=()
if [[ $# -eq 0 ]]; then
    mapfile -t ALL < <(ls "$DATA_DIR"/models/model_iter_*.pt 2>/dev/null | sort)
    if [[ ${#ALL[@]} -eq 0 ]]; then
        echo "[eval.sh] no checkpoints in $DATA_DIR/models/"
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
    echo "[eval.sh] batch mode: ${#TARGETS[@]} checkpoints (stride=$STRIDE of ${#ALL[@]})"
else
    target_arg="$1"
    if [[ "$target_arg" = /* ]]; then
        t="$target_arg"
    elif [[ "$target_arg" = *.pt ]]; then
        t="$DATA_DIR/models/$target_arg"
    else
        t="$DATA_DIR/models/${target_arg}.pt"
    fi
    [[ -f "$t" ]] || { echo "no target model at $t"; exit 1; }
    TARGETS=("$t")
fi

sim_flag=()
if [[ -n "${NUM_SIMULATIONS:-}" ]]; then
    sim_flag=(--num-simulations "$NUM_SIMULATIONS")
fi

# --- Count existing games for a given (a, b) pair -----------------------
count_existing() {
    local a="$1" b="$2"
    [[ -f "$OUT_FILE" ]] || { echo 0; return; }
    local n
    n="$(grep -cF -- "\"a\":\"$a\",\"b\":\"$b\"" "$OUT_FILE" 2>/dev/null || true)"
    echo "${n:-0}"
}

# --- Run matches --------------------------------------------------------
echo "[eval.sh] anchors: $(printf '%s\n' "${anchor_list[@]}" | xargs -n1 basename | paste -sd, -)"
echo "[eval.sh] output=$OUT_FILE  games/pair=$NUM_GAMES"

for target in "${TARGETS[@]}"; do
    target_abs="$(readlink -f "$target")"
    for anchor in "${anchor_list[@]}"; do
        anchor_abs="$(readlink -f "$anchor")"
        if [[ "$anchor_abs" = "$target_abs" ]]; then
            continue
        fi
        existing="$(count_existing "$target" "$anchor")"
        if (( existing >= NUM_GAMES )); then
            echo "[eval.sh] skip $(basename "$target") vs $(basename "$anchor"): already $existing games"
            continue
        fi
        remaining=$(( NUM_GAMES - existing ))
        echo "[eval.sh] === $(basename "$target")  vs  $(basename "$anchor")  ($remaining games) ==="
        "$EVAL_BIN" \
            --model-a "$target" \
            --model-b "$anchor" \
            --config "$EVAL_CFG" \
            --output "$OUT_FILE" \
            --num-games "$remaining" \
            "${sim_flag[@]}"
    done
done

# --- Regenerate Elo table + curve ---------------------------------------
echo "[eval.sh] updating Elo: $PLOT_FILE"
python "$ROOT/python/elo.py" --games "$OUT_FILE" --plot "$PLOT_FILE"
