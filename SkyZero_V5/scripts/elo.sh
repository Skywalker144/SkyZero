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
#   scripts/elo.sh latest                   # single: data/models/latest.pt
#   scripts/elo.sh model_iter_000300.pt     # single: data/models/<arg>
#   scripts/elo.sh /abs/path/to/model.pt    # single: absolute path
#
# Env overrides (all optional): NUM_GAMES, NEIGHBOR_K, NEIGHBOR_GAMES,
# STRIDE, NUM_SIMULATIONS, ANCHOR_DIR, ANCHORS, ELO_BIN, ELO_CFG,
# DATA_DIR, OUT_FILE, PLOT_FILE.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUT_FILE="${OUT_FILE:-$DATA_DIR/elo/games.jsonl}"
PLOT_FILE="${PLOT_FILE:-$DATA_DIR/elo/elo.png}"
ELO_BIN="${ELO_BIN:-$ROOT/cpp/build/gomoku_elo}"
ELO_CFG="${ELO_CFG:-$SCRIPT_DIR/elo.cfg}"

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

ANCHOR_DIR="${ANCHOR_DIR:-$(cfg_get ANCHOR_DIR anchors)}"
[[ "$ANCHOR_DIR" = /* ]] || ANCHOR_DIR="$ROOT/$ANCHOR_DIR"
STRIDE="${STRIDE:-$(cfg_get STRIDE 20)}"
NUM_GAMES="${NUM_GAMES:-$(cfg_get NUM_GAMES 40)}"
ANCHORS_CFG="${ANCHORS:-$(cfg_get ANCHORS '')}"
NEIGHBOR_K="${NEIGHBOR_K:-$(cfg_get NEIGHBOR_K 2)}"
NEIGHBOR_GAMES="${NEIGHBOR_GAMES:-$(cfg_get NEIGHBOR_GAMES 30)}"

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
    mapfile -t ALL < <(ls "$DATA_DIR"/models/model_iter_*.pt 2>/dev/null | sort)
    if [[ ${#ALL[@]} -eq 0 ]]; then
        echo "[elo.sh] no checkpoints in $DATA_DIR/models/"
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
    echo "[elo.sh] batch mode: ${#TARGETS[@]} checkpoints (stride=$STRIDE of ${#ALL[@]})"

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
        t="$DATA_DIR/models/$target_arg"
    else
        t="$DATA_DIR/models/${target_arg}.pt"
    fi
    [[ -f "$t" ]] || { echo "no target model at $t"; exit 1; }
    TARGETS=("$t")

    # Neighbor pairs: pick the K strided checkpoints with iter just below t
    # and just above. Compare by parsed iter number (last digit run in stem),
    # so that non-numeric paths like "latest.pt" sort to the end as the
    # "highest iter".
    if (( NEIGHBOR_K > 0 )); then
        mapfile -t ALL < <(ls "$DATA_DIR"/models/model_iter_*.pt 2>/dev/null | sort)
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

# Run gomoku_elo for one (a, b) pair, topping up to target_games. Skips if
# already filled. label is just for the log line ("anchor" or "neighbor").
run_pair_match() {
    local a="$1" b="$2" target_games="$3" label="$4"
    local existing remaining
    existing="$(count_existing "$a" "$b")"
    if (( existing >= target_games )); then
        echo "[elo.sh] skip $(basename "$a") vs $(basename "$b") ($label): already $existing games"
        return 0
    fi
    remaining=$(( target_games - existing ))
    echo "[elo.sh] === $(basename "$a") vs $(basename "$b") ($label, $remaining games) ==="
    "$ELO_BIN" \
        --model-a "$a" \
        --model-b "$b" \
        --config "$ELO_CFG" \
        --output "$OUT_FILE" \
        --num-games "$remaining" \
        "${sim_flag[@]}"
}

# --- Run matches --------------------------------------------------------
if (( ${#anchor_list[@]} == 0 )); then
    echo "[elo.sh] anchors: none (neighbor-only mode)"
else
    echo "[elo.sh] anchors: $(printf '%s\n' "${anchor_list[@]}" | xargs -n1 basename | paste -sd, -)"
fi
echo "[elo.sh] output=$OUT_FILE  anchor_games/pair=$NUM_GAMES  neighbor_games/pair=$NEIGHBOR_GAMES  K=$NEIGHBOR_K"
echo "[elo.sh] schedule: ${#TARGETS[@]} target(s) x ${#anchor_list[@]} anchor(s) + ${#NEIGHBOR_PAIRS[@]} neighbor pair(s)"

# Anchor matches
for target in "${TARGETS[@]}"; do
    target_abs="$(readlink -f "$target")"
    for anchor in "${anchor_list[@]}"; do
        anchor_abs="$(readlink -f "$anchor")"
        [[ "$anchor_abs" == "$target_abs" ]] && continue
        run_pair_match "$target" "$anchor" "$NUM_GAMES" "anchor"
    done
done

# Neighbor matches
for pair in "${NEIGHBOR_PAIRS[@]}"; do
    a="${pair%|*}"
    b="${pair#*|}"
    run_pair_match "$a" "$b" "$NEIGHBOR_GAMES" "neighbor"
done

# --- Regenerate Elo table + curve ---------------------------------------
echo "[elo.sh] updating Elo: $PLOT_FILE"
python "$ROOT/python/elo.py" --games "$OUT_FILE" --plot "$PLOT_FILE"
