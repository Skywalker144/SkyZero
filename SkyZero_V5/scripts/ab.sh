#!/usr/bin/env bash
# Pit a model against itself with two different MCTS configs (a.cfg vs
# b.cfg) and append per-game results to data/ab/<OUT_NAME>.jsonl. After
# every model in MODELS is filled to NUM_GAMES, regenerates the Elo-diff
# report + plot via python/ab.py.
#
# All parameters live in scripts/ab/ab.cfg (model list, total games,
# concurrency, board topology) and scripts/ab/{a,b}.cfg (per-side MCTS).
# Env vars of the same name override ab.cfg for one-off tweaks.
#
# Usage:
#   scripts/ab.sh                          # use MODELS from ab.cfg
#   scripts/ab.sh latest                   # single: data/models/latest.pt
#   scripts/ab.sh model_iter_000300.pt     # single: data/models/<arg>
#   scripts/ab.sh /abs/path/to/model.pt    # single: absolute path
#
# Env overrides: NUM_GAMES, MODELS, OUT_NAME, AB_BIN, AB_CFG, AB_CFG_A,
# AB_CFG_B, DATA_DIR, OUT_FILE, PLOT_FILE.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
AB_BIN="${AB_BIN:-$ROOT/cpp/build/gomoku_ab}"
AB_CFG="${AB_CFG:-$SCRIPT_DIR/ab/ab.cfg}"
AB_CFG_A="${AB_CFG_A:-$SCRIPT_DIR/ab/a.cfg}"
AB_CFG_B="${AB_CFG_B:-$SCRIPT_DIR/ab/b.cfg}"

[[ -x "$AB_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_ab"; exit 1; }
[[ -f "$AB_CFG"   ]] || { echo "no config at $AB_CFG"; exit 1; }
[[ -f "$AB_CFG_A" ]] || { echo "no config at $AB_CFG_A"; exit 1; }
[[ -f "$AB_CFG_B" ]] || { echo "no config at $AB_CFG_B"; exit 1; }

# --- Read a key from ab.cfg (last occurrence wins, strips comments/quotes).
cfg_get() {
    local key="$1" default="${2:-}"
    local val
    val="$(awk -F= -v k="$key" '
        { sub(/#.*/, "") }
        { gsub(/^[ \t]+|[ \t\r]+$/, "") }
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
    ' "$AB_CFG")"
    echo "${val:-$default}"
}

NUM_GAMES="${NUM_GAMES:-$(cfg_get NUM_GAMES 200)}"
OUT_NAME="${OUT_NAME:-$(cfg_get OUT_NAME ab)}"
MODELS_CFG="${MODELS:-$(cfg_get MODELS latest)}"

OUT_FILE="${OUT_FILE:-$DATA_DIR/ab/$OUT_NAME.jsonl}"
PLOT_FILE="${PLOT_FILE:-$DATA_DIR/ab/$OUT_NAME.png}"

mkdir -p "$(dirname "$OUT_FILE")"

# --- Resolve target list -----------------------------------------------
TARGETS=()
resolve_one() {
    local arg="$1"
    if [[ "$arg" = /* ]]; then
        echo "$arg"
    elif [[ "$arg" = *.pt ]]; then
        echo "$DATA_DIR/models/$arg"
    else
        echo "$DATA_DIR/models/${arg}.pt"
    fi
}

if [[ $# -gt 0 ]]; then
    t="$(resolve_one "$1")"
    [[ -f "$t" ]] || { echo "[ab.sh] no model at $t"; exit 1; }
    TARGETS=("$t")
else
    IFS=', ' read -r -a names <<< "$MODELS_CFG"
    for name in "${names[@]}"; do
        [[ -z "$name" ]] && continue
        t="$(resolve_one "$name")"
        [[ -f "$t" ]] || { echo "[ab.sh] model not found: $t"; exit 1; }
        TARGETS+=("$t")
    done
    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        echo "[ab.sh] MODELS resolved to empty list"
        exit 1
    fi
fi

# --- Count existing rows for a given (model, cfg_a, cfg_b) tuple --------
count_existing() {
    local model="$1" cfa="$2" cfb="$3"
    [[ -f "$OUT_FILE" ]] || { echo 0; return; }
    local n
    n="$(grep -cF -- "\"model\":\"$model\",\"cfg_a\":\"$cfa\",\"cfg_b\":\"$cfb\"" "$OUT_FILE" 2>/dev/null || true)"
    echo "${n:-0}"
}

# --- Run matches --------------------------------------------------------
echo "[ab.sh] models: $(printf '%s\n' "${TARGETS[@]}" | xargs -n1 basename | paste -sd, -)"
echo "[ab.sh] cfg-a=$AB_CFG_A  cfg-b=$AB_CFG_B"
echo "[ab.sh] output=$OUT_FILE  games/model=$NUM_GAMES"

# Warn once if any matching rows already exist. We dedup on cfg path, so an
# in-place edit to a.cfg / b.cfg between runs would silently mix two settings
# in the same OUT_NAME. Use a fresh OUT_NAME or wipe the jsonl when changing
# either cfg in place.
if [[ -f "$OUT_FILE" ]]; then
    prior=0
    for target in "${TARGETS[@]}"; do
        prior=$(( prior + $(count_existing "$target" "$AB_CFG_A" "$AB_CFG_B") ))
    done
    if (( prior > 0 )); then
        echo "[ab.sh] note: $prior prior rows match (model, $AB_CFG_A, $AB_CFG_B)."
        echo "[ab.sh]       If you edited either cfg in place since those rows were"
        echo "[ab.sh]       written, set a fresh OUT_NAME or rm $OUT_FILE."
    fi
fi

for target in "${TARGETS[@]}"; do
    existing="$(count_existing "$target" "$AB_CFG_A" "$AB_CFG_B")"
    if (( existing >= NUM_GAMES )); then
        echo "[ab.sh] skip $(basename "$target"): already $existing games"
        continue
    fi
    remaining=$(( NUM_GAMES - existing ))
    echo "[ab.sh] === $(basename "$target")  ($remaining games) ==="
    "$AB_BIN" \
        --model "$target" \
        --config-ab "$AB_CFG" \
        --config-a  "$AB_CFG_A" \
        --config-b  "$AB_CFG_B" \
        --output "$OUT_FILE" \
        --num-games "$remaining"
done

# --- Regenerate report + plot -------------------------------------------
echo "[ab.sh] updating report: $PLOT_FILE"
python "$ROOT/python/ab.py" --games "$OUT_FILE" --plot "$PLOT_FILE"
