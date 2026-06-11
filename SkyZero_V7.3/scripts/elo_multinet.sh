#!/usr/bin/env bash
# elo_multinet.sh — one combined Elo tournament across several co-trained
# networks, plotted as one Elo-vs-iteration curve PER network.
#
# Built for the net-capacity study: NETWORKS="b2c40,b3c64,b4c80,b5c96" all live
# in one DATA_DIR ($DATA_DIR/nets/<net>/), so a single Bradley-Terry fit puts
# every size on ONE Elo scale — provided the match graph is connected. We
# connect it three ways:
#   1. within-net neighbor chains   (near-50% pairs = high Elo resolution)
#   2. cross-net same-iter pairs    (adjacent sizes at the same iter; also
#                                     near-50%, and what ties the sizes together)
#   3. every strided checkpoint vs a common ANCHOR (optional; pins the zero and
#                                     gives cross-run comparability)
#
# Unlike elo.sh (one gomoku_elo process per pair, serial on a single GPU), this
# emits ONE schedule file and runs the gomoku_elo tournament mode once: each
# distinct checkpoint is loaded a single time and a shared worker pool plays all
# pairs concurrently, keeping the GPU fed across the whole tournament.
#
# Idempotent: re-running tops up only pairs short of their target game count
# (existing games in games.jsonl are reused), so you can re-run as training
# produces new checkpoints.
#
# Env overrides (all optional): NETWORKS, STRIDE, NUM_GAMES, NEIGHBOR_K,
# NEIGHBOR_GAMES, CROSS_GAMES, START_ITER, ANCHORS, ANCHOR_DIR, NUM_SIMULATIONS,
# CONFIG_DIR, DATA_DIR, ELO_BIN, ELO_CFG, OUT_FILE, PLOT_FILE, SERIES_REGEX.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[elo_multinet] no config dir at $CONFIG_DIR" >&2; exit 1; }

source "$CONFIG_DIR/paths.cfg"
source "$SCRIPT_DIR/env_paths.cfg"
BUILD_DIR="${BUILD_DIR:-$DATA_DIR/build}"
ELO_BIN="${ELO_BIN:-$BUILD_DIR/gomoku_elo}"
ELO_CFG="${ELO_CFG:-$CONFIG_DIR/elo.cfg}"
PY="${PY:-python}"

[[ -x "$ELO_BIN" ]] || { echo "build first: bash scripts/build.sh --target gomoku_elo (-> $BUILD_DIR)" >&2; exit 1; }
[[ -f "$ELO_CFG" ]] || { echo "no config at $ELO_CFG" >&2; exit 1; }

# --- Read a key from a cfg file (default elo.cfg; last occurrence wins,
# strips comments/quotes).
cfg_get() {
    local key="$1" default="${2:-}" file="${3:-$ELO_CFG}" val
    val="$(awk -F= -v k="$key" '
        { sub(/#.*/, "") }
        { gsub(/^[ \t]+|[ \t\r]+$/, "") }
        $0 == "" { next }
        { split($0, kv, "="); key = kv[1]; gsub(/^[ \t]+|[ \t]+$/, "", key)
          if (key != k) next
          val = substr($0, index($0, "=") + 1); gsub(/^[ \t]+|[ \t]+$/, "", val)
          if (val ~ /^".*"$/ || val ~ /^'\''.*'\''$/) val = substr(val, 2, length(val) - 2)
          last = val }
        END { print last }' "$file")"
    echo "${val:-$default}"
}

# Board size / rule are single-sourced from run.cfg (MAIN_BOARD_SIZE /
# MAIN_RULE), not duplicated in elo.cfg. run.cfg.local overrides run.cfg.
run_cfg_get() {
    local key="$1" default="${2:-}" val lv
    val="$(cfg_get "$key" "" "$CONFIG_DIR/run.cfg")"
    if [[ -f "$CONFIG_DIR/run.cfg.local" ]]; then
        lv="$(cfg_get "$key" "" "$CONFIG_DIR/run.cfg.local")"
        [[ -n "$lv" ]] && val="$lv"
    fi
    echo "${val:-$default}"
}
MAIN_BOARD_SIZE="${MAIN_BOARD_SIZE:-$(run_cfg_get MAIN_BOARD_SIZE 15)}"
MAIN_RULE="${MAIN_RULE:-$(run_cfg_get MAIN_RULE renju)}"

# Trailing iter number from a model path (matches python/elo.py:parse_iter).
parse_iter() {
    local stem="${1##*/}"; stem="${stem%.*}"; local n
    n="$(grep -oE '[0-9]+' <<< "$stem" | tail -n 1)"
    if [[ -z "$n" ]]; then echo 999999999; else echo "$((10#$n))"; fi
}

# --- Parameters ---------------------------------------------------------
IFS=', ' read -r -a NETS <<< "${NETWORKS:-$(cfg_get NETWORKS '')}"
(( ${#NETS[@]} > 0 )) || { echo "[elo_multinet] NETWORKS is empty (set env or cfg)" >&2; exit 1; }

STRIDE="${STRIDE:-$(cfg_get STRIDE 20)}"
START_ITER="${START_ITER:-$(cfg_get START_ITER 0)}"
NUM_GAMES="${NUM_GAMES:-$(cfg_get NUM_GAMES 40)}"          # games vs anchor
NEIGHBOR_K="${NEIGHBOR_K:-$(cfg_get NEIGHBOR_K 2)}"
NEIGHBOR_GAMES="${NEIGHBOR_GAMES:-$(cfg_get NEIGHBOR_GAMES 40)}"
CROSS_GAMES="${CROSS_GAMES:-$(cfg_get CROSS_GAMES 40)}"    # adjacent-net same-iter
ANCHOR_DIR="${ANCHOR_DIR:-$(cfg_get ANCHOR_DIR anchors)}"
[[ "$ANCHOR_DIR" = /* ]] || ANCHOR_DIR="$ROOT/$ANCHOR_DIR"
ANCHORS_CFG="${ANCHORS:-$(cfg_get ANCHORS '')}"
SERIES_REGEX="${SERIES_REGEX:-/nets/([^/]+)/}"

OUT_DIR="${OUT_DIR:-$DATA_DIR/elo/multinet}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/games.jsonl}"
PLOT_FILE="${PLOT_FILE:-$OUT_DIR/elo.png}"
TEXT_FILE="${TEXT_FILE:-$OUT_DIR/elo.txt}"
SCHED_FILE="$OUT_DIR/schedule.txt"
mkdir -p "$OUT_DIR"

sim_flag=()
[[ -n "${NUM_SIMULATIONS:-}" ]] && sim_flag=(--num-simulations "$NUM_SIMULATIONS")
# Board size / rule come from run.cfg (single source of truth), not elo.cfg.
topo_flag=(--board-size "$MAIN_BOARD_SIZE" --rule "$MAIN_RULE")

# --- Resolve common anchors (optional) ---------------------------------
anchor_list=()
if [[ -n "$ANCHORS_CFG" ]]; then
    IFS=', ' read -r -a anames <<< "$ANCHORS_CFG"
    for name in "${anames[@]}"; do
        [[ -z "$name" ]] && continue
        [[ "$name" = /* ]] && p="$name" || p="$ANCHOR_DIR/$name"
        [[ -f "$p" ]] || { echo "[elo_multinet] anchor not found: $p" >&2; exit 1; }
        anchor_list+=("$p")
    done
fi

# --- Per-net strided checkpoint lists ----------------------------------
# STRIDED_<net> arrays via namerefs; STRIDED_ITERS_<net> the matching iters.
declare -A NET_CKPTS    # net -> newline-joined strided paths
declare -A NET_ITERS    # net -> newline-joined iters (aligned with NET_CKPTS)
for net in "${NETS[@]}"; do
    mdir="$DATA_DIR/nets/$net"
    [[ -d "$mdir" ]] || { echo "[elo_multinet] missing $mdir (net=$net)" >&2; exit 1; }
    mapfile -t ALL < <(ls "$mdir"/scripted_iter_*.pt 2>/dev/null | sort)
    strided=(); iters=()
    i=0
    for f in "${ALL[@]}"; do
        it="$(parse_iter "$f")"
        (( it < START_ITER )) && { i=$((i+1)); continue; }
        if (( i % STRIDE == 0 )); then strided+=("$f"); iters+=("$it"); fi
        i=$((i+1))
    done
    # always include the newest checkpoint
    if (( ${#ALL[@]} > 0 )); then
        last="${ALL[-1]}"
        if (( ${#strided[@]} == 0 )) || [[ "${strided[-1]}" != "$last" ]]; then
            strided+=("$last"); iters+=("$(parse_iter "$last")")
        fi
    fi
    (( ${#strided[@]} > 0 )) || { echo "[elo_multinet] no checkpoints for net=$net (start_iter=$START_ITER)" >&2; exit 1; }
    NET_CKPTS[$net]="$(printf '%s\n' "${strided[@]}")"
    NET_ITERS[$net]="$(printf '%s\n' "${iters[@]}")"
    echo "[elo_multinet] $net: ${#strided[@]} strided checkpoints (stride=$STRIDE)"
done

# --- Build the pair list (a|b|target_games) ----------------------------
PAIRS=()
# 1) within-net neighbor chains
for net in "${NETS[@]}"; do
    mapfile -t ck <<< "${NET_CKPTS[$net]}"
    n=${#ck[@]}
    for ((i = 0; i < n; i++)); do
        for ((k = 1; k <= NEIGHBOR_K; k++)); do
            j=$((i + k)); (( j >= n )) && break
            PAIRS+=("${ck[i]}|${ck[j]}|$NEIGHBOR_GAMES")
        done
    done
done
# 2) cross-net same-iter pairs between ADJACENT sizes (ties the scale together)
for ((s = 0; s + 1 < ${#NETS[@]}; s++)); do
    na="${NETS[s]}"; nb="${NETS[s+1]}"
    mapfile -t cka <<< "${NET_CKPTS[$na]}"; mapfile -t ita <<< "${NET_ITERS[$na]}"
    mapfile -t ckb <<< "${NET_CKPTS[$nb]}"; mapfile -t itb <<< "${NET_ITERS[$nb]}"
    # index b by iter for matching
    declare -A bidx=()
    for ((j = 0; j < ${#ckb[@]}; j++)); do bidx[${itb[j]}]="${ckb[j]}"; done
    for ((i = 0; i < ${#cka[@]}; i++)); do
        match="${bidx[${ita[i]}]:-}"
        [[ -n "$match" ]] && PAIRS+=("${cka[i]}|$match|$CROSS_GAMES")
    done
    unset bidx
done
# 3) every strided checkpoint vs each common anchor
for net in "${NETS[@]}"; do
    mapfile -t ck <<< "${NET_CKPTS[$net]}"
    for f in "${ck[@]}"; do
        for anc in "${anchor_list[@]}"; do
            PAIRS+=("$f|$anc|$NUM_GAMES")
        done
    done
done

(( ${#anchor_list[@]} > 0 )) || echo "[elo_multinet] NOTE: no anchor set — scale is pinned to the earliest checkpoint (cross-run comparison not meaningful)."

# --- Dedup vs existing games, write schedule ---------------------------
count_existing() {
    local a="$1" b="$2"
    [[ -f "$OUT_FILE" ]] || { echo 0; return; }
    local n1 n2
    n1="$(grep -cF -- "\"a\":\"$a\",\"b\":\"$b\"" "$OUT_FILE" 2>/dev/null || true)"
    n2="$(grep -cF -- "\"a\":\"$b\",\"b\":\"$a\"" "$OUT_FILE" 2>/dev/null || true)"
    echo "$(( ${n1:-0} + ${n2:-0} ))"
}

: > "$SCHED_FILE"
pending=0
for pr in "${PAIRS[@]}"; do
    IFS='|' read -r a b tg <<< "$pr"
    have="$(count_existing "$a" "$b")"
    if (( have >= tg )); then continue; fi
    printf '%s|%s|%d\n' "$a" "$b" "$(( tg - have ))" >> "$SCHED_FILE"
    pending=$((pending + 1))
done

echo "[elo_multinet] ${#PAIRS[@]} pairs total, $pending pending -> $SCHED_FILE"

# --- Run the tournament (single process, internal concurrency) ---------
if (( pending > 0 )); then
    SHARD="$OUT_FILE.shard.jsonl"; : > "$SHARD"
    # Merge whatever the engine produced even on Ctrl+C / crash.
    cleanup() { [[ -s "$SHARD" ]] && grep -E '^\{.*\}$' "$SHARD" >> "$OUT_FILE" 2>/dev/null || true; rm -f "$SHARD"; }
    trap cleanup EXIT
    "$ELO_BIN" --match-schedule "$SCHED_FILE" --config "$ELO_CFG" --output "$SHARD" "${topo_flag[@]}" "${sim_flag[@]}"
    cleanup
    trap - EXIT
else
    echo "[elo_multinet] nothing pending; re-plotting from existing games."
fi

# --- Fit + plot (one curve per network) --------------------------------
[[ -s "$OUT_FILE" ]] || { echo "[elo_multinet] no games in $OUT_FILE" >&2; exit 1; }
"$PY" "$ROOT/python/elo.py" \
    --games "$OUT_FILE" \
    --plot "$PLOT_FILE" \
    --out-text "$TEXT_FILE" \
    --series-regex "$SERIES_REGEX"
echo "[elo_multinet] done. table=$TEXT_FILE plot=$PLOT_FILE"
