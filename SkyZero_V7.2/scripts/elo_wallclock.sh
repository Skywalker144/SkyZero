#!/usr/bin/env bash
# elo_wallclock.sh — unified-anchor Elo-vs-wallclock measurement across the
# auto_wallclock_sims.sh runs (one DATA_DIR per num_simulations point).
#
# For every finished run (EXPERIMENT_DONE marker) under wallclock_sims/:
#   1. stride its scripted checkpoints (always including the newest),
#   2. within-run neighbor chains (near-50% pairs = high Elo resolution),
#   3. every strided checkpoint vs the common ANCHORS (ties all runs to ONE
#      Elo scale; anchor origin doesn't bias the fit).
# Then ONE gomoku_elo tournament over the union schedule, a single BT-MLE fit
# (python/elo.py), and an Elo-vs-wallclock plot.
#
# X axis: training wall-clock hours, reconstructed from scripted_iter_*.pt
# mtimes — export wrote each file the moment the model existed, so mtime IS
# the wall-clock it became available. Don't touch/copy those files in place.
# Inter-checkpoint gaps > 30 min are treated as pauses (a run that was stopped
# and resumed later) and replaced by the run's median iter interval, so a
# resumed run's curve stays on the same time axis as uninterrupted ones.
#
# Idempotent like elo_multinet.sh: re-running tops up only pairs short of
# their target game count, so you can refine with a smaller STRIDE or re-run
# after more sweep points finish, reusing every game already played.
#
# Knobs are env-only (elo.cfg's STRIDE/NUM_GAMES target elo.sh's per-run use):
#   RUNS            comma/space list of run tags (default: all with marker)
#   NET             network name inside each run    (default b4c96)
#   STRIDE          take every Nth checkpoint       (default 40)
#   NEIGHBOR_K      chain neighbors per checkpoint  (default 2)
#   NEIGHBOR_GAMES  games per chain pair            (default 40)
#   NUM_GAMES       games per (checkpoint, anchor)  (default 40)
#   ANCHORS         filenames under ANCHOR_DIR      (default: the two
#                   nsim0064 snapshots created for this study)
#   NUM_SIMULATIONS eval sims override              (default: elo.cfg = 400)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
CONFIG_DIR="$ROOT/configs/wallclock_sims"
EXP_ROOT="$ROOT/wallclock_sims"

source "$SCRIPT_DIR/env_paths.cfg"
ELO_BIN="${ELO_BIN:-$ROOT/cpp/build/gomoku_elo}"
ELO_CFG="${ELO_CFG:-$CONFIG_DIR/elo.cfg}"
PY="${PY:-python}"

[[ -x "$ELO_BIN" ]] || { echo "build first: cmake --build $ROOT/cpp/build --target gomoku_elo" >&2; exit 1; }
[[ -f "$ELO_CFG" ]] || { echo "no config at $ELO_CFG" >&2; exit 1; }

NET="${NET:-b4c96}"
STRIDE="${STRIDE:-40}"
NEIGHBOR_K="${NEIGHBOR_K:-2}"
NEIGHBOR_GAMES="${NEIGHBOR_GAMES:-40}"
NUM_GAMES="${NUM_GAMES:-40}"
ANCHOR_DIR="${ANCHOR_DIR:-$ROOT/anchors}"
ANCHORS="${ANCHORS:-wallclock_b4c96_nsim0064_iter300.pt, wallclock_b4c96_nsim0064_iter677.pt}"

OUT_DIR="${OUT_DIR:-$EXP_ROOT/elo}"
OUT_FILE="$OUT_DIR/games.jsonl"
PLOT_FILE="$OUT_DIR/elo_wallclock.png"
TEXT_FILE="$OUT_DIR/elo.txt"
SCHED_FILE="$OUT_DIR/schedule.txt"
XMAP_FILE="$OUT_DIR/xmap.tsv"
mkdir -p "$OUT_DIR"

sim_flag=()
[[ -n "${NUM_SIMULATIONS:-}" ]] && sim_flag=(--num-simulations "$NUM_SIMULATIONS")

# Trailing iter number from a model path (matches python/elo.py:parse_iter).
parse_iter() {
    local stem="${1##*/}"; stem="${stem%.*}"; local n
    n="$(grep -oE '[0-9]+' <<< "$stem" | tail -n 1)"
    if [[ -z "$n" ]]; then echo 999999999; else echo "$((10#$n))"; fi
}

# --- Runs to measure -----------------------------------------------------
RUN_DIRS=()
if [[ -n "${RUNS:-}" ]]; then
    IFS=', ' read -r -a tags <<< "$RUNS"
    for t in "${tags[@]}"; do
        [[ -z "$t" ]] && continue
        [[ -d "$EXP_ROOT/$t" ]] || { echo "[elo_wallclock] no run dir $EXP_ROOT/$t" >&2; exit 1; }
        RUN_DIRS+=("$EXP_ROOT/$t")
    done
else
    for d in "$EXP_ROOT"/nsim*/; do
        [[ -f "$d/EXPERIMENT_DONE" ]] && RUN_DIRS+=("${d%/}")
    done
fi
(( ${#RUN_DIRS[@]} > 0 )) || { echo "[elo_wallclock] no finished runs under $EXP_ROOT" >&2; exit 1; }

# --- Anchors --------------------------------------------------------------
anchor_list=()
IFS=', ' read -r -a anames <<< "$ANCHORS"
for name in "${anames[@]}"; do
    [[ -z "$name" ]] && continue
    [[ "$name" = /* ]] && p="$name" || p="$ANCHOR_DIR/$name"
    [[ -f "$p" ]] || { echo "[elo_wallclock] anchor not found: $p" >&2; exit 1; }
    anchor_list+=("$p")
done
(( ${#anchor_list[@]} > 0 )) || { echo "[elo_wallclock] ANCHORS is empty — cross-run scale needs at least one" >&2; exit 1; }

# --- Per-run strided checkpoints + wallclock x-map ------------------------
declare -A RUN_CKPTS    # run dir -> newline-joined strided paths
: > "$XMAP_FILE"
for run in "${RUN_DIRS[@]}"; do
    mdir="$run/nets/$NET"
    iter0="$mdir/scripted_iter_000000.pt"
    [[ -f "$iter0" ]] || { echo "[elo_wallclock] missing $iter0 (need it as the t0 reference)" >&2; exit 1; }
    mapfile -t ALL < <(ls "$mdir"/scripted_iter_*.pt 2>/dev/null | sort)
    strided=()
    for ((i = 0; i < ${#ALL[@]}; i++)); do
        (( i % STRIDE == 0 )) && strided+=("${ALL[i]}")
    done
    last="${ALL[-1]}"
    [[ "${strided[-1]}" != "$last" ]] && strided+=("$last")
    RUN_CKPTS[$run]="$(printf '%s\n' "${strided[@]}")"
    "$PY" -c '
import glob, os, sys
mdir, sel = sys.argv[1], sys.argv[2:]
files = sorted(glob.glob(os.path.join(mdir, "scripted_iter_*.pt")))
mts = [os.path.getmtime(f) for f in files]
dts = [b - a for a, b in zip(mts, mts[1:])]
norm = sorted(d for d in dts if d <= 1800)
med = norm[len(norm) // 2] if norm else 0.0
elapsed, t = {files[0]: 0.0}, 0.0
for f, d in zip(files[1:], dts):
    t += d if d <= 1800 else med
    elapsed[f] = t
for f in sel:
    print(f"{f}\t{elapsed[f] / 3600:.4f}")
' "$mdir" "${strided[@]}" >> "$XMAP_FILE"
    echo "[elo_wallclock] $(basename "$run"): ${#strided[@]} strided checkpoints (stride=$STRIDE, last=$(basename "$last"))"
done

# --- Build the pair list (a|b|target_games) -------------------------------
PAIRS=()
# 1) within-run neighbor chains
for run in "${RUN_DIRS[@]}"; do
    mapfile -t ck <<< "${RUN_CKPTS[$run]}"
    n=${#ck[@]}
    for ((i = 0; i < n; i++)); do
        for ((k = 1; k <= NEIGHBOR_K; k++)); do
            j=$((i + k)); (( j >= n )) && break
            PAIRS+=("${ck[i]}|${ck[j]}|$NEIGHBOR_GAMES")
        done
    done
done
# 2) every strided checkpoint vs each common anchor
for run in "${RUN_DIRS[@]}"; do
    mapfile -t ck <<< "${RUN_CKPTS[$run]}"
    for f in "${ck[@]}"; do
        for anc in "${anchor_list[@]}"; do
            PAIRS+=("$f|$anc|$NUM_GAMES")
        done
    done
done

# --- Dedup vs existing games, write schedule ------------------------------
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

echo "[elo_wallclock] ${#PAIRS[@]} pairs total, $pending pending -> $SCHED_FILE"

# --- Run the tournament (single process, internal concurrency) ------------
if (( pending > 0 )); then
    SHARD="$OUT_FILE.shard.jsonl"; : > "$SHARD"
    # Merge whatever the engine produced even on Ctrl+C / crash.
    cleanup() { [[ -s "$SHARD" ]] && grep -E '^\{.*\}$' "$SHARD" >> "$OUT_FILE" 2>/dev/null || true; rm -f "$SHARD"; }
    trap cleanup EXIT
    "$ELO_BIN" --match-schedule "$SCHED_FILE" --config "$ELO_CFG" --output "$SHARD" "${sim_flag[@]}"
    cleanup
    trap - EXIT
else
    echo "[elo_wallclock] nothing pending; re-plotting from existing games."
fi

# --- Fit + plot (one curve per nsim run, x = wall-clock hours) -------------
[[ -s "$OUT_FILE" ]] || { echo "[elo_wallclock] no games in $OUT_FILE" >&2; exit 1; }
"$PY" "$ROOT/python/elo.py" \
    --games "$OUT_FILE" \
    --plot "$PLOT_FILE" \
    --out-text "$TEXT_FILE" \
    --series-regex '(nsim[0-9]+)' \
    --x-map "$XMAP_FILE" \
    --x-label "wall-clock hours" \
    --plot-max-se 300
echo "[elo_wallclock] done. table=$TEXT_FILE plot=$PLOT_FILE"
