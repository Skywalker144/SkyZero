#!/usr/bin/env bash
# AB sweep: PUCT vs Gumbel at 5 sim budgets, 200 games each.
# Both engines run noise-off (Dirichlet & Gumbel noise both 0) and
# stochastic-transform-on (per-NN-call symmetry randomization for
# game diversity).
#
# Outputs:
#   /tmp/ab_puct_sweep/{sims}.jsonl   per-sim-budget JSONL
#   /tmp/ab_puct_sweep/summary.txt    final aggregate winrates

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
BIN="$ROOT/build/gomoku_ab_puct"
MODEL="/home/sky/RL/SkyZero/SkyZero_V5/data_b10c128_b17_renju_run4/models/model_iter_000108.pt"

OUTDIR=/tmp/ab_puct_sweep
mkdir -p "$OUTDIR"
: > "$OUTDIR/summary.txt"

NUM_GAMES=200
SIM_BUDGETS=(100 150 200 400 800)

for SIMS in "${SIM_BUDGETS[@]}"; do
    A_CFG="$OUTDIR/a_${SIMS}.cfg"
    B_CFG="$OUTDIR/b_${SIMS}.cfg"
    sed "s/^NUM_SIMULATIONS=.*/NUM_SIMULATIONS=${SIMS}/" "$SCRIPT_DIR/a.cfg" > "$A_CFG"
    sed "s/^NUM_SIMULATIONS=.*/NUM_SIMULATIONS=${SIMS}/" "$SCRIPT_DIR/b.cfg" > "$B_CFG"

    OUT="$OUTDIR/${SIMS}.jsonl"
    : > "$OUT"

    START=$(date +%s)
    echo "=========================================================" | tee -a "$OUTDIR/summary.txt"
    echo "[sim_sweep] sims=$SIMS games=$NUM_GAMES start=$(date)" | tee -a "$OUTDIR/summary.txt"

    "$BIN" \
        --model "$MODEL" \
        --config-ab "$SCRIPT_DIR/ab.cfg" \
        --config-a "$A_CFG" \
        --config-b "$B_CFG" \
        --output "$OUT" \
        --num-games "$NUM_GAMES" \
        --seed "$((42 + SIMS))" \
        2>&1 | tail -3 | tee -a "$OUTDIR/summary.txt"

    END=$(date +%s)
    ELAPSED=$((END - START))

    # Tally A wins (PUCT) / draws / B wins (Gumbel) from JSONL.
    AW=$(grep -c '"winner_a":1' "$OUT" || true)
    DW=$(grep -c '"winner_a":0' "$OUT" || true)
    BW=$(grep -c '"winner_a":-1' "$OUT" || true)
    PUCT_SCORE=$(awk -v aw="$AW" -v dw="$DW" -v n="$NUM_GAMES" 'BEGIN{printf "%.3f", (aw + 0.5*dw)/n}')

    echo "[sim_sweep] sims=$SIMS done in ${ELAPSED}s | PUCT(A) wins=$AW draws=$DW Gumbel(B) wins=$BW | PUCT score=$PUCT_SCORE" \
        | tee -a "$OUTDIR/summary.txt"
done

echo "=========================================================" | tee -a "$OUTDIR/summary.txt"
echo "[sim_sweep] all done. Files in $OUTDIR/" | tee -a "$OUTDIR/summary.txt"
