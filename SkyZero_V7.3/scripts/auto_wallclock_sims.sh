#!/usr/bin/env bash
# ============================================================================
# auto_wallclock_sims.sh — Elo-vs-wallclock study across num_simulations.
#
# Trains b4c96 on 9x9 renju FROM SCRATCH for a FIXED wall-clock budget, once
# per num_simulations point, each into its own DATA_DIR. Built for a SINGLE
# GPU: runs are strictly sequential, so the "same device, same budget"
# fairness the study needs holds by construction.
#
#   Launch:    bash scripts/auto_wallclock_sims.sh
#   Resumable: each finished run drops an EXPERIMENT_DONE marker in its
#              DATA_DIR; re-launching skips finished runs. To cleanly REDO a
#              run, delete its data dir:  rm -rf wallclock_sims/nsim0032
#              (a Ctrl+C'd run has NO marker — re-launching RESUMES it for
#              another full budget, so prefer deleting+redoing interrupted runs.)
#
# What varies per run, and how it reaches the binaries:
#   * num_simulations -> exported as NUM_SIMULATIONS_STAGES; run.cfg reads
#     ${NUM_SIMULATIONS_STAGES:-...} so the value survives `source`, and
#     warmup.py passes it to the C++ as --num-simulations.
#   * DATA_DIR -> a unique dir per run so the trainings never touch each other
#     (paths.cfg reads ${DATA_DIR:-...}).
#
# Everything else (board, net=b4c96, 5h budget, pipeline granularity) lives in
# configs/wallclock_sims/run.cfg and is identical across all runs — sims is the
# ONLY knob that moves.
#
# Next step (separate): the unified-anchor Elo-vs-time measurement across the
# per-sims DATA_DIRs. Needs a 9x9 b4c96 anchor; not done here.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
CONFIG_DIR="$ROOT/configs/wallclock_sims"
EXP_ROOT="$ROOT/wallclock_sims"

# ---- knobs you may want to edit -------------------------------------------
# nsim points, in a spread (middle-outward) order so an early Ctrl+C still
# leaves the full range coarsely covered.
SIMS=(32 8 128 2 64 16 512)
PER_RUN_SECONDS=18000      # 5h wall-clock budget per run
# ---------------------------------------------------------------------------

[[ -d "$CONFIG_DIR" ]] || { echo "[auto] missing config dir: $CONFIG_DIR" >&2; exit 1; }
mkdir -p "$EXP_ROOT"

# Build once up front (board size 9) so a compile error surfaces now, not five
# hours into the first run. build.sh owns the first-time configure
# (libtorch/nvcc/arch flags) but does NOT re-point an existing build dir at a
# new experiment — so afterwards we force this experiment's CONFIG_DIR (and thus
# MAX_BOARD_SIZE=9) into the cmake cache exactly as run.sh does. Without this,
# the cache can keep a previous experiment's board size and build the wrong one.
echo "[auto] building C++ for this experiment (board size 9) ..."
bash "$SCRIPT_DIR/build.sh"
source "$SCRIPT_DIR/env_paths.cfg"
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DSKYZERO_CONFIG_DIR="$CONFIG_DIR"
cmake --build "$ROOT/cpp/build" -j
echo "[auto] board-9 build ready."

total=${#SIMS[@]}
idx=0
overall_start=$(date +%s)

for nsim in "${SIMS[@]}"; do
    idx=$(( idx + 1 ))
    tag="nsim$(printf '%04d' "$nsim")"
    run_data="$EXP_ROOT/$tag"
    marker="$run_data/EXPERIMENT_DONE"
    log="$EXP_ROOT/${tag}.runner.log"

    if [[ -f "$marker" ]]; then
        echo "[auto] ($idx/$total) SKIP  $tag — already done"
        continue
    fi

    echo "[auto] ($idx/$total) START $tag  nsim=$nsim budget=${PER_RUN_SECONDS}s"
    echo "[auto]        DATA_DIR=$run_data"
    echo "[auto]        log=$log"
    run_start=$(date +%s)

    # run.sh loops until MAX_TIME_SECONDS, then returns. The env prefix sets the
    # per-run knobs; run.cfg's ${VAR:-...} forms keep them through `source`.
    # Don't let one failed run abort the whole sweep.
    set +e
    CONFIG_DIR="$CONFIG_DIR" \
    DATA_DIR="$run_data" \
    NUM_SIMULATIONS_STAGES="$nsim" \
    NUM_SIMULATIONS_SCHEDULE="0" \
    MAX_TIME_SECONDS="$PER_RUN_SECONDS" \
        bash "$SCRIPT_DIR/run.sh" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -eq 0 ]]; then
        touch "$marker"
        echo "[auto] ($idx/$total) DONE  $tag in $(( $(date +%s) - run_start ))s"
    else
        echo "[auto] ($idx/$total) FAILED $tag rc=$rc — see $log; continuing to next run" >&2
    fi
done

echo "[auto] sweep finished in $(( $(date +%s) - overall_start ))s."
echo "[auto] per-run data + logs under: $EXP_ROOT/"
echo "[auto] next step: the unified-anchor Elo-vs-time measurement (separate script)."
