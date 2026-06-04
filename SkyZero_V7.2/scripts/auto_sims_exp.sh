#!/usr/bin/env bash
# ============================================================================
# auto_sims_exp.sh — sequential num_simulations x {puct,gumbel} sweep for the
# Gumbel low-budget convergence study.
#
# Trains b4c96 on 11x11 renju FROM SCRATCH for a fixed wall-clock budget, once
# for every (num_simulations, non_root_search_algo) combination, each into its
# own DATA_DIR. Built for a SINGLE GPU: runs are strictly sequential, so the
# "same device, same budget" fairness the experiment needs holds by
# construction.
#
#   Launch:    bash scripts/auto_sims_exp.sh
#   Resumable: each finished run drops a EXPERIMENT_DONE marker in its DATA_DIR;
#              re-launching skips finished runs. To cleanly REDO a run, delete
#              its data dir:  rm -rf data_sims_exp/nsim0032_puct
#              (a Ctrl+C'd run has no marker — re-launching would RESUME it for
#              another full budget, so prefer deleting+redoing interrupted runs).
#
# What varies per run, and how it reaches the binaries:
#   * num_simulations -> exported as NUM_SIMULATIONS_STAGES; run.cfg reads
#     ${NUM_SIMULATIONS_STAGES:-...} so the value survives `source`, and
#     warmup.py passes it to the C++ as --num-simulations.
#   * non_root_search_algo -> selected by CONFIG_DIR (the C++ reads it from the
#     run.cfg FILE, not the environment), via two sibling config dirs that
#     differ only in that one line.
#   * DATA_DIR -> a unique dir per run so the 10 trainings never touch each
#     other (paths.cfg reads ${DATA_DIR:-...}).
#
# Everything else (board, net, 4h budget, pipeline granularity) lives in
# configs/sims_exp_{puct,gumbel}/run.cfg.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
EXP_ROOT="$ROOT/data_sims_exp"

# ---- knobs you may want to edit -------------------------------------------
# nsim points, in a spread order so an early Ctrl+C still leaves the full range
# coarsely covered (middle, then both extremes, then fill-in).
SIMS=(32 2 512 8 128)
ALGOS=(puct gumbel)        # both non-root modes; each has its own config dir
PER_RUN_SECONDS=14400      # 4h wall-clock budget per run
# ---------------------------------------------------------------------------

declare -A CFGDIR=(
    [puct]="$ROOT/configs/sims_exp_puct"
    [gumbel]="$ROOT/configs/sims_exp_gumbel"
)

for a in "${ALGOS[@]}"; do
    [[ -d "${CFGDIR[$a]}" ]] || { echo "[auto] missing config dir: ${CFGDIR[$a]}" >&2; exit 1; }
done
mkdir -p "$EXP_ROOT"

# Build once up front (MAX_BOARD_SIZE=11) so a compile error surfaces now, not
# four hours into the first run. build.sh owns the first-time configure
# (libtorch/nvcc/arch flags) but does NOT re-point an existing build dir at a
# new experiment — so afterwards we force this experiment's CONFIG_DIR (and thus
# MAX_BOARD_SIZE=11) into the cmake cache exactly as run.sh does. Without this,
# the cache can keep a previous experiment's board size and build the wrong one.
echo "[auto] building C++ for this experiment (board size 11) ..."
bash "$SCRIPT_DIR/build.sh"
source "$SCRIPT_DIR/env_paths.cfg"
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DSKYZERO_CONFIG_DIR="${CFGDIR[puct]}"
cmake --build "$ROOT/cpp/build" -j
echo "[auto] board-11 build ready."

total=$(( ${#SIMS[@]} * ${#ALGOS[@]} ))
idx=0
overall_start=$(date +%s)

for nsim in "${SIMS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        idx=$(( idx + 1 ))
        tag="nsim$(printf '%04d' "$nsim")_${algo}"
        run_data="$EXP_ROOT/$tag"
        marker="$run_data/EXPERIMENT_DONE"
        log="$EXP_ROOT/${tag}.runner.log"

        if [[ -f "$marker" ]]; then
            echo "[auto] ($idx/$total) SKIP  $tag — already done"
            continue
        fi

        echo "[auto] ($idx/$total) START $tag  nsim=$nsim algo=$algo budget=${PER_RUN_SECONDS}s"
        echo "[auto]        DATA_DIR=$run_data"
        echo "[auto]        log=$log"
        run_start=$(date +%s)

        # run.sh loops until MAX_TIME_SECONDS, then returns. The env prefix sets
        # the per-run knobs; run.cfg's ${VAR:-...} forms keep them through
        # `source`. Don't let one failed run abort the whole sweep.
        set +e
        DATA_DIR="$run_data" \
        CONFIG_DIR="${CFGDIR[$algo]}" \
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
done

echo "[auto] sweep finished in $(( $(date +%s) - overall_start ))s."
echo "[auto] per-run data + logs under: $EXP_ROOT/"
echo "[auto] next step: the Elo-vs-time measurement (separate script)."
