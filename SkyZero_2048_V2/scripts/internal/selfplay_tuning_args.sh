#!/usr/bin/env bash
# Shared selfplay CLI "tuning" args — the search / MCTS / value / stochastic knobs
# read from the run.cfg env, assembled into a bash array SP_TUNING_ARGS. Sourced
# by BOTH the bounded per-iter path (internal/selfplay.sh, run.sh) and the
# continuous daemon (scripts/faster_run.sh) so the two selfplay producers can
# never drift. The caller adds the mode-specific args around it
# (--model / --out / --prefix / --iter / --games / --sims / --daemon / --log-dir
# / --progress-secs / ...). The run.cfg vars arrive via the caller's exported env;
# ${VAR:-default} keeps this runnable when a knob is unset.
SP_TUNING_ARGS=(
    --threads "${WORKERS_PER_GPU:-16}"
    --slot-games "${SLOT_GAMES:-64}"
    --server-threads "${INFERENCE_SERVERS_PER_GPU:-2}"
    --batch "${INFERENCE_BATCH_SIZE:-512}"
    --wait-us "${INFERENCE_WAIT_US:-150}"
    --gamma "${GAMMA:-0.999}"
    --td-steps "${TD_STEPS:-10}"
    --value-scale "${VALUE_SCALE:-30}"
    --c-puct "${C_PUCT:-1.25}"
    --c-puct-log "${C_PUCT_LOG:-0}"
    --c-puct-base "${C_PUCT_BASE:-500}"
    --fpu-reduction-max "${FPU_REDUCTION_MAX:-0}"
    --cpuct-stdev-scale "${CPUCT_UTILITY_STDEV_SCALE:-0}"
    --cpuct-stdev-prior "${CPUCT_UTILITY_STDEV_PRIOR:-0.20}"
    --cpuct-stdev-prior-weight "${CPUCT_UTILITY_STDEV_PRIOR_WEIGHT:-2.0}"
    --non-root-algo "${NON_ROOT_SEARCH_ALGO:-puct}"
    --enable-tree-reuse "${ENABLE_TREE_REUSE:-0}"
    --root-algo "${ROOT_SEARCH_ALGO:-gumbel}"
    --fast-root-algo "${FAST_ROOT_SEARCH_ALGO:-${ROOT_SEARCH_ALGO:-gumbel}}"
    --fast-non-root-algo "${FAST_NON_ROOT_SEARCH_ALGO:-${NON_ROOT_SEARCH_ALGO:-puct}}"
    --root-fpu-reduction-max "${ROOT_FPU_REDUCTION_MAX:-0}"
    --root-desired-per-child-visits-coeff "${ROOT_DESIRED_PER_CHILD_VISITS_COEFF:-0}"
    --chosen-move-temperature "${CHOSEN_MOVE_TEMPERATURE:-0}"
    --chosen-move-temperature-early "${CHOSEN_MOVE_TEMPERATURE_EARLY:-0}"
    --chosen-move-temperature-halflife "${CHOSEN_MOVE_TEMPERATURE_HALFLIFE:-19}"
    --root-policy-temperature "${ROOT_POLICY_TEMPERATURE:-1}"
    --root-policy-temperature-early "${ROOT_POLICY_TEMPERATURE_EARLY:-1}"
    --root-dirichlet-alpha "${ROOT_DIRICHLET_ALPHA:-0}"
    --root-noise-frac "${ROOT_NOISE_FRAC:-0.25}"
    --fast-search-prob "${FAST_SEARCH_PROB:-0}"
    --fast-search-target-weight "${FAST_SEARCH_TARGET_WEIGHT:-0}"
    --fast-search-fraction "${FAST_SEARCH_FRACTION:-0.16667}"
    --clear-tree-before-full-search "${CLEAR_TREE_BEFORE_FULL_SEARCH:-0}"
    --policy-surprise-weight "${POLICY_SURPRISE_DATA_WEIGHT:-0}"
    --value-surprise-weight "${VALUE_SURPRISE_DATA_WEIGHT:-0}"
    --side-position-prob "${SIDE_POSITION_PROB:-0}"
    --side-position-visits "${SIDE_POSITION_VISITS:-0}"
    --stochastic-transform-root "${ENABLE_STOCHASTIC_TRANSFORM_ROOT:-0}"
    --stochastic-transform-child "${ENABLE_STOCHASTIC_TRANSFORM_CHILD:-0}"
    --device "${DEVICE:-cuda}"
)
