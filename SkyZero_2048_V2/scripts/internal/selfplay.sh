#!/usr/bin/env bash
# One bounded self-play batch on one GPU, driving the 2048 parallel binary
# (selfplay2048_par). run.sh launches one of these per train card with
# MAIN_GPU=<card> and games = its share of the gate order. Multi-GPU spare-card
# production is selfplay_daemon.sh's job.
#
# The 2048 binary is CLI-arg driven (it does NOT parse run.cfg like the mainline
# selfplay_main), so this script maps every run.cfg knob to a flag. The run.cfg
# values arrive via run.sh's exported env (set -a); ${VAR:-default} keeps the
# script runnable standalone.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/log_common.sh"

iter="${1:?iter required}"
games="${2:?games required}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
BUILD_DIR="${BUILD_DIR:-$DATA_DIR/build}"
SELFPLAY_BIN="${SELFPLAY_BIN:-$BUILD_DIR/selfplay2048_par}"
PY="${PY:-python}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "$(_tag SelfPlay) binary not found or not executable: $SELFPLAY_BIN" >&2
    echo "Build it first: bash scripts/build.sh" >&2
    exit 1
fi

# Pin this batch to its card so the binary's --device cuda uses exactly it
# (and won't collide with the daemon on spare cards). run.sh sets MAIN_GPU.
MAIN_GPU="${MAIN_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$MAIN_GPU}"

# Per-iter NUM_SIMULATIONS warmup (falls back to SIMS when STAGES/SCHEDULE empty).
NSIM=$( cd "$ROOT/python" && "$PY" warmup.py num-simulations --data-dir "$DATA_DIR" )

# Unique npz prefix per (iter, card) so parallel train-card batches never
# overwrite each other's shards.
PREFIX="iter$(printf '%06d' "$iter")_gpu${MAIN_GPU}"

# Shared search/MCTS/value/stochastic tuning args (kept in sync with the daemon
# producer in faster_run.sh via this one file).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/selfplay_tuning_args.sh"

echo "$(_tag SelfPlay) iter=$iter games=$games num_simulations=$NSIM main_gpu=$MAIN_GPU"
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --out "$DATA_DIR/selfplay" \
    --prefix "$PREFIX" \
    --log-dir "$DATA_DIR/logs" \
    --iter "$iter" \
    --games "$games" \
    --sims "$NSIM" \
    "${SP_TUNING_ARGS[@]}" \
    --progress-secs "${PROGRESS_SECS:-0}"
