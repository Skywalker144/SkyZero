#!/usr/bin/env bash
# Periodic single-agent eval of the active network -> logs/eval.tsv (2048's
# analogue of V7.1's Gomoku Elo step: absolute score + tile reach-rates).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
network="${2:?network required (e.g. b6c96)}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
PY=${PY:-python}

export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

cd "$ROOT/python"
"$PY" evaluate.py --data-dir "$DATA_DIR" --network "$network" \
    --iter "$iter" --games "${EVAL_GAMES:-50}"
