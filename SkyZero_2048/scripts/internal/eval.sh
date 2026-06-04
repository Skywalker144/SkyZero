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

# Load run.cfg too (VALUE_SCALE/VALUE_TRANSFORM/EVAL_*) so the C++ eval is correct
# even when run standalone — not only when run.sh/faster_run.sh exported them.
set -a
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
set +a
source "$SCRIPTS_DIR/env_paths.cfg"

export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

# Batched C++ eval: the SAME selfplay2048_par binary in bounded + deterministic
# (--noise 0) mode — multi-worker + batched inference. --max-moves caps long
# deterministic games (which otherwise never reach terminal in time and stall the
# whole batch); cap=4000 covers up to the 8192 tile (~6min/100 games at 64 sims).
# Don't lower --sims to speed it up: 32 sims tanks search quality (reach4096
# 69%->36%) for the same wall-clock. --eval-log writes a python/evaluate.py-schema
# eval.tsv row (view_loss.py reads it unchanged). NO --out / --log-dir, so it never
# touches selfplay/ or selfplay_stats.tsv. MUST pass VALUE_SCALE/VALUE_TRANSFORM or
# the h()-space value head is mis-decoded. Deterministic on a fixed --seed 42 set.
EVAL_BIN="$ROOT/cpp/build/selfplay2048_par"
MODEL="$DATA_DIR/nets/$network/latest.pt"             # TorchScript export of this net
[[ -f "$MODEL" ]] || MODEL="$DATA_DIR/models/latest.pt"

"$EVAL_BIN" \
    --model "$MODEL" \
    --games "${EVAL_GAMES:-100}" \
    --sims "${EVAL_SIMS:-64}" \
    --noise 0 \
    --max-moves "${EVAL_MAX_MOVES:-4000}" \
    --threads "${EVAL_THREADS:-4}" \
    --slot-games "${EVAL_SLOT_GAMES:-32}" \
    --server-threads "${EVAL_SERVER_THREADS:-2}" \
    --batch "${BATCH:-512}" \
    --wait-us "${WAIT_US:-500}" \
    --value-scale "${VALUE_SCALE:-4000}" \
    --value-transform "${VALUE_TRANSFORM:-0}" \
    --gamma "${GAMMA:-0.999}" \
    --seed 42 \
    --device "${EVAL_DEVICE:-cuda}" \
    --progress-secs 0 \
    --iter "$iter" \
    --eval-network "$network" \
    --eval-log "$DATA_DIR/logs/eval.tsv"
