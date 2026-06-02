#!/usr/bin/env bash
# One iteration of C++ parallel self-play for 2048. Reads the active model from
# data/models/latest.pt (the mirror run.sh maintains), writes npz shards into
# data/selfplay/, and appends V7.1-schema rows to logs/{selfplay,selfplay_stats}.tsv.
#
# Runs on MAIN_GPU only; multi-GPU spread is selfplay_daemon.sh's job.
# (Phase B: the tsv is parsed from the binary's stdout via selfplay_log.py.
#  Phase C moves tsv-writing into the binary and adds --daemon.)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"

iter="${1:?iter required}"
games="${2:?games required}"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay2048_par}"
source "$SCRIPTS_DIR/env_paths.cfg"
source "$CONFIG_DIR/paths.cfg"
PY="${PY:-python}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "[selfplay.sh] binary not found: $SELFPLAY_BIN  (build: bash scripts/build.sh --target selfplay2048_par)" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${MAIN_GPU:-0}"

NSIM=$( cd "$ROOT/python" && "$PY" warmup.py num-simulations --data-dir "$DATA_DIR" )
printf -v prefix "iter%04d" "$iter"
mkdir -p "$DATA_DIR/selfplay" "$DATA_DIR/logs"

echo "[selfplay.sh] iter=$iter games=$games sims=$NSIM main_gpu=${MAIN_GPU:-0} threads=${THREADS:-6} slot_games=${SLOT_GAMES:-128}"

# The binary appends V7.1-schema rows to logs/{selfplay,selfplay_stats}.tsv itself
# (via --log-dir / --iter); stdout streams live to the terminal.
"$SELFPLAY_BIN" \
    --model "$DATA_DIR/models/latest.pt" \
    --out "$DATA_DIR/selfplay" \
    --prefix "$prefix" \
    --iter "$iter" \
    --games "$games" \
    --sims "$NSIM" \
    --threads "${THREADS:-6}" \
    --slot-games "${SLOT_GAMES:-128}" \
    --server-threads "${SERVER_THREADS:-3}" \
    --batch "${BATCH:-512}" \
    --wait-us "${WAIT_US:-500}" \
    --value-scale "${VALUE_SCALE:-4000}" \
    --td-steps "${TD_STEPS:-0}" \
    --device "${DEVICE:-cuda}" \
    --log-dir "$DATA_DIR/logs" \
    --progress-secs "${PROGRESS_SECS:-15}" \
    --noise 1
