#!/usr/bin/env bash
# Per-card transient selfplay with watchdog (V7.3 gate model, design §2.5).
#
# Runs selfplay_main --daemon pinned to ONE GPU, hot-reloading latest.pt like
# the persistent daemon. Under the gate model all production is daemon-side,
# so a crashed selfplay must come back by itself or the gate
# (compute_selfplay_target.py --wait) stalls — hence the restart loop.
# Lifecycle is owned by run.sh (start_card_selfplay / kill_card_selfplay).
#
# TERM/INT: forward to the child so it settles its selfplay.tsv row and
# flushes NPZs, then exit. NEVER `kill 0` here — this script shares run.sh's
# process group (the selfplay_daemon.sh-style trap would kill the whole run).
#
# Usage: card_selfplay.sh <gpu>
set -euo pipefail

g="${1:?gpu index required}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/log_common.sh"

CONFIG_DIR="${CONFIG_DIR:-$ROOT/configs/baseline}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"

set -a
# shellcheck disable=SC1091
source "$CONFIG_DIR/run.cfg"
[[ -f "$CONFIG_DIR/run.cfg.local" ]] && source "$CONFIG_DIR/run.cfg.local"
source "$CONFIG_DIR/paths.cfg"
source "$SCRIPTS_DIR/env_paths.cfg"
set +a

export DATA_DIR
PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay_main}"

if [[ ! -x "$SELFPLAY_BIN" ]]; then
    echo "$(_tag Card) binary not found or not executable: $SELFPLAY_BIN" >&2
    exit 1
fi

# One GPU: per-GPU counts apply as-is, all servers on this card.
SV="${INFERENCE_SERVERS_PER_GPU:-2}"
devices=""
for ((i=0; i<SV; i++)); do devices+="${devices:+,}$g"; done
export NUM_INFERENCE_SERVERS="$SV"
export NUM_WORKERS="${WORKERS_PER_GPU:-32}"
export INFERENCE_SERVER_DEVICES="$devices"

CHILD=""
trap 'trap - INT TERM; [[ -n "$CHILD" ]] && kill "$CHILD" 2>/dev/null && wait "$CHILD" 2>/dev/null; exit 0' INT TERM

echo "$(_tag Card) gpu $g selfplay up (servers=$SV workers=$NUM_WORKERS devices=$devices)"
while true; do
    "$SELFPLAY_BIN" --daemon \
        --model "$DATA_DIR/models/latest.pt" \
        --output-dir "$DATA_DIR/selfplay" \
        --config "$CONFIG_DIR/run.cfg" \
        --log-dir "$DATA_DIR/logs" \
        --model-watch-poll-ms "${DAEMON_RELOAD_POLL_MS:-2000}" \
        --sims-warmup-cmd "cd $ROOT/python && $PY warmup.py num-simulations --data-dir $DATA_DIR" &
    CHILD=$!
    rc=0
    wait "$CHILD" || rc=$?
    CHILD=""
    if [[ "$rc" -eq 0 || "$rc" -eq 130 ]]; then
        echo "$(_tag Card) gpu $g selfplay exited rc=$rc; done."
        exit 0
    fi
    echo "$(_tag Card) gpu $g selfplay_main crashed rc=$rc; restarting in 5s"
    sleep 5
done
