#!/usr/bin/env bash
# Persistent multi-GPU self-play daemon for 2048.
#
# The 2048 inference server is single-device, so this launches ONE daemon
# process per spare GPU (cuda:1..GPU_NUM-1, or SELFPLAY_DAEMON_GPUS). Each pins
# its GPU via CUDA_VISIBLE_DEVICES, continuously generates npz into
# data/selfplay/ (pid-tagged daemon_v<ver>_p<pid> prefixes so processes don't
# collide), and hot-reloads data/models/latest.pt whenever run.sh exports.
#
# Usage: GPU_NUM=4 bash scripts/internal/selfplay_daemon.sh
#   GPU_NUM<=1 -> exits (daemon not needed in single-GPU mode).
set -euo pipefail

trap 'trap - INT TERM; echo "[daemon-watchdog] stopping."; kill 0 2>/dev/null; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
ROOT="$(cd -- "$SCRIPTS_DIR/.." &> /dev/null && pwd)"
cd "$ROOT"

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
mkdir -p "$DATA_DIR"/{models,selfplay,logs}

PY=${PY:-python}
SELFPLAY_BIN="${SELFPLAY_BIN:-$ROOT/cpp/build/selfplay2048_par}"
[[ -x "$SELFPLAY_BIN" ]] || { echo "[daemon] binary not found: $SELFPLAY_BIN" >&2; exit 1; }

GPU_NUM="${GPU_NUM:-1}"
MAIN_GPU="${MAIN_GPU:-0}"

# Spare GPUs = all except MAIN_GPU, unless SELFPLAY_DAEMON_GPUS is pinned.
if [[ -z "${SELFPLAY_DAEMON_GPUS:-}" ]]; then
    [[ "$GPU_NUM" -le 1 ]] && { echo "[daemon] GPU_NUM<=1, daemon not needed; exiting."; exit 1; }
    gpus=""
    for ((i=0; i<GPU_NUM; i++)); do
        [[ "$i" -eq "$MAIN_GPU" ]] && continue
        [[ -n "$gpus" ]] && gpus+=","
        gpus+="$i"
    done
    SELFPLAY_DAEMON_GPUS="$gpus"
fi

POLL_MS="${DAEMON_RELOAD_POLL_MS:-2000}"
SIMS_WARMUP_CMD="cd $ROOT/python && $PY warmup.py num-simulations --data-dir $DATA_DIR"
echo "[daemon-watchdog] gpus=$SELFPLAY_DAEMON_GPUS poll_ms=$POLL_MS"

# One restart-on-crash watchdog per spare GPU.
run_one() {
    local gpu="$1"
    while true; do
        CUDA_VISIBLE_DEVICES="$gpu" "$SELFPLAY_BIN" --daemon \
            --model "$DATA_DIR/models/latest.pt" \
            --out "$DATA_DIR/selfplay" \
            --log-dir "$DATA_DIR/logs" \
            --model-watch-poll-ms "$POLL_MS" \
            --sims-warmup-cmd "$SIMS_WARMUP_CMD" \
            --sims "${SIMS:-64}" \
            --threads "${THREADS:-6}" \
            --slot-games "${SLOT_GAMES:-128}" \
            --server-threads "${SERVER_THREADS:-3}" \
            --batch "${BATCH:-512}" \
            --wait-us "${WAIT_US:-500}" \
            --value-scale "${VALUE_SCALE:-4000}" \
            --value-transform "${VALUE_TRANSFORM:-0}" \
            --gamma "${GAMMA:-0.999}" \
            --td-steps "${TD_STEPS:-0}" \
            --progress-secs "${PROGRESS_SECS:-30}" \
            --device cuda --noise 1 \
            && break
        rc=$?
        [[ "$rc" -eq 130 ]] && { echo "[daemon gpu=$gpu] interrupted; exiting."; break; }
        echo "[daemon gpu=$gpu] exited rc=$rc; restarting in 5s"; sleep 5
    done
}

IFS=',' read -ra DG <<< "$SELFPLAY_DAEMON_GPUS"
for gpu in "${DG[@]}"; do
    [[ -z "$gpu" ]] && continue
    run_one "$gpu" &
done
wait
