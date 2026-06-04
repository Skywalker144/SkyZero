#!/usr/bin/env bash
# One-shot retrain for Channel-Dodge after a gameplay change. V7.1-style layout:
# each run gets its own config (configs/<name>/run.cfg) and its own isolated
# output dir (runs/<name>/), so runs never contaminate each other.
#
# Workflow:
#   1. You edit ../SkyZeroWeb/channel-dodge.html (bullet speed, damage, spawns, ...)
#   2. Sync the changed gameplay constants into env_dodge.py (ask Claude, or by hand)
#   3. ./retrain.sh <name>            # from scratch — a config change invalidates old policies
#
# It uses configs/<name>/run.cfg (created from configs/dodge.cfg on first run, so
# you can then tweak that file per-experiment), trains the validated recipe with
# the multi-process vector env, then runs a robust 50-episode eval of best.pt.
#
# Usage:
#   ./retrain.sh                      # name defaults to "dodge"
#   ./retrain.sh fastbullet           # custom run name -> configs/fastbullet/, runs/fastbullet/
#   NUM_WORKERS=12 ./retrain.sh foo   # override cores
#   EXTRA="--set total_steps=15000000" ./retrain.sh foo
set -euo pipefail

cd "$(dirname "$0")"
PY=/home/sky/miniconda3/envs/pytorch/bin/python
NAME="${1:-dodge}"
WORKERS="${NUM_WORKERS:-8}"

CFG_DIR="configs/$NAME"
CFG="$CFG_DIR/run.cfg"
OUT="runs/$NAME"

# First run for this name: seed its config from the template and stamp RUN_NAME.
if [[ ! -f "$CFG" ]]; then
    mkdir -p "$CFG_DIR"
    sed "s/^RUN_NAME=.*/RUN_NAME=$NAME/" configs/dodge.cfg > "$CFG"
    echo "[retrain] created $CFG from template (edit it to tweak this experiment)"
fi
mkdir -p "$OUT"

echo "[retrain] run=$NAME  config=$CFG  out=$OUT  workers=$WORKERS  $(date)"
echo "[retrain] reminder: synced channel-dodge.html constants into env_dodge.py?"

# Train. stdout/stderr -> runs/<name>/train.log (also echoed to console via tee).
$PY -u train_ppo.py "$CFG" --set num_workers="$WORKERS" ${EXTRA:-} 2>&1 | tee "$OUT/train.log"

echo
echo "[retrain] done -> robust 50-episode eval of $OUT/best.pt"
$PY evaluate.py "$OUT/best.pt" --episodes 50 --device cpu | tee -a "$OUT/train.log"
