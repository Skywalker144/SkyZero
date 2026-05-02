"""Compute this iteration's TRAIN_STEPS_PER_EPOCH from the warmup schedule.

Mirrors compute_num_simulations.py: keys off cumulative selfplay rows in
last_run.tsv and applies a per-stage list configured via TRAIN_STEPS_STAGES +
TRAIN_STEPS_WARMUP_SAMPLES. Falls back to the TRAIN_STEPS_PER_EPOCH env var
when the list has < 2 entries (warmup disabled).

Prints the value to stdout (one integer) so run.sh can export it before
calling compute_games.py and train.sh — both consume TRAIN_STEPS_PER_EPOCH
and must agree within an iter (compute_games uses it to size needed_samples;
train.py uses it as the per-epoch step count).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from compute_games import read_history
from warmup import parse_stage_list, staged_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    last_run_tsv = data_dir / "logs" / "last_run.tsv"

    fallback = int(float(os.environ.get("TRAIN_STEPS_PER_EPOCH", "15000")))
    stages = parse_stage_list(os.environ.get("TRAIN_STEPS_STAGES"), cast=int)
    warmup_samples = int(float(os.environ.get("TRAIN_STEPS_WARMUP_SAMPLES", "0")))

    _, cum_rows = read_history(last_run_tsv)
    val = staged_value(cum_rows, warmup_samples, stages)
    steps = val if val is not None else fallback

    tag = "warmup" if val is not None else "steady-cfg"
    print(
        f"[compute_train_steps] cum_rows={cum_rows} "
        f"stages={stages} warmup_samples={warmup_samples} "
        f"-> TRAIN_STEPS_PER_EPOCH={steps}({tag})",
        file=sys.stderr,
    )
    print(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
