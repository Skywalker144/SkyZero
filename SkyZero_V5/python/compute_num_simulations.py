"""Compute this iteration's NUM_SIMULATIONS from the warmup schedule.

Mirrors compute_games.py: keys off cumulative selfplay rows in last_run.tsv
and applies a per-stage list configured via NUM_SIMULATIONS_STAGES +
NUM_SIM_WARMUP_SAMPLES. Falls back to the NUM_SIMULATIONS env var when the
list has < 2 entries (warmup disabled).

Prints the value to stdout (one integer) so selfplay.sh can pass it to
selfplay_main via --num-simulations.

Note: the persistent selfplay daemon (run_selfplay_daemon.sh) does not
consume this — it always uses cfg's NUM_SIMULATIONS by design.
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

    fallback = int(os.environ.get("NUM_SIMULATIONS", "400"))
    stages = parse_stage_list(os.environ.get("NUM_SIMULATIONS_STAGES"), cast=int)
    warmup_samples = int(float(os.environ.get("NUM_SIM_WARMUP_SAMPLES", "0")))

    _, cum_rows = read_history(last_run_tsv)
    val = staged_value(cum_rows, warmup_samples, stages)
    nsim = val if val is not None else fallback

    tag = "warmup" if val is not None else "steady-cfg"
    print(
        f"[compute_num_simulations] cum_rows={cum_rows} "
        f"stages={stages} warmup_samples={warmup_samples} "
        f"-> NUM_SIMULATIONS={nsim}({tag})",
        file=sys.stderr,
    )
    print(nsim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
