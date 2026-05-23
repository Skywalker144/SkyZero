"""Staged warmup for NUM_SIMULATIONS (sole remaining consumer).

NUM_SIMULATIONS_STAGES (values) is positionally matched with
NUM_SIMULATIONS_SCHEDULE (cumulative-selfplay-row thresholds). At any
cum_rows, the active stage is the one whose threshold is the largest
that has been crossed. Mirrors python/schedule.py's network scheduler.

  NUM_SIMULATIONS_STAGES="32, 64, 128, 400, 1000"
  NUM_SIMULATIONS_SCHEDULE="0, 5e6, 1.5e7, 3e7, 5e7"

Both env vars are required and must be the same length; otherwise this
script exits 1 (no fallback).

CLI for shell scripts (internal/selfplay.sh main loop; internal/selfplay_daemon.sh
as --sims-warmup-cmd for daemon's reload-time re-poll):

    python warmup.py num-simulations --data-dir DATA_DIR

Reads cum_rows from DATA_DIR/logs/selfplay.tsv (sums all producers).
Train rate is no longer staged — see python/bucket.py.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Callable


def parse_stage_list(s: str | None, cast: Callable = float) -> list:
    """Parse "v1, v2, v3" -> [cast(v1), cast(v2), cast(v3)]."""
    if not s:
        return []
    return [cast(p.strip()) for p in s.split(",") if p.strip()]


def staged_value(samples_seen: int, thresholds: list, stages: list):
    """Active stage value. Caller must validate stages/thresholds beforehand."""
    pairs = sorted(zip(thresholds, stages))
    chosen = pairs[0][1]
    for thr, val in pairs:
        if samples_seen >= thr:
            chosen = val
        else:
            break
    return chosen


def _read_cum_rows(selfplay_tsv: pathlib.Path) -> int:
    """Sum the rows column (col 3) across all producers in selfplay.tsv."""
    if not selfplay_tsv.exists():
        return 0
    total = 0
    try:
        for ln in selfplay_tsv.read_text().splitlines():
            if not ln.strip() or ln.startswith("producer"):
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                continue
            try:
                total += int(float(parts[3]))
            except ValueError:
                continue
    except OSError:
        pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="param", required=True)
    p = sub.add_parser("num-simulations")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    selfplay_tsv = data_dir / "logs" / "selfplay.tsv"

    stages = parse_stage_list(os.environ.get("NUM_SIMULATIONS_STAGES"), cast=int)
    thresholds = parse_stage_list(os.environ.get("NUM_SIMULATIONS_SCHEDULE"), cast=float)

    if not stages or not thresholds:
        print(
            "[compute_num_simulations] ERROR: NUM_SIMULATIONS_STAGES and "
            "NUM_SIMULATIONS_SCHEDULE must both be set "
            f"(stages={stages}, schedule={thresholds})",
            file=sys.stderr,
        )
        return 1
    if len(stages) != len(thresholds):
        print(
            "[compute_num_simulations] ERROR: NUM_SIMULATIONS_STAGES and "
            "NUM_SIMULATIONS_SCHEDULE length mismatch "
            f"({len(stages)} vs {len(thresholds)})",
            file=sys.stderr,
        )
        return 1

    cum_rows = _read_cum_rows(selfplay_tsv)
    out = staged_value(cum_rows, thresholds, stages)

    print(
        f"[compute_num_simulations] cum_rows={cum_rows} "
        f"stages={stages} schedule={thresholds} "
        f"-> NUM_SIMULATIONS={out}",
        file=sys.stderr,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
