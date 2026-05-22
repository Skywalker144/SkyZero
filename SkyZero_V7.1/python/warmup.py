"""Staged warmup for NUM_SIMULATIONS (sole remaining consumer).

A stage list of N values is evenly spaced across NUM_SIM_WARMUP_SAMPLES of
cumulative selfplay rows: the i-th value applies for progress in
[i/N, (i+1)/N), the last value sticks once progress >= 1.0.

Disabled when len(stages) < 2 or warmup_samples <= 0 — falls back to the
NUM_SIMULATIONS env var.

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


def staged_value(samples_seen: int, warmup_samples: int, stages: list):
    """Returns the stage value, or None if warmup disabled."""
    n = len(stages)
    if n < 2 or warmup_samples <= 0:
        return None
    progress = samples_seen / warmup_samples
    if progress >= 1.0:
        return stages[-1]
    if progress < 0.0:
        return stages[0]
    idx = min(n - 1, int(progress * n))
    return stages[idx]


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

    fallback = int(float(os.environ.get("NUM_SIMULATIONS", "400")))
    stages = parse_stage_list(os.environ.get("NUM_SIMULATIONS_STAGES"), cast=int)
    warmup_samples = int(float(os.environ.get("NUM_SIM_WARMUP_SAMPLES", "0")))

    cum_rows = _read_cum_rows(selfplay_tsv)
    val = staged_value(cum_rows, warmup_samples, stages)
    out = val if val is not None else fallback

    tag = "warmup" if val is not None else "steady-cfg"
    print(
        f"[compute_num_simulations] cum_rows={cum_rows} "
        f"stages={stages} warmup_samples={warmup_samples} "
        f"-> NUM_SIMULATIONS={out}({tag})",
        file=sys.stderr,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
