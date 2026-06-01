"""Per-iter NUM_SIMULATIONS resolver (mirrors SkyZero_V7.1/python/warmup.py).

Optional staged warmup: NUM_SIMULATIONS_STAGES (values) positionally matched
with NUM_SIMULATIONS_SCHEDULE (cumulative-selfplay-row thresholds). At any
cum_rows the active stage is the one whose threshold is the largest crossed.

Unlike V7.1 (which requires the stages), 2048 commonly runs a FIXED sim count,
so if the stage env vars are unset/empty/length-mismatched this falls back to
the scalar SIMS env (default 64). This keeps internal/selfplay*.sh structurally
identical to V7.1 while supporting the simple single-value case.

CLI (used by internal/selfplay.sh and as the daemon's --sims-warmup-cmd):
    python warmup.py num-simulations --data-dir DATA_DIR
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from bucket import read_cum_rows


def parse_stage_list(s, cast):
    if not s:
        return []
    return [cast(p.strip()) for p in s.replace(",", " ").split() if p.strip()]


def staged_value(samples_seen: int, thresholds: list, stages: list):
    pairs = sorted(zip(thresholds, stages))
    chosen = pairs[0][1]
    for thr, val in pairs:
        if samples_seen >= thr:
            chosen = val
        else:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="param", required=True)
    p = sub.add_parser("num-simulations")
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()

    fallback = int(float(os.environ.get("SIMS", "64")))
    stages = parse_stage_list(os.environ.get("NUM_SIMULATIONS_STAGES"), int)
    thresholds = parse_stage_list(os.environ.get("NUM_SIMULATIONS_SCHEDULE"), float)

    if not stages or not thresholds or len(stages) != len(thresholds):
        # Single fixed sim count (the common 2048 case).
        print(f"[warmup] no/!= staged sims -> SIMS={fallback}", file=sys.stderr)
        print(fallback)
        return 0

    cum_rows = read_cum_rows(pathlib.Path(args.data_dir) / "logs" / "selfplay.tsv")
    out = staged_value(cum_rows, thresholds, stages)
    print(f"[warmup] cum_rows={cum_rows} stages={stages} -> NUM_SIMULATIONS={out}",
          file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
