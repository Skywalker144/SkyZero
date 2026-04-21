"""Gate before training: ensure shuffled/current has at least MIN_SHUFFLED_ROWS rows.

Exit code 0 if ready, 2 otherwise (so run.sh can loop back to selfplay).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from data_processing import count_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--min-rows", type=int,
                        default=int(os.environ.get("MIN_SHUFFLED_ROWS", "10000")))
    args = parser.parse_args()

    shuffled_dir = pathlib.Path(args.data_dir) / "shuffled" / "current"
    if not shuffled_dir.exists():
        print(f"[wait_for_data] shuffled dir missing: {shuffled_dir}", file=sys.stderr)
        return 2

    total = 0
    for p in shuffled_dir.glob("*.npz"):
        total += count_rows(p)

    if total < args.min_rows:
        print(f"[wait_for_data] only {total} rows, need {args.min_rows}", file=sys.stderr)
        return 2

    print(f"[wait_for_data] OK: {total} rows available", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
