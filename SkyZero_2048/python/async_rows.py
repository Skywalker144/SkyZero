"""Count self-play rows newer than a timestamp — the cadence signal for the
async consumer (scripts/faster_run.sh).

In the async architecture the self-play daemon(s) run continuously and only
append a logs/selfplay.tsv row at model-reload boundaries, so bucket.py's
tsv-based row accounting can't tell the consumer "how much fresh data exists
yet" (cold-start deadlock: no reload until we train, no train until we see
rows). Instead the consumer polls the disk directly: this script sums the rows
of every selfplay/*.npz whose st_mtime is strictly greater than --since.

Only freshly-written shards are read, so the cost scales with new data, not the
whole window. Mid-write / truncated / 0-byte npz raise inside count_rows and are
skipped (counted as 0), matching shuffle.py's robustness — they get re-counted
on the next poll once fully written.

Prints a single integer (total new rows) to stdout.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from data_processing import count_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data2048"))
    p.add_argument("--since", type=float, default=0.0,
                   help="only count files with st_mtime > since (unix seconds, sub-second ok)")
    args = p.parse_args()

    selfplay = pathlib.Path(args.data_dir) / "selfplay"
    total = 0
    if selfplay.exists():
        for f in selfplay.glob("*.npz"):
            try:
                if f.stat().st_mtime <= args.since:
                    continue
            except OSError:
                continue
            try:
                total += count_rows(f)
            except Exception:  # noqa: BLE001 - mid-write/corrupt -> retried next poll
                continue
    print(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
