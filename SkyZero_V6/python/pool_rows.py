"""Total selfplay-pool row count (main + daemon) — single source of truth for cum_rows.

Two layered notions:
  - `current_pool_rows`  = rows physically present on disk under data_dir/selfplay/*.npz
  - `cumulative_produced` = rows ever produced (monotonic) = current_pool_rows + pruned
    Used by warmup and replay-ratio staging so window-based deletion never causes
    a warmup regression.

`pool_rows.tsv` snapshots the cumulative quantity (not the on-disk quantity), so the
delta between consecutive rows is the actual produced-rows-this-iter, immune to any
pruning that happened in between. predict_daemon_rows therefore Just Works.

Reads shuffle.py's summary cache (`data/shuffled/summary.json`) read-only; for
NPZs not yet in the cache (e.g. daemon files written since last shuffle), falls
back to `count_rows()` (NPZ header read, fast). Never writes the cache — that's
shuffle.py's job, and writing here would race with OVERLAP_SHUFFLE=1's bg shuffle.

CLI:
    python pool_rows.py current  --data-dir DIR              # stdout: current on-disk total
    python pool_rows.py produced --data-dir DIR              # stdout: cumulative produced
    python pool_rows.py snapshot --data-dir DIR --iter K     # append cumulative row to logs/pool_rows.tsv
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import shuffle
from data_processing import count_rows


POOL_TSV_NAME = "pool_rows.tsv"
POOL_TSV_HEADER = "iter\ttotal_pool_rows"

PRUNED_TSV_NAME = "pruned_rows.tsv"
PRUNED_TSV_HEADER = "timestamp\titer\tpruned_files\tpruned_rows"


def current_pool_rows(data_dir: pathlib.Path) -> int:
    """Sum rows across data_dir/selfplay/*.npz. Read-only on shuffle's cache."""
    selfplay_dir = data_dir / "selfplay"
    summary_path = data_dir / "shuffled" / "summary.json"
    if not selfplay_dir.exists():
        return 0
    summary = shuffle._load_summary(summary_path)
    cached = summary.get(str(selfplay_dir.resolve()), {}).get("files", {})
    total = 0
    for p in selfplay_dir.glob("*.npz"):
        try:
            mt = p.stat().st_mtime
        except FileNotFoundError:
            continue
        entry = cached.get(p.name)
        if entry and entry.get("mtime") == mt and "rows" in entry:
            total += int(entry["rows"])
        else:
            try:
                total += count_rows(p)
            except Exception:
                continue
    return total


def cumulative_pruned_rows(data_dir: pathlib.Path) -> int:
    """Sum of every pruned-rows event ever recorded. 0 when log absent (no pruning yet)."""
    path = data_dir / "logs" / PRUNED_TSV_NAME
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("timestamp"):
            continue
        parts = line.split("\t")
        # columns: timestamp, iter, pruned_files, pruned_rows
        if len(parts) < 4:
            continue
        try:
            total += int(parts[3])
        except ValueError:
            continue
    return total


def cumulative_produced(data_dir: pathlib.Path) -> int:
    """Total rows ever produced by selfplay (main + daemon).

    Monotonic: equals current on-disk pool + all rows ever pruned. Survives both
    PRUNE_OUTSIDE_WINDOW deletions and (defensively) manual `rm` of NPZs that
    were already accounted for in pruned_rows.tsv.
    """
    return current_pool_rows(data_dir) + cumulative_pruned_rows(data_dir)


def append_prune_event(
    data_dir: pathlib.Path,
    iter_no: int,
    pruned_files: int,
    pruned_rows: int,
) -> pathlib.Path:
    """Append one row to data_dir/logs/pruned_rows.tsv. Caller deletes the NPZs
    *before* calling — only successful deletes get recorded so cumulative_pruned
    stays bounded above by actual disk activity."""
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / PRUNED_TSV_NAME
    had_header = path.exists()
    with path.open("a") as f:
        if not had_header:
            f.write(PRUNED_TSV_HEADER + "\n")
        f.write(f"{int(time.time())}\t{iter_no}\t{pruned_files}\t{pruned_rows}\n")
    return path


def append_snapshot(data_dir: pathlib.Path, iter_no: int, total: int) -> pathlib.Path:
    """Append `iter\ttotal` to data_dir/logs/pool_rows.tsv (header on first write).

    `total` is expected to be the cumulative-produced quantity. Stored value is
    monotonic by construction (caller is the CLI `snapshot`, which always passes
    cumulative_produced)."""
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / POOL_TSV_NAME
    had_header = path.exists()
    with path.open("a") as f:
        if not had_header:
            f.write(POOL_TSV_HEADER + "\n")
        f.write(f"{iter_no}\t{total}\n")
    return path


def read_snapshots(data_dir: pathlib.Path) -> list[tuple[int, int]]:
    """Return [(iter, total_pool_rows), ...] sorted by iter ascending. Empty on missing/malformed."""
    path = data_dir / "logs" / POOL_TSV_NAME
    if not path.exists():
        return []
    out: list[tuple[int, int]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("iter"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    out.sort(key=lambda t: t[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    cur = sub.add_parser("current")
    cur.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    prod = sub.add_parser("produced")
    prod.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    snap = sub.add_parser("snapshot")
    snap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    snap.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if args.cmd == "current":
        print(current_pool_rows(data_dir))
        return 0
    if args.cmd == "produced":
        print(cumulative_produced(data_dir))
        return 0
    if args.cmd == "snapshot":
        total = cumulative_produced(data_dir)
        path = append_snapshot(data_dir, args.iter, total)
        print(f"[pool_rows] iter={args.iter} cum_produced={total} -> {path}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
