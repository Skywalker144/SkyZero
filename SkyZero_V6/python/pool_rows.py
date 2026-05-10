"""Total selfplay-pool row count (main + daemon) — single source of truth for cum_rows.

Replaces the legacy `sum(last_run.tsv[*].rows)` lookups, which only accounted for
main-GPU per-iter selfplay and missed daemon-produced NPZs entirely. All
downstream consumers (warmup.py / compute_games.py / slots.py) now route here.

Reads shuffle.py's summary cache (`data/shuffled/summary.json`) read-only; for
NPZs not yet in the cache (e.g. daemon files written since last shuffle), falls
back to `count_rows()` (NPZ header read, fast). Never writes the cache — that's
shuffle.py's job, and writing here would race with OVERLAP_SHUFFLE=1's bg shuffle.

CLI:
    python pool_rows.py current  --data-dir DIR              # stdout: int total
    python pool_rows.py snapshot --data-dir DIR --iter K     # append row to logs/pool_rows.tsv
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import shuffle
from data_processing import count_rows


POOL_TSV_NAME = "pool_rows.tsv"
POOL_TSV_HEADER = "iter\ttotal_pool_rows"


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


def append_snapshot(data_dir: pathlib.Path, iter_no: int, total: int) -> pathlib.Path:
    """Append `iter\ttotal` to data_dir/logs/pool_rows.tsv (header on first write)."""
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
    snap = sub.add_parser("snapshot")
    snap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    snap.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if args.cmd == "current":
        print(current_pool_rows(data_dir))
        return 0
    if args.cmd == "snapshot":
        total = current_pool_rows(data_dir)
        path = append_snapshot(data_dir, args.iter, total)
        print(f"[pool_rows] iter={args.iter} total={total} -> {path}", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
