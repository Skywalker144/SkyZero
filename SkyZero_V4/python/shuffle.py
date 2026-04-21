"""Scan data/selfplay/*.npz, compute power-law window, sample+shuffle, write shards.

The shuffle stage is stateless: each iteration rescans disk. Data augmentation
is *not* applied here — train.py does it on-the-fly.

Window formula (same as CSkyZero_V3 replaybuffer.h):
    total = sum of rows across *.npz on disk
    if total <= linear_threshold: window = total
    else: window = linear_threshold * (total / linear_threshold) ** alpha
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import shutil
import sys

import numpy as np

from data_processing import NpzBatch, concat_batches, count_rows, load_npz, save_npz


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def list_selfplay_npz_newest_first(selfplay_dir: pathlib.Path) -> list[pathlib.Path]:
    files = [p for p in selfplay_dir.glob("*.npz") if p.is_file()]
    files.sort(key=lambda p: p.name, reverse=True)  # names encode iter + part
    return files


def compute_window_size(total: int, linear_threshold: int, alpha: float) -> int:
    if total <= linear_threshold:
        return total
    return int(linear_threshold * ((total / linear_threshold) ** alpha))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--shard-rows", type=int, default=200_000,
                        help="Max rows per output shard.")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    selfplay_dir = data_dir / "selfplay"
    shuffled_dir = data_dir / "shuffled" / "current"

    linear_threshold = _env_int("LINEAR_THRESHOLD", 2_000_000)
    alpha = _env_float("REPLAY_ALPHA", 0.8)

    if not selfplay_dir.exists():
        print(f"[shuffle] selfplay dir missing: {selfplay_dir}", file=sys.stderr)
        return 1

    files = list_selfplay_npz_newest_first(selfplay_dir)
    if not files:
        print("[shuffle] no selfplay files found", file=sys.stderr)
        return 1

    rows_per_file = [count_rows(p) for p in files]
    total = sum(rows_per_file)
    window = compute_window_size(total, linear_threshold, alpha)

    print(f"[shuffle] total_rows={total} window={window} "
          f"(linear_threshold={linear_threshold}, alpha={alpha})")

    rng = np.random.default_rng(None if args.seed < 0 else args.seed)

    # Take newest files until we cover `window` rows. Partial-sample the last.
    chosen_files: list[pathlib.Path] = []
    chosen_rows: list[int] = []
    covered = 0
    partial_n: int | None = None
    for p, r in zip(files, rows_per_file):
        if covered >= window:
            break
        take = min(r, window - covered)
        chosen_files.append(p)
        chosen_rows.append(take)
        if take < r:
            partial_n = take
        covered += take

    batches: list[NpzBatch] = []
    for p, take in zip(chosen_files, chosen_rows):
        b = load_npz(p)
        if take < len(b):
            idx = rng.choice(len(b), size=take, replace=False)
            idx.sort()
            b = b.select(idx)
        batches.append(b)

    full = concat_batches(batches)
    n = len(full)
    perm = rng.permutation(n)
    full = full.select(perm)

    # Clean output dir
    if shuffled_dir.exists():
        shutil.rmtree(shuffled_dir)
    shuffled_dir.mkdir(parents=True, exist_ok=True)

    shard_rows = args.shard_rows
    num_shards = max(1, math.ceil(n / shard_rows))
    for k in range(num_shards):
        lo = k * shard_rows
        hi = min(n, lo + shard_rows)
        idx = np.arange(lo, hi)
        shard = full.select(idx)
        path = shuffled_dir / f"shard_{k:04d}.npz"
        save_npz(path, shard)

    print(f"[shuffle] wrote {num_shards} shards, {n} rows total to {shuffled_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
