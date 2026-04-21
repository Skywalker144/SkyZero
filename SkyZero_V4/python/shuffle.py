"""Scan data/selfplay/*.npz, compute power-law window, sample+shuffle, write shards.

The shuffle stage is stateless: each iteration rescans disk. Data augmentation
is *not* applied here — train.py does it on-the-fly.

Window formula (same as CSkyZero_V3 replaybuffer.h):
    total = sum of rows across *.npz on disk
    if total <= linear_threshold: window = total
    else: window = linear_threshold * (total / linear_threshold) ** alpha

Memory strategy: two-pass external shuffle (KataGo-style).
    Pass 1 (scatter): stream each input file, randomly assign every row to one
        of K buckets, append per-bucket chunks to disk. Only one input file is
        resident at a time.
    Pass 2 (per-bucket shuffle): for each bucket, load its chunks, permute,
        write the final shard. Only one bucket (~shard_rows) is resident.
    Peak RAM = O(max(input_file_rows, shard_rows)), independent of window.
"""
from __future__ import annotations

import argparse
import gc
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
    scatter_dir = data_dir / "shuffled" / "_scatter"

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

    # Take newest files until we cover `window` rows. Partial-sample the oldest.
    chosen_files: list[pathlib.Path] = []
    chosen_takes: list[int] = []
    covered = 0
    for p, r in zip(files, rows_per_file):
        if covered >= window:
            break
        take = min(r, window - covered)
        chosen_files.append(p)
        chosen_takes.append(take)
        covered += take

    shard_rows = args.shard_rows
    K = max(1, math.ceil(covered / shard_rows))
    print(f"[shuffle] pass1: scattering {covered} rows from {len(chosen_files)} "
          f"files into {K} buckets")

    # Fresh scatter dir.
    if scatter_dir.exists():
        shutil.rmtree(scatter_dir)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for k in range(K):
        (scatter_dir / f"bucket_{k:04d}").mkdir()

    # Pass 1: scatter.
    for file_idx, (p, take) in enumerate(zip(chosen_files, chosen_takes)):
        b = load_npz(p)
        n_rows = len(b)
        if take < n_rows:
            sel = rng.choice(n_rows, size=take, replace=False)
            sel.sort()
            b = b.select(sel)
        assign = rng.integers(0, K, size=len(b))
        for k in np.unique(assign):
            mask = assign == k
            sub = b.select(mask)
            out = scatter_dir / f"bucket_{int(k):04d}" / f"file_{file_idx:06d}.npz"
            save_npz(out, sub)
            del sub
        del b, assign
        gc.collect()

    # Pass 2: per-bucket shuffle.
    if shuffled_dir.exists():
        shutil.rmtree(shuffled_dir)
    shuffled_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    num_shards_written = 0
    for k in range(K):
        bucket_dir = scatter_dir / f"bucket_{k:04d}"
        chunk_paths = sorted(bucket_dir.glob("*.npz"))
        if not chunk_paths:
            continue
        parts = [load_npz(cp) for cp in chunk_paths]
        b = concat_batches(parts)
        del parts
        perm = rng.permutation(len(b))
        b = b.select(perm)
        out = shuffled_dir / f"shard_{k:04d}.npz"
        save_npz(out, b)
        total_written += len(b)
        num_shards_written += 1
        del b, perm
        gc.collect()

    shutil.rmtree(scatter_dir, ignore_errors=True)

    print(f"[shuffle] wrote {num_shards_written} shards, "
          f"{total_written} rows total to {shuffled_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
