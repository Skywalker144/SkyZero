"""Scan data/selfplay/*.npz, compute power-law window, sample+shuffle, write shards.

Architecture mirrors KataGo's shuffle.py:

  1. Summary-JSON cache of per-file (mtime, row count) — rescanning an unchanged
     selfplay dir is nearly free.
  2. Parallel row-count for cache-miss files.
  3. Pass-1 (shardify): group input files into ~worker_group_size-row chunks, run
     each chunk in a worker-pool task that loads + concats + random-assigns rows
     to K buckets, writing **one file per bucket** per task. Intermediate file
     count is num_groups * K (not num_inputs * K).
  4. Pass-2 (merge): one worker-pool task per bucket; loads all tmp chunks,
     joint-shuffles, writes the final shard.

Deviation from KataGo: output uses np.savez (uncompressed) because V4 rows are
small (~2.7 KB/row) and deflate is a net loss — mirrors cpp/npz_writer.h's
ZIP_CM_STORE choice.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import shutil
import sys
import time

import numpy as np

from data_processing import (
    NpzBatch,
    concat_batches,
    count_rows,
    joint_shuffle_take_first_n,
    load_npz,
    save_npz,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def list_selfplay_npz_newest_first(selfplay_dir: pathlib.Path) -> list[pathlib.Path]:
    # mtime descending. Daemon-produced files (`daemon_v*_*.npz`) and main-loop
    # files (`iter_*_*.npz`) interleave by wall-clock production order; lex sort
    # by name would group all daemon files after iter files regardless of when
    # each was written.
    files = [p for p in selfplay_dir.glob("*.npz") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def compute_window_size(total: int, linear_threshold: int, alpha: float) -> int:
    if total <= linear_threshold:
        return total
    return int(linear_threshold * ((total / linear_threshold) ** alpha))


# ---------------------------------------------------------------------------
# Summary cache
# ---------------------------------------------------------------------------

def _load_summary(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_summary(path: pathlib.Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(summary, f)
    tmp.replace(path)


def _count_rows_worker(path_str: str) -> tuple[str, int]:
    return path_str, count_rows(path_str)


def discover_and_count(
    selfplay_dir: pathlib.Path,
    summary_path: pathlib.Path | None,
    pool: mp.pool.Pool | None,
) -> list[tuple[pathlib.Path, int]]:
    """Return [(path, rows), ...] newest-first. Uses summary cache when valid."""
    files = list_selfplay_npz_newest_first(selfplay_dir)
    if not files:
        return []

    key = str(selfplay_dir.resolve())
    summary = _load_summary(summary_path) if summary_path else {}
    cached_files = summary.get(key, {}).get("files", {}) if summary else {}

    known: dict[str, int] = {}
    unknown: list[pathlib.Path] = []
    for p in files:
        name = p.name
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        entry = cached_files.get(name)
        if entry and entry.get("mtime") == mtime and "rows" in entry:
            known[name] = entry["rows"]
        else:
            unknown.append(p)

    if unknown:
        if pool is not None and len(unknown) > 1:
            for path_str, rows in pool.imap_unordered(
                _count_rows_worker, [str(p) for p in unknown]
            ):
                known[pathlib.Path(path_str).name] = rows
        else:
            for p in unknown:
                known[p.name] = count_rows(p)

    # Rebuild cache for this dir with fresh mtimes + counts.
    new_files_entry: dict[str, dict] = {}
    result: list[tuple[pathlib.Path, int]] = []
    for p in files:
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        rows = known.get(p.name)
        if rows is None:
            continue
        new_files_entry[p.name] = {"mtime": mtime, "rows": rows}
        result.append((p, rows))

    if summary_path is not None:
        summary[key] = {
            "dir_mtime": selfplay_dir.stat().st_mtime,
            "files": new_files_entry,
        }
        _save_summary(summary_path, summary)

    return result


# ---------------------------------------------------------------------------
# Pass-1: shardify
# ---------------------------------------------------------------------------

def _slice_batch(b: NpzBatch, s: int, e: int) -> NpzBatch:
    return NpzBatch(
        state=b.state[s:e],
        global_features=b.global_features[s:e],
        policy_target=b.policy_target[s:e],
        opponent_policy_target=b.opponent_policy_target[s:e],
        opponent_policy_mask=b.opponent_policy_mask[s:e],
        value_target=b.value_target[s:e],
        td_value_target=b.td_value_target[s:e],
        futurepos_target=b.futurepos_target[s:e],
        sample_weight=b.sample_weight[s:e],
    )


def _shardify_job(args):
    task_idx, files_with_takes, num_shards, tmp_root, task_seed = args
    rng = np.random.default_rng(None if task_seed is None or task_seed < 0 else task_seed)
    parts: list[NpzBatch] = []
    for path_str, take in files_with_takes:
        b = load_npz(path_str)
        n = len(b)
        if take < n:
            sel = rng.choice(n, size=take, replace=False)
            sel.sort()
            b = b.select(sel)
        parts.append(b)
    batch = parts[0] if len(parts) == 1 else concat_batches(parts)
    del parts
    total = len(batch)
    if total == 0:
        return 0

    assign = rng.integers(0, num_shards, size=total)
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=num_shards)
    ends = np.cumsum(counts)
    batch = batch.select(order)

    tmp_root_p = pathlib.Path(tmp_root)
    start = 0
    for k in range(num_shards):
        end = int(ends[k])
        if end > start:
            sub = _slice_batch(batch, start, end)
            out = tmp_root_p / f"bucket_{k:04d}" / f"task_{task_idx:06d}.npz"
            save_npz(out, sub)
        start = end
    return total


def group_files_by_rows(
    files_with_takes: list[tuple[pathlib.Path, int]],
    worker_group_size: int,
) -> list[list[tuple[str, int]]]:
    """Pack (path, take) pairs into groups whose summed takes reach worker_group_size."""
    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_rows = 0
    for path, take in files_with_takes:
        if take <= 0:
            continue
        current.append((str(path), take))
        current_rows += take
        if current_rows >= worker_group_size:
            groups.append(current)
            current = []
            current_rows = 0
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Pass-2: merge shard
# ---------------------------------------------------------------------------

def _merge_shard_job(args):
    bucket_dir_str, out_str, seed = args
    bucket_dir = pathlib.Path(bucket_dir_str)
    out = pathlib.Path(out_str)
    chunk_paths = sorted(bucket_dir.glob("*.npz"))
    if not chunk_paths:
        return 0
    parts = [load_npz(cp) for cp in chunk_paths]
    batch = parts[0] if len(parts) == 1 else concat_batches(parts)
    del parts
    rng = np.random.default_rng(None if seed < 0 else seed)
    batch = joint_shuffle_take_first_n(len(batch), batch, rng)
    save_npz(out, batch)
    return len(batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--shard-rows", type=int, default=200_000,
                        help="Max rows per output shard.")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num-workers", type=int,
                        default=_env_int("SHUFFLE_WORKERS", 0),
                        help="Pool size for parallel shardify+merge (0/1 = serial).")
    parser.add_argument("--worker-group-size", type=int,
                        default=_env_int("SHUFFLE_WORKER_GROUP_SIZE", 200_000),
                        help="Target rows per shardify task.")
    parser.add_argument("--no-summary-cache", action="store_true",
                        help="Disable per-file row-count cache.")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    selfplay_dir = data_dir / "selfplay"
    shuffled_dir = data_dir / "shuffled" / "current"
    scatter_dir = data_dir / "shuffled" / "_scatter"
    summary_path = None if args.no_summary_cache else data_dir / "shuffled" / "summary.json"

    linear_threshold = _env_int("LINEAR_THRESHOLD", 2_000_000)
    alpha = _env_float("REPLAY_ALPHA", 0.8)

    if not selfplay_dir.exists():
        print(f"[shuffle] selfplay dir missing: {selfplay_dir}", file=sys.stderr)
        return 1

    num_workers = max(1, args.num_workers or 1)
    ctx = mp.get_context("spawn")

    t0 = time.perf_counter()

    if num_workers > 1:
        pool = ctx.Pool(processes=num_workers)
    else:
        pool = None

    try:
        files_rows = discover_and_count(selfplay_dir, summary_path, pool)
    finally:
        # Keep the pool alive across phases to avoid spawn overhead.
        pass

    if not files_rows:
        print("[shuffle] no selfplay files found", file=sys.stderr)
        if pool is not None:
            pool.close()
            pool.join()
        return 1

    t_discover = time.perf_counter() - t0

    total = sum(r for _, r in files_rows)
    window = compute_window_size(total, linear_threshold, alpha)

    print(f"[shuffle] total_rows={total} window={window} "
          f"(linear_threshold={linear_threshold}, alpha={alpha}) "
          f"discover={t_discover:.2f}s")

    rng = np.random.default_rng(None if args.seed < 0 else args.seed)

    # Take newest files until we cover `window` rows. Partial-take the oldest.
    chosen: list[tuple[pathlib.Path, int]] = []
    covered = 0
    for p, r in files_rows:
        if covered >= window:
            break
        take = min(r, window - covered)
        chosen.append((p, take))
        covered += take

    shard_rows = args.shard_rows
    K = max(1, math.ceil(covered / shard_rows))

    groups = group_files_by_rows(chosen, args.worker_group_size)
    num_groups = len(groups)
    print(f"[shuffle] pass1: {covered} rows from {len(chosen)} files -> "
          f"{num_groups} groups x {K} buckets (workers={num_workers})")

    # Fresh scatter dir.
    if scatter_dir.exists():
        shutil.rmtree(scatter_dir)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for k in range(K):
        (scatter_dir / f"bucket_{k:04d}").mkdir()

    # Per-task seeds so parallel runs stay deterministic.
    if args.seed >= 0:
        task_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_groups)]
    else:
        task_seeds = [-1] * num_groups

    shardify_args = [
        (i, groups[i], K, str(scatter_dir), task_seeds[i])
        for i in range(num_groups)
    ]

    t1 = time.perf_counter()
    if pool is not None:
        pass1_total = sum(pool.imap_unordered(_shardify_job, shardify_args))
    else:
        pass1_total = sum(_shardify_job(a) for a in shardify_args)
    t_pass1 = time.perf_counter() - t1
    print(f"[shuffle] pass1 done: {pass1_total} rows scattered in {t_pass1:.2f}s")

    # Pass 2: per-bucket merge.
    if shuffled_dir.exists():
        shutil.rmtree(shuffled_dir)
    shuffled_dir.mkdir(parents=True, exist_ok=True)

    merge_args = []
    for k in range(K):
        bucket_dir = scatter_dir / f"bucket_{k:04d}"
        out = shuffled_dir / f"shard_{k:04d}.npz"
        bucket_seed = int(rng.integers(0, 2**31 - 1)) if args.seed >= 0 else -1
        merge_args.append((str(bucket_dir), str(out), bucket_seed))

    t2 = time.perf_counter()
    total_written = 0
    num_shards_written = 0
    if pool is not None:
        for written in pool.imap_unordered(_merge_shard_job, merge_args):
            if written > 0:
                total_written += written
                num_shards_written += 1
    else:
        for a in merge_args:
            written = _merge_shard_job(a)
            if written > 0:
                total_written += written
                num_shards_written += 1
    t_pass2 = time.perf_counter() - t2

    if pool is not None:
        pool.close()
        pool.join()

    shutil.rmtree(scatter_dir, ignore_errors=True)

    total_t = time.perf_counter() - t0
    print(f"[shuffle] pass2 done: {num_shards_written} shards, "
          f"{total_written} rows in {t_pass2:.2f}s")
    print(f"[shuffle] wrote {num_shards_written} shards, "
          f"{total_written} rows total to {shuffled_dir} "
          f"(total={total_t:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
