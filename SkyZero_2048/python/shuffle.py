"""Power-law-window shuffle for 2048 self-play data — a faithful port of
SkyZero_V7.1/python/shuffle.py (KataGo's 3-parameter window), adapted to the
2048 npz schema (see data_processing.py).

Scan <DATA_DIR>/selfplay/*.npz newest-first, compute the power-law window over
cumulative rows, sub-sample at the keep ratio, shardify+merge into
<DATA_DIR>/shuffled/current/shard_*.npz, and delete out-of-window files.

Run as:  python shuffle.py --data-dir DIR
Exit 2 = total rows < MIN_ROWS (caller skips training this iter).
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
    NpzBatch, concat_batches, count_rows, joint_shuffle_take_first_n, load_npz, save_npz,
)


def _env_int(name: str, default: int) -> int:
    return int(float(os.environ.get(name, str(default))))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def list_selfplay_npz_newest_first(selfplay_dir: pathlib.Path) -> list[pathlib.Path]:
    files = [p for p in selfplay_dir.glob("*.npz") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def compute_window_size(total: int, min_rows: int, exponent: float, expand_per_row: float) -> int:
    """KataGo 3-param power law: W(N) = (N^E - M^E)/(E*M^(E-1))*IWPR + M."""
    if total <= min_rows:
        return total
    M, E, IWPR = float(min_rows), exponent, expand_per_row
    scaled = (total ** E - M ** E) / (E * M ** (E - 1))
    return int(scaled * IWPR + M)


# ---- summary cache (per-file row counts keyed by mtime) ----
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


def discover_and_count(selfplay_dir, summary_path, pool):
    files = list_selfplay_npz_newest_first(selfplay_dir)
    if not files:
        return []
    key = str(selfplay_dir.resolve())
    summary = _load_summary(summary_path) if summary_path else {}
    cached = summary.get(key, {}).get("files", {}) if summary else {}
    known: dict[str, int] = {}
    unknown: list[pathlib.Path] = []
    for p in files:
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        e = cached.get(p.name)
        if e and e.get("mtime") == mtime and "rows" in e:
            known[p.name] = e["rows"]
        else:
            unknown.append(p)
    if unknown:
        if pool is not None and len(unknown) > 1:
            for path_str, rows in pool.imap_unordered(_count_rows_worker, [str(p) for p in unknown]):
                known[pathlib.Path(path_str).name] = rows
        else:
            for p in unknown:
                known[p.name] = count_rows(p)
    new_entry: dict[str, dict] = {}
    result: list[tuple[pathlib.Path, int]] = []
    for p in files:
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        rows = known.get(p.name)
        if rows is None:
            continue
        new_entry[p.name] = {"mtime": mtime, "rows": rows}
        result.append((p, rows))
    if summary_path is not None:
        summary[key] = {"dir_mtime": selfplay_dir.stat().st_mtime, "files": new_entry}
        _save_summary(summary_path, summary)
    return result


# ---- pass 1: shardify ----
def _slice_batch(b: NpzBatch, s: int, e: int) -> NpzBatch:
    return NpzBatch(b.state[s:e], b.policy_target[s:e], b.value_target[s:e], b.sample_weight[s:e])


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
            save_npz(tmp_root_p / f"bucket_{k:04d}" / f"task_{task_idx:06d}.npz",
                     _slice_batch(batch, start, end))
        start = end
    return total


def group_files_by_rows(files_with_takes, worker_group_size):
    groups, current, current_rows = [], [], 0
    for path, take in files_with_takes:
        if take <= 0:
            continue
        current.append((str(path), take))
        current_rows += take
        if current_rows >= worker_group_size:
            groups.append(current)
            current, current_rows = [], 0
    if current:
        groups.append(current)
    return groups


# ---- pass 2: merge ----
def _merge_shard_job(args):
    bucket_dir_str, out_str, seed = args
    chunk_paths = sorted(pathlib.Path(bucket_dir_str).glob("*.npz"))
    if not chunk_paths:
        return 0
    parts = [load_npz(cp) for cp in chunk_paths]
    batch = parts[0] if len(parts) == 1 else concat_batches(parts)
    del parts
    rng = np.random.default_rng(None if seed < 0 else seed)
    batch = joint_shuffle_take_first_n(len(batch), batch, rng)
    save_npz(pathlib.Path(out_str), batch)
    return len(batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data2048_nbt"))
    parser.add_argument("--shard-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=_env_int("SHUFFLE_WORKERS", 12))
    parser.add_argument("--worker-group-size", type=int, default=_env_int("SHUFFLE_WORKER_GROUP_SIZE", 80_000))
    parser.add_argument("--no-summary-cache", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    selfplay_dir = data_dir / "selfplay"
    shuffled_dir = data_dir / "shuffled" / "current"
    scatter_dir = data_dir / "shuffled" / "_scatter"
    summary_path = None if args.no_summary_cache else data_dir / "shuffled" / "summary.json"

    min_rows = _env_int("MIN_ROWS", 250_000)
    exponent = _env_float("TAPER_WINDOW_EXPONENT", 0.65)
    expand_per_row = _env_float("EXPAND_WINDOW_PER_ROW", 0.4)
    keep_target_rows = _env_int("KEEP_TARGET_ROWS", 20_000_000)

    if not selfplay_dir.exists():
        print(f"[Shuffle] selfplay dir missing: {selfplay_dir}", file=sys.stderr)
        return 1

    num_workers = max(1, args.num_workers or 1)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_workers) if num_workers > 1 else None
    t0 = time.perf_counter()

    files_rows = discover_and_count(selfplay_dir, summary_path, pool)
    if not files_rows:
        print("[Shuffle] no selfplay files found", file=sys.stderr)
        if pool:
            pool.close(); pool.join()
        return 1

    total = sum(r for _, r in files_rows)
    if total < min_rows:
        print(f"[Shuffle] total_rows={total} < MIN_ROWS={min_rows} -- skipping training", file=sys.stderr)
        if shuffled_dir.exists():
            shutil.rmtree(shuffled_dir)
        if pool:
            pool.close(); pool.join()
        return 2

    pool_size = compute_window_size(total, min_rows, exponent, expand_per_row)
    keep = min(pool_size, keep_target_rows)
    ratio = keep / pool_size if pool_size > 0 else 1.0
    print(f"[Shuffle] total_rows={total} pool={pool_size} keep={keep} ratio={ratio:.4f} "
          f"(min={min_rows}, E={exponent}, IWPR={expand_per_row}, keep_target={keep_target_rows})")

    rng = np.random.default_rng(None if args.seed < 0 else args.seed)
    chosen: list[tuple[pathlib.Path, int]] = []
    pool_covered = covered = 0
    oow_paths: list[pathlib.Path] = []
    for p, r in files_rows:
        if pool_covered >= pool_size:
            oow_paths.append(p)
            continue
        elig = min(r, pool_size - pool_covered)
        take = int(round(elig * ratio))
        if take > 0:
            chosen.append((p, take))
            covered += take
        pool_covered += elig

    if _env_int("ENABLE_OOW_CLEANUP", 1) != 0 and oow_paths:
        removed = 0
        for op in oow_paths:
            try:
                op.unlink(); removed += 1
            except OSError as e:
                print(f"[Shuffle] OOW cleanup could not remove {op}: {e}", file=sys.stderr)
        if removed:
            print(f"[Shuffle] OOW cleanup: deleted {removed} files past pool boundary")

    K = max(1, math.ceil(covered / args.shard_rows))
    groups = group_files_by_rows(chosen, args.worker_group_size)
    print(f"[Shuffle] pass1: {covered} rows from {len(chosen)} files -> {len(groups)} groups x {K} buckets")

    if scatter_dir.exists():
        shutil.rmtree(scatter_dir)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for k in range(K):
        (scatter_dir / f"bucket_{k:04d}").mkdir()

    task_seeds = ([int(rng.integers(0, 2**31 - 1)) for _ in groups] if args.seed >= 0 else [-1] * len(groups))
    shardify_args = [(i, groups[i], K, str(scatter_dir), task_seeds[i]) for i in range(len(groups))]
    if pool:
        sum(pool.imap_unordered(_shardify_job, shardify_args))
    else:
        for a in shardify_args:
            _shardify_job(a)

    if shuffled_dir.exists():
        shutil.rmtree(shuffled_dir)
    shuffled_dir.mkdir(parents=True, exist_ok=True)
    merge_args = []
    for k in range(K):
        seed = int(rng.integers(0, 2**31 - 1)) if args.seed >= 0 else -1
        merge_args.append((str(scatter_dir / f"bucket_{k:04d}"), str(shuffled_dir / f"shard_{k:04d}.npz"), seed))
    total_written = num_shards_written = 0
    res = (pool.imap_unordered(_merge_shard_job, merge_args) if pool else map(_merge_shard_job, merge_args))
    for w in res:
        if w > 0:
            total_written += w
            num_shards_written += 1
    if pool:
        pool.close(); pool.join()
    shutil.rmtree(scatter_dir, ignore_errors=True)
    print(f"[Shuffle] wrote {num_shards_written} shards, {total_written} rows to {shuffled_dir} "
          f"({time.perf_counter()-t0:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
