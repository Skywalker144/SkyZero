"""Compute this iteration's main-GPU selfplay game count from the (warmed) replay ratio.

Formula:
    needed_total   = ceil(batch_size * train_steps_per_epoch / ratio_eff)
    daemon_pred    = predicted daemon rows this iter (dead-reckoning from last iter)
    needed_main    = max(0, needed_total - daemon_pred)
    raw_games      = ceil(needed_main / avg_game_len)
    N_games        = clip(raw_games, MIN_GAMES, MAX_GAMES)

ratio_eff is picked from REPLAY_RATIO_STAGES (per-stage list across
REPLAY_RATIO_WARMUP_SAMPLES of cumulative selfplay rows). When the list has
< 2 entries, falls back to TARGET_REPLAY_RATIO (no warmup).

Multi-GPU dead-reckoning (Plan C): the daemon (run_selfplay_daemon.sh) produces
NPZs in parallel without touching last_run.tsv. We use last iter's actual daemon
contribution — derived from `pool_rows.tsv` (total pool snapshot per iter) minus
`last_run.tsv[that_iter].rows` (main-only this iter) — as the prediction for
this iter, and deduct it from main's quota so the *combined* production tracks
target ratio instead of overshooting by ~3x.

cum_rows for warmup staging comes from pool_rows.cumulative_produced
(main + daemon, monotonic — immune to PRUNE_OUTSIDE_WINDOW deletion), not from
last_run.tsv summation. avg_game_len still comes from last_run.tsv's last row —
that's main-GPU's empirical len/game, which is the right scale to convert main's
row quota into a game count.

TRAIN_STEPS_PER_EPOCH is itself staged (see warmup.py train-steps, run by
run.sh just before this script and exported into env), so during warmup
both TS and ratio shrink together and `needed_total = B*TS/ratio` ends
up *smaller* on early iters, not larger. The MIN_ROWS floor in cold-start
mode keeps the very first training run from being skipped (see also
shuffle.py's MIN_ROWS gate).

Prints N_games to stdout (one integer) so run.sh can `GAMES=$(python ...)`.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import pool_rows
from warmup import parse_stage_list, staged_value


def _env_int(name: str, default: int) -> int:
    # int(float(...)) so cfg values like "2e6" / "6e6" parse as integers.
    return int(float(os.environ.get(name, str(default))))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def read_avg_game_len(last_run_tsv: pathlib.Path) -> float | None:
    """Last iter's main-GPU avg game length (rows/games), or None if no usable row."""
    try:
        if not last_run_tsv.exists():
            return None
        lines = [ln.strip() for ln in last_run_tsv.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        data = lines[1:] if lines[0].startswith("iter") else lines
        if not data:
            return None
        last = data[-1].split("\t")
        try:
            games = float(last[1])
            rows = float(last[2])
        except (IndexError, ValueError):
            return None
        return rows / games if games > 0 else None
    except Exception:
        return None


def read_main_rows_for_iter(last_run_tsv: pathlib.Path, target_iter: int) -> int | None:
    """Look up the `rows` column for a specific iter in last_run.tsv. None if absent."""
    try:
        if not last_run_tsv.exists():
            return None
        for line in last_run_tsv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("iter"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                if int(parts[0]) == target_iter:
                    return int(float(parts[2]))
            except ValueError:
                continue
    except Exception:
        return None
    return None


def predict_daemon_rows(data_dir: pathlib.Path, last_run_tsv: pathlib.Path) -> int:
    """Predict daemon's row contribution for the upcoming iter via dead-reckoning.

    Uses the last finished iter's actual daemon delta:
        delta_total   = pool_rows[-1].total - pool_rows[-2].total  # cumulative-produced diff = iter K produced
        main_prev     = last_run.tsv[pool_rows[-1].iter].rows
        daemon_prev   = max(0, delta_total - main_prev)
    Cumulative-produced semantics make the diff immune to PRUNE_OUTSIDE_WINDOW.
    Returns 0 (current behavior, main fills full quota) when there's <2 snapshots
    or last_run.tsv lacks the matching iter — both cases are early/cold-start.
    """
    snaps = pool_rows.read_snapshots(data_dir)
    if len(snaps) < 2:
        return 0
    prev_iter, prev_total = snaps[-1]
    _, prev_prev_total = snaps[-2]
    delta_total = max(0, prev_total - prev_prev_total)
    main_prev = read_main_rows_for_iter(last_run_tsv, prev_iter)
    if main_prev is None:
        return 0
    return max(0, delta_total - main_prev)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    last_run_tsv = data_dir / "logs" / "last_run.tsv"

    batch_size = _env_int("BATCH_SIZE", 256)
    train_steps = _env_int("TRAIN_STEPS_PER_EPOCH", 100)
    ratio_steady = _env_float("TARGET_REPLAY_RATIO", 6.0)
    min_games = _env_int("MIN_GAMES", 200)
    max_games = _env_int("MAX_GAMES", 8000)
    bootstrap = _env_float("AVG_GAME_LEN_BOOTSTRAP", 50.0)
    min_rows = _env_int("MIN_ROWS", 250000)

    stages = parse_stage_list(os.environ.get("REPLAY_RATIO_STAGES"), cast=float)
    warmup_samples = _env_int("REPLAY_RATIO_WARMUP_SAMPLES", 0)

    cum_rows = pool_rows.cumulative_produced(data_dir)
    avg_game_len = read_avg_game_len(last_run_tsv)
    warmed = staged_value(cum_rows, warmup_samples, stages)
    ratio = warmed if warmed is not None else ratio_steady

    needed_total = math.ceil(batch_size * train_steps / max(ratio, 1e-6))
    daemon_predicted = predict_daemon_rows(data_dir, last_run_tsv)

    if avg_game_len is None:
        needed_main = max(min_rows, max(0, needed_total - daemon_predicted))
        avg_game_len_used = bootstrap
        mode = "cold-start"
    else:
        needed_main = max(1, needed_total - daemon_predicted)
        avg_game_len_used = avg_game_len
        mode = "steady"
    raw_games = math.ceil(needed_main / max(avg_game_len_used, 1.0))
    n_games = max(min_games, min(max_games, raw_games))

    ratio_tag = "warmup" if warmed is not None else "steady-cfg"
    print(
        f"[compute_games] mode={mode} ratio={ratio:.2f}({ratio_tag}) "
        f"cum_rows={cum_rows} needed_total={needed_total} "
        f"daemon_predicted={daemon_predicted} needed_main={needed_main} "
        f"avg_game_len={avg_game_len_used:.1f} raw_games={raw_games} -> N_games={n_games}",
        file=sys.stderr,
    )
    print(n_games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
