"""Compute this iteration's selfplay game count from the (warmed) replay ratio.

Formula:
    needed_samples = ceil(batch_size * train_steps_per_epoch / ratio_eff)
    raw_games      = ceil(needed_samples / avg_game_len)
    N_games        = clip(raw_games, MIN_GAMES, MAX_GAMES)

ratio_eff is picked from REPLAY_RATIO_STAGES (per-stage list across
REPLAY_RATIO_WARMUP_SAMPLES of cumulative selfplay rows). When the list has
< 2 entries, falls back to TARGET_REPLAY_RATIO (no warmup).

Cold start (no last_run.tsv history): cum_rows=0 picks the first stage
value, which is typically smallest -> largest needed_samples. A MIN_ROWS
floor still applies on the first iter so the very first training run isn't
skipped (see shuffle.py's MIN_ROWS gate).

Prints N_games to stdout (one integer) so run.sh can `GAMES=$(python ...)`.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

from warmup import parse_stage_list, staged_value


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def read_history(last_run_tsv: pathlib.Path) -> tuple[float | None, int]:
    """Parse last_run.tsv (columns: iter games rows ...).

    Returns (avg_game_len of last iter, cumulative rows across all iters).
    avg_game_len is None when no usable history exists.
    """
    try:
        if not last_run_tsv.exists():
            return None, 0
        lines = [ln.strip() for ln in last_run_tsv.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, 0
        data = lines[1:] if lines[0].startswith("iter") else lines
        if not data:
            return None, 0
        cum_rows = 0
        for ln in data:
            try:
                cum_rows += int(float(ln.split("\t")[2]))
            except (IndexError, ValueError):
                continue
        last = data[-1].split("\t")
        try:
            games = float(last[1])
            rows = float(last[2])
        except (IndexError, ValueError):
            return None, cum_rows
        avg = rows / games if games > 0 else None
        return avg, cum_rows
    except Exception:
        return None, 0


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

    avg_game_len, cum_rows = read_history(last_run_tsv)
    warmed = staged_value(cum_rows, warmup_samples, stages)
    ratio = warmed if warmed is not None else ratio_steady

    if avg_game_len is None:
        needed_samples = max(min_rows, math.ceil(batch_size * train_steps / max(ratio, 1e-6)))
        avg_game_len_used = bootstrap
        mode = "cold-start"
    else:
        needed_samples = max(1, math.ceil(batch_size * train_steps / max(ratio, 1e-6)))
        avg_game_len_used = avg_game_len
        mode = "steady"
    raw_games = math.ceil(needed_samples / max(avg_game_len_used, 1.0))
    n_games = max(min_games, min(max_games, raw_games))

    ratio_tag = "warmup" if warmed is not None else "steady-cfg"
    print(
        f"[compute_games] mode={mode} ratio={ratio:.2f}({ratio_tag}) "
        f"cum_rows={cum_rows} needed_samples={needed_samples} "
        f"avg_game_len={avg_game_len_used:.1f} raw_games={raw_games} -> N_games={n_games}",
        file=sys.stderr,
    )
    print(n_games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
