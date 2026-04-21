"""Compute this iteration's selfplay game count from target_replay_ratio.

Formula:
    needed_samples = ceil(batch_size * train_steps_per_epoch / target_replay_ratio)
    avg_game_len   = last_run.tsv's rows/games (fallback: AVG_GAME_LEN_BOOTSTRAP)
    N_games        = clip(ceil(needed_samples / avg_game_len), MIN_GAMES, MAX_GAMES)

Prints N_games to stdout (one integer) so run.sh can `GAMES=$(python ...)`.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def read_avg_game_len(last_run_tsv: pathlib.Path, bootstrap: float) -> float:
    """Parse last line of last_run.tsv; columns: iter games rows ...

    Returns rows/games for the most recent iteration, or bootstrap on failure.
    """
    try:
        if not last_run_tsv.exists():
            return bootstrap
        lines = [ln.strip() for ln in last_run_tsv.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            return bootstrap
        # Skip header if present
        data = lines[1:] if lines[0].startswith("iter") else lines
        if not data:
            return bootstrap
        fields = data[-1].split("\t")
        games = float(fields[1])
        rows = float(fields[2])
        if games <= 0:
            return bootstrap
        return rows / games
    except Exception:
        return bootstrap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    last_run_tsv = data_dir / "logs" / "last_run.tsv"

    batch_size = _env_int("BATCH_SIZE", 256)
    train_steps = _env_int("TRAIN_STEPS_PER_EPOCH", 100)
    ratio = _env_float("TARGET_REPLAY_RATIO", 6.0)
    min_games = _env_int("MIN_GAMES", 200)
    max_games = _env_int("MAX_GAMES", 8000)
    bootstrap = _env_float("AVG_GAME_LEN_BOOTSTRAP", 50.0)

    needed_samples = max(1, math.ceil(batch_size * train_steps / max(ratio, 1e-6)))
    avg_game_len = read_avg_game_len(last_run_tsv, bootstrap)
    raw_games = math.ceil(needed_samples / max(avg_game_len, 1.0))
    n_games = max(min_games, min(max_games, raw_games))

    # stderr for humans, stdout for scripts
    print(
        f"[compute_games] needed_samples={needed_samples} avg_game_len={avg_game_len:.1f} "
        f"raw_games={raw_games} -> N_games={n_games}",
        file=sys.stderr,
    )
    print(n_games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
