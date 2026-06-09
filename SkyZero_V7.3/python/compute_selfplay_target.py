"""Fixed-train / adaptive-selfplay accounting (V7.3).

Replaces the V7.2 token bucket (python/bucket.py). Train volume is now FIXED
(train_steps = TRAIN_SAMPLES_PER_EPOCH / BATCH_SIZE every iter); this module
decides how many games the MAIN loop should produce to keep the cumulative
replay ratio at TARGET_REPLAY_RATIO.

Each iter (called by run.sh BEFORE selfplay):
    target_cum  += needed                 (needed = TRAIN_SAMPLES_PER_EPOCH / RATIO)
    trained_cum += TRAIN_SAMPLES_PER_EPOCH
    deficit = target_cum - cum_rows                 (cum_rows from selfplay.tsv)
    games   = 0 if deficit <= 0 else clip(ceil(deficit / avg_len), MIN, MAX)

Cold start (first call): target_cum = max(MIN_ROWS, needed) — produce enough to
begin training; shuffle skips below MIN_ROWS anyway.

cum_rows is read from selfplay.tsv (the single APPEND-ONLY cumulative truth,
main + daemon) — NOT by scanning data/selfplay/*.npz, which OOW-cleanup keeps
trimming to the in-window size (non-monotonic). avg_len = cumulative rows/games
across all producers (robust mean game length; equals the avg_len column's
running weighted average).

Daemon batch-settle lag: the daemon's rows only land in selfplay.tsv on model
reload, so cum_rows can lag ~1 iter behind disk. Accepted (design doc §2.4):
the cumulative accounting self-corrects next iter; at worst the main loop
slightly over-tops-up near the break-even card count.

State persists in data/selfplay_target.json (consume-once family — resume must
NOT recompute it, or it desyncs from selfplay.tsv's cumulative rows):
    target_cum   — float, cumulative target rows
    trained_cum  — float, cumulative trained samples (for ratio reporting)

CLI (one shot, called by run.sh):
    python compute_selfplay_target.py --data-dir DIR
        Updates state, prints the main-loop game count to stdout (0 = skip
        main selfplay this iter; daemon already over-produced).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

from bucket import read_cum_rows  # reuse the exact tsv row-sum (col 3, all producers)
from log_util import tag

TAG = tag("Target", sys.stderr)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(float(os.environ.get(name, str(default))))


def read_avg_game_len(selfplay_tsv: pathlib.Path) -> float:
    """Cumulative mean game length = total rows / total games across all
    producers (main + daemon). Used to convert a row deficit into a game
    count. Returns 0.0 when no games have been logged yet."""
    if not selfplay_tsv.exists():
        return 0.0
    tot_rows = tot_games = 0
    try:
        for ln in selfplay_tsv.read_text().splitlines():
            if not ln.strip() or ln.startswith("producer"):
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                continue
            try:
                tot_games += int(float(parts[2]))
                tot_rows += int(float(parts[3]))
            except ValueError:
                continue
    except OSError:
        return 0.0
    return (tot_rows / tot_games) if tot_games > 0 else 0.0


def load_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state_path: pathlib.Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    state_path = data_dir / "selfplay_target.json"
    selfplay_tsv = data_dir / "logs" / "selfplay.tsv"

    samples_per_epoch = _env_float("TRAIN_SAMPLES_PER_EPOCH", 128_000)
    ratio = _env_float("TARGET_REPLAY_RATIO", 6.0)
    min_rows = _env_float("MIN_ROWS", 250_000)
    min_games = _env_int("MIN_GAMES", 0)
    max_games = _env_int("MAX_GAMES", 2000)
    needed = samples_per_epoch / ratio if ratio > 0 else samples_per_epoch

    state = load_state(state_path)
    cum_rows = read_cum_rows(selfplay_tsv)

    # Advance the cumulative target. First call: cold-start to MIN_ROWS.
    if "target_cum" not in state:
        target_cum = max(min_rows, needed)
        trained_cum = samples_per_epoch
    else:
        target_cum = float(state["target_cum"]) + needed
        trained_cum = float(state.get("trained_cum", 0.0)) + samples_per_epoch

    deficit = target_cum - cum_rows
    avg_len = read_avg_game_len(selfplay_tsv)

    if deficit <= 0:
        games = 0  # daemon already over-produced; skip main selfplay this iter
    elif avg_len <= 0:
        games = max_games  # no length info yet (cold start) — fill fast
    else:
        games = int(math.ceil(deficit / avg_len))
        games = max(min_games, min(max_games, games))

    state["target_cum"] = target_cum
    state["trained_cum"] = trained_cum
    save_state(state_path, state)

    eff_ratio = (trained_cum / cum_rows) if cum_rows > 0 else 0.0
    print(
        f"{TAG} cum_rows={cum_rows} target_cum={target_cum:.0f} "
        f"deficit={deficit:.0f} avg_len={avg_len:.1f} "
        f"eff_ratio={eff_ratio:.2f} -> games={games}",
        file=sys.stderr,
    )
    print(games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
