#!/usr/bin/env python3
"""
Compute dynamic selfplay game count based on train_per_data ratio.

Given the current training progress (global_step_samples from checkpoint) and
the total selfplay rows on disk, calculate how many new selfplay games are
needed to maintain the target ratio before the next training epoch.

Ratio semantics (KataGomo-style train_per_data):
  train_per_data = 2.0 means every 1 new selfplay row permits 2 training rows.
  Required new selfplay rows = samples_per_epoch / train_per_data
"""
import sys
import os
import argparse
import multiprocessing

import torch

from shuffle import compute_num_rows


def count_total_selfplay_rows(selfplay_dir, num_processes=8):
    """Count total rows across all NPZ files in selfplay_dir (recursively)."""
    npz_files = []
    for dirpath, _, filenames in os.walk(selfplay_dir):
        for f in filenames:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, f))

    if not npz_files:
        return 0, 0  # total_rows, num_files

    with multiprocessing.Pool(min(num_processes, len(npz_files))) as pool:
        results = pool.map(compute_num_rows, npz_files)

    total_rows = 0
    valid_files = 0
    for _, num_rows in results:
        if num_rows is not None and num_rows > 0:
            total_rows += num_rows
            valid_files += 1
    return total_rows, valid_files


def load_global_step_samples(traindir):
    """Load global_step_samples from the latest checkpoint. Returns None if no checkpoint."""
    ckpt_path = os.path.join(traindir, "latest.ckpt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_state = ckpt.get("train_state", {})
    return train_state.get("global_step_samples", 0)


def main():
    parser = argparse.ArgumentParser(description="Compute dynamic selfplay game count")
    parser.add_argument("--traindir", required=True, help="Training checkpoint directory")
    parser.add_argument("--selfplay-dir", required=True, help="Selfplay data directory")
    parser.add_argument("--train-per-data", type=float, default=2.0,
                        help="Training rows permitted per selfplay row (default: 2.0)")
    parser.add_argument("--samples-per-epoch", type=int, default=2000000,
                        help="Training samples consumed per epoch")
    parser.add_argument("--default-games", type=int, default=4000,
                        help="Default game count when no history exists")
    parser.add_argument("--min-games", type=int, default=500,
                        help="Minimum games per iteration")
    parser.add_argument("--max-games", type=int, default=4000,
                        help="Maximum games per iteration")
    parser.add_argument("--avg-rows-per-game", type=int, default=60,
                        help="Estimated rows per game (default: 60 for gomoku/renju)")
    args = parser.parse_args()

    # If no checkpoint yet, use default
    global_step_samples = load_global_step_samples(args.traindir)
    if global_step_samples is None:
        print(args.default_games)
        return

    # Count existing selfplay data
    total_sp_rows, num_files = count_total_selfplay_rows(args.selfplay_dir)
    if total_sp_rows == 0 or num_files == 0:
        print(args.default_games)
        return

    # Cumulative balance: total training done vs total selfplay generated.
    # Ideal: global_step_samples <= total_sp_rows * train_per_data
    # After next epoch: (global_step_samples + samples_per_epoch) <= (total_sp_rows + new_rows) * train_per_data
    # => new_rows >= (global_step_samples + samples_per_epoch) / train_per_data - total_sp_rows
    # Negative means remote workers have already over-supplied; main host can fall to MIN_GAMES.
    cumulative_needed = (global_step_samples + args.samples_per_epoch) / args.train_per_data - total_sp_rows
    needed_rows = max(0, cumulative_needed)

    # Convert rows to games
    games = int(needed_rows / args.avg_rows_per_game)
    games = max(args.min_games, min(games, args.max_games))

    # Log to stderr for debugging (stdout is the result)
    print(
        f"[compute_games] trained={global_step_samples}, sp_rows={total_sp_rows}, "
        f"needed_rows={needed_rows:.0f}, games={games}",
        file=sys.stderr,
    )

    print(games)


if __name__ == "__main__":
    main()
