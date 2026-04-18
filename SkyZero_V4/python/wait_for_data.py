#!/usr/bin/env python3
"""
Train-side gate: block until enough new selfplay rows exist to satisfy the
target train_per_data ratio for the next training epoch.

Gating condition:
    total_sp_rows * train_per_data >= global_step_samples + samples_per_epoch

global_step_samples is read from latest.ckpt and reflects what training has
already consumed. total_sp_rows is recounted from disk on every check so that
data synced from remote selfplay workers is picked up.
"""
import sys
import os
import time
import argparse
import datetime

from compute_games import count_total_selfplay_rows, load_global_step_samples


def main():
    parser = argparse.ArgumentParser(description="Wait for sufficient selfplay data before training")
    parser.add_argument("--traindir", required=True)
    parser.add_argument("--selfplay-dir", required=True)
    parser.add_argument("--train-per-data", type=float, default=2.0)
    parser.add_argument("--samples-per-epoch", type=int, default=2000000)
    parser.add_argument("--check-interval", type=int, default=30,
                        help="Seconds between checks while waiting")
    parser.add_argument("--max-wait", type=int, default=0,
                        help="Maximum total seconds to wait (0 = unlimited)")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit with code 2 if insufficient (no waiting)")
    args = parser.parse_args()

    start_time = time.time()
    last_rows = -1
    last_check_time = start_time

    while True:
        global_step_samples = load_global_step_samples(args.traindir) or 0
        total_sp_rows, _ = count_total_selfplay_rows(args.selfplay_dir)
        budget = total_sp_rows * args.train_per_data - global_step_samples

        if budget >= args.samples_per_epoch:
            print(
                f"[wait_for_data] OK: budget={budget:.0f} >= need={args.samples_per_epoch} "
                f"(trained={global_step_samples}, sp_rows={total_sp_rows})",
                flush=True,
            )
            return 0

        now = time.time()
        speed_msg = ""
        if last_rows >= 0:
            new_rows = total_sp_rows - last_rows
            elapsed = now - last_check_time
            if elapsed > 0:
                speed_msg = f", speed={new_rows / elapsed:.1f} rows/s"
        last_rows = total_sp_rows
        last_check_time = now

        print(
            f"[wait_for_data] waiting: budget={budget:.0f} < need={args.samples_per_epoch} "
            f"(trained={global_step_samples}, sp_rows={total_sp_rows}){speed_msg} "
            f"@ {datetime.datetime.now().strftime('%H:%M:%S')}",
            flush=True,
        )

        if args.once:
            return 2

        if args.max_wait > 0 and (now - start_time) >= args.max_wait:
            print(f"[wait_for_data] max-wait {args.max_wait}s exceeded, exiting with error", file=sys.stderr)
            return 1

        time.sleep(args.check_interval)


if __name__ == "__main__":
    sys.exit(main())
