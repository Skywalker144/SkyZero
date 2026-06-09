"""KataGo-style train-rate token bucket (see ../KataGo/python/train.py).

Replaces the V5-era compute_games + REPLAY_RATIO_STAGES + TRAIN_STEPS_STAGES
machinery. The bucket is the single source of truth for when training runs:

  fill:    bucket += new_rows * MAX_TRAIN_PER_DATA  (clipped to MAX_BUCKET)
  consume: if bucket >= TRAIN_SAMPLES_PER_EPOCH:
               bucket -= TRAIN_SAMPLES_PER_EPOCH
               train one epoch

`new_rows` is the disk-truth row count delta (from selfplay.tsv,
producer-agnostic), so daemon contributions feed the bucket transparently.

State persists in data/bucket.json (shared across all networks — the bucket
just controls the train/selfplay cadence and is orthogonal to which network
is currently training):
    train_bucket_level     — float, current bucket fill
    rows_at_last_fill      — int, cum rows snapshot from last fill

CLI (one shot, called by run.sh):
    python bucket.py --data-dir DIR
        Updates state, decides this iter's train_steps. Writes the new
        bucket state. Prints train_steps to stdout (0 = skip train).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from log_util import tag

TAG = tag("Bucket", sys.stderr)


def _env_int(name: str, default: int) -> int:
    return int(float(os.environ.get(name, str(default))))


def read_cum_rows(selfplay_tsv: pathlib.Path) -> int:
    """Sum the rows column (col 3) across all producers."""
    if not selfplay_tsv.exists():
        return 0
    total = 0
    try:
        for ln in selfplay_tsv.read_text().splitlines():
            if not ln.strip() or ln.startswith("producer"):
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                continue
            try:
                total += int(float(parts[3]))
            except ValueError:
                continue
    except OSError:
        pass
    return total


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
    state_path = data_dir / "bucket.json"
    selfplay_tsv = data_dir / "logs" / "selfplay.tsv"

    batch_size = _env_int("BATCH_SIZE", 256)
    samples_per_epoch = _env_int("TRAIN_SAMPLES_PER_EPOCH", 3_840_000)
    max_per_data = float(os.environ.get("MAX_TRAIN_PER_DATA", "4.0"))
    max_bucket = _env_int("MAX_TRAIN_BUCKET_SIZE", 2 * samples_per_epoch)

    state = load_state(state_path)
    cum_rows = read_cum_rows(selfplay_tsv)

    # First-time init: seed bucket to one epoch so training can start as
    # soon as MIN_ROWS is met (KataGo train.py:788 initializes the same way).
    if "train_bucket_level" not in state:
        state["train_bucket_level"] = float(samples_per_epoch)
        state["rows_at_last_fill"] = cum_rows

    # Fill.
    last_fill = int(state.get("rows_at_last_fill", cum_rows))
    new_rows = max(0, cum_rows - last_fill)
    fill = new_rows * max_per_data
    bucket = float(state["train_bucket_level"]) + fill
    # Cap at max_bucket, but never below samples_per_epoch (KataGo train.py:1069).
    cap = max(max_bucket, samples_per_epoch)
    if bucket > cap:
        bucket = cap

    # Consume: drain all epochs that fit (KataGo -stop-when-train-bucket-limited).
    # Combining N epochs into a single train.py invocation avoids paying
    # checkpoint-save / export overhead between epochs that would otherwise
    # eat 4-5 of the per-iter quota when the bucket is saturated.
    # 0.99 absorbs float jitter (KataGo train.py:1269).
    threshold = 0.99 * samples_per_epoch
    epochs = 0
    while bucket >= threshold:
        bucket -= samples_per_epoch
        epochs += 1
    train_steps = epochs * (samples_per_epoch // batch_size)

    state["train_bucket_level"] = bucket
    state["rows_at_last_fill"] = cum_rows
    save_state(state_path, state)

    print(
        f"{TAG} cum_rows={cum_rows} new_rows={new_rows} "
        f"fill=+{fill:.0f} bucket={bucket:.0f}/{cap} epochs={epochs} "
        f"-> train_steps={train_steps}",
        file=sys.stderr,
    )
    print(train_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
