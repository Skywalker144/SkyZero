"""KataGo-style train-rate token bucket — faithful port of
SkyZero_V7.1/python/bucket.py (schema-agnostic; unchanged logic).

  fill:    bucket += new_rows * MAX_TRAIN_PER_DATA  (clipped to MAX_BUCKET)
  consume: while bucket >= TRAIN_SAMPLES_PER_EPOCH: bucket -= it; epochs += 1
  train_steps = epochs * (TRAIN_SAMPLES_PER_EPOCH // BATCH_SIZE)

new_rows is the disk-truth delta from <DATA_DIR>/logs/selfplay.tsv. State in
<DATA_DIR>/bucket.json. Prints train_steps to stdout (0 = skip training).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def _env_int(name: str, default: int) -> int:
    return int(float(os.environ.get(name, str(default))))


def read_cum_rows(selfplay_tsv: pathlib.Path) -> int:
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
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data2048_nbt"))
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    state_path = data_dir / "bucket.json"
    selfplay_tsv = data_dir / "logs" / "selfplay.tsv"

    batch_size = _env_int("BATCH_SIZE", 128)
    samples_per_epoch = _env_int("TRAIN_SAMPLES_PER_EPOCH", 3_840_000)
    max_per_data = float(os.environ.get("MAX_TRAIN_PER_DATA", "4.0"))
    max_bucket = _env_int("MAX_TRAIN_BUCKET_SIZE", 2 * samples_per_epoch)

    state = load_state(state_path)
    cum_rows = read_cum_rows(selfplay_tsv)

    if "train_bucket_level" not in state:
        state["train_bucket_level"] = float(samples_per_epoch)
        state["rows_at_last_fill"] = cum_rows

    last_fill = int(state.get("rows_at_last_fill", cum_rows))
    new_rows = max(0, cum_rows - last_fill)
    fill = new_rows * max_per_data
    bucket = float(state["train_bucket_level"]) + fill
    cap = max(max_bucket, samples_per_epoch)
    if bucket > cap:
        bucket = cap

    threshold = 0.99 * samples_per_epoch
    epochs = 0
    while bucket >= threshold:
        bucket -= samples_per_epoch
        epochs += 1
    train_steps = epochs * (samples_per_epoch // batch_size)

    state["train_bucket_level"] = bucket
    state["rows_at_last_fill"] = cum_rows
    save_state(state_path, state)

    print(f"[bucket] cum_rows={cum_rows} new_rows={new_rows} fill=+{fill:.0f} "
          f"bucket={bucket:.0f}/{cap} epochs={epochs} -> train_steps={train_steps}",
          file=sys.stderr)
    print(train_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
