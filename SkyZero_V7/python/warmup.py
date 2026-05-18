"""Shared list-driven warmup helper.

A warmup is parameterized by (1) a list of values, one per stage, and
(2) a window length in cumulative selfplay rows. Stages are evenly spaced:
with N values, the i-th value (0-indexed) applies for progress in
[i/N, (i+1)/N). Once progress >= 1.0, the last value is used (steady state).

Disabled (returns None) when len(stages) < 2 or warmup_samples <= 0; the
caller is expected to fall back to its existing steady-state value.

Also serves as a CLI for shell scripts to fetch the current-iter value of
a staged parameter:

    python warmup.py num-simulations --data-dir DATA_DIR
    python warmup.py train-steps     --data-dir DATA_DIR

Each subcommand reads cumulative selfplay rows from
DATA_DIR/logs/last_run.tsv, looks up the staged value, and prints one
integer to stdout. Falls back to the steady-state env var when the stage
list has < 2 entries (warmup disabled).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Callable


def parse_stage_list(s: str | None, cast: Callable = float) -> list:
    """Parse "v1, v2, v3" -> [cast(v1), cast(v2), cast(v3)].

    Empty / whitespace-only strings yield []. Skips empty fields produced
    by trailing commas. Raises if a non-empty field fails to cast.
    """
    if not s:
        return []
    return [cast(p.strip()) for p in s.split(",") if p.strip()]


def staged_value(samples_seen: int, warmup_samples: int, stages: list):
    """Returns the stage value for samples_seen, or None if warmup disabled.

    Disabled when len(stages) < 2 or warmup_samples <= 0.
    """
    n = len(stages)
    if n < 2 or warmup_samples <= 0:
        return None
    progress = samples_seen / warmup_samples
    if progress >= 1.0:
        return stages[-1]
    if progress < 0.0:
        return stages[0]
    idx = min(n - 1, int(progress * n))
    return stages[idx]


PARAMS = {
    "num-simulations": dict(
        stages_env="NUM_SIMULATIONS_STAGES",
        warmup_env="NUM_SIM_WARMUP_SAMPLES",
        fallback_env="NUM_SIMULATIONS",
        fallback_default="400",
        tag="compute_num_simulations",
        out_name="NUM_SIMULATIONS",
    ),
    "train-steps": dict(
        stages_env="TRAIN_STEPS_STAGES",
        warmup_env="TRAIN_STEPS_WARMUP_SAMPLES",
        fallback_env="TRAIN_STEPS_PER_EPOCH",
        fallback_default="15000",
        tag="compute_train_steps",
        out_name="TRAIN_STEPS_PER_EPOCH",
    ),
}


def _compute(cfg: dict, data_dir: pathlib.Path) -> int:
    from compute_games import read_history

    last_run_tsv = data_dir / "logs" / "last_run.tsv"

    fallback = int(float(os.environ.get(cfg["fallback_env"], cfg["fallback_default"])))
    stages = parse_stage_list(os.environ.get(cfg["stages_env"]), cast=int)
    warmup_samples = int(float(os.environ.get(cfg["warmup_env"], "0")))

    _, cum_rows = read_history(last_run_tsv)
    val = staged_value(cum_rows, warmup_samples, stages)
    out = val if val is not None else fallback

    tag = "warmup" if val is not None else "steady-cfg"
    print(
        f"[{cfg['tag']}] cum_rows={cum_rows} "
        f"stages={stages} warmup_samples={warmup_samples} "
        f"-> {cfg['out_name']}={out}({tag})",
        file=sys.stderr,
    )
    print(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="param", required=True)
    for name in PARAMS:
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = parser.parse_args()
    return _compute(PARAMS[args.param], pathlib.Path(args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
