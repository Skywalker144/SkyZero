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
import json
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
    # KataGo paper visit warmup: cheap visits ramp 100→200 over the same
    # NUM_SIM_WARMUP_SAMPLES window the full visits use, so cheap/full move
    # in lockstep through the warmup.
    "cheap-search-visits": dict(
        stages_env="CHEAP_SEARCH_VISITS_STAGES",
        warmup_env="NUM_SIM_WARMUP_SAMPLES",
        fallback_env="CHEAP_SEARCH_VISITS",
        fallback_default="100",
        tag="compute_cheap_search_visits",
        out_name="CHEAP_SEARCH_VISITS",
    ),
}


def _compute_value(cfg: dict, data_dir: pathlib.Path) -> tuple[int, int, str]:
    """Returns (value, cum_rows, tag). Silent — no stdout/stderr.

    cum_rows is the cumulative-produced selfplay-row count (main + daemon),
    via pool_rows.cumulative_produced — monotonic, immune to PRUNE_OUTSIDE_WINDOW
    deletion. Matches the rows-ever-generated quantity that staged warmup expects.
    """
    import pool_rows

    fallback = int(float(os.environ.get(cfg["fallback_env"], cfg["fallback_default"])))
    stages = parse_stage_list(os.environ.get(cfg["stages_env"]), cast=int)
    warmup_samples = int(float(os.environ.get(cfg["warmup_env"], "0")))

    cum_rows = pool_rows.cumulative_produced(data_dir)
    val = staged_value(cum_rows, warmup_samples, stages)
    out = val if val is not None else fallback
    tag = "warmup" if val is not None else "steady-cfg"
    return out, cum_rows, tag


def _compute(cfg: dict, data_dir: pathlib.Path) -> int:
    out, cum_rows, tag = _compute_value(cfg, data_dir)
    stages = parse_stage_list(os.environ.get(cfg["stages_env"]), cast=int)
    warmup_samples = int(float(os.environ.get(cfg["warmup_env"], "0")))
    print(
        f"[{cfg['tag']}] cum_rows={cum_rows} "
        f"stages={stages} warmup_samples={warmup_samples} "
        f"-> {cfg['out_name']}={out}({tag})",
        file=sys.stderr,
    )
    print(out)
    return 0


def _write_sidecar(data_dir: pathlib.Path, out_path: pathlib.Path) -> int:
    nsim, cum_rows, _ = _compute_value(PARAMS["num-simulations"], data_dir)
    cheap, _, _ = _compute_value(PARAMS["cheap-search-visits"], data_dir)
    payload = {"num_simulations": int(nsim), "cheap_search_visits": int(cheap)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    os.replace(tmp, out_path)

    print(
        f"[warmup-sidecar] wrote {out_path}: "
        f"num_sim={nsim} cheap={cheap} cum_rows={cum_rows}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="param", required=True)
    for name in PARAMS:
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ws = sub.add_parser("write-sidecar")
    ws.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ws.add_argument("--out", default=None,
                    help="output JSON path (default: <data-dir>/logs/warmup_current.json)")
    args = parser.parse_args()
    if args.param == "write-sidecar":
        data_dir = pathlib.Path(args.data_dir)
        out_path = pathlib.Path(args.out) if args.out else data_dir / "logs" / "warmup_current.json"
        return _write_sidecar(data_dir, out_path)
    return _compute(PARAMS[args.param], pathlib.Path(args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
