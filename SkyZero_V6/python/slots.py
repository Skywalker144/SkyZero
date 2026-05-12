"""Multi-slot model bookkeeping.

A "slot" is one of N model variants trained side-by-side from the same
selfplay data. Each slot has its own checkpoints/<slot>/ and models/<slot>/
subtree; selfplay always reads models/latest.pt, which run.sh promotes from
exactly one slot per iter (the "active" slot, picked by cumulative selfplay
sample count crossing per-slot thresholds).

Config — four parallel comma-separated lists in scripts/run.cfg, exported
by run.sh as env vars:

    MODEL_SLOTS="b8c128, b10c128, b12c128, b12c196"
    MODEL_BLOCKS="8, 10, 12, 12"
    MODEL_CHANNELS="128, 128, 128, 196"
    MODEL_ACTIVATE_SAMPLES="0, 1e8, 5e8, 2e9"

Activation rule: among slots whose activate_samples <= cumulative selfplay
samples, pick the slot with the largest activate_samples. Lists must be
strictly monotonic in activate_samples and the first entry must be 0
(otherwise no slot is active at cum=0). Switching is forward-only — once
a higher-threshold slot has activated, lower-threshold slots are not
selected again.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    name: str
    num_blocks: int
    num_channels: int
    activate_samples: int


def _split(s: str) -> list[str]:
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def load_slots() -> list[Slot]:
    names = _split(os.environ.get("MODEL_SLOTS", ""))
    blocks = _split(os.environ.get("MODEL_BLOCKS", ""))
    channels = _split(os.environ.get("MODEL_CHANNELS", ""))
    actives = _split(os.environ.get("MODEL_ACTIVATE_SAMPLES", ""))
    if not names:
        raise RuntimeError(
            "MODEL_SLOTS not set — export it (and the parallel MODEL_BLOCKS / "
            "MODEL_CHANNELS / MODEL_ACTIVATE_SAMPLES lists) from run.cfg.")
    n = len(names)
    if not (len(blocks) == len(channels) == len(actives) == n):
        raise RuntimeError(
            f"slot list length mismatch: MODEL_SLOTS={n} MODEL_BLOCKS={len(blocks)} "
            f"MODEL_CHANNELS={len(channels)} MODEL_ACTIVATE_SAMPLES={len(actives)}")
    slots: list[Slot] = []
    for name, b, c, a in zip(names, blocks, channels, actives):
        slots.append(Slot(
            name=name,
            num_blocks=int(b),
            num_channels=int(c),
            activate_samples=int(float(a)),
        ))
    if slots[0].activate_samples != 0:
        raise RuntimeError(
            f"first slot must have activate_samples=0; got {slots[0].activate_samples} "
            f"({slots[0].name})")
    for i in range(1, n):
        if slots[i].activate_samples <= slots[i - 1].activate_samples:
            raise RuntimeError(
                "MODEL_ACTIVATE_SAMPLES must be strictly increasing; "
                f"slot[{i}]={slots[i].name}({slots[i].activate_samples}) <= "
                f"slot[{i-1}]={slots[i-1].name}({slots[i-1].activate_samples})")
    return slots


def get_slot(name: str, slots: list[Slot] | None = None) -> Slot:
    slots = slots if slots is not None else load_slots()
    for s in slots:
        if s.name == name:
            return s
    raise KeyError(f"unknown slot {name!r}; defined: {[s.name for s in slots]}")


def cumulative_selfplay_samples(data_dir: pathlib.Path) -> int:
    """Cumulative selfplay rows ever produced (main + daemon).

    Used by `pick_active` against `MODEL_ACTIVATE_SAMPLES`. Routes through
    pool_rows.cumulative_produced so PRUNE_OUTSIDE_WINDOW deletion can never
    cause the active slot to regress to a smaller model. Multi-GPU runs see
    cum_rows grow faster than single-GPU because daemon contributions are
    counted — `MODEL_ACTIVATE_SAMPLES` thresholds may need recalibration.
    """
    import pool_rows
    return pool_rows.cumulative_produced(data_dir)


def pick_active(slots: list[Slot], cum_samples: int) -> Slot:
    chosen = slots[0]
    for s in slots:
        if s.activate_samples <= cum_samples:
            chosen = s
        else:
            break
    return chosen


# ---------------------------------------------------------------------------
# Path helpers — single source of truth for slot subdirectory layout.
# ---------------------------------------------------------------------------

def slot_ckpt_dir(data_dir: pathlib.Path, slot_name: str) -> pathlib.Path:
    return data_dir / "checkpoints" / slot_name


def slot_models_dir(data_dir: pathlib.Path, slot_name: str) -> pathlib.Path:
    return data_dir / "models" / slot_name


def slot_train_log(data_dir: pathlib.Path, slot_name: str) -> pathlib.Path:
    return data_dir / "logs" / f"train_{slot_name}.tsv"


def slot_iter_snapshot(data_dir: pathlib.Path, slot_name: str, it: int) -> pathlib.Path:
    return slot_models_dir(data_dir, slot_name) / f"model_iter_{it:06d}.pt"


# ---------------------------------------------------------------------------
# Promote: copy active slot's iter snapshot to models/latest.pt (atomic) and
# refresh latest.meta.json. Selfplay polls latest.pt mtime.
# ---------------------------------------------------------------------------

def promote(data_dir: pathlib.Path, it: int, slot_name: str | None = None) -> Slot:
    slots = load_slots()
    if slot_name is None:
        cum = cumulative_selfplay_samples(data_dir)
        active = pick_active(slots, cum)
    else:
        active = get_slot(slot_name, slots)
    src = slot_iter_snapshot(data_dir, active.name, it)
    if not src.exists():
        raise FileNotFoundError(f"iter snapshot missing for active slot: {src}")
    latest = data_dir / "models" / "latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, latest)
    meta = {"iter": int(it), "slot": active.name, "mtime": latest.stat().st_mtime}
    meta_path = data_dir / "models" / "latest.meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with meta_tmp.open("w") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)
    return active


# ---------------------------------------------------------------------------
# CLI — invoked from shell scripts.
# ---------------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="parse + assert slot config; exit 0 if ok")
    sub.add_parser("list-slots", help="print slot names, one per line")

    p_active = sub.add_parser("active", help="print active slot name for current cum samples")
    p_active.add_argument("--data-dir", required=True)

    p_promote = sub.add_parser(
        "promote",
        help="copy active slot's iter snapshot to models/latest.pt + meta")
    p_promote.add_argument("--data-dir", required=True)
    p_promote.add_argument("--iter", type=int, required=True)
    p_promote.add_argument("--slot", default=None,
                           help="override active slot pick (debug)")

    p_cum = sub.add_parser("cum-samples", help="print cumulative selfplay samples")
    p_cum.add_argument("--data-dir", required=True)

    args = p.parse_args()

    if args.cmd == "validate":
        load_slots()
        return 0
    if args.cmd == "list-slots":
        for s in load_slots():
            print(s.name)
        return 0
    if args.cmd == "active":
        slots = load_slots()
        cum = cumulative_selfplay_samples(pathlib.Path(args.data_dir))
        print(pick_active(slots, cum).name)
        return 0
    if args.cmd == "cum-samples":
        print(cumulative_selfplay_samples(pathlib.Path(args.data_dir)))
        return 0
    if args.cmd == "promote":
        active = promote(pathlib.Path(args.data_dir), args.iter, slot_name=args.slot)
        print(active.name)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
