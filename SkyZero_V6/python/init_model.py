"""Random-init every slot defined in run.cfg, plus seed data/models/latest.pt.

For each slot S in MODEL_SLOTS:
    data/checkpoints/<S>/model_latest.pt   (state_dict, iter=-1)
    data/models/<S>/model_iter_000000.pt   (TorchScript)

Then copies the first slot's iter_000000 to data/models/latest.pt — at
cum_samples=0 that slot is the active one for the upcoming first selfplay
round. Idempotent: skips entirely when data/models/latest.pt exists unless
--force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

from model_config import net_config_for_slot
from nets import build_model
from slots import (
    load_slots,
    slot_ckpt_dir,
    slot_iter_snapshot,
    slot_models_dir,
)


def _trace_random_init(cfg) -> torch.jit.ScriptModule:
    model = build_model(cfg)
    model.initialize()
    model.eval()
    with torch.no_grad():
        example_state  = torch.zeros(1, cfg.num_planes, cfg.board_size, cfg.board_size,
                                     dtype=torch.float32)
        example_state[:, 0] = 1.0
        example_global = torch.zeros(1, cfg.num_global_features, dtype=torch.float32)
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)
    return scripted, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    latest = data_dir / "models" / "latest.pt"

    if latest.exists() and not args.force:
        print(f"[init_model] already exists: {latest} — skipping (use --force to redo)")
        return 0

    slots = load_slots()
    print(f"[init_model] initializing {len(slots)} slot(s): {[s.name for s in slots]}")

    for slot in slots:
        cfg = net_config_for_slot(slot.name)
        scripted, model = _trace_random_init(cfg)

        ckpt_dir = slot_ckpt_dir(data_dir, slot.name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "model_latest.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    "global_step": 0, "iter": -1}, ckpt_path)

        models_dir = slot_models_dir(data_dir, slot.name)
        models_dir.mkdir(parents=True, exist_ok=True)
        iter0 = slot_iter_snapshot(data_dir, slot.name, 0)
        scripted.save(str(iter0))
        print(f"[init_model]   {slot.name} (b{slot.num_blocks}c{slot.num_channels}): "
              f"wrote {ckpt_path} and {iter0}")

    # Promote slot 0 (activate=0) → models/latest.pt for the first selfplay round.
    active = slots[0]
    src = slot_iter_snapshot(data_dir, active.name, 0)
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    tmp.write_bytes(src.read_bytes())
    os.replace(tmp, latest)

    meta = {"iter": 0, "slot": active.name, "mtime": latest.stat().st_mtime}
    meta_path = data_dir / "models" / "latest.meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with meta_tmp.open("w") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)

    print(f"[init_model] promoted slot {active.name} → {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
