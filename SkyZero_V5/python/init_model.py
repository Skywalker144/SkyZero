"""Produce a random-init TorchScript model at data/models/latest.pt.

Run once before the first iteration of run.sh. Idempotent if target exists
unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

from model_config import net_config_from_env
from nets_v2 import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Output TorchScript path.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path = pathlib.Path(args.out) if args.out else (models_dir / "latest.pt")

    if out_path.exists() and not args.force:
        print(f"[init_model] already exists: {out_path}")
        return 0

    cfg = net_config_from_env()
    model = build_model(cfg)
    model.initialize()   # RepVGG init + set_norm_scales (NOTES.md §3 traps)
    model.eval()
    with torch.no_grad():
        # nets_v2 forward signature: (input_spatial, input_global)
        example_state  = torch.zeros(1, cfg.num_planes, cfg.board_size, cfg.board_size, dtype=torch.float32)
        example_state[:, 0] = 1.0   # mask plane: full board
        example_global = torch.zeros(1, cfg.num_global_features, dtype=torch.float32)
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    scripted.save(str(tmp_path))
    os.replace(tmp_path, out_path)

    # Per-iter trail
    iter0_path = models_dir / "model_iter_000000.pt"
    if not iter0_path.exists():
        scripted.save(str(iter0_path))

    # Also write state_dict to data/checkpoints/model_latest.pt so train.py
    # starts from the same weights self-play just used (critical: otherwise
    # iter-0 training fits against data from a different random net).
    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model_latest.pt"
    torch.save({"model_state_dict": model.state_dict(), "global_step": 0, "iter": -1}, ckpt_path)

    print(f"[init_model] wrote TorchScript to {out_path} and state_dict to {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
