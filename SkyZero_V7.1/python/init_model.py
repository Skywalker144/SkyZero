"""Produce a random-init TorchScript + state_dict for one network.

Run once per network before the first iteration of run.sh. Idempotent
unless --force is given. Writes to data/nets/<network>/:
    latest.pt              TorchScript (consumed by selfplay; run.sh mirrors
                           the active network's to data/models/latest.pt)
    model_latest.pt        state_dict (consumed by train.py)
    scripted_iter_000000.pt  archival snapshot
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

from model_config import net_config_from_name
from nets import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--network", required=True,
                        help="Network name, e.g. b5c128.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    net_dir = data_dir / "nets" / args.network
    net_dir.mkdir(parents=True, exist_ok=True)

    scripted_path = net_dir / "latest.pt"
    ckpt_path = net_dir / "model_latest.pt"

    if scripted_path.exists() and ckpt_path.exists() and not args.force:
        print(f"[init_model] already exists for {args.network}: {scripted_path}")
        return 0

    cfg = net_config_from_name(args.network)
    model = build_model(cfg)
    model.initialize()   # RepVGG init + set_norm_scales (NOTES.md §3 traps)
    model.eval()
    with torch.no_grad():
        example_state  = torch.zeros(1, cfg.num_planes, cfg.board_size, cfg.board_size, dtype=torch.float32)
        example_state[:, 0] = 1.0   # mask plane: full board
        example_global = torch.zeros(1, cfg.num_global_features, dtype=torch.float32)
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)

    tmp_path = scripted_path.with_suffix(scripted_path.suffix + ".tmp")
    scripted.save(str(tmp_path))
    os.replace(tmp_path, scripted_path)

    # Per-iter trail.
    iter0_path = net_dir / "scripted_iter_000000.pt"
    if not iter0_path.exists():
        scripted.save(str(iter0_path))

    # state_dict so train.py starts from the same weights self-play uses
    # (critical: otherwise iter-0 training fits against data from a
    # different random net).
    torch.save({"model_state_dict": model.state_dict(), "global_step": 0, "iter": -1}, ckpt_path)

    print(f"[init_model] {args.network}: wrote {scripted_path} and {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
