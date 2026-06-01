"""Produce a random-init TorchScript + state_dict for one 2048 network.

Run once per network before the first iteration of run.sh (idempotent unless
--force). Mirrors SkyZero_V7.1/python/init_model.py, adapted to 2048: fixed 4x4
board, no global-feature input, scalar value head, BatchNorm (default init —
no RepVGG/set_norm_scales step). Writes to data/nets/<network>/:
    latest.pt                TorchScript (selfplay; run.sh mirrors active -> models/)
    model_latest.pt          state_dict (train.py starts from these weights)
    scripted_iter_000000.pt  archival snapshot
    latest.meta.json         {iter:-1, network, value_scale}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

from model_config import config_from_name
from nets import build_net


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--network", required=True, help="Network name, e.g. b6c96.")
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

    cfg = config_from_name(args.network)
    cfg.device = "cpu"
    model = build_net(cfg)
    model.eval()
    with torch.no_grad():
        example = torch.zeros(1, cfg.num_planes, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(model, example)

    tmp = scripted_path.with_suffix(scripted_path.suffix + ".tmp")
    scripted.save(str(tmp))
    os.replace(tmp, scripted_path)

    iter0 = net_dir / "scripted_iter_000000.pt"
    if not iter0.exists():
        scripted.save(str(iter0))

    # state_dict so train.py starts from the SAME weights selfplay uses
    # (else iter-0 training fits data from a different random net).
    torch.save({"model_state_dict": model.state_dict(), "global_step": 0, "iter": -1}, ckpt_path)

    meta = {"iter": -1, "network": args.network, "value_scale": cfg.value_scale, "mtime": time.time()}
    (net_dir / "latest.meta.json").write_text(json.dumps(meta))

    print(f"[init_model] {args.network}: wrote {scripted_path} and {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
