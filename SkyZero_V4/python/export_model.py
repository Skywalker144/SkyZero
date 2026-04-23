"""Export a trained nn.Module checkpoint as TorchScript for C++ selfplay.

Expected workflow: train.py saves state_dict to data/checkpoints/model_latest.pt,
this script loads it, traces to TorchScript, and atomically replaces
data/models/latest.pt (plus writes a per-iter snapshot).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

from model_config import net_config_from_env
from nets import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="state_dict .pt path")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = net_config_from_env()
    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    # Prefer SWA weights when present — mirrors KataGomo's export path
    # (load_model.load_swa_model_state_dict). `swa_model_state_dict` comes
    # from a torch.optim.swa_utils.AveragedModel, whose params live under
    # the "module." prefix plus an "n_averaged" buffer we strip.
    if isinstance(state, dict) and state.get("swa_model_state_dict") is not None:
        swa_sd = state["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa_sd.items() if k.startswith("module.")}
        model.load_state_dict(stripped)
        print(f"[export_model] loaded SWA weights from {args.ckpt}")
    else:
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[export_model] loaded raw weights from {args.ckpt}")
    model.eval()

    with torch.no_grad():
        example = torch.zeros(1, cfg.num_planes, cfg.board_size, cfg.board_size, dtype=torch.float32)
        scripted = torch.jit.trace(model, example, strict=False)

    iter_path = models_dir / f"model_iter_{args.iter:06d}.pt"
    scripted.save(str(iter_path))

    latest = models_dir / "latest.pt"
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    scripted.save(str(tmp))
    os.replace(tmp, latest)

    print(f"[export_model] wrote {iter_path} and updated {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
