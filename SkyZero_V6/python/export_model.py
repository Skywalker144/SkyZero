"""Export a trained nn.Module checkpoint as TorchScript for C++ selfplay.

Per-slot: reads data/checkpoints/<slot>/model_latest.pt, traces, writes the
TorchScript snapshot to data/models/<slot>/model_iter_NNNNNN.pt. Does NOT
touch data/models/latest.pt — that promotion is run.sh's responsibility
(via slots.py promote, which only fires after every slot has exported and
the active slot has been picked).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

from checkpoint_utils import zero_pad_for_v6
from model_config import net_config_for_slot
from nets import build_model
from slots import slot_ckpt_dir, slot_iter_snapshot, slot_models_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", required=True)
    parser.add_argument("--ckpt", default=None,
                        help="state_dict .pt path (defaults to "
                             "<data-dir>/checkpoints/<slot>/model_latest.pt)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else (
        slot_ckpt_dir(data_dir, args.slot) / "model_latest.pt")

    cfg = net_config_for_slot(args.slot)
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    # Prefer SWA weights when present — mirrors KataGomo's export path
    # (load_model.load_swa_model_state_dict). `swa_model_state_dict` comes
    # from a torch.optim.swa_utils.AveragedModel, whose params live under
    # the "module." prefix plus an "n_averaged" buffer we strip.
    if isinstance(state, dict) and state.get("swa_model_state_dict") is not None:
        swa_sd = state["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa_sd.items() if k.startswith("module.")}
        model.load_state_dict(zero_pad_for_v6(
            stripped, cfg.num_planes, cfg.num_global_features))
        print(f"[export_model] loaded SWA weights from {ckpt_path}")
    else:
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(zero_pad_for_v6(
            state, cfg.num_planes, cfg.num_global_features))
        print(f"[export_model] loaded raw weights from {ckpt_path}")
    # TRAP 3 (NOTES.md §3.3): NormMask.scale not in state_dict. Without this,
    # WDL logits magnitude blows up ~10× (overconfident, "model looks broken").
    model.set_norm_scales()
    model.eval()

    with torch.no_grad():
        # nets.KataGoNet forward signature: (input_spatial, input_global)
        example_state  = torch.zeros(1, cfg.num_planes, cfg.board_size, cfg.board_size, dtype=torch.float32)
        example_state[:, 0] = 1.0   # mask plane: full board
        example_global = torch.zeros(1, cfg.num_global_features, dtype=torch.float32)
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)

    models_dir = slot_models_dir(data_dir, args.slot)
    models_dir.mkdir(parents=True, exist_ok=True)
    iter_path = slot_iter_snapshot(data_dir, args.slot, args.iter)
    scripted.save(str(iter_path))
    print(f"[export_model] {args.slot}: wrote {iter_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
