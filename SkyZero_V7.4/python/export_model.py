"""Export a trained nn.Module checkpoint as TorchScript for C++ selfplay.

Per-network: train.py saves state_dict to data/nets/<network>/model_latest.pt,
this script loads it, traces to TorchScript, and atomically replaces
data/nets/<network>/latest.pt (plus writes a per-iter snapshot). The active
network's latest.pt is mirrored to data/models/latest.pt by run.sh so the
C++ selfplay binary keeps consuming a stable path.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

from log_util import tag
from model_config import net_config_from_name
from nets import build_model

TAG = tag("Export")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="state_dict .pt path (default: <data>/nets/<network>/model_latest.pt)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--network", required=True,
                        help="Network name, e.g. b5c128. Selects nets/<name>/ "
                             "namespace and parses architecture via "
                             "net_config_from_name.")
    parser.add_argument("--iter", type=int, required=True)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir or os.environ.get("DATA_DIR", "data"))
    net_dir = data_dir / "nets" / args.network
    net_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else (net_dir / "model_latest.pt")

    cfg = net_config_from_name(args.network)
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    # Prefer SWA weights when present — mirrors KataGomo's export path
    # (load_model.load_swa_model_state_dict). `swa_model_state_dict` comes
    # from a torch.optim.swa_utils.AveragedModel, whose params live under
    # the "module." prefix plus an "n_averaged" buffer we strip.
    if isinstance(state, dict) and state.get("swa_model_state_dict") is not None:
        swa_sd = state["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa_sd.items() if k.startswith("module.")}
        model.load_state_dict(stripped)
        print(f"{TAG} loaded SWA weights from {ckpt_path}")
    else:
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"{TAG} loaded raw weights from {ckpt_path}")
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

    iter_path = net_dir / f"scripted_iter_{args.iter:06d}.pt"
    scripted.save(str(iter_path))

    latest = net_dir / "latest.pt"
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    scripted.save(str(tmp))
    os.replace(tmp, latest)

    # Sidecar consumed by the selfplay daemon to tag NPZ filenames with the
    # model version it was generated under. Atomic via tmp+os.replace so a
    # daemon polling latest.pt mtime never sees a torn meta. run.sh mirrors
    # the active network's pair (latest.pt + latest.meta.json) into
    # data/models/ so the C++ binary keeps reading a stable path.
    meta = {"iter": int(args.iter), "network": args.network,
            "mtime": latest.stat().st_mtime}
    meta_path = net_dir / "latest.meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with meta_tmp.open("w") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)

    print(f"{TAG} wrote {iter_path} and updated {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
