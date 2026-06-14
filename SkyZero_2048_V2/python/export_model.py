"""Export a per-network state_dict checkpoint to TorchScript for the C++ binary.

V7.1-style: each network lives in <DATA_DIR>/nets/<network>/ with a state_dict
checkpoint (model_latest.pt) and this script traces it to TorchScript:

    nets/<net>/model_latest.pt  ->  nets/<net>/latest.pt        (atomic, the live TS)
                                    nets/<net>/scripted_iter_<iter>.pt  (archival)
                                    nets/<net>/latest.meta.json  (iter/network/value_scale)

The traced module takes (B, NUM_PLANES, 4, 4) float input and returns
(policy_logits[B,4], value[B]) in SCALED units; the C++ side multiplies by
value_scale (passed via --value-scale) to get raw points.

    python export_model.py --data-dir DIR --network b6c96 --iter N
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from model_config import config_from_name
from nets import build_net


def export(data_dir, network: str, iter_: int, device: str = "cpu") -> bool:
    net_dir = pathlib.Path(data_dir) / "nets" / network
    cfg = config_from_name(network)
    cfg.device = device
    net = build_net(cfg).to(device)
    ckpt = torch.load(net_dir / "model_latest.pt", map_location=device, weights_only=False)
    # Prefer SWA (EMA-averaged) weights when present — mirrors the mainline
    # export path. They live under a "module." prefix (torch AveragedModel),
    # plus an "n_averaged" buffer we drop.
    if isinstance(ckpt, dict) and ckpt.get("swa_model_state_dict") is not None:
        swa = ckpt["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa.items() if k.startswith("module.")}
        net.load_state_dict(stripped)
        print("[export] loaded SWA weights")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    net.set_norm_scales()   # fixscale nbt: scales aren't in state_dict (no-op for BN net)
    net.eval()

    example = torch.zeros(1, cfg.num_planes, 4, 4, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(net, example)

    out = net_dir / "latest.pt"
    tmp = out.with_suffix(".pt.tmp")
    traced.save(str(tmp))
    tmp.replace(out)                                   # atomic publish
    traced.save(str(net_dir / f"scripted_iter_{iter_:06d}.pt"))

    meta = {"iter": iter_, "network": network,
            "value_scale": cfg.value_scale, "mtime": time.time()}
    meta_tmp = net_dir / "latest.meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta))
    meta_tmp.replace(net_dir / "latest.meta.json")

    # reload sanity: traced output must match the eager net.
    reloaded = torch.jit.load(str(out), map_location=device)
    with torch.no_grad():
        p0, v0 = net(example)
        p1, v1 = reloaded(example)
    ok = bool(torch.allclose(p0, p1, atol=1e-5) and torch.allclose(v0, v1, atol=1e-5))
    print(f"[export] {network} iter={iter_} -> {out} "
          f"(value_scale={cfg.value_scale}, reload_match={ok})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    export(args.data_dir, args.network, args.iter, args.device)


if __name__ == "__main__":
    main()
