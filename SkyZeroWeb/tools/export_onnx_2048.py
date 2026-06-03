#!/usr/bin/env python3
"""Export a SkyZero_2048 Net2048 checkpoint to ONNX for browser inference.

The 2048 net (python/nets.py:Net2048) takes (B, 16, 4, 4) one-hot exponent
planes and returns (policy_logits[B,4], value[B]) where `value` is in SCALED
units (raw expected-discounted-score / value_scale). The browser AI wants raw
points so its expectimax Q = reward + gamma*V stays in one unit, so this
wrapper folds the value_scale multiply into the graph: the ONNX `value` output
is already in raw 2048 points.

Usage:
    python tools/export_onnx_2048.py \\
        --ckpt ../SkyZero_2048/data2048_td/nets/b3c64/scripted_iter_000007.pt \\
        --net b3c64 --out models/2048.onnx

`--net b3c64` parses to blocks=3, channels=64; all other Config fields (value
scale, widths) use the SkyZero_2048 defaults. Pass --value-scale to override if
the checkpoint was trained with a non-default VALUE_SCALE.

Mirrors tools/export_onnx.py's gotchas: opset 18 (Mish), dynamo=False (so the
weights inline into a single self-contained .onnx rather than a sibling
.onnx.data file that onnxruntime-web can't follow), dynamic batch axis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import onnx


# Locate the SkyZero_2048 python/ package so we can import the net definition.
_2048_PY = Path(__file__).resolve().parents[2] / "SkyZero_2048" / "python"
if not _2048_PY.is_dir():
    raise SystemExit(f"Expected SkyZero_2048 python at {_2048_PY}")
sys.path.insert(0, str(_2048_PY))

from nets import build_net               # noqa: E402
from model_config import config_from_name, Config  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Wraps Net2048 and rescales the value head back to raw 2048 points.

    Two value-target conventions (must match the run.cfg the ckpt was trained
    with), so the ONNX `value` output is always raw 2048 points:
      - linear (value_transform=False): head regresses raw/value_scale, so we
        just multiply by value_scale.
      - MuZero h() (value_transform=True): head regresses h(raw)/value_scale, so
        we multiply by value_scale to recover h(raw) then apply h_inv to get raw
        points. h/h_inv mirror python/value_transform.py (eps = 1e-3).
    """

    EPS = 1e-3

    def __init__(self, model: torch.nn.Module, value_scale: float,
                 value_transform: bool = False) -> None:
        super().__init__()
        self.model = model
        self.value_scale = float(value_scale)
        self.value_transform = bool(value_transform)

    def _h_inv(self, y: torch.Tensor) -> torch.Tensor:
        """h-space -> raw points (exact inverse of value_transform.to_h)."""
        eps = self.EPS
        z = (torch.sqrt(1.0 + 4.0 * eps * (torch.abs(y) + 1.0 + eps)) - 1.0) / (2.0 * eps)
        return torch.sign(y) * (z * z - 1.0)

    def forward(self, x: torch.Tensor):
        policy_logits, value_scaled = self.model(x)
        value_points = value_scaled * self.value_scale          # raw or h-space
        if self.value_transform:
            value_points = self._h_inv(value_points)            # h-space -> raw
        # Force a (B, 1) shape so onnxruntime-web sees a 2-D tensor regardless
        # of how the scalar head squeezes.
        return policy_logits, value_points.reshape(-1, 1)


def load_state(ckpt_path: Path) -> dict:
    """Extract a state_dict from a SkyZero_2048 .pt file.

      - TorchScript module (export_model.py -> scripted_iter_*.pt) -> .state_dict()
      - {"model": state_dict} or {"model_state_dict": ...} wrapper      -> inner
      - bare state_dict (OrderedDict of name -> Tensor)                 -> as-is
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, torch.jit.ScriptModule):
        print(f"[export] loading from TorchScript module {ckpt_path.name}")
        return state.state_dict()
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                print(f"[export] using '{key}' from {ckpt_path.name}")
                return state[key]
        if state and all(isinstance(v, torch.Tensor) for v in state.values()):
            print(f"[export] loading bare state_dict from {ckpt_path.name}")
            return state
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def export_one(ckpt: Path, out: Path, net: str,
               value_scale: float | None, value_transform: bool, opset: int) -> None:
    cfg: Config = config_from_name(net)
    if value_scale is not None:
        cfg.value_scale = value_scale
    cfg.value_transform = value_transform
    print(f"[export] net={net} blocks={cfg.blocks} channels={cfg.channels} "
          f"c_mid={cfg.c_mid} c_gpool={cfg.c_gpool} value_scale={cfg.value_scale} "
          f"value_transform={cfg.value_transform}")

    model = build_net(cfg)
    state_dict = load_state(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[export] missing keys: {missing}")
    if unexpected:
        print(f"[export] unexpected keys: {unexpected}")
    model.eval()

    wrapper = ExportWrapper(model, cfg.value_scale, cfg.value_transform).eval()

    dummy = torch.zeros(1, cfg.num_planes, 4, 4, dtype=torch.float32)
    with torch.no_grad():
        p, v = wrapper(dummy)
        print(f"[export] policy: {tuple(p.shape)}, value: {tuple(v.shape)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "input":         {0: "B"},
            "policy_logits": {0: "B"},
            "value":         {0: "B"},
        },
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))
    print(f"[export] wrote {out}  (check passed)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="source checkpoint (.pt)")
    ap.add_argument("--out", type=Path, required=True, help="output .onnx path")
    ap.add_argument("--net", type=str, required=True, help="net name, e.g. b3c64")
    ap.add_argument("--value-scale", type=float, default=None,
                    help="override Config.value_scale (default: net's config)")
    ap.add_argument("--value-transform", action="store_true",
                    help="ckpt trained with VALUE_TRANSFORM=1 (h-space target); "
                         "fold the MuZero h_inv into the graph so value stays raw")
    # opset 18 is the minimum that supports Mish; onnxruntime-web 1.17 has it.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        ap.error(f"checkpoint not found: {args.ckpt}")
    export_one(args.ckpt, args.out, args.net, args.value_scale,
               args.value_transform, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
