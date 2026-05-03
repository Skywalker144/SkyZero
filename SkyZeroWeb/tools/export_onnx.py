#!/usr/bin/env python3
"""Export SkyZero V5 checkpoints to ONNX for browser inference.

Wraps SkyZero_V5's nets.KataGoNet, drops the value_td and intermediate_*
outputs (UI doesn't display them), and reorders to a (1, 4, 17, 17) +
(1, 12) → (policy_logits, value_wdl_logits, value_futurepos_pretanh)
signature. Spatial dims fixed at MAX_BOARD_SIZE so the same .onnx serves
all board sizes 13-17 via the mask plane.

Two usage modes:

    # Single file (explicit):
    python tools/export_onnx.py \\
        --ckpt /path/to/model.pt --out models/level3.onnx \\
        --num-blocks 10 --num-channels 128

    # Batch (auto-detects b{N}c{M} from filenames):
    python tools/export_onnx.py --src /path/to/v5/anchors \\
        b4c64iter180.pt b10c128iter80.pt b10c128b15iter234.pt
        # → models/level1.onnx, models/level2.onnx, models/level3.onnx

    # Batch with explicit slot mapping:
    python tools/export_onnx.py --src /path/to/v5/anchors \\
        b4c64iter180.pt:lv1 b10c128iter80.pt:lv3 b10c128b15iter234.pt:lv5
        # → models/level1.onnx, models/level3.onnx, models/level5.onnx
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import onnx


# Make SkyZero_V5/python importable regardless of CWD.
SKYZERO_V5_PY = Path(__file__).resolve().parents[2] / "SkyZero_V5" / "python"
if not SKYZERO_V5_PY.is_dir():
    raise SystemExit(f"Expected V5 python at {SKYZERO_V5_PY}")
sys.path.insert(0, str(SKYZERO_V5_PY))

from nets import build_model               # noqa: E402
from model_config import NetConfig         # noqa: E402


# Default output dir for batch mode (SkyZeroWeb/models/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

# Filename arch regex: matches "b10c128", "b4c64", etc.
ARCH_RE = re.compile(r"b(\d+)c(\d+)")


class ExportWrapper(torch.nn.Module):
    """Wraps KataGoNet and emits only the heads the web UI consumes."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor):
        out = self.model(input_spatial, input_global)
        # Drop: value_td (9), intermediate_* (4 keys). Keep: policy / wdl / futurepos.
        return (
            out["policy"],              # (B, 4, H*W)
            out["value_wdl"],           # (B, 3) — logits
            out["value_futurepos"],     # (B, 2, H, W) — pre-tanh
        )


def load_state(ckpt_path: Path) -> dict:
    """Extract a state_dict from a V5 .pt file.

    Handles four V5 .pt flavors:
      - Bare state_dict (OrderedDict of name → Tensor)   → use directly
        (V5 selfplay daemon writes data/.../checkpoints/model_iter_NNNNNN.pt this way)
      - Wrapped checkpoint with `swa_model_state_dict`   → strip 'module.' prefix
      - Wrapped checkpoint with `model_state_dict`       → return inner dict
      - TorchScript module (anchors/, models/latest.pt)  → call .state_dict()
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # TorchScript module case (V5 anchors/ and data/.../models/latest.pt are scripted)
    if isinstance(state, torch.jit.ScriptModule):
        print(f"[export_onnx] loading from TorchScript module at {ckpt_path}")
        return state.state_dict()

    if isinstance(state, dict) and state.get("swa_model_state_dict") is not None:
        swa_sd = state["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa_sd.items() if k.startswith("module.")}
        print(f"[export_onnx] using SWA weights from {ckpt_path}")
        return stripped
    if isinstance(state, dict) and "model_state_dict" in state:
        print(f"[export_onnx] using regular model_state_dict from {ckpt_path}")
        return state["model_state_dict"]

    # Bare state_dict: every value is a Tensor and at least one key looks like a
    # KataGoNet param name. This is what V5's selfplay daemon writes for
    # data/.../checkpoints/model_iter_NNNNNN.pt — the raw model.state_dict().
    if isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values()):
        print(f"[export_onnx] loading bare state_dict from {ckpt_path}")
        return state

    raise ValueError(
        f"Checkpoint {ckpt_path} is not a recognized format "
        "(expected ScriptModule, swa_model_state_dict, model_state_dict, or bare state_dict)"
    )


def detect_arch(filename: str) -> tuple[int, int] | None:
    """Parse 'b10c128iter80.pt' → (10, 128). Returns None if unparseable."""
    m = ARCH_RE.search(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def export_one(
    ckpt: Path,
    out: Path,
    num_blocks: int,
    num_channels: int,
    max_board_size: int = 17,
    num_planes: int = 5,
    num_global_features: int = 12,
    opset: int = 18,
) -> None:
    """Export one V5 checkpoint to ONNX. Used by both single and batch modes."""
    cfg = NetConfig(
        board_size=max_board_size,
        num_planes=num_planes,
        num_blocks=num_blocks,
        num_channels=num_channels,
        num_global_features=num_global_features,
    )
    model = build_model(cfg)

    state_dict = load_state(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[export_onnx] missing keys: {missing}")
    if unexpected:
        print(f"[export_onnx] unexpected keys: {unexpected}")

    # V5 trap 3: NormMask scales not in state_dict → re-derive from arch.
    model.set_norm_scales()
    model.eval()

    wrapper = ExportWrapper(model).eval()

    M = max_board_size
    spatial = torch.zeros(1, num_planes, M, M, dtype=torch.float32)
    spatial[:, 0] = 1.0   # mask: full board
    global_in = torch.zeros(1, num_global_features, dtype=torch.float32)

    with torch.no_grad():
        p, w, f = wrapper(spatial, global_in)
        print(f"[export_onnx] policy: {tuple(p.shape)}, wdl: {tuple(w.shape)}, futurepos: {tuple(f.shape)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: PyTorch 2.9's new dynamo exporter splits weights into a
    # sibling .onnx.data file (external-data format). onnxruntime-web loads
    # from in-memory bytes and can't follow filesystem .data references → the
    # browser would fail. The legacy TorchScript-based exporter inlines
    # everything into a single self-contained .onnx file. Use it.
    torch.onnx.export(
        wrapper,
        (spatial, global_in),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_spatial", "input_global"],
        output_names=["policy_logits", "value_wdl_logits", "value_futurepos_pretanh"],
        dynamic_axes={
            "input_spatial":              {0: "B"},
            "input_global":               {0: "B"},
            "policy_logits":              {0: "B"},
            "value_wdl_logits":           {0: "B"},
            "value_futurepos_pretanh":    {0: "B"},
        },
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(out)))
    print(f"[export_onnx] wrote {out}  (check passed)")


def parse_batch_spec(spec: str, default_idx: int) -> tuple[str, str]:
    """Parse 'filename.pt' or 'filename.pt:lvN' into (filename, tier_id).

    Without ':lvN', tier_id defaults to f'lv{default_idx + 1}'.
    """
    if ":" in spec:
        filename, tier = spec.rsplit(":", 1)
        if not re.fullmatch(r"lv\d+", tier):
            raise ValueError(f"Bad tier id '{tier}' in '{spec}' — expected lv1, lv2, ...")
        return filename, tier
    return spec, f"lv{default_idx + 1}"


def tier_to_outname(tier_id: str) -> str:
    """'lv3' → 'level3.onnx'."""
    n = tier_id[2:]   # strip 'lv'
    return f"level{n}.onnx"


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # Single-file mode (legacy, still supported)
    ap.add_argument("--ckpt", type=Path, help="single-file mode: source checkpoint")
    ap.add_argument("--out",  type=Path, help="single-file mode: output .onnx path")

    # Batch mode
    ap.add_argument("--src", type=Path, default=None,
                    help="batch mode: source dir containing .pt files")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR,
                    help=f"batch mode: output dir (default: {DEFAULT_MODELS_DIR})")
    ap.add_argument("files", nargs="*",
                    help="batch mode: filenames (e.g., 'b10c128iter80.pt' or "
                         "'b10c128iter80.pt:lv3')")

    # Architecture overrides (single-file mode; batch auto-detects)
    ap.add_argument("--num-blocks",   type=int, default=None)
    ap.add_argument("--num-channels", type=int, default=None)
    ap.add_argument("--max-board-size", type=int, default=17)
    ap.add_argument("--num-planes", type=int, default=5)
    ap.add_argument("--num-global-features", type=int, default=12)
    # opset 18 is the minimum that supports Mish (V5's default activation).
    # ORT 1.16+ supports it; jsdelivr's onnxruntime-web 1.17 does too.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    # ----- Single-file mode -----
    if args.ckpt is not None:
        if args.out is None:
            ap.error("--ckpt requires --out")
        nb = args.num_blocks   if args.num_blocks   is not None else 10
        nc = args.num_channels if args.num_channels is not None else 128
        export_one(
            ckpt=args.ckpt, out=args.out,
            num_blocks=nb, num_channels=nc,
            max_board_size=args.max_board_size,
            num_planes=args.num_planes,
            num_global_features=args.num_global_features,
            opset=args.opset,
        )
        return 0

    # ----- Batch mode -----
    if not args.files:
        ap.error("provide either --ckpt + --out (single mode) "
                 "or --src + filenames (batch mode); see --help")
    if args.src is None:
        ap.error("batch mode requires --src DIR")
    if not args.src.is_dir():
        ap.error(f"--src dir not found: {args.src}")

    # Resolve each spec
    plan = []
    for i, spec in enumerate(args.files):
        filename, tier = parse_batch_spec(spec, i)
        ckpt = args.src / filename
        if not ckpt.is_file():
            ap.error(f"file not found: {ckpt}")
        out = args.out_dir / tier_to_outname(tier)

        if args.num_blocks is not None and args.num_channels is not None:
            nb, nc = args.num_blocks, args.num_channels
        else:
            arch = detect_arch(filename)
            if arch is None:
                ap.error(f"cannot auto-detect arch from '{filename}' "
                         "(need 'b{N}c{M}' in name) — pass --num-blocks/--num-channels explicitly")
            nb, nc = arch
        plan.append((ckpt, out, nb, nc, tier))

    def _short(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    print(f"[export_onnx] batch plan ({len(plan)} files):")
    for ckpt, out, nb, nc, tier in plan:
        print(f"  {tier:>4}  b{nb}c{nc}  {ckpt.name}  →  {_short(out)}")
    print()

    for i, (ckpt, out, nb, nc, tier) in enumerate(plan, 1):
        print(f"--- [{i}/{len(plan)}] {tier} : {ckpt.name} (b{nb}c{nc}) ---")
        export_one(
            ckpt=ckpt, out=out,
            num_blocks=nb, num_channels=nc,
            max_board_size=args.max_board_size,
            num_planes=args.num_planes,
            num_global_features=args.num_global_features,
            opset=args.opset,
        )
        print()

    print(f"[export_onnx] batch done — wrote {len(plan)} file(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
