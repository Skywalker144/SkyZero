#!/usr/bin/env python3
"""
Export a PyTorch training checkpoint to TorchScript for C++ selfplay.

Loads the SWA model (or regular model) from a checkpoint, wraps it in
ExportWrapper to produce tuple output, traces it, and saves as .pt.
"""

import argparse
import torch
from nets import ResNet, ExportWrapper


def main():
    parser = argparse.ArgumentParser(description="Export SkyZero_V4 model to TorchScript")
    parser.add_argument("-checkpoint", required=True, help="Path to model.ckpt file")
    parser.add_argument("-output", required=True, help="Output .pt TorchScript file path")
    parser.add_argument("-board-size", type=int, default=15)
    parser.add_argument("-num-planes", type=int, default=4)
    parser.add_argument("-num-blocks", type=int, default=4)
    parser.add_argument("-num-channels", type=int, default=128)
    parser.add_argument("-use-swa", action="store_true", help="Use SWA model weights if available")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")

    model = ResNet(args.board_size, args.num_planes, args.num_blocks, args.num_channels)

    if args.use_swa and "swa_model" in state:
        print("Using SWA model weights")
        # AveragedModel wraps with 'module.' prefix and adds n_averaged
        swa_state = state["swa_model"]
        # Strip 'module.' prefix from SWA state dict keys
        cleaned = {}
        for k, v in swa_state.items():
            if k == "n_averaged":
                continue
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            cleaned[new_key] = v
        model.load_state_dict(cleaned)
    elif "model" in state:
        print("Using regular model weights")
        model.load_state_dict(state["model"])
    else:
        raise ValueError("Checkpoint has neither 'model' nor 'swa_model' key")

    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, args.num_planes, args.board_size, args.board_size)

    # Verify forward pass works
    with torch.no_grad():
        out = wrapper(dummy_input)
        print(f"  policy_logits shape: {out[0].shape}")
        print(f"  opp_policy_logits shape: {out[1].shape}")
        print(f"  value_logits shape: {out[2].shape}")

    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(args.output)
    print(f"Exported TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
