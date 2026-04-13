#!/usr/bin/env python3
"""Create a randomly-initialized TorchScript model for bootstrapping the training loop."""

import argparse
import torch
from nets import ResNet, ExportWrapper


def main():
    parser = argparse.ArgumentParser(description="Create initial random model for SkyZero_V4")
    parser.add_argument("-output", required=True, help="Output .pt file path")
    parser.add_argument("-board-size", type=int, default=15)
    parser.add_argument("-num-planes", type=int, default=4)
    parser.add_argument("-num-blocks", type=int, default=4)
    parser.add_argument("-num-channels", type=int, default=128)
    args = parser.parse_args()

    model = ResNet(args.board_size, args.num_planes, args.num_blocks, args.num_channels)
    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, args.num_planes, args.board_size, args.board_size)
    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(args.output)

    print(f"Saved initial model to {args.output}")
    print(f"  board_size={args.board_size}, num_planes={args.num_planes}")
    print(f"  num_blocks={args.num_blocks}, num_channels={args.num_channels}")


if __name__ == "__main__":
    main()
