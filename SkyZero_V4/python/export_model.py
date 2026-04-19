#!/usr/bin/env python3
"""
Export a PyTorch training checkpoint to TorchScript for C++ selfplay.

Loads the model from a checkpoint, wraps it in ExportWrapper to produce
tuple output, traces it, and saves as .pt.

Before tracing, recalibrates BatchNorm running stats on a sample of training
data — essential when total training is small relative to KataGo's 50M+ sample
regime, because the running_mean/running_std EMA (momentum=0.001) has not
converged and eval-mode outputs diverge badly from train-mode outputs.
"""

import argparse
import glob
import os

import numpy as np
import torch

import torch.nn as nn

from nets import Model, ExportWrapper
from model_config import CONFIG_BY_NAME


def iter_calibration_batches(data_dir, batch_size, max_batches):
    """Yield float32 (B, C, H, W) input tensors from .npz files under data_dir."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    produced = 0
    for path in files:
        if produced >= max_batches:
            break
        data = np.load(path)
        x = data["encodedInputNCHW"]  # int8, [N, C, H, W]
        n = x.shape[0]
        for i in range(0, n, batch_size):
            if produced >= max_batches:
                break
            chunk = x[i : i + batch_size]
            if chunk.shape[0] < batch_size:
                break  # Skip ragged tail — BN stats are more accurate on a full batch.
            yield torch.from_numpy(chunk.astype(np.float32))
            produced += 1


def recalibrate_batchnorm(model, batches, device):
    """Recalibrate BatchNorm2d running stats using cumulative averaging.

    Sets each BN's momentum to None (PyTorch's cumulative moving average mode),
    zeros the tracked counters, then runs the model in train() + no_grad so
    running_mean / running_var become an unbiased mean of batch stats.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bn_modules:
        print("  No BatchNorm2d modules — skipping calibration")
        return 0

    original_momenta = {m: m.momentum for m in bn_modules}
    for m in bn_modules:
        m.momentum = None  # Triggers cumulative moving average.
        m.reset_running_stats()

    was_training = model.training
    model.train()
    n_seen = 0
    try:
        with torch.no_grad():
            for batch in batches:
                n_seen += 1
                model(batch.to(device))
    finally:
        for m in bn_modules:
            m.momentum = original_momenta[m]
        model.train(was_training)
    return n_seen


def main():
    parser = argparse.ArgumentParser(description="Export SkyZero_V4 model to TorchScript")
    parser.add_argument("-checkpoint", required=True, help="Path to model.ckpt file")
    parser.add_argument("-output", required=True, help="Output .pt TorchScript file path")
    parser.add_argument("-board-size", type=int, default=15)
    parser.add_argument("-num-planes", type=int, default=4)
    parser.add_argument("-model-config", type=str, default="b6c96", help="Model config name")
    parser.add_argument("-calibration-data-dir", type=str, default=None,
                        help="Dir of .npz files used to recalibrate BN stats. "
                             "If unset or missing, calibration is skipped.")
    parser.add_argument("-calibration-batches", type=int, default=500,
                        help="Max batches for BN recalibration. 0 disables.")
    parser.add_argument("-calibration-batch-size", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu")

    model_config = CONFIG_BY_NAME[args.model_config]
    model = Model(model_config, args.board_size, args.num_planes)

    model.load_state_dict(state["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    should_calibrate = (
        args.calibration_batches > 0
        and args.calibration_data_dir
        and os.path.isdir(args.calibration_data_dir)
    )
    if should_calibrate:
        print(f"Recalibrating BN on up to {args.calibration_batches} batches of "
              f"{args.calibration_batch_size} from {args.calibration_data_dir} "
              f"(device={device}) ...")
        model.to(device)
        batches = iter_calibration_batches(
            args.calibration_data_dir,
            args.calibration_batch_size,
            args.calibration_batches,
        )
        n = recalibrate_batchnorm(model, batches, device)
        print(f"  Consumed {n} batches.")
        model.to("cpu")
    else:
        reason = "disabled" if args.calibration_batches <= 0 else (
            "no dir set" if not args.calibration_data_dir else
            f"dir missing: {args.calibration_data_dir}"
        )
        print(f"Skipping BN recalibration ({reason}).")

    model.eval()

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, args.num_planes, args.board_size, args.board_size)

    # Verify forward pass works and sanity-check value output.
    with torch.no_grad():
        out = wrapper(dummy_input)
        print(f"  policy_logits shape: {out[0].shape}")
        print(f"  opp_policy_logits shape: {out[1].shape}")
        print(f"  value_logits shape: {out[2].shape}")
        print(f"  value_error_pred shape: {out[3].shape}")
        probs = torch.softmax(out[2].float(), dim=-1).squeeze(0).tolist()
        print(f"  empty-board value_probs (W/D/L): "
              f"[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]")

    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(args.output)
    print(f"Exported TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
