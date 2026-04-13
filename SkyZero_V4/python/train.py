#!/usr/bin/env python3
"""
SkyZero_V4 Training Script.

Loads shuffled NPZ data, trains the neural network for one epoch, and exports
a checkpoint for the export script to convert to TorchScript.
"""

import sys
import os
import argparse
import time
import logging
import json
import datetime
import shutil
import glob
import math
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast

from nets import ResNet
from data_processing import read_npz_training_data, collect_npz_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_cross_entropy(logits, targets, weights):
    """Weighted cross-entropy: mean(weights * sum(-targets * log_softmax(logits), dim=-1))"""
    log_probs = torch.log_softmax(logits, dim=-1)
    per_sample = -torch.sum(targets * log_probs, dim=-1)
    return torch.mean(weights * per_sample)


def save_checkpoint(model, swa_model, optimizer, scaler, train_state, path):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_state": train_state,
    }
    if swa_model is not None:
        state["swa_model"] = swa_model.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, model, swa_model, optimizer, scaler, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if swa_model is not None and "swa_model" in state:
        swa_model.load_state_dict(state["swa_model"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    train_state = state.get("train_state", {})
    logging.info(f"Loaded checkpoint from {path}")
    return train_state


def main():
    parser = argparse.ArgumentParser(description="SkyZero_V4 Training")
    parser.add_argument("-traindir", required=True, help="Directory for training state and logs")
    parser.add_argument("-datadir", required=True, help="Directory with train/ and val/ subdirs of shuffled NPZ")
    parser.add_argument("-exportdir", required=False, help="Directory to export checkpoints for model export")
    parser.add_argument("-exportprefix", default="skyzero", help="Prefix for exported model names")
    parser.add_argument("-pos-len", type=int, required=True, help="Board size (e.g. 15)")
    parser.add_argument("-batch-size", type=int, required=True)
    parser.add_argument("-num-planes", type=int, default=4, help="Number of input planes")
    parser.add_argument("-num-blocks", type=int, default=4)
    parser.add_argument("-num-channels", type=int, default=128)
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-weight-decay", type=float, default=3e-5)
    parser.add_argument("-max-grad-norm", type=float, default=1.0)
    parser.add_argument("-max-epochs-this-instance", type=int, default=1)
    parser.add_argument("-samples-per-epoch", type=int, default=None, help="Cap samples per epoch (None = all data)")
    parser.add_argument("-use-fp16", action="store_true")
    parser.add_argument("-swa-scale", type=float, default=None, help="EMA scale for SWA (e.g. 1.0 means new_factor=1.0)")
    parser.add_argument("-lookahead-k", type=int, default=None, help="Lookahead steps (e.g. 6)")
    parser.add_argument("-lookahead-alpha", type=float, default=0.5)
    parser.add_argument("-policy-loss-weight", type=float, default=1.0)
    parser.add_argument("-opp-policy-loss-weight", type=float, default=0.15)
    parser.add_argument("-value-loss-weight", type=float, default=1.0)
    parser.add_argument("-no-export", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.traindir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    pos_len = args.pos_len
    batch_size = args.batch_size
    board_area = pos_len * pos_len

    # Create model
    model = ResNet(pos_len, args.num_planes, args.num_blocks, args.num_channels)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # FP16
    scaler = GradScaler() if args.use_fp16 else None

    # SWA
    swa_model = None
    if args.swa_scale is not None:
        new_factor = 1.0 / args.swa_scale if args.swa_scale > 0 else 1.0
        ema_avg = lambda avg_param, cur_param, num_averaged: avg_param + new_factor * (cur_param - avg_param)
        swa_model = AveragedModel(model, avg_fn=ema_avg)

    # Load existing checkpoint if present
    train_state = {"global_step_samples": 0, "epoch": 0, "swa_sample_accum": 0.0}
    ckpt_path = os.path.join(args.traindir, "latest.ckpt")
    if os.path.exists(ckpt_path):
        train_state = load_checkpoint(ckpt_path, model, swa_model, optimizer, scaler, device)

    # Lookahead cache
    lookahead_cache = None
    if args.lookahead_k is not None:
        lookahead_cache = {}
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                lookahead_cache[param] = param.data.clone()

    # Training metrics log
    train_log_path = os.path.join(args.traindir, "train_metrics.json")

    # Main training loop
    for epoch_idx in range(args.max_epochs_this_instance):
        train_state["epoch"] = train_state.get("epoch", 0) + 1
        logging.info(f"=== Epoch {train_state['epoch']} ===")

        # Collect training data files
        tdatadir = os.path.join(args.datadir, "train")
        vdatadir = os.path.join(args.datadir, "val")

        train_files = collect_npz_files(tdatadir) if os.path.exists(tdatadir) else []
        if len(train_files) == 0:
            logging.warning("No training data files found!")
            break

        np.random.shuffle(train_files)

        # Training
        model.train()
        running_loss = defaultdict(float)
        batch_count = 0
        samples_this_epoch = 0
        t0 = time.perf_counter()
        lookahead_counter = 0

        for batch in read_npz_training_data(train_files, batch_size, pos_len, device, randomize_symmetries=True):
            encoded = batch["encodedInputNCHW"]          # [B, C, H, W]
            policy_targets = batch["policyTargetsN"]     # [B, board_area]
            opp_policy_targets = batch["opponentPolicyTargetsN"]  # [B, board_area]
            value_targets = batch["valueTargetsN"]       # [B, 3]
            sample_weights = batch["sampleWeightsN"]     # [B]

            optimizer.zero_grad()

            if args.use_fp16:
                with autocast():
                    outputs = model(encoded)
                    policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                    opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                    value_logits = outputs["value_logits"]

                    p_loss = weighted_cross_entropy(policy_logits, policy_targets, sample_weights)
                    o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, sample_weights)
                    v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)
                    total_loss = args.policy_loss_weight * p_loss + args.opp_policy_loss_weight * o_loss + args.value_loss_weight * v_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(encoded)
                policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                value_logits = outputs["value_logits"]

                p_loss = weighted_cross_entropy(policy_logits, policy_targets, sample_weights)
                o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, sample_weights)
                v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)
                total_loss = args.policy_loss_weight * p_loss + args.opp_policy_loss_weight * o_loss + args.value_loss_weight * v_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            batch_count += 1
            samples_this_epoch += batch_size
            train_state["global_step_samples"] += batch_size

            running_loss["policy"] += p_loss.item()
            running_loss["opp_policy"] += o_loss.item()
            running_loss["value"] += v_loss.item()
            running_loss["total"] += total_loss.item()

            # Lookahead
            in_between_lookaheads = False
            if args.lookahead_k is not None:
                lookahead_counter += 1
                if lookahead_counter >= args.lookahead_k:
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            slow = lookahead_cache[param]
                            slow.add_(param.data.detach() - slow, alpha=args.lookahead_alpha)
                            param.data.copy_(slow)
                    lookahead_counter = 0
                else:
                    in_between_lookaheads = True

            # SWA
            if swa_model is not None and not in_between_lookaheads:
                train_state["swa_sample_accum"] = train_state.get("swa_sample_accum", 0) + batch_size
                swa_period = max(batch_size * 10, 10000)  # Update SWA every ~10k samples
                if train_state["swa_sample_accum"] >= swa_period:
                    train_state["swa_sample_accum"] = 0
                    swa_model.update_parameters(model)

            # Log every 100 batches
            if batch_count % 100 == 0:
                avg = {k: v / 100 for k, v in running_loss.items()}
                t1 = time.perf_counter()
                speed = (100 * batch_size) / (t1 - t0)
                logging.info(
                    f"  step={batch_count} samples={samples_this_epoch} "
                    f"loss={avg['total']:.4f} p={avg['policy']:.4f} o={avg['opp_policy']:.4f} v={avg['value']:.4f} "
                    f"speed={speed:.0f} samp/s"
                )
                # Write to JSON log
                with open(train_log_path, "a") as f:
                    log_entry = {
                        "step": batch_count,
                        "global_samples": train_state["global_step_samples"],
                        "epoch": train_state["epoch"],
                        "policy_loss": avg["policy"],
                        "opp_policy_loss": avg["opp_policy"],
                        "value_loss": avg["value"],
                        "total_loss": avg["total"],
                        "speed": speed,
                        "time": datetime.datetime.now().isoformat(),
                    }
                    f.write(json.dumps(log_entry) + "\n")
                running_loss = defaultdict(float)
                t0 = time.perf_counter()

            # Cap samples per epoch
            if args.samples_per_epoch is not None and samples_this_epoch >= args.samples_per_epoch:
                break

        # Discard lookahead in-flight updates at epoch end
        if args.lookahead_k is not None:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    param.data.copy_(lookahead_cache[param])

        logging.info(f"Epoch done: {batch_count} batches, {samples_this_epoch} samples")

        # Save checkpoint
        save_checkpoint(model, swa_model, optimizer, scaler, train_state, ckpt_path)

        # Validation
        val_files = collect_npz_files(vdatadir) if os.path.exists(vdatadir) else []
        if len(val_files) > 0:
            logging.info("Running validation...")
            model.eval()
            val_loss = defaultdict(float)
            val_count = 0
            with torch.no_grad():
                for batch in read_npz_training_data(val_files, batch_size, pos_len, device, randomize_symmetries=True):
                    encoded = batch["encodedInputNCHW"]
                    policy_targets = batch["policyTargetsN"]
                    opp_policy_targets = batch["opponentPolicyTargetsN"]
                    value_targets = batch["valueTargetsN"]
                    sample_weights = batch["sampleWeightsN"]

                    outputs = model(encoded)
                    policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                    opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                    value_logits = outputs["value_logits"]

                    p_loss = weighted_cross_entropy(policy_logits, policy_targets, sample_weights)
                    o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, sample_weights)
                    v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)

                    val_loss["policy"] += p_loss.item()
                    val_loss["opp_policy"] += o_loss.item()
                    val_loss["value"] += v_loss.item()
                    val_loss["total"] += (p_loss + 0.15 * o_loss + v_loss).item()
                    val_count += 1

            if val_count > 0:
                avg = {k: v / val_count for k, v in val_loss.items()}
                logging.info(
                    f"Validation: loss={avg['total']:.4f} p={avg['policy']:.4f} "
                    f"o={avg['opp_policy']:.4f} v={avg['value']:.4f}"
                )
                val_log_path = os.path.join(args.traindir, "val_metrics.json")
                with open(val_log_path, "a") as f:
                    log_entry = {
                        "epoch": train_state["epoch"],
                        "global_samples": train_state["global_step_samples"],
                        **{f"val_{k}": v for k, v in avg.items()},
                        "time": datetime.datetime.now().isoformat(),
                    }
                    f.write(json.dumps(log_entry) + "\n")
            model.train()

        # Export model for export.sh
        if not args.no_export and args.exportdir is not None:
            modelname = "%s-s%d-e%d" % (args.exportprefix, train_state["global_step_samples"], train_state["epoch"])
            savepath = os.path.join(args.exportdir, modelname)
            savepathtmp = savepath + ".tmp"
            if not os.path.exists(savepath):
                os.makedirs(savepathtmp, exist_ok=True)
                save_checkpoint(model, swa_model, optimizer, scaler, train_state,
                                os.path.join(savepathtmp, "model.ckpt"))
                time.sleep(1)
                os.rename(savepathtmp, savepath)
                logging.info(f"Exported checkpoint to {savepath}")
            else:
                logging.info(f"Export path already exists, skipping: {savepath}")

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
