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

from nets import Model, ExportWrapper
from model_config import CONFIG_BY_NAME, SKYZERO_B6C96
from data_processing import read_npz_training_data, collect_npz_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_cross_entropy(logits, targets, weights):
    """Weighted cross-entropy: mean(weights * sum(-targets * log_softmax(logits), dim=-1))"""
    log_probs = torch.log_softmax(logits, dim=-1)
    per_sample = -torch.sum(targets * log_probs, dim=-1)
    return torch.mean(weights * per_sample)


def compute_value_error_loss(value_logits, value_error_logit, value_targets, sample_weights, delta=0.4):
    """Shortterm value error head loss.

    Predicts (utility_pred - actual_outcome)^2 with Huber loss. utility_pred and
    actual_outcome are detached from the value_logits gradient path so the
    error head supervises a target derived from the (current) value head's own
    output without distorting it. Reference: KataGomo metrics_pytorch.py:238-245.
    """
    with torch.no_grad():
        value_probs = torch.softmax(value_logits.float(), dim=-1)
        utility_pred = value_probs[:, 0] - value_probs[:, 2]
        actual_outcome = value_targets[:, 0] - value_targets[:, 2]
        actual_sq_err = (utility_pred - actual_outcome) ** 2
    pred = torch.nn.functional.softplus(value_error_logit.float().squeeze(-1))
    huber = torch.nn.functional.huber_loss(pred, actual_sq_err, reduction="none", delta=delta)
    return torch.mean(sample_weights * huber)


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


def compute_warmup_scale(global_step_samples, norm_kind):
    """LR warmup schedule for fixbrenorm (from KataGomo train.py:604-622)."""
    if norm_kind in ("brenorm", "fixbrenorm"):
        if global_step_samples < 250000: return 1.0 / 20.0
        if global_step_samples < 500000: return 1.0 / 14.0
        if global_step_samples < 750000: return 1.0 / 10.0
        if global_step_samples < 1000000: return 1.0 / 7.0
        if global_step_samples < 1250000: return 1.0 / 5.0
        if global_step_samples < 1500000: return 1.0 / 3.0
        if global_step_samples < 1750000: return 1.0 / 2.0
        if global_step_samples < 2000000: return 1.0 / 1.4
        return 1.0
    else:
        # fixup/fixscale warmup
        if global_step_samples < 1000000: return 1.0 / 5.0
        if global_step_samples < 2000000: return 1.0 / 3.0
        if global_step_samples < 4000000: return 1.0 / 2.0
        if global_step_samples < 6000000: return 1.0 / 1.4
        return 1.0


def lr_scale_auto_factor(global_step_samples):
    """Step-wise LR decay over long training (from KataGomo train.py:257-269)."""
    if global_step_samples < 60000000: return 8.0
    if global_step_samples < 110000000: return 4.0
    if global_step_samples < 160000000: return 2.0
    if global_step_samples < 200000000: return 1.0
    return 0.25


def compute_adaptive_gnorm_cap(norm_kind, lr_scale, global_step_samples):
    """Adaptive gradient clipping (from KataGomo train.py:1068-1089)."""
    if norm_kind in ("fixup", "fixscale"):
        gnorm_cap = 2500.0
    else:
        gnorm_cap = 5500.0
    effective_lr_scale = lr_scale * lr_scale_auto_factor(global_step_samples)
    gnorm_cap = gnorm_cap / math.sqrt(max(1e-7, effective_lr_scale))
    return gnorm_cap


def update_lr(optimizer, base_lr, lr_scale, warmup_scale):
    """Update optimizer LR per parameter group."""
    for param_group in optimizer.param_groups:
        group_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = base_lr * warmup_scale * lr_scale * lr_scale_auto_factor(0) * group_scale


def update_lr_and_wd(optimizer, base_lr, lr_scale, global_step_samples, norm_kind, use_auto_lr):
    """Update LR and weight decay for all parameter groups."""
    warmup = compute_warmup_scale(global_step_samples, norm_kind)
    auto = lr_scale_auto_factor(global_step_samples) if use_auto_lr else 1.0
    for param_group in optimizer.param_groups:
        group_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = base_lr * warmup * lr_scale * auto * group_scale
    return warmup, auto


def maybe_update_brenorm_params(model, train_state, last_brenorm_update, norm_kind,
                                 brenorm_target_rmax, brenorm_target_dmax,
                                 brenorm_avg_momentum, brenorm_adjustment_scale):
    """Gradually adjust brenorm rmax/dmax (from KataGomo train.py:665-680)."""
    if norm_kind not in ("brenorm", "fixbrenorm"):
        return last_brenorm_update

    if "brenorm_rmax" not in train_state:
        train_state["brenorm_rmax"] = 1.0
    if "brenorm_dmax" not in train_state:
        train_state["brenorm_dmax"] = 0.0

    num_samples_elapsed = train_state["global_step_samples"] - last_brenorm_update
    factor = math.exp(-num_samples_elapsed / brenorm_adjustment_scale)
    train_state["brenorm_rmax"] += (1.0 - factor) * (brenorm_target_rmax - train_state["brenorm_rmax"])
    train_state["brenorm_dmax"] += (1.0 - factor) * (brenorm_target_dmax - train_state["brenorm_dmax"])

    model.set_brenorm_params(brenorm_avg_momentum, train_state["brenorm_rmax"], train_state["brenorm_dmax"])
    return train_state["global_step_samples"]


def main():
    parser = argparse.ArgumentParser(description="SkyZero_V4 Training")
    parser.add_argument("-traindir", required=True, help="Directory for training state and logs")
    parser.add_argument("-datadir", required=True, help="Directory with train/ and val/ subdirs of shuffled NPZ")
    parser.add_argument("-exportdir", required=False, help="Directory to export checkpoints for model export")
    parser.add_argument("-exportprefix", default="skyzero", help="Prefix for exported model names")
    parser.add_argument("-pos-len", type=int, required=True, help="Board size (e.g. 15)")
    parser.add_argument("-batch-size", type=int, required=True)
    parser.add_argument("-num-planes", type=int, default=4, help="Number of input planes")
    parser.add_argument("-model-config", type=str, default="b6c96", help="Model config name (b6c96, b4c32, b10c128)")
    parser.add_argument("-lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("-lr-scale", type=float, default=1.0, help="LR scale multiplier")
    parser.add_argument("-lr-scale-auto", action="store_true", help="Enable auto LR decay over training")
    parser.add_argument("-weight-decay", type=float, default=3e-5)
    parser.add_argument("-max-epochs-this-instance", type=int, default=1)
    parser.add_argument("-samples-per-epoch", type=int, default=None, help="Cap samples per epoch (None = all data)")
    parser.add_argument("-use-fp16", action="store_true")
    parser.add_argument("-swa-scale", type=float, default=None, help="EMA scale for SWA (e.g. 1.0 means new_factor=1.0)")
    parser.add_argument("-lookahead-k", type=int, default=None, help="Lookahead steps (e.g. 6)")
    parser.add_argument("-lookahead-alpha", type=float, default=0.5)
    parser.add_argument("-policy-loss-weight", type=float, default=1.0)
    parser.add_argument("-opp-policy-loss-weight", type=float, default=0.15)
    parser.add_argument("-value-loss-weight", type=float, default=1.2)
    parser.add_argument("-value-error-loss-weight", type=float, default=2.0,
                        help="Weight for shortterm value-error head (KataGo-style)")
    parser.add_argument("-brenorm-target-rmax", type=float, default=3.0)
    parser.add_argument("-brenorm-target-dmax", type=float, default=5.0)
    parser.add_argument("-brenorm-avg-momentum", type=float, default=0.001)
    parser.add_argument("-brenorm-adjustment-scale", type=float, default=50000000)
    parser.add_argument("-no-export", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.traindir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    pos_len = args.pos_len
    batch_size = args.batch_size
    board_area = pos_len * pos_len

    # Create model
    model_config = CONFIG_BY_NAME[args.model_config]
    norm_kind = model_config["norm_kind"]
    model = Model(model_config, pos_len, args.num_planes)
    model.initialize()
    model.to(device)
    model.train()

    # Parameter groups with different weight decay and LR scales
    reg_dict = {}
    model.add_reg_dict(reg_dict)
    wd = args.weight_decay
    param_groups = [
        {"params": reg_dict["normal"], "weight_decay": wd, "lr_scale": 1.0, "group_name": "normal"},
        {"params": reg_dict["normal_gamma"], "weight_decay": wd * 0.5, "lr_scale": 1.0, "group_name": "normal_gamma"},
        {"params": reg_dict["output"], "weight_decay": wd * 0.5, "lr_scale": 0.5, "group_name": "output"},
        {"params": reg_dict["noreg"], "weight_decay": 0.0, "lr_scale": 1.0, "group_name": "noreg"},
        {"params": reg_dict["output_noreg"], "weight_decay": 0.0, "lr_scale": 0.5, "group_name": "output_noreg"},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

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

    # Initialize brenorm params
    last_brenorm_update_samples = train_state.get("global_step_samples", 0)
    last_brenorm_update_samples = maybe_update_brenorm_params(
        model, train_state, last_brenorm_update_samples, norm_kind,
        args.brenorm_target_rmax, args.brenorm_target_dmax,
        args.brenorm_avg_momentum, args.brenorm_adjustment_scale,
    )

    # Initialize LR
    update_lr_and_wd(optimizer, args.lr, args.lr_scale,
                     train_state.get("global_step_samples", 0), norm_kind, args.lr_scale_auto)

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
            policy_weights = batch["policyWeightsN"]              # [B] (PCR)
            opp_policy_weights = batch["oppPolicyWeightsN"]       # [B] (PCR)
            policy_sw = sample_weights * policy_weights
            opp_policy_sw = sample_weights * opp_policy_weights

            optimizer.zero_grad()

            if args.use_fp16:
                with autocast():
                    outputs = model(encoded)
                    policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                    opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                    value_logits = outputs["value_logits"]
                    value_error_logit = outputs["value_error_logit"]

                    p_loss = weighted_cross_entropy(policy_logits, policy_targets, policy_sw)
                    o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, opp_policy_sw)
                    v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)
                    ve_loss = compute_value_error_loss(value_logits, value_error_logit, value_targets, sample_weights)
                    total_loss = (args.policy_loss_weight * p_loss
                                  + args.opp_policy_loss_weight * o_loss
                                  + args.value_loss_weight * v_loss
                                  + args.value_error_loss_weight * ve_loss)

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                gnorm_cap = compute_adaptive_gnorm_cap(norm_kind, args.lr_scale, train_state["global_step_samples"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), gnorm_cap)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(encoded)
                policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                value_logits = outputs["value_logits"]
                value_error_logit = outputs["value_error_logit"]

                p_loss = weighted_cross_entropy(policy_logits, policy_targets, policy_sw)
                o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, opp_policy_sw)
                v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)
                ve_loss = compute_value_error_loss(value_logits, value_error_logit, value_targets, sample_weights)
                total_loss = (args.policy_loss_weight * p_loss
                              + args.opp_policy_loss_weight * o_loss
                              + args.value_loss_weight * v_loss
                              + args.value_error_loss_weight * ve_loss)

                total_loss.backward()
                gnorm_cap = compute_adaptive_gnorm_cap(norm_kind, args.lr_scale, train_state["global_step_samples"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), gnorm_cap)
                optimizer.step()

            batch_count += 1
            samples_this_epoch += batch_size
            train_state["global_step_samples"] += batch_size

            running_loss["policy"] += p_loss.item()
            running_loss["opp_policy"] += o_loss.item()
            running_loss["value"] += v_loss.item()
            running_loss["value_error"] += ve_loss.item()
            running_loss["total"] += total_loss.item()

            # Update LR every 10 batches (during first 50M samples)
            if batch_count % 10 == 0 and train_state["global_step_samples"] <= 50000000:
                update_lr_and_wd(optimizer, args.lr, args.lr_scale,
                                 train_state["global_step_samples"], norm_kind, args.lr_scale_auto)

            # Update brenorm params every 500 batches
            if batch_count % 500 == 0:
                last_brenorm_update_samples = maybe_update_brenorm_params(
                    model, train_state, last_brenorm_update_samples, norm_kind,
                    args.brenorm_target_rmax, args.brenorm_target_dmax,
                    args.brenorm_avg_momentum, args.brenorm_adjustment_scale,
                )

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
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"  step={batch_count} samples={samples_this_epoch} "
                    f"loss={avg['total']:.4f} p={avg['policy']:.4f} o={avg['opp_policy']:.4f} "
                    f"v={avg['value']:.4f} ve={avg['value_error']:.4f} "
                    f"lr={current_lr:.2e} speed={speed:.0f} samp/s"
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
                        "value_error_loss": avg["value_error"],
                        "total_loss": avg["total"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "brenorm_rmax": train_state.get("brenorm_rmax", 1.0),
                        "brenorm_dmax": train_state.get("brenorm_dmax", 0.0),
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
                    policy_weights = batch["policyWeightsN"]
                    opp_policy_weights = batch["oppPolicyWeightsN"]
                    policy_sw = sample_weights * policy_weights
                    opp_policy_sw = sample_weights * opp_policy_weights

                    outputs = model(encoded)
                    policy_logits = outputs["policy_logits"].reshape(-1, board_area)
                    opp_logits = outputs["opponent_policy_logits"].reshape(-1, board_area)
                    value_logits = outputs["value_logits"]
                    value_error_logit = outputs["value_error_logit"]

                    p_loss = weighted_cross_entropy(policy_logits, policy_targets, policy_sw)
                    o_loss = weighted_cross_entropy(opp_logits, opp_policy_targets, opp_policy_sw)
                    v_loss = weighted_cross_entropy(value_logits, value_targets, sample_weights)
                    ve_loss = compute_value_error_loss(value_logits, value_error_logit, value_targets, sample_weights)

                    val_loss["policy"] += p_loss.item()
                    val_loss["opp_policy"] += o_loss.item()
                    val_loss["value"] += v_loss.item()
                    val_loss["value_error"] += ve_loss.item()
                    val_loss["total"] += (
                        args.policy_loss_weight * p_loss
                        + args.opp_policy_loss_weight * o_loss
                        + args.value_loss_weight * v_loss
                        + args.value_error_loss_weight * ve_loss
                    ).item()
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
