"""Train one epoch over data/shuffled/current/.

One "epoch" here = TRAIN_STEPS_PER_EPOCH mini-batch SGD steps. After training,
saves model state_dict to data/checkpoints/model_latest.pt and appends an
entry to data/logs/train.tsv. run.sh invokes export_model.py afterwards to
produce a TorchScript artifact for the next selfplay round.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from data_processing import load_npz, random_d4_inplace
from model_config import net_config_from_env
from nets import build_model


# ---------------------------------------------------------------------------
# Streaming batch iterator — mirrors KataGo's data_processing_pytorch.py.
# Peak RAM ≈ 2 * one shard (current + prefetched next). Independent of total
# window size.
# ---------------------------------------------------------------------------

def _collate_to_device(batch, s: int, e: int, device: torch.device, non_blocking: bool):
    """Slice rows [s, e) out of an NpzBatch and move to device as a dict of tensors."""
    return {
        "state": torch.from_numpy(batch.state[s:e]).to(device, non_blocking=non_blocking),
        "global_features": torch.from_numpy(batch.global_features[s:e]).to(device, non_blocking=non_blocking),
        "policy_target": torch.from_numpy(batch.policy_target[s:e]).to(device, non_blocking=non_blocking),
        "opponent_policy_target": torch.from_numpy(batch.opponent_policy_target[s:e]).to(device, non_blocking=non_blocking),
        "opponent_policy_mask": torch.from_numpy(batch.opponent_policy_mask[s:e]).to(device, non_blocking=non_blocking),
        "value_target": torch.from_numpy(batch.value_target[s:e]).to(device, non_blocking=non_blocking),
        "td_value_target": torch.from_numpy(batch.td_value_target[s:e]).to(device, non_blocking=non_blocking),
        "futurepos_target": torch.from_numpy(batch.futurepos_target[s:e]).to(device, non_blocking=non_blocking),
        "sample_weight": torch.from_numpy(batch.sample_weight[s:e]).to(device, non_blocking=non_blocking),
    }


def iterate_batches(
    shard_dir: pathlib.Path,
    batch_size: int,
    device: torch.device,
    *,
    seed: int | None = None,
    infinite: bool = True,
):
    """Infinite generator yielding batch dicts on `device`.

    Each epoch: random-shuffle the shard file list, then consume each shard
    sequentially (within-shard rows are already permuted by shuffle.py pass-2),
    prefetching the next shard in a background thread. Tail rows that don't
    fill a full batch are dropped (like DataLoader's drop_last=True).
    """
    files = sorted(shard_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no shards in {shard_dir}")
    rng = np.random.default_rng(seed)
    non_blocking = (device.type == "cuda")

    with ThreadPoolExecutor(max_workers=1) as ex:
        while True:
            order = list(files)
            rng.shuffle(order)
            fut = ex.submit(load_npz, order[0])
            for i in range(len(order)):
                batch_np = fut.result()
                fut = ex.submit(load_npz, order[i + 1]) if i + 1 < len(order) else None
                n = len(batch_np)
                n_full = (n // batch_size) * batch_size
                for s in range(0, n_full, batch_size):
                    yield _collate_to_device(batch_np, s, s + batch_size, device, non_blocking)
                # batch_np goes out of scope here once yielded tensors release it
            if not infinite:
                break


# ---------------------------------------------------------------------------
# Loss: weighted soft-target cross entropy.
# ---------------------------------------------------------------------------

def weighted_soft_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """logits: (B, C), target: (B, C) (soft distribution, sum<=1), weight: (B,)."""
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target * log_probs).sum(dim=-1)
    denom = weight.sum().clamp_min(1e-8)
    return (per_sample * weight).sum() / denom


def soft_policy_target(p: torch.Tensor) -> torch.Tensor:
    """KataGo soft-policy target: (p + 1e-7)^0.25, renormalized.

    Mirrors KataGomo metrics_pytorch.py:462-464. Flattens the visit-count
    distribution so the soft head learns "everything reasonable" rather than
    just the top move — empirically helps generalization.
    """
    soft = (p + 1e-7).clamp_min(0.0).pow(0.25)
    return soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def weighted_td_value_ce(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Multi-horizon TD value loss (KataGomo metrics_pytorch.py:73-79).

    pred_logits / target_probs: (B, 3, 3) — 3 horizons (long/mid/short) × WLD.
    Per horizon: CE - H(target). Returns sample-weighted mean over batch.
    """
    log_p = F.log_softmax(pred_logits, dim=-1)               # (B, 3, 3)
    ce = -(target_probs * log_p).sum(dim=-1)                 # (B, 3)
    H_t = -(target_probs * (target_probs + 1e-30).log()).sum(dim=-1)
    per_h = ce - H_t                                         # (B, 3)
    per_sample = per_h.sum(dim=-1)                           # (B,)
    denom = weight.sum().clamp_min(1e-8)
    return (per_sample * weight).sum() / denom


def weighted_futurepos_mse(
    pred_pretanh: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Futurepos loss (KataGomo metrics_pytorch.py:114-130).

    pred_pretanh / target: (B, 2, H, W). mask: (B, 1, H, W) on-board indicator.
    Channel 0 (+8 step) weight 1.0, channel 1 (+32 step) weight 0.25.
    Per-sample sum normalized by sqrt(mask_sum_hw); sample-weighted mean.
    """
    pred = torch.tanh(pred_pretanh)
    err = (pred - target).pow(2) * mask                      # (B, 2, H, W)
    ch_w = err.new_tensor([1.0, 0.25]).view(1, 2, 1, 1)
    err = err * ch_w
    mask_sum_hw = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)     # (B,)
    per_sample = err.sum(dim=(1, 2, 3)) / mask_sum_hw.sqrt()
    denom = weight.sum().clamp_min(1e-8)
    return (per_sample * weight).sum() / denom


# ---------------------------------------------------------------------------
@dataclass
class TrainArgs:
    data_dir: pathlib.Path
    iter: int
    batch_size: int
    train_steps: int
    lr: float
    weight_decay: float
    grad_clip: float
    policy_loss_weight: float
    opponent_policy_loss_weight: float
    soft_policy_loss_weight: float
    value_loss_weight: float
    td_value_loss_weight: float
    futurepos_loss_weight: float
    intermediate_loss_scale: float
    device: torch.device
    num_workers: int
    amp: bool
    enable_swa: bool
    swa_scale: float
    swa_period_steps: int
    # Lookahead optimizer (KataGo-style in-loop, not torch_optimizer wrapper).
    # k = 0 disables. With k>0, opt's LR is divided by alpha to compensate for
    # the slow-weight averaging — user's LR env var stays the "effective" LR.
    lookahead_k: int
    lookahead_alpha: float


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--iter", type=int, required=True)
    # Deprecated: data loading is now a streaming generator with its own
    # single-thread prefetch (see iterate_batches). Accepted for backward
    # compatibility; value is ignored.
    p.add_argument("--num-workers", type=int,
                   default=int(os.environ.get("TRAIN_NUM_WORKERS", "0")))
    a = p.parse_args()
    return TrainArgs(
        data_dir=pathlib.Path(a.data_dir),
        iter=a.iter,
        batch_size=int(os.environ.get("BATCH_SIZE", "256")),
        train_steps=int(os.environ.get("TRAIN_STEPS_PER_EPOCH", "100")),
        lr=float(os.environ.get("LR", "1e-4")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "3e-5")),
        grad_clip=float(os.environ.get("GRAD_CLIP", "1.0")),
        policy_loss_weight=float(os.environ.get("POLICY_LOSS_WEIGHT", "1.0")),
        opponent_policy_loss_weight=float(os.environ.get("OPP_POLICY_LOSS_WEIGHT", "0.15")),
        soft_policy_loss_weight=float(os.environ.get("SOFT_POLICY_LOSS_WEIGHT", "8.0")),
        value_loss_weight=float(os.environ.get("VALUE_LOSS_WEIGHT", "1.0")),
        td_value_loss_weight=float(os.environ.get("TD_VALUE_LOSS_WEIGHT", "0.72")),
        futurepos_loss_weight=float(os.environ.get("FUTUREPOS_LOSS_WEIGHT", "0.25")),
        intermediate_loss_scale=float(os.environ.get("INTERMEDIATE_LOSS_SCALE", "0.3")),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_workers=a.num_workers,
        amp=int(os.environ.get("ENABLE_AMP", "1")) != 0,
        enable_swa=int(os.environ.get("ENABLE_SWA", "0")) != 0,
        swa_scale=float(os.environ.get("SWA_SCALE", "8")),
        swa_period_steps=int(os.environ.get("SWA_PERIOD_STEPS", "200")),
        lookahead_k=int(os.environ.get("LOOKAHEAD_K", "0")),
        lookahead_alpha=float(os.environ.get("LOOKAHEAD_ALPHA", "0.5")),
    )


def _load_or_init_model(args: TrainArgs) -> tuple[torch.nn.Module, int, dict | None, dict | None, dict | None, int]:
    """Returns (model, global_step, optim_state, scaler_state, swa_state, swa_accum_steps).
    Any of the state dicts are None if absent (fresh run or legacy ckpt).
    """
    cfg = net_config_from_env()
    model = build_model(cfg)
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "model_latest.pt"
    global_step = 0
    optim_state: dict | None = None
    scaler_state: dict | None = None
    swa_state: dict | None = None
    swa_accum_steps = 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            global_step = int(state.get("global_step", 0))
            optim_state = state.get("optimizer_state_dict")
            scaler_state = state.get("scaler_state_dict")
            swa_state = state.get("swa_model_state_dict")
            swa_accum_steps = int(state.get("swa_accum_steps", 0))
        else:
            model.load_state_dict(state)
        # TRAP 3 (NOTES.md §3.3): NormMask.scale is plain Python float, not in
        # state_dict. Must restore it after load_state_dict.
        model.set_norm_scales()
        tags = []
        if optim_state: tags.append("+optim")
        if scaler_state: tags.append("+scaler")
        if swa_state: tags.append("+swa")
        print(f"[train] loaded checkpoint {latest} (global_step={global_step}"
              f"{''.join(', '+t for t in tags)})")
    else:
        # Fresh model: must call initialize() to set RepVGG weights and all
        # NormMask.scale values; otherwise weights are PyTorch defaults and
        # FixscaleNorm.scale is None → broken forward.
        model.initialize()
        print("[train] starting from fresh model (no checkpoint found); model.initialize() called")
    return model, global_step, optim_state, scaler_state, swa_state, swa_accum_steps


def _write_log(log_path: pathlib.Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(row.keys())
    line = "\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values())
    exists = log_path.exists()
    with log_path.open("a") as f:
        if not exists:
            f.write(header + "\n")
        f.write(line + "\n")


def main() -> int:
    args = parse_args()
    cfg = net_config_from_env()
    shard_dir = args.data_dir / "shuffled" / "current"

    it = iterate_batches(shard_dir, args.batch_size, args.device, infinite=True)

    model, global_step, optim_state, scaler_state, swa_state, swa_accum_steps = _load_or_init_model(args)
    print(f"[train] moving model to {args.device}...", flush=True)
    t = time.time()
    model = model.to(args.device)
    model.train()
    print(f"[train] model on {args.device} in {time.time() - t:.1f}s", flush=True)
    # Lookahead (KataGo train.py:933-939): when enabled, divide opt's LR by
    # alpha so the slow-weight effective LR matches the user-configured LR.
    lookahead_enabled = args.lookahead_k > 0 and args.lookahead_alpha > 0.0
    opt_lr = args.lr / args.lookahead_alpha if lookahead_enabled else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=opt_lr, weight_decay=args.weight_decay)
    if optim_state is not None:
        try:
            opt.load_state_dict(optim_state)
        except Exception as e:
            print(f"[train] warning: failed to restore optimizer state ({e}); starting fresh")
        # load_state_dict overwrites LR with whatever was saved; re-apply so
        # toggling Lookahead between runs (or changing args.lr) takes effect.
        for group in opt.param_groups:
            group["lr"] = opt_lr

    # Lookahead slow-weight cache (KataGo train.py:852-858). Initialized from
    # current fast weights; not persisted across checkpoints — on resume we
    # re-init from the loaded fast weights, which is equivalent to a fresh
    # Lookahead phase. This loses at most `k-1` slow-weight averaging steps.
    lookahead_cache: dict | None = None
    lookahead_counter = 0
    in_between_lookaheads = False
    if lookahead_enabled:
        lookahead_cache = {}
        for group in opt.param_groups:
            for p in group["params"]:
                lookahead_cache[p] = p.data.clone().detach()
        print(f"[train] Lookahead enabled: k={args.lookahead_k} alpha={args.lookahead_alpha} "
              f"(opt LR scaled to {opt_lr:.3e} from effective {args.lr:.3e})")

    use_amp = args.amp and args.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if scaler_state is not None and use_amp:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as e:
            print(f"[train] warning: failed to restore GradScaler state ({e})")

    # SWA (EMA variant, mirrors KataGomo train.py:426-430, 1164-1171). We pass
    # `use_buffers=True` so BN running_mean / running_var are EMA-averaged too —
    # SkyZero keeps BN at start_layer/trunk_tip/heads, so the SWA weights must
    # carry matching stats or inference will diverge.
    swa_model: torch.optim.swa_utils.AveragedModel | None = None
    if args.enable_swa:
        from torch.optim.swa_utils import AveragedModel
        new_factor = 1.0 / args.swa_scale
        def ema_avg(avg_p, cur_p, num_averaged):
            return avg_p + new_factor * (cur_p - avg_p)
        swa_model = AveragedModel(model, avg_fn=ema_avg, use_buffers=True)
        if swa_state is not None:
            try:
                swa_model.load_state_dict(swa_state)
            except Exception as e:
                print(f"[train] warning: failed to restore SWA state ({e}); starting fresh")
                swa_accum_steps = 0
        print(f"[train] SWA enabled: scale={args.swa_scale} period_steps={args.swa_period_steps}")
    else:
        swa_accum_steps = 0

    LOSS_KEYS = ("policy", "opp_policy", "soft_policy", "soft_opp_policy",
                 "value", "td_value", "futurepos", "int_total", "total")
    accum = {k: 0.0 for k in LOSS_KEYS}
    window = max(1, min(10, args.train_steps // 4))
    first_sum = {k: 0.0 for k in LOSS_KEYS}
    last_sum = {k: 0.0 for k in LOSS_KEYS}
    last_buf: list[dict] = []

    # Progress print cadence: ~20 updates across the epoch, rounded to a nice
    # multiple. Gives visible feedback without spamming.
    report_every = max(1, args.train_steps // 20)

    print(f"[train] warming up first batch (loading first shard)...", flush=True)
    t_warmup = time.time()
    first_batch = next(it)
    print(f"[train] first batch ready in {time.time() - t_warmup:.1f}s; "
          f"running {args.train_steps} steps", flush=True)

    start = time.time()
    step = 0
    while step < args.train_steps:
        batch = first_batch if step == 0 else next(it)
        first_batch = None  # drop reference

        state = batch["state"].float()
        global_features = batch["global_features"]
        policy_target = batch["policy_target"]
        opp_policy_target = batch["opponent_policy_target"]
        opp_mask = batch["opponent_policy_mask"]
        value_target = batch["value_target"]
        td_value_target = batch["td_value_target"].view(-1, 3, 3)
        # int8 → float32 (KataGomo target is in {-1,0,+1}, matches tanh range)
        futurepos_target = batch["futurepos_target"].float()
        sample_weight = batch["sample_weight"]

        # D4 augmentation transforms all spatial channels (incl. mask plane ch 0
        # and the futurepos targets, which are D4-equivariant occupancy maps).
        # global_features / value_target / td_value_target / sample_weight are
        # D4-invariant.
        state, policy_target, opp_policy_target, futurepos_target = random_d4_inplace(
            state, policy_target, opp_policy_target, futurepos_target, cfg.board_size
        )

        B = state.shape[0]
        # On-board mask (B, 1, H, W) for futurepos masking. Same source as the
        # network's internal mask (input plane 0 in nets_v2.py:653).
        fp_mask = state[:, 0:1, :, :]
        # Soft targets are derived in fp32 once per batch (not transformed by
        # D4 — already done above on the underlying main/opp targets).
        soft_main_target = soft_policy_target(policy_target)
        soft_opp_target = soft_policy_target(opp_policy_target)
        opp_weight = sample_weight * opp_mask
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(state, global_features)
            # nets.PolicyHead (slim, 4 outputs):
            #   idx 0 = main_policy
            #   idx 1 = opp_policy        (KataGomo C1, our renamed "aux")
            #   idx 2 = soft_main_policy
            #   idx 3 = soft_opp_policy
            # Heads dropped (vs full_nets): aux/opt — see KataGomo
            # metrics_pytorch.py:553/590, gated by target_weight_ownership=0.
            policy_all = out["policy"]
            int_policy_all = out.get("intermediate_policy")
            int_value_logits = out.get("intermediate_value_wdl")
            int_value_td = out.get("intermediate_value_td")
            int_value_fp = out.get("intermediate_value_futurepos")

            def head_losses(p_all: torch.Tensor, v_logits: torch.Tensor,
                            v_td: torch.Tensor, v_fp: torch.Tensor):
                p_main = p_all[:, 0, :].view(B, -1).float()
                p_opp = p_all[:, 1, :].view(B, -1).float()
                p_soft_main = p_all[:, 2, :].view(B, -1).float()
                p_soft_opp = p_all[:, 3, :].view(B, -1).float()
                return (
                    weighted_soft_ce(p_main, policy_target, sample_weight),
                    weighted_soft_ce(p_opp, opp_policy_target, opp_weight),
                    weighted_soft_ce(p_soft_main, soft_main_target, sample_weight),
                    weighted_soft_ce(p_soft_opp, soft_opp_target, opp_weight),
                    weighted_soft_ce(v_logits.float(), value_target, sample_weight),
                    weighted_td_value_ce(v_td.view(B, 3, 3).float(), td_value_target, sample_weight),
                    weighted_futurepos_mse(v_fp.float(), futurepos_target, fp_mask, sample_weight),
                )

            (policy_loss, opp_policy_loss, soft_policy_loss,
             soft_opp_policy_loss, value_loss, td_value_loss, futurepos_loss) = head_losses(
                policy_all, out["value_wdl"], out["value_td"], out["value_futurepos"])

            # KataGomo per-head weight (metrics_pytorch.py):
            #   soft_main: 1.0 × soft_scale       → policy_w × soft_w
            #   soft_opp:  0.15 × soft_scale      → opp_w  × soft_w
            soft_main_w = args.policy_loss_weight * args.soft_policy_loss_weight
            soft_opp_w = args.opponent_policy_loss_weight * args.soft_policy_loss_weight
            main_total = (args.policy_loss_weight * policy_loss
                          + args.opponent_policy_loss_weight * opp_policy_loss
                          + soft_main_w * soft_policy_loss
                          + soft_opp_w * soft_opp_policy_loss
                          + args.value_loss_weight * value_loss
                          + args.td_value_loss_weight * td_value_loss
                          + args.futurepos_loss_weight * futurepos_loss)

            if (int_policy_all is not None and int_value_logits is not None
                    and int_value_td is not None and int_value_fp is not None
                    and args.intermediate_loss_scale > 0):
                (i_policy, i_opp, i_soft, i_soft_opp, i_value, i_td, i_fp) = head_losses(
                    int_policy_all, int_value_logits, int_value_td, int_value_fp)
                int_total = (args.policy_loss_weight * i_policy
                             + args.opponent_policy_loss_weight * i_opp
                             + soft_main_w * i_soft
                             + soft_opp_w * i_soft_opp
                             + args.value_loss_weight * i_value
                             + args.td_value_loss_weight * i_td
                             + args.futurepos_loss_weight * i_fp)
            else:
                int_total = torch.zeros((), device=state.device, dtype=main_total.dtype)

            total_loss = main_total + args.intermediate_loss_scale * int_total

        opt.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        # Lookahead sync (KataGo train.py:1521-1543). Every k fast-opt steps,
        # update slow weights toward fast: slow ← slow + α(fast - slow), then
        # copy slow back to fast. Between syncs, in_between_lookaheads=True
        # gates SWA so it only snapshots synced weights.
        if lookahead_enabled:
            lookahead_counter += 1
            in_between_lookaheads = True
            if lookahead_counter >= args.lookahead_k:
                with torch.no_grad():
                    for group in opt.param_groups:
                        for p in group["params"]:
                            slow = lookahead_cache[p]
                            slow.add_(p.data - slow, alpha=args.lookahead_alpha)
                            p.data.copy_(slow)
                lookahead_counter = 0
                in_between_lookaheads = False

        if swa_model is not None and not in_between_lookaheads:
            swa_accum_steps += 1
            if swa_accum_steps >= args.swa_period_steps:
                swa_accum_steps = 0
                swa_model.update_parameters(model)

        losses = {
            "policy": float(policy_loss.detach()),
            "opp_policy": float(opp_policy_loss.detach()),
            "soft_policy": float(soft_policy_loss.detach()),
            "soft_opp_policy": float(soft_opp_policy_loss.detach()),
            "value": float(value_loss.detach()),
            "td_value": float(td_value_loss.detach()),
            "futurepos": float(futurepos_loss.detach()),
            "int_total": float(int_total.detach()),
            "total": float(total_loss.detach()),
        }
        for k, v in losses.items():
            accum[k] += v
        if step < window:
            for k, v in losses.items():
                first_sum[k] += v
        last_buf.append(losses)
        if len(last_buf) > window:
            last_buf.pop(0)
        step += 1
        global_step += B

        if step % report_every == 0 or step == args.train_steps:
            elapsed = time.time() - start
            sps = step * B / elapsed if elapsed > 0 else 0.0
            print(f"[train]   step {step}/{args.train_steps} "
                  f"loss={losses['total']:.3f} (p={losses['policy']:.3f} "
                  f"sp={losses['soft_policy']:.3f} int={losses['int_total']:.3f} "
                  f"v={losses['value']:.3f} tv={losses['td_value']:.3f} "
                  f"fp={losses['futurepos']:.3f}) sps={sps:.0f} "
                  f"t={elapsed:.1f}s", flush=True)

    dt = time.time() - start
    avg = {k: v / max(1, step) for k, v in accum.items()}
    first_n = min(window, step)
    first_avg = {k: first_sum[k] / max(1, first_n) for k in accum}
    last_n = len(last_buf)
    for b in last_buf:
        for k, v in b.items():
            last_sum[k] += v
    last_avg = {k: last_sum[k] / max(1, last_n) for k in accum}

    print(f"[train] iter={args.iter} steps={step} samples_seen={step * args.batch_size} "
          f"t={dt:.1f}s (first/last avg over {window} steps)")
    name_w = max(len(k) for k in accum)
    for k in ("total", "policy", "opp_policy", "soft_policy",
              "soft_opp_policy", "value", "td_value", "futurepos", "int_total"):
        print(f"[train]   {k:<{name_w}} : {first_avg[k]:8.4f} -> {last_avg[k]:8.4f}")

    # Save checkpoint
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "model_latest.pt"
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "swa_model_state_dict": swa_model.state_dict() if swa_model is not None else None,
        "swa_accum_steps": swa_accum_steps,
        "global_step": global_step,
        "iter": args.iter,
    }, tmp)
    os.replace(tmp, latest)

    # Per-iter snapshot (light — no optimizer). When SWA is enabled, the
    # snapshot is the EMA-averaged weights so selfplay / export pick those up
    # (mirrors KataGomo's save path in train.py:291-292). `swa_model.module`
    # is the underlying bare model.
    snap = ckpt_dir / f"model_iter_{args.iter:06d}.pt"
    if swa_model is not None:
        torch.save(swa_model.module.state_dict(), snap)
    else:
        torch.save(model.state_dict(), snap)

    # State json
    state_json = ckpt_dir / "state.json"
    state_json.write_text(json.dumps({
        "iter": args.iter,
        "global_step_samples": global_step,
    }, indent=2))

    # Log
    _write_log(args.data_dir / "logs" / "train.tsv", {
        "iter": args.iter,
        "steps": step,
        "global_step_samples": global_step,
        "policy_loss": avg["policy"],
        "opp_policy_loss": avg["opp_policy"],
        "soft_policy_loss": avg["soft_policy"],
        "soft_opp_policy_loss": avg["soft_opp_policy"],
        "value_loss": avg["value"],
        "td_value_loss": avg["td_value"],
        "futurepos_loss": avg["futurepos"],
        "intermediate_loss": avg["int_total"],
        "total_loss": avg["total"],
        "seconds": dt,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
