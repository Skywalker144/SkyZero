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
        "policy_target": torch.from_numpy(batch.policy_target[s:e]).to(device, non_blocking=non_blocking),
        "opponent_policy_target": torch.from_numpy(batch.opponent_policy_target[s:e]).to(device, non_blocking=non_blocking),
        "opponent_policy_mask": torch.from_numpy(batch.opponent_policy_mask[s:e]).to(device, non_blocking=non_blocking),
        "value_target": torch.from_numpy(batch.value_target[s:e]).to(device, non_blocking=non_blocking),
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
    value_loss_weight: float
    device: torch.device
    num_workers: int
    amp: bool


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
        value_loss_weight=float(os.environ.get("VALUE_LOSS_WEIGHT", "1.0")),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_workers=a.num_workers,
        amp=int(os.environ.get("ENABLE_AMP", "1")) != 0,
    )


def _load_or_init_model(args: TrainArgs) -> tuple[torch.nn.Module, int, dict | None, dict | None]:
    """Returns (model, global_step, optim_state, scaler_state).
    optim_state / scaler_state are None if absent (fresh run or legacy ckpt).
    """
    cfg = net_config_from_env()
    model = build_model(cfg)
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "model_latest.pt"
    global_step = 0
    optim_state: dict | None = None
    scaler_state: dict | None = None
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            global_step = int(state.get("global_step", 0))
            optim_state = state.get("optimizer_state_dict")
            scaler_state = state.get("scaler_state_dict")
        else:
            model.load_state_dict(state)
        print(f"[train] loaded checkpoint {latest} (global_step={global_step}"
              f"{', +optim' if optim_state else ''}"
              f"{', +scaler' if scaler_state else ''})")
    else:
        print("[train] starting from fresh model (no checkpoint found)")
    return model, global_step, optim_state, scaler_state


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

    model, global_step, optim_state, scaler_state = _load_or_init_model(args)
    print(f"[train] moving model to {args.device}...", flush=True)
    t = time.time()
    model = model.to(args.device)
    model.train()
    print(f"[train] model on {args.device} in {time.time() - t:.1f}s", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim_state is not None:
        try:
            opt.load_state_dict(optim_state)
        except Exception as e:
            print(f"[train] warning: failed to restore optimizer state ({e}); starting fresh")

    use_amp = args.amp and args.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if scaler_state is not None and use_amp:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as e:
            print(f"[train] warning: failed to restore GradScaler state ({e})")

    accum = {"policy": 0.0, "opp_policy": 0.0, "value": 0.0, "total": 0.0}
    window = max(1, min(10, args.train_steps // 4))
    first_sum = {"policy": 0.0, "opp_policy": 0.0, "value": 0.0, "total": 0.0}
    last_sum = {"policy": 0.0, "opp_policy": 0.0, "value": 0.0, "total": 0.0}
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
        policy_target = batch["policy_target"]
        opp_policy_target = batch["opponent_policy_target"]
        opp_mask = batch["opponent_policy_mask"]
        value_target = batch["value_target"]
        sample_weight = batch["sample_weight"]

        state, policy_target, opp_policy_target = random_d4_inplace(
            state, policy_target, opp_policy_target, cfg.board_size
        )

        B = state.shape[0]
        with torch.amp.autocast("cuda", enabled=use_amp):
            policy_logits, opp_policy_logits, value_logits = model(state)
            policy_logits = policy_logits.view(B, -1)
            opp_policy_logits = opp_policy_logits.view(B, -1)
            # Compute losses in fp32 regardless — softmax+log on half is noisy.
            policy_loss = weighted_soft_ce(policy_logits.float(), policy_target, sample_weight)
            opp_policy_loss = weighted_soft_ce(opp_policy_logits.float(), opp_policy_target, sample_weight * opp_mask)
            value_loss = weighted_soft_ce(value_logits.float(), value_target, sample_weight)
            total_loss = (args.policy_loss_weight * policy_loss
                          + args.opponent_policy_loss_weight * opp_policy_loss
                          + args.value_loss_weight * value_loss)

        opt.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        losses = {
            "policy": float(policy_loss.detach()),
            "opp_policy": float(opp_policy_loss.detach()),
            "value": float(value_loss.detach()),
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
                  f"v={losses['value']:.3f}) sps={sps:.0f} "
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
    for k in ("total", "policy", "opp_policy", "value"):
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
        "global_step": global_step,
        "iter": args.iter,
    }, tmp)
    os.replace(tmp, latest)

    # Per-iter snapshot (light — no optimizer)
    snap = ckpt_dir / f"model_iter_{args.iter:06d}.pt"
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
        "value_loss": avg["value"],
        "total_loss": avg["total"],
        "seconds": dt,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
