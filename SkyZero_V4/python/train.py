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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_processing import NPZ_KEYS, random_d4_inplace
from model_config import net_config_from_env
from nets import build_model


# ---------------------------------------------------------------------------
# Dataset backed by one or more shuffled NPZ shards (loaded into RAM).
# ---------------------------------------------------------------------------

class ShuffledShardDataset(Dataset):
    def __init__(self, shard_dir: pathlib.Path) -> None:
        files = sorted(shard_dir.glob("*.npz"))
        if not files:
            raise RuntimeError(f"no shards in {shard_dir}")
        arrays = {k: [] for k in NPZ_KEYS}
        for p in files:
            with np.load(p) as f:
                for k in NPZ_KEYS:
                    arrays[k].append(np.asarray(f[k]))
        self.data = {k: np.concatenate(arrays[k], axis=0) for k in NPZ_KEYS}
        self.n = int(self.data["state"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self.data["state"][i].astype(np.int8)),
            "policy_target": torch.from_numpy(self.data["policy_target"][i]),
            "opponent_policy_target": torch.from_numpy(self.data["opponent_policy_target"][i]),
            "opponent_policy_mask": torch.tensor(self.data["opponent_policy_mask"][i], dtype=torch.float32),
            "value_target": torch.from_numpy(self.data["value_target"][i]),
            "sample_weight": torch.tensor(self.data["sample_weight"][i], dtype=torch.float32),
        }


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


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--num-workers", type=int, default=2)
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
    )


def _load_or_init_model(args: TrainArgs) -> tuple[torch.nn.Module, int]:
    cfg = net_config_from_env()
    model = build_model(cfg)
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "model_latest.pt"
    global_step = 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            global_step = int(state.get("global_step", 0))
        else:
            model.load_state_dict(state)
        print(f"[train] loaded checkpoint {latest} (global_step={global_step})")
    else:
        print("[train] starting from fresh model (no checkpoint found)")
    return model, global_step


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

    ds = ShuffledShardDataset(shard_dir)
    print(f"[train] dataset size: {len(ds)} rows")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model, global_step = _load_or_init_model(args)
    model = model.to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    it = iter(loader)
    accum = {"policy": 0.0, "opp_policy": 0.0, "value": 0.0, "total": 0.0}
    start = time.time()
    step = 0
    while step < args.train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        state = batch["state"].to(args.device, non_blocking=True).float()
        policy_target = batch["policy_target"].to(args.device, non_blocking=True)
        opp_policy_target = batch["opponent_policy_target"].to(args.device, non_blocking=True)
        opp_mask = batch["opponent_policy_mask"].to(args.device, non_blocking=True)
        value_target = batch["value_target"].to(args.device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(args.device, non_blocking=True)

        state, policy_target, opp_policy_target = random_d4_inplace(
            state, policy_target, opp_policy_target, cfg.board_size
        )

        policy_logits, opp_policy_logits, value_logits = model(state)
        B = state.shape[0]
        policy_logits = policy_logits.view(B, -1)
        opp_policy_logits = opp_policy_logits.view(B, -1)

        policy_loss = weighted_soft_ce(policy_logits, policy_target, sample_weight)
        opp_policy_loss = weighted_soft_ce(opp_policy_logits, opp_policy_target, sample_weight * opp_mask)
        value_loss = weighted_soft_ce(value_logits, value_target, sample_weight)

        total_loss = (args.policy_loss_weight * policy_loss
                      + args.opponent_policy_loss_weight * opp_policy_loss
                      + args.value_loss_weight * value_loss)

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        accum["policy"] += float(policy_loss.detach())
        accum["opp_policy"] += float(opp_policy_loss.detach())
        accum["value"] += float(value_loss.detach())
        accum["total"] += float(total_loss.detach())
        step += 1
        global_step += B

    dt = time.time() - start
    avg = {k: v / max(1, step) for k, v in accum.items()}
    print(f"[train] iter={args.iter} steps={step} samples_seen={step * args.batch_size} "
          f"t={dt:.1f}s | policy={avg['policy']:.4f} opp={avg['opp_policy']:.4f} "
          f"value={avg['value']:.4f} total={avg['total']:.4f}")

    # Save checkpoint
    ckpt_dir = args.data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "model_latest.pt"
    tmp = latest.with_suffix(latest.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
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
