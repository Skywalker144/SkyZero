"""Training: soft-target policy cross-entropy + scalar value regression.

Replay buffer holds raw int8 states + improved-policy targets + raw discounted
value targets. States are encoded and D4-augmented on the fly each batch; value
targets are regressed in SCALED units (target / value_scale) with a Huber loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import game as G
import value_transform
from augment import augment_batch
from model_config import Config


class ReplayBuffer:
    def __init__(self, window: int) -> None:
        self.window = window
        self.states = np.zeros((0, 16), dtype=np.int8)
        self.policies = np.zeros((0, 4), dtype=np.float32)
        self.values = np.zeros((0,), dtype=np.float32)

    def add(self, states, policies, values) -> None:
        self.states = np.concatenate([self.states, states])[-self.window:]
        self.policies = np.concatenate([self.policies, policies])[-self.window:]
        self.values = np.concatenate([self.values, values])[-self.window:]

    def __len__(self) -> int:
        return self.states.shape[0]


def _encode_states(states: np.ndarray) -> np.ndarray:
    # states: (B,16) int8 -> (B,16,4,4) float32 one-hot exponent planes
    exps = np.clip(states.astype(np.int64), 0, G.NUM_PLANES - 1)
    B = states.shape[0]
    enc = np.zeros((B, G.NUM_PLANES, G.AREA), dtype=np.float32)
    rows = np.arange(B)[:, None]
    cols = np.arange(G.AREA)[None, :]
    enc[rows, exps, cols] = 1.0
    return enc.reshape(B, G.NUM_PLANES, G.SIZE, G.SIZE)


def load_shuffled(shards_dir) -> dict:
    """Load all <shards_dir>/*.npz (the shuffle window for this iter) into
    arrays for sampling. Mirrors training off V7.1's shuffled/current/."""
    import pathlib
    from data_processing import load_npz, concat_batches
    paths = sorted(pathlib.Path(shards_dir).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"no shuffled shards in {shards_dir}")
    b = concat_batches([load_npz(p) for p in paths])
    return {"states": b.state, "policies": b.policy_target,
            "values": b.value_target.reshape(-1), "weights": b.sample_weight}


def _train_loop(net, opt, states, policies, values, weights, cfg: Config, steps: int) -> dict:
    """`steps` gradient steps sampling uniformly from the given arrays, with
    on-the-fly D4 augmentation (board + action relabel) and per-sample weights."""
    net.train()
    dev = cfg.device
    n = states.shape[0]
    p_loss_sum = v_loss_sum = 0.0
    rng = np.random.default_rng()
    for _ in range(steps):
        idx = rng.integers(0, n, size=cfg.batch_size)
        x = torch.from_numpy(_encode_states(states[idx])).to(dev)
        pol_t = torch.from_numpy(policies[idx]).to(dev)
        raw_v = torch.from_numpy(values[idx]).to(dev)
        if cfg.value_transform:
            raw_v = value_transform.to_h_torch(raw_v)   # compress before scaling
        val_t = raw_v / cfg.value_scale
        w = torch.from_numpy(weights[idx].astype(np.float32)).to(dev)
        w = w / w.mean().clamp_min(1e-8)            # normalize so scale ~ unweighted

        x, pol_t = augment_batch(x, pol_t)

        logits, value = net(x)
        logp = F.log_softmax(logits, dim=1)
        p_loss = (-(pol_t * logp).sum(dim=1) * w).mean()
        v_loss = (F.huber_loss(value, val_t, delta=1.0, reduction="none") * w).mean()
        loss = p_loss + cfg.value_loss_weight * v_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        p_loss_sum += float(p_loss.detach())
        v_loss_sum += float(v_loss.detach())
    return {"policy_loss": p_loss_sum / steps, "value_loss": v_loss_sum / steps}


def train_on_shuffled(net, opt, data: dict, cfg: Config, steps: int) -> dict:
    """V7.1-style: train `steps` (from the token bucket) on the shuffle window."""
    return _train_loop(net, opt, data["states"], data["policies"],
                       data["values"], data["weights"], cfg, steps)


def train_steps(net, opt, buf: ReplayBuffer, cfg: Config) -> dict:
    """Pure-Python loop path (loop.py): fixed steps on the in-memory buffer."""
    w = np.ones(len(buf), dtype=np.float32)
    return _train_loop(net, opt, buf.states, buf.policies, buf.values, w,
                       cfg, cfg.train_steps_per_iter)


# ---------------------------------------------------------------------------
# V7.1-style per-network CLI (driven by scripts/internal/train.sh).
#   python train.py --data-dir DIR --network b6c96 --iter N
# env TRAIN_STEPS_PER_EPOCH (from bucket.py) = number of gradient steps this iter
# (0 -> skip training, just re-stamp the checkpoint/iter). Trains off the
# shuffle window in <DIR>/shuffled/current/, resuming <DIR>/nets/<net>/model_latest.pt.
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    import json
    import os
    import pathlib
    import time

    import torch

    from model_config import config_from_name
    from nets import build_net

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()

    cfg = config_from_name(args.network)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"
    steps = int(float(os.environ.get("TRAIN_STEPS_PER_EPOCH", "0")))

    data_dir = pathlib.Path(args.data_dir)
    net_dir = data_dir / "nets" / args.network
    net_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = net_dir / "model_latest.pt"

    net = build_net(cfg).to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    global_step = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            try:
                opt.load_state_dict(ck["optimizer_state_dict"])
            except ValueError:
                pass  # optimizer shape drift (shouldn't happen) -> fresh opt
        global_step = int(ck.get("global_step", 0))

    t0 = time.time()
    losses = {"policy_loss": float("nan"), "value_loss": float("nan")}
    if steps > 0:
        data = load_shuffled(net_dir.parent.parent / "shuffled" / "current")
        losses = train_on_shuffled(net, opt, data, cfg, steps)
        global_step += steps * cfg.batch_size

    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "global_step": global_step, "iter": args.iter}, tmp)
    tmp.replace(ckpt_path)
    (net_dir / "state.json").write_text(json.dumps({"iter": args.iter, "global_step": global_step}))

    tsv = net_dir / "train.tsv"
    new = not tsv.exists()
    with open(tsv, "a") as f:
        if new:
            f.write("iter\tsteps\tglobal_step\tpolicy_loss\tvalue_loss\tseconds\n")
        f.write(f"{args.iter}\t{steps}\t{global_step}\t{losses['policy_loss']:.4f}\t"
                f"{losses['value_loss']:.4f}\t{time.time() - t0:.0f}\n")
    print(f"[train] {args.network} iter={args.iter} steps={steps} "
          f"ploss={losses['policy_loss']:.3f} vloss={losses['value_loss']:.3f} "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
