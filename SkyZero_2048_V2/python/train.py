"""Training: soft-target policy cross-entropy + scalar value regression.

Replay buffer holds raw int8 states + improved-policy targets + raw discounted
value targets. States are encoded and D4-augmented on the fly each batch; value
targets are regressed in SCALED units (h(target) / value_scale) with a Huber loss.

Training niceties ported from the mainline (SkyZero_V7.8) train.py — AMP, SWA
(EMA-averaged weights, mirrors KataGomo), in-loop Lookahead, and KataGomo-style
5-stage LR warmup — adapted here to the 2048 scalar-value loss (no WDL/multi-head
machinery). All are env-gated (ENABLE_AMP / ENABLE_SWA / LOOKAHEAD_K /
ENABLE_LR_WARMUP); leaving the knobs at their defaults reproduces the plain
Adam+Huber loop. SWA weights, when present, are stored in model_latest.pt and
preferred by export_model.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import game as G
import value_transform
from augment import augment_batch
from model_config import Config, _env_float, _env_int


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


# ---------------------------------------------------------------------------
# Training niceties (AMP / SWA / Lookahead / LR warmup), ported from V7.8.
# ---------------------------------------------------------------------------

def lr_warmup_factor(samples_seen: int, warmup_samples: int) -> float:
    """KataGomo-style 5-stage LR warmup factor (mainline train.py:594-603).

        progress < 1/6 -> 0.20 ; < 1/3 -> 0.33 ; < 2/3 -> 0.50 ; < 1 -> 0.71 ; else 1.0

    Scales LR across the warmup window so Adam's second-moment estimates
    stabilize before full-LR steps. Returns 1.0 immediately if warmup disabled.
    """
    if warmup_samples <= 0:
        return 1.0
    progress = samples_seen / warmup_samples
    if progress < 1.0 / 6.0:
        return 0.20
    if progress < 1.0 / 3.0:
        return 0.33
    if progress < 2.0 / 3.0:
        return 0.50
    if progress < 1.0:
        return 0.71
    return 1.0


@dataclass
class Niceties:
    """AMP/SWA/Lookahead/warmup machinery, built once in main() and threaded
    through the training loop. Carries the cross-iter counters (global_step,
    swa_accum_steps) that must be persisted to the checkpoint. When a Niceties
    is *not* passed to _train_loop, the loop runs the plain Adam+Huber path."""
    use_amp: bool
    scaler: "torch.amp.GradScaler"
    swa_model: object | None
    swa_period_steps: int
    swa_accum_steps: int
    lookahead_k: int
    lookahead_alpha: float
    lookahead_cache: dict | None
    warmup_samples: int
    base_lr: float
    global_step: int


def _make_batch(states, policies, values, weights, idx, cfg: Config, dev):
    """Encode + D4-augment one sampled minibatch. Returns (x, pol_t, val_t, w)."""
    x = torch.from_numpy(_encode_states(states[idx])).to(dev)
    pol_t = torch.from_numpy(policies[idx]).to(dev)
    raw_v = torch.from_numpy(values[idx]).to(dev)
    # value target = MuZero h(raw) / VALUE_SCALE (always on; head learns h-space).
    val_t = value_transform.to_h_torch(raw_v) / cfg.value_scale
    w = torch.from_numpy(weights[idx].astype(np.float32)).to(dev)
    w = w / w.mean().clamp_min(1e-8)                # normalize so scale ~ unweighted
    x, pol_t = augment_batch(x, pol_t)
    return x, pol_t, val_t, w


def _train_loop(net, opt, states, policies, values, weights, cfg: Config,
                steps: int, nice: Niceties | None = None) -> dict:
    """`steps` gradient steps sampling uniformly from the given arrays, with
    on-the-fly D4 augmentation (board + action relabel) and per-sample weights.

    With `nice` provided, runs AMP + Lookahead + SWA + LR warmup (mutating the
    carried counters in `nice`); without it, the plain Adam+Huber loop."""
    net.train()
    dev = cfg.device
    n = states.shape[0]
    p_loss_sum = v_loss_sum = 0.0
    rng = np.random.default_rng()
    use_amp = bool(nice and nice.use_amp)
    lookahead_counter = 0
    for _ in range(steps):
        if nice is not None and nice.warmup_samples > 0:
            wf = lr_warmup_factor(nice.global_step, nice.warmup_samples)
            for g in opt.param_groups:
                g["lr"] = nice.base_lr * wf

        idx = rng.integers(0, n, size=cfg.batch_size)
        x, pol_t, val_t, w = _make_batch(states, policies, values, weights, idx, cfg, dev)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, value = net(x)
            logp = F.log_softmax(logits, dim=1)
            p_loss = (-(pol_t * logp).sum(dim=1) * w).mean()
            v_loss = (F.huber_loss(value, val_t, delta=1.0, reduction="none") * w).mean()
            loss = p_loss + cfg.value_loss_weight * v_loss

        opt.zero_grad(set_to_none=True)
        if nice is not None:
            nice.scaler.scale(loss).backward()
            nice.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            nice.scaler.step(opt)
            nice.scaler.update()

            # Lookahead (mainline train.py:1521-1543): every k fast steps,
            # slow += alpha(fast - slow) then copy slow back to fast.
            if nice.lookahead_cache is not None:
                lookahead_counter += 1
                if lookahead_counter >= nice.lookahead_k:
                    with torch.no_grad():
                        for g in opt.param_groups:
                            for p in g["params"]:
                                slow = nice.lookahead_cache[p]
                                slow.add_(p.data - slow, alpha=nice.lookahead_alpha)
                                p.data.copy_(slow)
                    lookahead_counter = 0

            # SWA (EMA): accumulate every step; only snapshot on a synced step
            # (lookahead_counter==0) so the EMA carries lookahead-synced weights.
            if nice.swa_model is not None:
                nice.swa_accum_steps += 1
                if nice.swa_accum_steps >= nice.swa_period_steps and lookahead_counter == 0:
                    nice.swa_accum_steps = 0
                    nice.swa_model.update_parameters(net)

            nice.global_step += cfg.batch_size
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()

        p_loss_sum += float(p_loss.detach())
        v_loss_sum += float(v_loss.detach())
    return {"policy_loss": p_loss_sum / steps, "value_loss": v_loss_sum / steps}


def train_on_shuffled(net, opt, data: dict, cfg: Config, steps: int,
                      nice: Niceties | None = None) -> dict:
    """V7.1-style: train `steps` (from the gate/bucket) on the shuffle window."""
    return _train_loop(net, opt, data["states"], data["policies"],
                       data["values"], data["weights"], cfg, steps, nice)


def train_steps(net, opt, buf: ReplayBuffer, cfg: Config) -> dict:
    """Pure-Python loop path: fixed steps on the in-memory buffer (plain loop)."""
    w = np.ones(len(buf), dtype=np.float32)
    return _train_loop(net, opt, buf.states, buf.policies, buf.values, w,
                       cfg, cfg.train_steps_per_iter)


# ---------------------------------------------------------------------------
# V7.1-style per-network CLI (driven by scripts/internal/train.sh).
#   python train.py --data-dir DIR --network b6c96 --iter N
# env TRAIN_STEPS_PER_EPOCH (from compute_selfplay_target / run.sh) = number of
# gradient steps this iter (0 -> skip training, just re-stamp the checkpoint).
# Trains off the shuffle window in <DIR>/shuffled/current/, resuming
# <DIR>/nets/<net>/model_latest.pt.
# ---------------------------------------------------------------------------
def _build_niceties(net, opt, cfg: Config, *, opt_lr: float, global_step: int,
                    scaler_state, swa_state, swa_accum_steps: int) -> Niceties:
    """Assemble AMP/SWA/Lookahead/warmup state from env knobs + restored ckpt."""
    use_amp = (_env_int("ENABLE_AMP", 1) != 0) and cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if scaler_state is not None and use_amp:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as e:  # noqa: BLE001
            print(f"[train] warning: failed to restore GradScaler state ({e})")

    swa_model = None
    if _env_int("ENABLE_SWA", 1) != 0:
        from torch.optim.swa_utils import AveragedModel
        new_factor = 1.0 / _env_float("SWA_SCALE", 8.0)

        def ema_avg(avg_p, cur_p, num_averaged):
            return avg_p + new_factor * (cur_p - avg_p)

        # use_buffers=True so BatchNorm running stats are EMA-averaged too.
        swa_model = AveragedModel(net, avg_fn=ema_avg, use_buffers=True)
        if swa_state is not None:
            try:
                swa_model.load_state_dict(swa_state)
            except Exception as e:  # noqa: BLE001
                print(f"[train] warning: failed to restore SWA state ({e}); fresh")
                swa_accum_steps = 0
    else:
        swa_accum_steps = 0

    # Lookahead slow-weight cache, re-init from the loaded fast weights each
    # run (not persisted; mainline does the same — loses <=k-1 averaging steps).
    lookahead_k = _env_int("LOOKAHEAD_K", 6)
    lookahead_alpha = _env_float("LOOKAHEAD_ALPHA", 0.5)
    lookahead_cache = None
    if lookahead_k > 0 and lookahead_alpha > 0.0:
        lookahead_cache = {}
        for grp in opt.param_groups:
            for p in grp["params"]:
                lookahead_cache[p] = p.data.clone().detach()

    warmup_samples = (int(_env_float("LR_WARMUP_SAMPLES", 6e6))
                      if _env_int("ENABLE_LR_WARMUP", 0) != 0 else 0)

    return Niceties(
        use_amp=use_amp, scaler=scaler, swa_model=swa_model,
        swa_period_steps=_env_int("SWA_PERIOD_STEPS", 200),
        swa_accum_steps=swa_accum_steps,
        lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha,
        lookahead_cache=lookahead_cache,
        warmup_samples=warmup_samples, base_lr=opt_lr, global_step=global_step,
    )


def main() -> int:
    import argparse
    import json
    import pathlib
    import time

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

    # Lookahead LR compensation (mainline train.py:933-939): divide the
    # optimizer LR by alpha so the slow-weight effective LR matches cfg.lr.
    lookahead_k = _env_int("LOOKAHEAD_K", 6)
    lookahead_alpha = _env_float("LOOKAHEAD_ALPHA", 0.5)
    lookahead_on = lookahead_k > 0 and lookahead_alpha > 0.0
    opt_lr = cfg.lr / lookahead_alpha if lookahead_on else cfg.lr
    opt = torch.optim.AdamW(net.parameters(), lr=opt_lr, weight_decay=cfg.weight_decay)

    global_step = 0
    scaler_state = swa_state = None
    swa_accum_steps = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        net.set_norm_scales()   # fixscale nbt: scales aren't in state_dict (no-op for BN net)
        if "optimizer_state_dict" in ck:
            try:
                opt.load_state_dict(ck["optimizer_state_dict"])
            except ValueError:
                pass  # optimizer shape drift (shouldn't happen) -> fresh opt
            for grp in opt.param_groups:    # re-apply LR (load_state_dict clobbers it)
                grp["lr"] = opt_lr
        global_step = int(ck.get("global_step", 0))
        scaler_state = ck.get("scaler_state_dict")
        swa_state = ck.get("swa_model_state_dict")
        swa_accum_steps = int(ck.get("swa_accum_steps", 0))
    else:
        net.initialize()        # fresh: RepVGG/fixscale init for nbt (no-op for BN net)

    nice = _build_niceties(net, opt, cfg, opt_lr=opt_lr, global_step=global_step,
                           scaler_state=scaler_state, swa_state=swa_state,
                           swa_accum_steps=swa_accum_steps)

    t0 = time.time()
    losses = {"policy_loss": float("nan"), "value_loss": float("nan")}
    if steps > 0:
        data = load_shuffled(net_dir.parent.parent / "shuffled" / "current")
        losses = train_on_shuffled(net, opt, data, cfg, steps, nice)
    global_step = nice.global_step

    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": nice.scaler.state_dict() if nice.use_amp else None,
                "swa_model_state_dict": (nice.swa_model.state_dict()
                                         if nice.swa_model is not None else None),
                "swa_accum_steps": nice.swa_accum_steps,
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
