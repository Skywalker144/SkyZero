#!/usr/bin/env python3
"""Behaviour eval for the SAC coefficient sweep: not just survival, but *how* it moves.

For one or more checkpoints, runs greedy episodes on the GPU env and reports the
three knobs the user cares about:
  - smoothness  -> jitter = mean |Δaction| step-to-step (lower = less twitchy)
  - economy     -> speed when SAFE vs speed when in DANGER (rest-when-safe => safe<<danger)
  - centering   -> mean distance from arena centre in px (lower = more centred)
plus survival (score/surv) as a guard that behaviour shaping didn't wreck play.

    python eval_behavior.py runs/sac_tune_A/final.pt runs/sac_tune_B/final.pt ...
"""
import argparse, math, sys
import numpy as np
import torch
from env_dodge_gpu import VecDodgeGPU
from sac_net import SquashedGaussianActor

ACT_DIM = 2
CX, CY = 225.0, 300.0
DANGER_PX, SAFE_PX = 90.0, 180.0      # nearest-threat distance bands


@torch.no_grad()
def eval_ckpt(path, dev, n=128, max_steps=18000, seed=12345):
    ck = torch.load(path, map_location=dev)
    actor = SquashedGaussianActor(271, ACT_DIM, (256, 256)).to(dev)
    actor.load_state_dict(ck["actor"]); actor.eval()

    env = VecDodgeGPU(n, device=dev, max_steps=max_steps)   # raw env: shaping off, we measure behaviour
    obs = env.reset(seed=seed)
    prev = torch.zeros(n, ACT_DIM, device=dev)
    scores, survs = [], []
    jit_sum = torch.zeros(n, device=dev); steps_alive = torch.zeros(n, device=dev)
    spd_safe_s = torch.zeros(n, device=dev); spd_safe_c = torch.zeros(n, device=dev)
    spd_dng_s = torch.zeros(n, device=dev);  spd_dng_c = torch.zeros(n, device=dev)
    cdist_s = torch.zeros(n, device=dev)
    started = False
    done_count, steps = 0, 0
    while done_count < n and steps < max_steps + 5:
        a = actor.mean_action(obs)
        mag = torch.hypot(a[:, 0], a[:, 1]).clamp(max=1.0)             # executed speed
        if started:
            jit_sum += torch.hypot(a[:, 0] - prev[:, 0], a[:, 1] - prev[:, 1])
        prev = a.clone(); started = True
        # nearest active projectile distance (dominant threat)
        dx, dy = env.p_x - env.px[:, None], env.p_y - env.py[:, None]
        d2 = torch.where(env.p_act, dx * dx + dy * dy, torch.full_like(dx, float("inf")))
        nd = d2.min(dim=1).values.sqrt()
        safe = nd > SAFE_PX; dng = nd < DANGER_PX
        spd_safe_s += torch.where(safe, mag, torch.zeros_like(mag)); spd_safe_c += safe.float()
        spd_dng_s += torch.where(dng, mag, torch.zeros_like(mag));   spd_dng_c += dng.float()
        cdist_s += torch.hypot(env.px - CX, env.py - CY)
        steps_alive += 1
        obs, r, term, trunc, info = env.step(a)
        ep = info["episodes"]; dm = ep["done"]
        if bool(dm.any()):
            scores += ep["score"][dm].tolist(); survs += ep["survived"][dm].tolist()
            done_count += int(dm.sum())
        steps += 1
    sc, sv = np.array(scores[:n]), np.array(survs[:n])
    jitter = (jit_sum / steps_alive.clamp(min=1)).mean().item()
    cdist = (cdist_s / steps_alive.clamp(min=1)).mean().item()
    safe_spd = (spd_safe_s.sum() / spd_safe_c.sum().clamp(min=1)).item()
    dng_spd = (spd_dng_s.sum() / spd_dng_c.sum().clamp(min=1)).item()
    cfg = ck.get("config", {})
    return dict(path=path, score=float(sc.mean()), surv=float(sv.mean()), jitter=jitter,
                cdist=cdist, safe_spd=safe_spd, dng_spd=dng_spd,
                ratio=safe_spd / max(dng_spd, 1e-6),
                accel=cfg.get("accel_penalty"), speed=cfg.get("speed_penalty"),
                center=cfg.get("center_weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=18000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for p in args.ckpts:
        try:
            rows.append(eval_ckpt(p, dev, n=args.n, max_steps=args.max_steps))
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
    hdr = f"{'run':<26}{'a/s/c':>14}{'score':>8}{'surv':>7}{'jitter':>8}{'cdist':>8}{'safe_v':>8}{'dng_v':>7}{'ratio':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        asc = f"{r['accel']}/{r['speed']}/{r['center']}"
        name = r["path"].replace("runs/", "").replace("/final.pt", "").replace("/best.pt", "")
        print(f"{name:<26}{asc:>14}{r['score']:>8.0f}{r['surv']:>7.0f}{r['jitter']:>8.3f}"
              f"{r['cdist']:>8.0f}{r['safe_spd']:>8.2f}{r['dng_spd']:>7.2f}{r['ratio']:>7.2f}")
    print("\nlower jitter = smoother | lower cdist = more centred | "
          "safe_v<<dng_v (low ratio) = rests when safe")


if __name__ == "__main__":
    main()
