"""GPU-native SAC on the vectorized Channel-Dodge env (env_dodge_gpu.VecDodgeGPU).

The champion algorithm on the fast env: a few hundred GPU envs feed a GPU-resident
replay buffer; SAC's many gradient updates (the learner is the bottleneck, not the
env) all run on device — nothing crosses to CPU. SAC's sample efficiency (~10M
steps to a strong policy) + the GPU env's speed -> a strong agent in minutes.

    python train_sac_gpu.py
    python train_sac_gpu.py --num-envs 512 --gradient-steps 2 --total-steps 12000000 \
        --accel-penalty 0.03 --speed-penalty 0.02 --run-name sac_gpu

Why only ~512 envs (not thousands like PPO): SAC reuses data via replay, so beyond
a few hundred envs you over-collect relative to the gradient budget.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_dodge_gpu import VecDodgeGPU
from sac_net import SquashedGaussianActor, TwinQ
from common import Logger

ACT_DIM = 2
TRAIN_FIELDS = ["step", "score_avg", "surv_avg", "ret_avg", "q_loss", "actor_loss",
                "alpha", "q_mean", "speed_avg", "entropy"]


class GPUReplay:
    """All-on-GPU ring replay buffer (no CPU transfer)."""

    def __init__(self, cap, obs_dim, act_dim, dev):
        self.obs = torch.zeros(cap, obs_dim, device=dev)
        self.nobs = torch.zeros(cap, obs_dim, device=dev)
        self.act = torch.zeros(cap, act_dim, device=dev)
        self.rew = torch.zeros(cap, device=dev)
        self.term = torch.zeros(cap, device=dev)
        self.cap, self.dev = cap, dev
        self.pos, self.size = 0, 0

    def add_batch(self, o, a, r, no, t):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.dev) + self.pos) % self.cap
        self.obs[idx] = o; self.act[idx] = a; self.rew[idx] = r
        self.nobs[idx] = no; self.term[idx] = t
        self.pos = (self.pos + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.size, (b,), device=self.dev)
        return self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.term[i]


def parse_cli():
    p = argparse.ArgumentParser(description="GPU-native SAC on Channel-Dodge")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--total-steps", type=int, default=12_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gradient-steps", type=int, default=2)
    p.add_argument("--learning-starts", type=int, default=25_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target-entropy-scale", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=18000)
    p.add_argument("--accel-penalty", type=float, default=0.0)
    p.add_argument("--jerk-penalty", type=float, default=0.0)
    p.add_argument("--speed-penalty", type=float, default=0.0)
    p.add_argument("--center-weight", type=float, default=0.0)
    p.add_argument("--stationary-bonus", type=float, default=0.005)
    p.add_argument("--reverse-penalty", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--log-every", type=int, default=200_000)
    p.add_argument("--run-name", default="sac_gpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="", help="checkpoint to resume actor/critic/alpha from")
    return p.parse_args()


@torch.no_grad()
def gpu_eval(actor, dev, env_kwargs, episodes=30, max_steps=18000, seed=999):
    n = max(episodes, 64)
    ev = VecDodgeGPU(n, device=dev, max_steps=max_steps, **env_kwargs)
    obs = ev.reset(seed=seed)
    scores, survs, rets, steps = [], [], [], 0
    while len(scores) < episodes and steps < max_steps + 5:
        obs, r, term, trunc, info = ev.step(actor.mean_action(obs))
        ep = info["episodes"]; dm = ep["done"]
        if bool(dm.any()):
            scores += ep["score"][dm].tolist(); survs += ep["survived"][dm].tolist()
            rets += ep["ret"][dm].tolist()
        steps += 1
    s, v, rt = np.array(scores[:episodes]), np.array(survs[:episodes]), np.array(rets[:episodes])
    return {"score_mean": float(s.mean()), "score_max": int(s.max()),
            "surv_mean": float(v.mean()), "surv_max": float(v.max()),
            "ret_mean": float(rt.mean()), "hp_mean": float("nan"), "len_mean": float(v.mean() * 30)}


def main():
    args = parse_cli()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    N = args.num_envs
    env_kwargs = dict(stationary_bonus=args.stationary_bonus, reverse_penalty=args.reverse_penalty,
                      accel_penalty=args.accel_penalty, jerk_penalty=args.jerk_penalty,
                      speed_penalty=args.speed_penalty, center_weight=args.center_weight)
    venv = VecDodgeGPU(N, device=dev, max_steps=args.max_steps, **env_kwargs)
    obs_dim = venv.obs_dim

    actor = SquashedGaussianActor(obs_dim, ACT_DIM, (256, 256)).to(dev)
    critic = TwinQ(obs_dim, ACT_DIM, (256, 256)).to(dev)
    target = TwinQ(obs_dim, ACT_DIM, (256, 256)).to(dev)
    target.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    target_entropy = -ACT_DIM * args.target_entropy_scale
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    al_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    alpha = log_alpha.exp().item()

    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        target.load_state_dict(ck["critic"])
        if "log_alpha" in ck:                      # restore temperature (else entropy term blows up the policy)
            with torch.no_grad():
                log_alpha.copy_(ck["log_alpha"].to(dev))
            alpha = log_alpha.exp().item()
        print(f"[resume] loaded actor/critic{'/alpha' if 'log_alpha' in ck else ''} from {args.resume} (alpha={alpha:.4f})")

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(out_dir, TRAIN_FIELDS, args.run_name, enable_plot=True)
    best_path, final_path = os.path.join(out_dir, "best.pt"), os.path.join(out_dir, "final.pt")
    buf = GPUReplay(args.buffer_size, obs_dim, ACT_DIM, dev)
    print(f"[setup] GPU-SAC N={N} buffer={args.buffer_size:,} batch={args.batch_size} "
          f"grad/tick={args.gradient_steps} dev={dev} "
          f"params={sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters()):,}")

    obs = venv.reset(seed=args.seed)
    ep_sc, ep_sv, ep_rt = deque(maxlen=2000), deque(maxlen=2000), deque(maxlen=2000)
    speed_hist = deque(maxlen=400)
    gstep, best_eval = 0, -float("inf")
    next_log, next_eval = args.log_every, args.eval_every
    last = {"q_loss": 0.0, "actor_loss": 0.0, "q_mean": 0.0, "entropy": 0.0}
    t0 = time.time()

    while gstep < args.total_steps:
        # ---- collect one step from all N envs ----
        if (not args.resume) and gstep < args.learning_starts:
            actions = torch.rand(N, ACT_DIM, device=dev) * 2 - 1   # random warmup (fresh runs only)
        else:
            with torch.no_grad():
                actions, _ = actor.sample(obs)                     # resume: use the loaded policy from step 0
        obs2, rew, term, trunc, info = venv.step(actions)
        done = (term | trunc)
        next_obs = torch.where(done[:, None], info["final_obs"], obs2)
        buf.add_batch(obs, actions, rew, next_obs, term.float())
        obs = obs2
        gstep += N
        # executed speed = magnitude clamped to the unit disk (matches the env), NOT per-component
        speed_hist.append(float(torch.clamp(torch.hypot(actions[:, 0], actions[:, 1]), max=1.0).mean()))
        ep = info["episodes"]; dm = ep["done"]
        if bool(dm.any()):
            ep_sc.extend(ep["score"][dm].tolist()); ep_sv.extend(ep["survived"][dm].tolist())
            ep_rt.extend(ep["ret"][dm].tolist())

        # ---- learn ----
        if gstep >= args.learning_starts:
            for _ in range(args.gradient_steps):
                o, a, r, no, t = buf.sample(args.batch_size)
                with torch.no_grad():
                    na, nlp = actor.sample(no)
                    q1t, q2t = target(no, na)
                    minqt = torch.min(q1t, q2t) - alpha * nlp
                    tq = r + args.gamma * (1.0 - t) * minqt
                q1, q2 = critic(o, a)
                q_loss = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
                c_opt.zero_grad(set_to_none=True); q_loss.backward(); c_opt.step()

                api, lp = actor.sample(o)
                q1pi, q2pi = critic(o, api)
                minqpi = torch.min(q1pi, q2pi)
                a_loss = (alpha * lp - minqpi).mean()
                a_opt.zero_grad(set_to_none=True); a_loss.backward(); a_opt.step()

                alpha_loss = -(log_alpha * (lp + target_entropy).detach()).mean()
                al_opt.zero_grad(set_to_none=True); alpha_loss.backward(); al_opt.step()
                alpha = log_alpha.exp().item()

                with torch.no_grad():
                    for p, tp in zip(critic.parameters(), target.parameters()):
                        tp.mul_(1.0 - args.tau).add_(args.tau * p)
            last = {"q_loss": float(q_loss.item()), "actor_loss": float(a_loss.item()),
                    "q_mean": float(minqpi.detach().mean().item()),
                    "entropy": float((-lp).detach().mean().item())}

        if gstep >= next_log:
            next_log += args.log_every
            sps = int(gstep / (time.time() - t0))
            sc = np.mean(ep_sc) if ep_sc else float("nan")
            sv = np.mean(ep_sv) if ep_sv else float("nan")
            rt = np.mean(ep_rt) if ep_rt else float("nan")
            print(f"[sac_gpu step {gstep:>10,}] score(avg)={sc:7.1f} surv={sv:5.1f}s ret={rt:7.2f} "
                  f"q_loss={last['q_loss']:.3f} a_loss={last['actor_loss']:.2f} alpha={alpha:.3f} "
                  f"q={last['q_mean']:6.2f} spd={np.mean(speed_hist):.2f} sps={sps:,}")
            logger.log_train(step=gstep, score_avg=sc, surv_avg=sv, ret_avg=rt,
                             q_loss=last["q_loss"], actor_loss=last["actor_loss"], alpha=alpha,
                             q_mean=last["q_mean"], speed_avg=float(np.mean(speed_hist)), entropy=last["entropy"])

        if gstep >= next_eval:
            next_eval += args.eval_every
            actor.eval()
            stats = gpu_eval(actor, dev, env_kwargs, episodes=30, max_steps=args.max_steps)
            actor.train()
            print(f"[sac_gpu EVAL {gstep:>10,}] score(avg/max)={stats['score_mean']:7.1f}/{stats['score_max']} "
                  f"surv(avg/max)={stats['surv_mean']:5.1f}/{stats['surv_max']:5.1f}s")
            logger.log_eval(step=gstep, **stats)
            png = logger.plot()
            if png:
                print(f"[plot {gstep:>10,}] -> {png}")
            ck = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                  "log_alpha": log_alpha.detach().cpu(), "obs_shape": (obs_dim,),
                  "action_mode": "continuous", "algo": "sac", "config": {"hidden": (256, 256), **vars(args)}}
            if stats["score_mean"] > best_eval:
                best_eval = stats["score_mean"]; torch.save(ck, best_path)
            torch.save(ck, final_path)

    print(f"[done] GPU-SAC best eval {best_eval:.1f} -> {best_path}")


if __name__ == "__main__":
    main()
