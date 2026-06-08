"""Train a SAC agent on the *continuous* Channel-Dodge action mode — the off-policy
counterpart to the continuous-PPO champion.

Soft Actor-Critic: a squashed-Gaussian policy, twin Q critics with clipped-double-Q
targets, soft (Polyak) target updates, and auto-tuned entropy temperature. Shares
the multi-process collection + replay + logging harness with the value agents, so
its curves overlay the PPO ones directly.

    python train_sac.py configs/sac.cfg
    python train_sac.py configs/sac.cfg --set total_steps=8000000
"""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_rl import SACConfig, load_config, parse_overrides
from env_dodge import make_vec_env
from sac_net import SquashedGaussianActor, TwinQ
from replay_buffer import UniformReplay, NStepCollector
from common import Logger, evaluate

ACT_DIM = 2
TRAIN_FIELDS = ["step", "score_avg", "surv_avg", "ret_avg", "q_loss", "actor_loss",
                "alpha", "q_mean", "speed_avg", "entropy"]


def parse_cli():
    p = argparse.ArgumentParser(description="SAC on Channel-Dodge (continuous)")
    p.add_argument("config", nargs="?", default="configs/sac.cfg")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--print-config", action="store_true")
    return p.parse_args()


def main():
    args = parse_cli()
    if args.config and not os.path.exists(args.config):
        print(f"[train] config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(SACConfig, args.config, parse_overrides(args.overrides))
    print(f"[train] config: {args.config}")
    print(cfg.pretty())
    if args.print_config:
        return

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg.device == "auto" else cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env_kwargs = dict(obs_mode=cfg.obs_mode, action_mode="continuous",
                      max_steps=(cfg.max_steps or None),
                      stationary_bonus=cfg.stationary_bonus, reverse_penalty=cfg.reverse_penalty,
                      accel_penalty=cfg.accel_penalty, jerk_penalty=cfg.jerk_penalty,
                      speed_penalty=cfg.speed_penalty, center_weight=cfg.center_weight)
    # smoothness annealing: start the *training* envs at 0 penalty and ramp to the
    # target so the critic stays calibrated (full-strength-from-start shocks it).
    anneal = cfg.smooth_anneal_steps > 0
    train_env_kwargs = dict(env_kwargs)
    if anneal:
        train_env_kwargs["accel_penalty"] = 0.0
        train_env_kwargs["jerk_penalty"] = 0.0
    envs = make_vec_env(cfg.num_envs, num_workers=cfg.num_workers, **train_env_kwargs)
    obs_shape = envs.obs_shape
    obs_dim = obs_shape[0]

    actor = SquashedGaussianActor(obs_dim, ACT_DIM, tuple(cfg.hidden)).to(device)
    critic = TwinQ(obs_dim, ACT_DIM, tuple(cfg.hidden)).to(device)
    target_critic = TwinQ(obs_dim, ACT_DIM, tuple(cfg.hidden)).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    target_entropy = -ACT_DIM * cfg.target_entropy_scale
    if cfg.autotune_alpha:
        log_alpha = torch.zeros(1, device=device, requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr, eps=cfg.adam_eps)
        alpha = log_alpha.exp().item()
    else:
        log_alpha, alpha_opt, alpha = None, None, cfg.alpha

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        actor.load_state_dict(ckpt["actor"]); critic.load_state_dict(ckpt["critic"])
        target_critic.load_state_dict(ckpt["critic"])
        print(f"[resume] loaded actor+critic from {cfg.resume}")

    buffer = UniformReplay(cfg.buffer_size, obs_shape, act_shape=(ACT_DIM,))
    collectors = [NStepCollector(cfg.n_step, cfg.gamma) for _ in range(cfg.num_envs)]

    n_params = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"[setup] SAC obs_shape={obs_shape} device={device} params={n_params:,} "
          f"autotune_alpha={cfg.autotune_alpha} target_entropy={target_entropy}")
    print(f"[setup] num_envs={cfg.num_envs} workers={cfg.num_workers} buffer={cfg.buffer_size:,} "
          f"batch={cfg.batch_size} grad_steps/tick={cfg.gradient_steps} n_step={cfg.n_step}")

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_path, final_path = os.path.join(out_dir, "best.pt"), os.path.join(out_dir, "final.pt")
    with open(os.path.join(out_dir, "run.cfg"), "w") as f:
        f.write(cfg.to_cfg())
    print(f"[setup] out_dir={out_dir}")
    logger = Logger(out_dir, TRAIN_FIELDS, cfg.tag, enable_plot=cfg.plot)
    meta = asdict(cfg)

    def save(path):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "config": meta, "obs_mode": cfg.obs_mode, "obs_shape": obs_shape,
                    "action_mode": "continuous", "algo": "sac"}, path)

    def greedy_act(obs_np):
        t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        return actor.mean_action(t).squeeze(0).cpu().numpy()

    def run_eval():
        actor.eval()
        stats = evaluate(greedy_act, env_kwargs, cfg.eval_episodes)
        actor.train()
        return stats

    running_obs = envs.reset(seed=cfg.seed)
    ep_scores, ep_survs, ep_rets = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    speed_hist = deque(maxlen=200)   # recent per-tick mean move speed |mv| (energy diagnostic)
    global_step, best_eval = 0, -float("inf")
    next_log, next_eval, next_smooth = cfg.log_every, cfg.eval_every, 0
    smooth_frac = 1.0 if not anneal else 0.0
    last = {"q_loss": 0.0, "actor_loss": 0.0, "q_mean": 0.0, "entropy": 0.0}
    t0 = time.time()
    tick = 0

    try:
        while cfg.total_steps <= 0 or global_step < cfg.total_steps:
            tick += 1
            # ---- action selection ----
            if global_step < cfg.learning_starts:
                actions = np.random.uniform(-1.0, 1.0, size=(cfg.num_envs, ACT_DIM)).astype(np.float32)
            else:
                with torch.no_grad():
                    t = torch.as_tensor(running_obs, dtype=torch.float32, device=device)
                    a, _ = actor.sample(t)
                    actions = a.cpu().numpy()

            # ---- step env, store n-step transitions ----
            next_obs_np, reward, term, trunc, infos = envs.step(actions)
            reward = reward.astype(np.float32) * cfg.reward_scale
            for i in range(cfg.num_envs):
                ended = bool(term[i] or trunc[i])
                s_next = infos["final_obs"][i] if ended else next_obs_np[i]
                for tr in collectors[i].push(running_obs[i], actions[i].astype(np.float32),
                                             float(reward[i]), s_next, bool(term[i])):
                    buffer.add(tr)
                if ended:
                    for tr in collectors[i].flush():
                        buffer.add(tr)
            running_obs = next_obs_np
            global_step += cfg.num_envs
            speed_hist.append(float(np.minimum(np.linalg.norm(actions, axis=1), 1.0).mean()))
            for ep in infos["episodes"]:
                ep_scores.append(ep["score"]); ep_survs.append(ep["survived"]); ep_rets.append(ep["return"])

            # ---- anneal the smoothness penalty 0 -> target (env-side, every ~10k steps) ----
            if anneal and global_step >= next_smooth:
                next_smooth += 10000
                smooth_frac = min(1.0, global_step / cfg.smooth_anneal_steps)
                envs.set_env_attr("accel_penalty", smooth_frac * cfg.accel_penalty)
                envs.set_env_attr("jerk_penalty", smooth_frac * cfg.jerk_penalty)

            # ---- learn ----
            if global_step >= cfg.learning_starts and tick % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    batch = buffer.sample(cfg.batch_size)
                    obs = torch.as_tensor(batch["obs"], device=device)
                    act = torch.as_tensor(batch["actions"], device=device)
                    ret = torch.as_tensor(batch["returns"], device=device)
                    nobs = torch.as_tensor(batch["next_obs"], device=device)
                    disc = torch.as_tensor(batch["disc"], device=device)
                    term_t = torch.as_tensor(batch["terminal"], device=device)

                    # critic update
                    with torch.no_grad():
                        next_a, next_logp = actor.sample(nobs)
                        q1t, q2t = target_critic(nobs, next_a)
                        min_qt = torch.min(q1t, q2t) - alpha * next_logp
                        target_q = ret + disc * (1.0 - term_t) * min_qt
                    q1, q2 = critic(obs, act)
                    q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                    critic_opt.zero_grad(set_to_none=True)
                    q_loss.backward()
                    if cfg.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                    critic_opt.step()

                    # actor update
                    a_pi, logp_pi = actor.sample(obs)
                    q1pi, q2pi = critic(obs, a_pi)
                    min_qpi = torch.min(q1pi, q2pi)
                    actor_loss = (alpha * logp_pi - min_qpi).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if cfg.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    actor_opt.step()

                    # temperature update
                    if cfg.autotune_alpha:
                        alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                        alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_opt.step()
                        alpha = log_alpha.exp().item()

                    # soft target update
                    with torch.no_grad():
                        for p, tp in zip(critic.parameters(), target_critic.parameters()):
                            tp.mul_(1.0 - cfg.tau).add_(cfg.tau * p)
                last = {"q_loss": float(q_loss.item()), "actor_loss": float(actor_loss.item()),
                        "q_mean": float(min_qpi.detach().mean().item()),
                        "entropy": float((-logp_pi).detach().mean().item())}

            # ---- logging ----
            if global_step >= next_log:
                next_log += cfg.log_every
                sps = int(global_step / (time.time() - t0))
                sc = np.mean(ep_scores) if ep_scores else float("nan")
                sv = np.mean(ep_survs) if ep_survs else float("nan")
                rt = np.mean(ep_rets) if ep_rets else float("nan")
                print(f"[sac step {global_step:>9,}] score(avg)={sc:6.1f} surv(avg)={sv:5.1f}s "
                      f"ret={rt:7.2f} q_loss={last['q_loss']:.3f} a_loss={last['actor_loss']:.3f} "
                      f"alpha={alpha:.3f} acc={smooth_frac * cfg.accel_penalty:.3f} "
                      f"q={last['q_mean']:6.2f} buf={buffer.size:>7,} sps={sps}")
                logger.log_train(step=global_step, score_avg=sc, surv_avg=sv, ret_avg=rt,
                                 q_loss=last["q_loss"], actor_loss=last["actor_loss"],
                                 alpha=alpha, q_mean=last["q_mean"],
                                 speed_avg=(float(np.mean(speed_hist)) if speed_hist else float("nan")),
                                 entropy=last["entropy"])

            # ---- eval ----
            if global_step >= next_eval:
                next_eval += cfg.eval_every
                stats = run_eval()
                print(f"[sac EVAL {global_step:>9,}] "
                      f"score(avg/max)={stats['score_mean']:6.1f}/{stats['score_max']:>4} "
                      f"surv(avg/max)={stats['surv_mean']:5.1f}/{stats['surv_max']:5.1f}s "
                      f"ret={stats['ret_mean']:7.2f} hp={stats['hp_mean']:4.1f} len={stats['len_mean']:.0f}")
                logger.log_eval(step=global_step, **stats)
                png = logger.plot()
                if png:
                    print(f"[plot {global_step:>9,}] -> {png}")
                if stats["score_mean"] > best_eval:
                    best_eval = stats["score_mean"]
                    save(best_path)
                    print(f"[eval {global_step:>9,}] new best avg score {best_eval:.1f} -> {best_path}")
                save(final_path)
    except KeyboardInterrupt:
        print(f"\n[train] interrupted at step {global_step:,} -- saving")

    save(final_path)
    envs.close()
    print(f"[done] SAC: saved -> {final_path}  (best eval avg score={best_eval:.1f})")


if __name__ == "__main__":
    main()
