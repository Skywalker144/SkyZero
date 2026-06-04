"""Train a PPO agent on Channel-Dodge (bullet-hell survival).

CleanRL-style PPO (clipped surrogate + GAE + minibatch epochs + value clipping +
entropy bonus + LR anneal), wired to the SkyZero ``.cfg`` config style and to the
``ChannelDodgeEnv`` vector env.  Handles both observation modes (vector MLP / grid
CNN) and both action modes (discrete categorical / continuous Gaussian), and does
proper truncation bootstrapping (a timed-out episode is bootstrapped from the
value of its real terminal observation, not treated as a hard terminal).

    python train_ppo.py configs/dodge.cfg
    python train_ppo.py configs/dodge.cfg --set total_steps=5000000 --set num_envs=32
    python train_ppo.py configs/dodge.cfg --print-config

Periodic greedy evaluation prints in-game score / survival time / return; metrics
go to ``checkpoints/<tag>_{train,eval}.csv`` with a dashboard PNG rebuilt each eval.
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from config import load_config, parse_overrides
from env_dodge import ChannelDodgeEnv, make_vec_env, NUM_ACTIONS
from networks import ActorCritic


def parse_cli():
    p = argparse.ArgumentParser(description="PPO on Channel-Dodge (config-file driven)")
    p.add_argument("config", nargs="?", default="configs/dodge.cfg")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--print-config", action="store_true")
    return p.parse_args()


def make_agent(cfg, obs_shape):
    continuous = cfg.action_mode == "continuous"
    return ActorCritic(
        obs_shape, continuous=continuous,
        num_actions=(None if continuous else NUM_ACTIONS),
        act_dim=(2 if continuous else None),
        hidden=tuple(cfg.hidden), channels=tuple(cfg.channels),
    )


@torch.no_grad()
def evaluate(agent, cfg, device, episodes, seed_base=1_000_000):
    env = ChannelDodgeEnv(obs_mode=cfg.obs_mode, action_mode=cfg.action_mode,
                          max_steps=(cfg.max_steps or None),
                          stationary_bonus=cfg.stationary_bonus, reverse_penalty=cfg.reverse_penalty)
    scores, survs, lengths, hps, rets = [], [], [], [], []
    for k in range(episodes):
        obs, info = env.reset(seed=seed_base + k)
        done = False
        total = 0.0
        while not done:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = agent.act_greedy(t).squeeze(0).cpu().numpy()
            if cfg.action_mode == "discrete":
                a = int(a)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        scores.append(info["score"]); survs.append(info["survived"])
        lengths.append(info["steps"]); hps.append(info["hp"]); rets.append(total)
    return {
        "score_mean": float(np.mean(scores)), "score_max": int(np.max(scores)),
        "surv_mean": float(np.mean(survs)), "surv_max": float(np.max(survs)),
        "ret_mean": float(np.mean(rets)), "hp_mean": float(np.mean(hps)),
        "len_mean": float(np.mean(lengths)),
    }


# --------------------------------------------------------------------------- #
# CSV logging + dashboard
# --------------------------------------------------------------------------- #
TRAIN_FIELDS = ["step", "score_avg", "surv_avg", "ret_avg", "pg_loss", "v_loss",
                "entropy", "approx_kl", "lr"]
EVAL_FIELDS = ["step", "score_mean", "score_max", "surv_mean", "surv_max",
               "ret_mean", "hp_mean", "len_mean"]


class DodgeLogger:
    def __init__(self, out_dir, tag="", enable_plot=True):
        os.makedirs(out_dir, exist_ok=True)
        self.tag, self.enable_plot = tag, enable_plot
        self.train_csv = os.path.join(out_dir, "train.csv")
        self.eval_csv = os.path.join(out_dir, "eval.csv")
        self.png = os.path.join(out_dir, "progress.png")
        self._init(self.train_csv, TRAIN_FIELDS)
        self._init(self.eval_csv, EVAL_FIELDS)

    @staticmethod
    def _init(path, fields):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    @staticmethod
    def _append(path, fields, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in fields])

    def log_train(self, **row):
        self._append(self.train_csv, TRAIN_FIELDS, row)

    def log_eval(self, **row):
        self._append(self.eval_csv, EVAL_FIELDS, row)

    def plot(self):
        if not self.enable_plot:
            return None
        try:
            return _plot_dashboard(self.train_csv, self.eval_csv, self.png, self.tag)
        except Exception as e:
            print(f"[plot] skipped ({type(e).__name__}: {e})")
            return None


def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
            for k in rows[0]}


def _plot_dashboard(train_csv, eval_csv, out_png, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, ev = _read_csv(train_csv), _read_csv(eval_csv)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Channel-Dodge PPO progress — {tag}", fontsize=13)

    ax = axes[0, 0]
    if tr is not None:
        ax.plot(tr["step"], tr["score_avg"], color="tab:blue", alpha=0.45, lw=1.2, label="train avg")
    if ev is not None:
        ax.plot(ev["step"], ev["score_mean"], "o-", color="tab:red", label="eval mean")
        ax.plot(ev["step"], ev["score_max"], "x--", color="tab:orange", alpha=0.6, label="eval max")
    ax.set_title("In-game score"); ax.set_xlabel("env steps"); ax.set_ylabel("score")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if tr is not None:
        ax.plot(tr["step"], tr["surv_avg"], color="tab:blue", alpha=0.45, lw=1.2, label="train avg")
    if ev is not None:
        ax.plot(ev["step"], ev["surv_mean"], "o-", color="tab:green", label="eval mean")
        ax.plot(ev["step"], ev["surv_max"], "x--", color="tab:olive", alpha=0.6, label="eval max")
    ax.set_title("Survival time (s)"); ax.set_xlabel("env steps"); ax.set_ylabel("seconds")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    if ev is not None:
        ax.plot(ev["step"], ev["ret_mean"], "o-", color="tab:brown", label="eval return")
    if tr is not None:
        ax.plot(tr["step"], tr["ret_avg"], color="tab:purple", alpha=0.45, lw=1.2, label="train return")
    ax.set_title("Episode return"); ax.set_xlabel("env steps"); ax.set_ylabel("return")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    if tr is not None:
        ax.plot(tr["step"], tr["entropy"], color="tab:gray", lw=1, label="entropy")
        ax2 = ax.twinx()
        ax2.plot(tr["step"], tr["approx_kl"], color="tab:red", lw=1, alpha=0.6, label="approx_kl")
        ax2.set_ylabel("approx_kl")
    ax.set_title("Policy entropy / KL"); ax.set_xlabel("env steps"); ax.set_ylabel("entropy")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def main():
    args = parse_cli()
    if args.config and not os.path.exists(args.config):
        print(f"[train] config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(args.config, parse_overrides(args.overrides))
    print(f"[train] config: {args.config}")
    print(cfg.pretty())
    if args.print_config:
        return

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg.device == "auto" else cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    continuous = cfg.action_mode == "continuous"
    # Build the (possibly multi-process) vector env BEFORE moving the model to
    # CUDA: SubprocVectorDodgeEnv forks, and forking a CUDA-uninitialized parent
    # keeps the NumPy-only workers clean.
    envs = make_vec_env(cfg.num_envs, num_workers=cfg.num_workers,
                        obs_mode=cfg.obs_mode, action_mode=cfg.action_mode,
                        max_steps=(cfg.max_steps or None),
                        stationary_bonus=cfg.stationary_bonus, reverse_penalty=cfg.reverse_penalty)
    obs_shape = envs.obs_shape
    agent = make_agent(cfg, obs_shape).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    if cfg.resume:
        if not os.path.exists(cfg.resume):
            print(f"[resume] checkpoint not found: {cfg.resume}", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(cfg.resume, map_location=device)
        agent.load_state_dict(ckpt["model"])
        try:
            opt.load_state_dict(ckpt["opt"])
            for g in opt.param_groups:          # reset LR (the saved one may be ~0 post-anneal)
                g["lr"] = cfg.lr
            opt_msg = "model+optimizer"
        except Exception as e:
            opt_msg = f"model only (optimizer skipped: {type(e).__name__})"
        print(f"[resume] loaded {opt_msg} from {cfg.resume}")

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[setup] obs_mode={cfg.obs_mode} action_mode={cfg.action_mode} "
          f"obs_shape={obs_shape} device={device} params={n_params:,}")
    print(f"[setup] num_envs={cfg.num_envs} num_workers={cfg.num_workers} "
          f"num_steps={cfg.num_steps} batch={cfg.batch_size} "
          f"minibatch={cfg.minibatch_size} epochs={cfg.update_epochs}")

    # Per-run isolated output directory: runs/<run_name>/ (V7.1-style — one run
    # never contaminates another). Snapshot the resolved config for reproducibility.
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    final_path = os.path.join(out_dir, "final.pt")
    with open(os.path.join(out_dir, "run.cfg"), "w") as f:
        f.write(cfg.to_cfg())
    print(f"[setup] out_dir={out_dir}")
    meta = asdict(cfg)
    logger = DodgeLogger(out_dir, cfg.tag, enable_plot=cfg.plot)

    def save(path):
        torch.save({"model": agent.state_dict(), "opt": opt.state_dict(),
                    "config": meta, "obs_mode": cfg.obs_mode,
                    "action_mode": cfg.action_mode, "obs_shape": obs_shape}, path)

    # rollout storage
    N, E = cfg.num_steps, cfg.num_envs
    obs_buf = torch.zeros((N, E, *obs_shape), dtype=torch.float32, device=device)
    if continuous:
        act_buf = torch.zeros((N, E, agent.act_dim), dtype=torch.float32, device=device)
    else:
        act_buf = torch.zeros((N, E), dtype=torch.long, device=device)
    logp_buf = torch.zeros((N, E), device=device)
    rew_buf = torch.zeros((N, E), device=device)
    done_buf = torch.zeros((N, E), device=device)
    val_buf = torch.zeros((N, E), device=device)

    next_obs = torch.as_tensor(envs.reset(seed=cfg.seed), dtype=torch.float32, device=device)
    next_done = torch.zeros(E, device=device)

    batch = cfg.batch_size
    total_updates = (cfg.total_steps // batch) if cfg.total_steps > 0 else 0
    if total_updates == 0:
        print("[train] total_steps<=0 -> training indefinitely; press Ctrl-C to stop and save")

    # rolling train stats (over recently finished episodes)
    from collections import deque
    ep_scores, ep_survs, ep_rets = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

    global_step = 0
    best_eval = -float("inf")
    t0 = time.time()
    update = 0
    try:
        while total_updates == 0 or update < total_updates:
            update += 1
            if cfg.anneal_lr and total_updates > 0:
                frac = 1.0 - (update - 1.0) / total_updates
                for g in opt.param_groups:
                    g["lr"] = frac * cfg.lr

            # ----- collect a rollout -----
            for t in range(N):
                global_step += E
                obs_buf[t] = next_obs
                done_buf[t] = next_done
                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(next_obs)
                val_buf[t] = value
                act_buf[t] = action
                logp_buf[t] = logp

                act_np = action.cpu().numpy()
                next_obs_np, reward, term, trunc, infos = envs.step(act_np)

                # truncation bootstrap: fold gamma * V(terminal_obs) into the reward
                reward = reward.astype(np.float32) * cfg.reward_scale
                trunc_only = np.where(trunc & ~term)[0]
                if len(trunc_only):
                    finals = np.stack([infos["final_obs"][i] for i in trunc_only])
                    with torch.no_grad():
                        vfin = agent.get_value(
                            torch.as_tensor(finals, dtype=torch.float32, device=device)
                        ).cpu().numpy()
                    reward[trunc_only] += cfg.gamma * vfin

                rew_buf[t] = torch.as_tensor(reward, device=device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor((term | trunc).astype(np.float32), device=device)

                for ep in infos["episodes"]:
                    ep_scores.append(ep["score"]); ep_survs.append(ep["survived"])
                    ep_rets.append(ep["return"])

            # ----- GAE -----
            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros_like(rew_buf, device=device)
                lastgaelam = 0
                for t in reversed(range(N)):
                    if t == N - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = val_buf[t + 1]
                    delta = rew_buf[t] + cfg.gamma * nextvalues * nextnonterminal - val_buf[t]
                    advantages[t] = lastgaelam = (
                        delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam)
                returns = advantages + val_buf

            # ----- flatten the batch -----
            b_obs = obs_buf.reshape((-1, *obs_shape))
            b_logp = logp_buf.reshape(-1)
            b_act = act_buf.reshape((-1, agent.act_dim)) if continuous else act_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            b_val = val_buf.reshape(-1)

            # ----- PPO update -----
            idx = np.arange(batch)
            clipfracs = []
            approx_kl = pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(idx)
                for start in range(0, batch, cfg.minibatch_size):
                    mb = idx[start:start + cfg.minibatch_size]
                    _, newlogp, entropy, newval = agent.get_action_and_value(
                        b_obs[mb], b_act[mb])
                    logratio = newlogp - b_logp[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                    mb_adv = b_adv[mb]
                    if cfg.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    if cfg.clip_vloss:
                        v_unclipped = (newval - b_ret[mb]) ** 2
                        v_clipped = b_val[mb] + torch.clamp(
                            newval - b_val[mb], -cfg.clip_coef, cfg.clip_coef)
                        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    opt.step()

                if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl:
                    break

            # ----- logging -----
            if update % cfg.log_every == 0:
                sps = int(global_step / (time.time() - t0))
                sc = np.mean(ep_scores) if ep_scores else float("nan")
                sv = np.mean(ep_survs) if ep_survs else float("nan")
                rt = np.mean(ep_rets) if ep_rets else float("nan")
                lr_now = opt.param_groups[0]["lr"]
                print(f"[upd {update:>5} | step {global_step:>9,}] "
                      f"score(avg)={sc:6.1f} surv(avg)={sv:5.1f}s ret={rt:6.2f} "
                      f"ent={entropy_loss.item():.3f} kl={approx_kl.item():.4f} "
                      f"v_loss={v_loss.item():.3f} sps={sps}")
                logger.log_train(step=global_step, score_avg=sc, surv_avg=sv, ret_avg=rt,
                                 pg_loss=pg_loss.item(), v_loss=v_loss.item(),
                                 entropy=entropy_loss.item(), approx_kl=approx_kl.item(), lr=lr_now)

            # ----- eval -----
            if cfg.eval_every and update % cfg.eval_every == 0:
                stats = evaluate(agent, cfg, device, cfg.eval_episodes)
                print(f"[eval  step {global_step:>9,}] "
                      f"score(avg/max)={stats['score_mean']:6.1f}/{stats['score_max']:>4} "
                      f"surv(avg/max)={stats['surv_mean']:5.1f}/{stats['surv_max']:5.1f}s "
                      f"ret={stats['ret_mean']:6.2f} hp={stats['hp_mean']:4.1f} len={stats['len_mean']:.0f}")
                logger.log_eval(step=global_step, score_mean=stats["score_mean"],
                                score_max=stats["score_max"], surv_mean=stats["surv_mean"],
                                surv_max=stats["surv_max"], ret_mean=stats["ret_mean"],
                                hp_mean=stats["hp_mean"], len_mean=stats["len_mean"])
                png = logger.plot()
                if png:
                    print(f"[plot  step {global_step:>9,}] dashboard -> {png}")
                if stats["score_mean"] > best_eval:
                    best_eval = stats["score_mean"]
                    save(best_path)
                    print(f"[eval  step {global_step:>9,}] new best avg score {best_eval:.1f} -> {best_path}")
                save(final_path)
    except KeyboardInterrupt:
        print(f"\n[train] interrupted at update {update} (step {global_step:,}) -- saving")

    save(final_path)
    envs.close()
    print(f"[done] saved final model -> {final_path}  (best eval avg score={best_eval:.1f})")


if __name__ == "__main__":
    main()
