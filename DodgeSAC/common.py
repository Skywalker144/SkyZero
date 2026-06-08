"""Shared training infrastructure for the off-policy agents: greedy evaluation,
CSV logging, and the progress dashboard — the value-based (Rainbow/QR-DQN) and SAC
trainers all use these so the runs are directly comparable to the PPO ones.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from env_dodge import ChannelDodgeEnv


# --------------------------------------------------------------------------- #
# greedy evaluation (same protocol as train_ppo.evaluate)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(act_fn, env_kwargs, episodes, seed_base=1_000_000):
    """``act_fn(obs_np) -> action`` (deterministic). Returns score/surv/return stats."""
    env = ChannelDodgeEnv(**env_kwargs)
    scores, survs, lengths, hps, rets = [], [], [], [], []
    for k in range(episodes):
        obs, info = env.reset(seed=seed_base + k)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs))
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


EVAL_FIELDS = ["step", "score_mean", "score_max", "surv_mean", "surv_max",
               "ret_mean", "hp_mean", "len_mean"]


# --------------------------------------------------------------------------- #
# CSV logging + dashboard
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, out_dir, train_fields, tag="", enable_plot=True):
        os.makedirs(out_dir, exist_ok=True)
        self.tag, self.enable_plot = tag, enable_plot
        self.train_fields = list(train_fields)
        self.train_csv = os.path.join(out_dir, "train.csv")
        self.eval_csv = os.path.join(out_dir, "eval.csv")
        self.png = os.path.join(out_dir, "progress.png")
        self._init(self.train_csv, self.train_fields)
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
        self._append(self.train_csv, self.train_fields, row)

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
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Channel-Dodge progress — {tag}", fontsize=13)

    ax = axes[0, 0]
    if tr is not None and "score_avg" in tr:
        ax.plot(tr["step"], tr["score_avg"], color="tab:blue", alpha=0.45, lw=1.2, label="train avg")
    if ev is not None:
        ax.plot(ev["step"], ev["score_mean"], "o-", color="tab:red", label="eval mean")
        ax.plot(ev["step"], ev["score_max"], "x--", color="tab:orange", alpha=0.6, label="eval max")
    ax.set_title("In-game score"); ax.set_xlabel("env steps"); ax.set_ylabel("score")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # [0,1] Critic — value estimate (Q) + critic loss (the two are NOT redundant with
    # score: Q is the discounted-value scale, loss is fit stability).
    ax = axes[0, 1]
    loss_col = next((c for c in ("q_loss", "v_loss", "loss") if tr is not None and c in tr), None)
    if tr is not None and "q_mean" in tr:                 # value/SAC: Q + critic loss
        ax.plot(tr["step"], tr["q_mean"], color="tab:purple", lw=1.3, label="Q (value est.)")
        ax.set_ylabel("Q"); ax.set_title("Critic: value & loss")
        if loss_col:
            ax2 = ax.twinx(); ax2.plot(tr["step"], tr[loss_col], color="tab:gray", lw=1, alpha=0.6)
            ax2.set_ylabel(loss_col + " (gray)")
    elif loss_col:                                        # PPO: value loss alone
        ax.plot(tr["step"], tr[loss_col], color="tab:gray", lw=1.2, label=loss_col)
        ax.set_ylabel(loss_col); ax.set_title("Value loss")
    else:
        ax.set_title("Critic / loss")
    ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper left")

    # [1,0] Exploration state — SAC: temperature alpha (+ policy entropy);
    # value agents: epsilon (+ PER beta).
    ax = axes[1, 0]
    twin = None
    if tr is not None and "alpha" in tr:                  # SAC: temperature (+ entropy)
        ax.plot(tr["step"], tr["alpha"], color="tab:orange", lw=1.3, label="alpha (temp)")
        ax.set_ylabel("alpha"); ax.set_title("Temperature / entropy")
        twin = "entropy" if "entropy" in tr else None
    elif tr is not None and "eps" in tr:                  # value: epsilon (+ PER beta)
        ax.plot(tr["step"], tr["eps"], color="tab:orange", lw=1.3, label="epsilon")
        ax.set_ylabel("epsilon"); ax.set_title("Exploration / PER beta")
        twin = "beta" if "beta" in tr else None
    elif tr is not None and "entropy" in tr:              # PPO: policy entropy (+ approx_kl)
        ax.plot(tr["step"], tr["entropy"], color="tab:orange", lw=1.3, label="policy entropy")
        ax.set_ylabel("entropy"); ax.set_title("Policy entropy / approx-KL")
        twin = "approx_kl" if "approx_kl" in tr else None
    else:
        ax.set_title("Exploration")
    if twin:
        ax2 = ax.twinx(); ax2.plot(tr["step"], tr[twin], color="tab:green", lw=1, alpha=0.7)
        ax2.set_ylabel(twin + " (green)")
    ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper left")

    # [1,1] Behavior — mean move speed |v| (0=rest, 1=max): the energy signal.
    # Falls back to survival for runs that didn't log speed.
    ax = axes[1, 1]
    if tr is not None and "speed_avg" in tr:
        ax.plot(tr["step"], tr["speed_avg"], color="tab:red", lw=1.3, label="mean |v| (train)")
        ax.set_ylim(0, 1.02); ax.set_ylabel("mean move speed  (0=rest, 1=max)")
        ax.set_title("Move speed (energy)")
        ax.legend(fontsize=8, loc="upper left")
    elif ev is not None:
        ax.plot(ev["step"], ev["surv_mean"], "o-", color="tab:green", label="eval surv")
        ax.set_ylabel("seconds"); ax.set_title("Survival time (s)")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("env steps"); ax.grid(alpha=0.3)

    # [0,2] Episode reward (return) — unlike in-game score (which saturates at the
    # 600s time cap), the shaped return keeps moving as economy/smoothness/centering
    # improve, so it shows progress the score curve can't.
    ax = axes[0, 2]
    if tr is not None and "ret_avg" in tr:
        ax.plot(tr["step"], tr["ret_avg"], color="tab:green", lw=1.3, label="train return")
    if ev is not None and "ret_mean" in ev and np.isfinite(ev["ret_mean"]).any():
        ax.plot(ev["step"], ev["ret_mean"], "o-", color="tab:olive", label="eval return")
    ax.set_title("Episode reward (return)"); ax.set_xlabel("env steps"); ax.set_ylabel("return")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="lower right")

    # [1,2] Survival time.
    ax = axes[1, 2]
    if ev is not None and "surv_mean" in ev:
        ax.plot(ev["step"], ev["surv_mean"], "o-", color="tab:green", label="eval surv")
        if "surv_max" in ev:
            ax.plot(ev["step"], ev["surv_max"], "x--", color="tab:olive", alpha=0.6, label="eval max")
    ax.set_title("Survival time (s)"); ax.set_xlabel("env steps"); ax.set_ylabel("seconds")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
