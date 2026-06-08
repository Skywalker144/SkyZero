"""Load a trained SAC checkpoint and play Channel-Dodge greedily.

The obs/action modes and net sizes are read from the config embedded in the
checkpoint by the trainer (train_sac_gpu.py / train_sac.py).

    python evaluate.py runs/sac_gpu_smooth/best.pt --episodes 50
    python evaluate.py runs/sac_gpu_smooth/best.pt --episodes 1 --render
"""

import argparse

import numpy as np
import torch

from env_dodge import ChannelDodgeEnv
from sac_net import SquashedGaussianActor


def build_actfn(ckpt, device):
    cfg = ckpt.get("config", {})
    obs_shape = tuple(ckpt["obs_shape"])
    hidden = tuple(cfg.get("hidden", (256, 256)))

    actor = SquashedGaussianActor(obs_shape[0], 2, hidden).to(device)
    actor.load_state_dict(ckpt["actor"]); actor.eval()

    def act(obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return actor.mean_action(t).squeeze(0).cpu().numpy()
    return act, "continuous", ckpt.get("algo", "sac")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained dodge agent")
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=1_000_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    act, action_mode, algo = build_actfn(ckpt, device)
    cfg = ckpt.get("config", {})
    env = ChannelDodgeEnv(obs_mode=ckpt.get("obs_mode", "vector"), action_mode=action_mode,
                          max_steps=(cfg.get("max_steps") or None),
                          stationary_bonus=cfg.get("stationary_bonus", 0.005),
                          reverse_penalty=cfg.get("reverse_penalty", 0.0),
                          render_mode=("ansi" if args.render else None))

    scores, survs, rets = [], [], []
    for k in range(args.episodes):
        obs, info = env.reset(seed=args.seed + k)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            total += r
            done = term or trunc
        scores.append(info["score"]); survs.append(info["survived"]); rets.append(total)
        print(f"  ep {k:3d}: score={info['score']:5d} surv={info['survived']:6.1f}s ret={total:8.2f}")

    print(f"\n[{algo}] {args.episodes} eps: score {np.mean(scores):.1f} (max {int(np.max(scores))}) | "
          f"surv {np.mean(survs):.1f}s (max {np.max(survs):.1f}) | ret {np.mean(rets):.2f}")


if __name__ == "__main__":
    main()
