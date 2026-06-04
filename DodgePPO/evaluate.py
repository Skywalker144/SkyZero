"""Load a trained PPO checkpoint and play Channel-Dodge greedily.

The obs/action modes and network sizes are read from the config embedded in the
checkpoint by ``train_ppo.py``.

    python evaluate.py checkpoints/dodge_best.pt --episodes 50
    python evaluate.py checkpoints/dodge_best.pt --episodes 1 --render
"""

import argparse
import time

import numpy as np
import torch

from config import Config
from env_dodge import ChannelDodgeEnv, NUM_ACTIONS
from networks import ActorCritic


def build_agent(ckpt, device):
    cfgd = ckpt.get("config", {})
    continuous = ckpt.get("action_mode", cfgd.get("action_mode", "discrete")) == "continuous"
    obs_shape = tuple(ckpt["obs_shape"])
    hidden = tuple(cfgd.get("hidden", (256, 256)))
    channels = tuple(cfgd.get("channels", (32, 64)))
    agent = ActorCritic(obs_shape, continuous=continuous,
                        num_actions=(None if continuous else NUM_ACTIONS),
                        act_dim=(2 if continuous else None),
                        hidden=hidden, channels=channels).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    return agent, continuous


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained PPO dodge agent")
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--render", action="store_true", help="print an ANSI status line per step")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to pause per step when rendering")
    ap.add_argument("--seed", type=int, default=1_000_000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent, continuous = build_agent(ckpt, device)
    obs_mode = ckpt.get("obs_mode", ckpt.get("config", {}).get("obs_mode", "vector"))
    action_mode = "continuous" if continuous else "discrete"
    max_steps = ckpt.get("config", {}).get("max_steps", 4000) or None
    print(f"[eval] {args.checkpoint}  obs_mode={obs_mode} action_mode={action_mode}")

    env = ChannelDodgeEnv(obs_mode=obs_mode, action_mode=action_mode, max_steps=max_steps,
                          render_mode=("ansi" if args.render else None))
    scores, survs, lengths, rets = [], [], [], []
    for k in range(args.episodes):
        obs, info = env.reset(seed=args.seed + k)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a = agent.act_greedy(t).squeeze(0).cpu().numpy()
            if not continuous:
                a = int(a)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            if args.render and args.sleep:
                time.sleep(args.sleep)
        scores.append(info["score"]); survs.append(info["survived"])
        lengths.append(info["steps"]); rets.append(total)

    scores, survs = np.array(scores), np.array(survs)
    print(f"\n{args.episodes} episodes:")
    print(f"  score   mean={scores.mean():6.1f}  max={int(scores.max()):>4}  min={int(scores.min()):>4}")
    print(f"  survive mean={survs.mean():5.1f}s  max={survs.max():5.1f}s")
    print(f"  return  mean={np.mean(rets):6.2f}  len mean={np.mean(lengths):.0f} steps")


if __name__ == "__main__":
    main()
