"""Record a trained agent playing Channel-Dodge into a replay file for viewer.html.

Plays N greedy episodes in the *exact* training env and snapshots the full game
state every frame (player, every projectile / laser / bomb / pickup, score, and the
agent's action), then writes ``<out>.js`` defining ``window.REPLAY`` so the viewer
works by just opening the HTML file (no server / no file:// fetch).

    python record_play.py runs/sac/best.pt                       # -> sac_play.js
    python record_play.py runs/sac/best.pt --episodes 3 --max-frames 3600 --out sac_play
    python record_play.py runs/rainbow/best.pt --out rainbow_play

Each frame is rounded to 1 decimal and uses short keys to keep the file small.
"""

import argparse
import json
import math

import numpy as np
import torch

from env_dodge import ChannelDodgeEnv, TYPES, PLAYER_R, LASER_HALFWIDTH
from evaluate import build_actfn

# threat-type -> short integer code (viewer maps code -> color)
KIND_CODE = {"bullet": 0, "split": 1, "aimed": 2, "cannon": 3}


def _r1(x):
    return round(float(x), 1)


def snapshot(env, action_vec):
    p = env.player
    proj = [{"k": KIND_CODE.get(b["kind"], 0), "x": _r1(b["x"]), "y": _r1(b["y"]),
             "r": _r1(b["r"])} for b in env.projectiles]
    lasers = [{"ox": _r1(L["ox"]), "oy": _r1(L["oy"]), "a": round(float(L["ang"]), 4),
               "ph": 0 if L["phase"] == "charge" else 1, "hw": _r1(L["halfWidth"])}
              for L in env.lasers]
    bombs = [{"x": _r1(b["x"]), "y": _r1(b["y"]), "br": _r1(b["blastR"]),
              "ph": 0 if b["phase"] == "fuse" else 1,
              "f": _r1(max(0.0, b["fuse"]))} for b in env.bombs]
    picks = [{"x": _r1(k["x"]), "y": _r1(k["y"])} for k in env.pickups]
    return {
        "px": _r1(p["x"]), "py": _r1(p["y"]), "hp": _r1(p["hp"]),
        "iv": 1 if p["invuln"] > 0 else 0,
        "sc": int(env.score), "t": _r1(env.elapsed),
        "ax": round(float(action_vec[0]), 3), "ay": round(float(action_vec[1]), 3),
        "pr": proj, "la": lasers, "bo": bombs, "pk": picks,
    }


def main():
    ap = argparse.ArgumentParser(description="Record an agent playing for the web viewer")
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=3600, help="cap frames/episode (~30fps)")
    ap.add_argument("--out", default="sac_play", help="output basename (writes <out>.js)")
    ap.add_argument("--seed", type=int, default=1_000_000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    act, action_mode, algo = build_actfn(ckpt, device)
    cfg = ckpt.get("config", {})
    env = ChannelDodgeEnv(obs_mode=ckpt.get("obs_mode", "vector"), action_mode=action_mode,
                          max_steps=(cfg.get("max_steps") or None),
                          stationary_bonus=cfg.get("stationary_bonus", 0.005),
                          reverse_penalty=cfg.get("reverse_penalty", 0.0))

    episodes = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        frames, done = [], False
        while not done and len(frames) < args.max_frames:
            a = act(obs)
            mv = env._action_to_move(a)          # 2D move vector for the action arrow
            obs, r, term, trunc, info = env.step(a)
            frames.append(snapshot(env, mv))
            done = term or trunc
        # jitter = mean frame-to-frame change of the action command (lower = smoother)
        jit = float(np.mean([math.hypot(frames[i]["ax"] - frames[i - 1]["ax"],
                                        frames[i]["ay"] - frames[i - 1]["ay"])
                             for i in range(1, len(frames))])) if len(frames) > 1 else 0.0
        episodes.append({"score": int(info["score"]), "survived": round(float(info["survived"]), 1),
                         "frames": frames, "died": bool(term), "jitter": round(jit, 4)})
        print(f"  ep {ep}: score={info['score']} surv={info['survived']:.1f}s "
              f"frames={len(frames)} jitter(|Δact|)={jit:.4f} {'(died)' if term else '(cap/timeout)'}")

    replay = {
        "meta": {"W": env.W, "H": env.H, "dt": env.dt, "player_r": PLAYER_R,
                 "laser_hw": LASER_HALFWIDTH, "algo": algo, "action_mode": action_mode,
                 "checkpoint": args.checkpoint},
        "episodes": episodes,
    }
    out_path = args.out + ".js"
    with open(out_path, "w") as f:
        f.write("window.REPLAY = ")
        json.dump(replay, f, separators=(",", ":"))
        f.write(";\n")
    total = sum(len(e["frames"]) for e in episodes)
    import os
    print(f"[record] {len(episodes)} eps, {total} frames -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
