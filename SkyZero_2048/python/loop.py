"""Training loop orchestrator for 2048 Stochastic Gumbel AlphaZero.

One iteration: self-play (batched MCTS) -> append to replay buffer -> train ->
periodic eval -> checkpoint. Run with:

    python loop.py --iters 40 --data data2048
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import time

import numpy as np
import torch

import selfplay
from model_config import CONFIG, Config
from evaluate import evaluate
from mcts import net_evaluator
from nets import build_net
from train import ReplayBuffer, train_steps


def save_checkpoint(path: pathlib.Path, net, cfg: Config, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(),
                "cfg": dataclasses.asdict(cfg),
                "meta": meta}, path)


def load_net(path: str | pathlib.Path, device: str = "cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    cfg.device = device
    net = build_net(cfg).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, cfg, ckpt.get("meta", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=CONFIG.num_iters)
    ap.add_argument("--data", type=str, default="data2048")
    ap.add_argument("--games", type=int, default=CONFIG.games_per_iter)
    ap.add_argument("--sims", type=int, default=CONFIG.num_simulations)
    ap.add_argument("--device", type=str, default=CONFIG.device)
    ap.add_argument("--workers", type=int, default=min((os.cpu_count() or 4) - 2, 16),
                    help="parallel self-play worker processes (1 = serial)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = dataclasses.replace(CONFIG, num_iters=args.iters, games_per_iter=args.games,
                              num_simulations=args.sims, device=args.device)
    data_dir = pathlib.Path(args.data)
    ckpt_path = data_dir / "latest.pt"
    log_path = data_dir / "train_log.tsv"

    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"
        print("[warn] cuda unavailable -> cpu")

    start_iter = 0
    if args.resume and ckpt_path.exists():
        net, loaded_cfg, meta = load_net(ckpt_path, cfg.device)
        start_iter = int(meta.get("iter", 0)) + 1
        print(f"[resume] from iter {start_iter}")
    else:
        net = build_net(cfg).to(cfg.device)
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("iter\tsp_avg\tsp_best\tbesttile\tpolicy_loss\tvalue_loss\teval_avg\teval_reach2048\tseconds\n")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buf = ReplayBuffer(cfg.replay_window)

    par = None
    if args.workers > 1:
        from selfplay_parallel import ParallelSelfplay
        par = ParallelSelfplay(cfg, args.workers)
        print(f"[selfplay] {args.workers} parallel CPU workers")

    for it in range(start_iter, cfg.num_iters):
        t0 = time.time()
        net.eval()
        if par is not None:
            sp = par.generate(net, cfg, cfg.games_per_iter, seed=1_000_000 + it * 10_000)
        else:
            ev = net_evaluator(net, cfg)
            sp = selfplay.generate(ev, cfg, cfg.games_per_iter, seed=1_000_000 + it * 10_000)
        buf.add(sp.states, sp.policies, sp.values)

        losses = train_steps(net, opt, buf, cfg)

        sp_avg = float(np.mean(sp.scores))
        sp_best = int(np.max(sp.scores))
        best_tile = int(np.max(sp.max_tiles))

        eval_avg = eval_reach = float("nan")
        if (it + 1) % 5 == 0 or it == cfg.num_iters - 1:
            net.eval()
            res = evaluate(net_evaluator(net, cfg), cfg, cfg.eval_games, seed=42)
            eval_avg = res.avg_score
            eval_reach = res.reach_rates.get(2048, 0.0)
            print(f"  [eval] {res.summary()}")

        save_checkpoint(ckpt_path, net, cfg, {"iter": it})
        dt = time.time() - t0
        line = (f"{it}\t{sp_avg:.0f}\t{sp_best}\t{best_tile}\t"
                f"{losses['policy_loss']:.4f}\t{losses['value_loss']:.4f}\t"
                f"{eval_avg:.0f}\t{eval_reach:.3f}\t{dt:.0f}")
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(f"iter {it}: sp_avg={sp_avg:.0f} best={sp_best} tile={best_tile} "
              f"ploss={losses['policy_loss']:.3f} vloss={losses['value_loss']:.3f} "
              f"buf={len(buf)} ({dt:.0f}s)")

    if par is not None:
        par.close()
    print("done.")


if __name__ == "__main__":
    main()
