"""Evaluation: play N games (deterministic, no Gumbel noise) and report
single-agent metrics — average score and tile reach-rates — since Elo (a
two-player notion) does not apply to 2048."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

import game as G
from model_config import Config
from mcts import EvalFn, GameSearch, batch_search


@dataclass
class EvalResult:
    avg_score: float
    median_score: float
    max_score: int
    avg_max_tile: float
    reach_rates: dict[int, float]   # tile value -> fraction of games reaching it
    scores: list[int]
    max_tiles: list[int]

    def summary(self) -> str:
        rr = " ".join(f"{k}:{v*100:.0f}%" for k, v in sorted(self.reach_rates.items()))
        return (f"avg={self.avg_score:.0f} med={self.median_score:.0f} "
                f"max={self.max_score} | reach {rr}")


def evaluate(eval_fn: EvalFn, cfg: Config, num_games: int, seed: int = 0) -> EvalResult:
    ecfg = copy.copy(cfg)
    ecfg.gumbel_noise = False     # deterministic play for evaluation
    scores: list[int] = []
    max_tiles: list[int] = []

    done_games = 0
    next_seed = seed
    while done_games < num_games:
        batch = min(cfg.selfplay_batch, num_games - done_games)
        rngs = [np.random.default_rng(next_seed + i) for i in range(batch)]
        next_seed += batch
        states = [G.initial_state(rngs[i]) for i in range(batch)]
        score = [0] * batch
        done = [G.is_terminal(states[i]) for i in range(batch)]
        while not all(done):
            active = [i for i in range(batch) if not done[i]]
            searches = [GameSearch(states[i], ecfg, rngs[i]) for i in active]
            batch_search(searches, eval_fn, ecfg)
            for gs, i in zip(searches, active):
                a = gs.best_action()
                if a < 0:
                    done[i] = True
                    continue
                after, reward, _ = G.apply_move(states[i], a)
                score[i] += reward
                states[i] = G.spawn_random(after, rngs[i])
                if G.is_terminal(states[i]):
                    done[i] = True
        for i in range(batch):
            scores.append(score[i])
            max_tiles.append(1 << G.max_tile_exp(states[i]))
        done_games += batch

    scores_arr = np.asarray(scores)
    tiles_arr = np.asarray(max_tiles)
    milestones = [256, 512, 1024, 2048, 4096, 8192]
    reach = {m: float((tiles_arr >= m).mean()) for m in milestones}
    return EvalResult(
        avg_score=float(scores_arr.mean()),
        median_score=float(np.median(scores_arr)),
        max_score=int(scores_arr.max()),
        avg_max_tile=float(tiles_arr.mean()),
        reach_rates=reach,
        scores=scores,
        max_tiles=max_tiles,
    )


# ---------------------------------------------------------------------------
# V7.1-style CLI (driven by scripts/internal/eval.sh): evaluate the active
# network and append a row to <DATA_DIR>/logs/eval.tsv (2048's single-agent
# analogue of the Gomoku Elo step — no head-to-head, just absolute metrics).
#   python evaluate.py --data-dir DIR --network b6c96 --iter N --games 50
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    import pathlib
    import time

    import torch

    from model_config import config_from_name
    from mcts import net_evaluator
    from nets import build_net

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()

    cfg = config_from_name(args.network)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"
    data_dir = pathlib.Path(args.data_dir)
    ckpt = data_dir / "nets" / args.network / "model_latest.pt"
    net = build_net(cfg).to(cfg.device)
    ck = torch.load(ckpt, map_location=cfg.device, weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()

    res = evaluate(net_evaluator(net, cfg), cfg, args.games, seed=42)
    print(f"[eval] {args.network} iter={args.iter} {res.summary()}")

    ms = [256, 512, 1024, 2048, 4096, 8192]
    eval_tsv = data_dir / "logs" / "eval.tsv"
    eval_tsv.parent.mkdir(parents=True, exist_ok=True)
    new = not eval_tsv.exists()
    rcols = "\t".join(f"r{m}" for m in ms)
    rvals = "\t".join(f"{res.reach_rates.get(m, 0.0):.4f}" for m in ms)
    with open(eval_tsv, "a") as f:
        if new:
            f.write("iter\ttimestamp\tnetwork\tgames\tavg_score\tmedian_score\t"
                    f"max_score\tavg_max_tile\t{rcols}\n")
        f.write(f"{args.iter}\t{int(time.time())}\t{args.network}\t{args.games}\t"
                f"{res.avg_score:.0f}\t{res.median_score:.0f}\t{res.max_score}\t"
                f"{res.avg_max_tile:.0f}\t{rvals}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
