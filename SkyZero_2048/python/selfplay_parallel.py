"""Multiprocessing self-play: fan game generation across CPU cores.

The MCTS is pure-Python and GIL-bound, so a single process pins one core and
leaves the GPU ~idle (the 4x4 net is tiny — GPU is never the bottleneck). Here
each worker is its own interpreter doing CPU inference pinned to ONE torch
thread, so N workers use N cores ~linearly. The GPU stays free for training.

The pool is persistent (created once); each iteration ships the current weights
(a ~2 MB CPU state_dict) to the workers via the task args.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os

import numpy as np

import selfplay
from model_config import Config

# Set once per worker process.
_W: dict = {}


def _init_worker() -> None:
    # Keep each worker single-threaded so N workers don't oversubscribe cores.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # workers never touch the GPU
    import torch
    torch.set_num_threads(1)
    from nets import build_net  # noqa: imported here so torch loads in-worker
    _W["build_net"] = build_net


def _run_chunk(args):
    state_dict, cfg_dict, num_games, seed = args
    import torch
    cfg = Config(**cfg_dict)
    cfg.device = "cpu"
    net = _W["build_net"](cfg)
    net.load_state_dict(state_dict)
    net.eval()
    from mcts import net_evaluator
    with torch.no_grad():
        ev = net_evaluator(net, cfg, "cpu")
        sp = selfplay.generate(ev, cfg, num_games, seed)
    return sp.states, sp.policies, sp.values, sp.scores, sp.max_tiles


class ParallelSelfplay:
    def __init__(self, cfg: Config, num_workers: int) -> None:
        self.num_workers = max(1, num_workers)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.num_workers, initializer=_init_worker)

    def generate(self, net, cfg: Config, num_games: int, seed: int) -> selfplay.SelfplayData:
        # CPU state_dict so it pickles without CUDA tensors.
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        cfg_dict = dataclasses.asdict(cfg)
        # Split games across workers; give each a disjoint seed range.
        nw = self.num_workers
        base = num_games // nw
        rem = num_games % nw
        tasks = []
        s = seed
        for i in range(nw):
            g = base + (1 if i < rem else 0)
            if g == 0:
                continue
            tasks.append((sd, cfg_dict, g, s))
            s += g * 1000  # disjoint seed space per worker
        results = self.pool.map(_run_chunk, tasks)

        states = np.concatenate([r[0] for r in results]) if results else np.zeros((0, 16), np.int8)
        policies = np.concatenate([r[1] for r in results]) if results else np.zeros((0, 4), np.float32)
        values = np.concatenate([r[2] for r in results]) if results else np.zeros((0,), np.float32)
        scores = [x for r in results for x in r[3]]
        max_tiles = [x for r in results for x in r[4]]
        return selfplay.SelfplayData(states, policies, values, scores, max_tiles)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
