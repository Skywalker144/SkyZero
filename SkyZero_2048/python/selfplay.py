"""Self-play data generation via batched afterstate Gumbel MCTS.

Each move records (state, improved_policy). After a game ends, value targets are
the Monte-Carlo discounted future score from each state (exact, since games are
played to terminal): G_t = r_t + gamma * G_{t+1}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import game as G
from model_config import Config
from mcts import EvalFn, GameSearch, batch_search


@dataclass
class SelfplayData:
    states: np.ndarray     # (N, 16) int8
    policies: np.ndarray   # (N, 4)  float32  (improved-policy targets)
    values: np.ndarray     # (N,)    float32  (raw discounted future score)
    scores: list[int]      # per-game final score
    max_tiles: list[int]   # per-game max tile (value, not exponent)


def generate(eval_fn: EvalFn, cfg: Config, num_games: int, seed: int) -> SelfplayData:
    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    scores: list[int] = []
    max_tiles: list[int] = []

    done_games = 0
    next_seed = seed
    while done_games < num_games:
        batch = min(cfg.selfplay_batch, num_games - done_games)
        rngs = [np.random.default_rng(next_seed + i) for i in range(batch)]
        next_seed += batch
        states = [G.initial_state(rngs[i]) for i in range(batch)]
        # per-game trajectory of (state, improved_policy, reward)
        traj_states: list[list[np.ndarray]] = [[] for _ in range(batch)]
        traj_pol: list[list[np.ndarray]] = [[] for _ in range(batch)]
        traj_rew: list[list[int]] = [[] for _ in range(batch)]
        score = [0] * batch
        done = [G.is_terminal(states[i]) for i in range(batch)]

        while not all(done):
            active = [i for i in range(batch) if not done[i]]
            searches = [GameSearch(states[i], cfg, rngs[i]) for i in active]
            batch_search(searches, eval_fn, cfg)
            for gs, i in zip(searches, active):
                a = gs.best_action()
                if a < 0:
                    done[i] = True
                    continue
                traj_states[i].append(states[i].copy())
                traj_pol[i].append(gs.improved_policy().astype(np.float32))
                after, reward, _ = G.apply_move(states[i], a)
                traj_rew[i].append(reward)
                score[i] += reward
                states[i] = G.spawn_random(after, rngs[i])
                if G.is_terminal(states[i]):
                    done[i] = True

        # finalize: MC discounted returns
        for i in range(batch):
            g = 0.0
            vals = [0.0] * len(traj_rew[i])
            for t in range(len(traj_rew[i]) - 1, -1, -1):
                g = traj_rew[i][t] + cfg.gamma * g
                vals[t] = g
            if traj_states[i]:
                all_states.append(np.stack(traj_states[i]))
                all_policies.append(np.stack(traj_pol[i]))
                all_values.append(np.asarray(vals, dtype=np.float32))
            scores.append(score[i])
            max_tiles.append(1 << G.max_tile_exp(states[i]))
        done_games += batch

    return SelfplayData(
        states=np.concatenate(all_states).astype(np.int8) if all_states else np.zeros((0, 16), np.int8),
        policies=np.concatenate(all_policies) if all_policies else np.zeros((0, 4), np.float32),
        values=np.concatenate(all_values) if all_values else np.zeros((0,), np.float32),
        scores=scores,
        max_tiles=max_tiles,
    )
