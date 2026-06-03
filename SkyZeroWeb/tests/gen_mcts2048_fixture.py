#!/usr/bin/env python3
"""Generate a cross-check fixture for mcts2048.js against python/mcts.py.

Runs the reference SkyZero_2048 GameSearch (afterstate Stochastic Gumbel
AlphaZero MCTS) with gumbel_noise OFF (deterministic) over states pulled from
ai2048_fixture.json, using a fixed synthetic evaluator that mcts2048.js
reproduces bit-for-bit (so the whole search — node structure, PUCT, chance
descent, Gumbel sequential halving, backup, improved policy — must agree).

Synthetic evaluator (must stay identical to test_mcts2048.mjs):
    value(state)   = sum over cells of 2**exp (exp > 0)   [raw points]
    logits(state)[a] = apply_move(state, a).reward * 0.01  [0 if illegal]

Run from the SkyZeroWeb dir with the torch-env python (numpy):
    $PY tests/gen_mcts2048_fixture.py
Writes tests/mcts2048_fixture.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_2048_PY = Path(__file__).resolve().parents[2] / "SkyZero_2048" / "python"
if not _2048_PY.is_dir():
    raise SystemExit(f"Expected SkyZero_2048 python at {_2048_PY}")
sys.path.insert(0, str(_2048_PY))

import game as G          # noqa: E402
import mcts as M          # noqa: E402
from model_config import Config  # noqa: E402

HERE = Path(__file__).resolve().parent
NUM_STATES = 60
NUM_SIMS = 64


def make_eval():
    def _eval(states):
        b = len(states)
        logits = np.zeros((b, 4), dtype=np.float64)
        values = np.zeros(b, dtype=np.float64)
        for k, s in enumerate(states):
            s = np.asarray(s, dtype=np.int8)
            values[k] = float(sum((1 << int(e)) for e in s if e > 0))
            for a in range(4):
                _after, reward, _changed = G.apply_move(s, a)
                logits[k, a] = reward * 0.01
        return logits, values
    return _eval


def main() -> int:
    cases = json.loads((HERE / "ai2048_fixture.json").read_text())
    states = [c["state"] for c in cases if not c["terminal"]][:NUM_STATES]

    cfg = Config(num_simulations=NUM_SIMS, gamma=0.999, c_puct=1.25,
                 gumbel_c_visit=50.0, gumbel_c_scale=1.0, gumbel_noise=False)
    eval_fn = make_eval()

    out = []
    for st in states:
        arr = np.asarray(st, dtype=np.int8)
        gs = M.GameSearch(arr, cfg, np.random.default_rng(0))
        M.batch_search([gs], eval_fn, cfg)
        out.append({
            "state": st,
            "best_action": int(gs.best_action()),
            "visits": [int(x) for x in gs.visit_counts()],
            "improved_policy": [float(x) for x in gs.improved_policy()],
            "root_value": float(gs.root_value()),
            "nn_policy": [float(x) for x in gs.nn_policy()],
        })

    (HERE / "mcts2048_fixture.json").write_text(json.dumps(out))
    print(f"wrote {HERE/'mcts2048_fixture.json'}  ({len(out)} states, "
          f"sims={NUM_SIMS}, gumbel_noise=False)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
