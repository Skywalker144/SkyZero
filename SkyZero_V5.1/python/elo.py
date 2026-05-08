#!/usr/bin/env python
"""Elo ratings from gomoku_elo JSONL match logs.

Reads game records produced by cpp/gomoku_elo (one JSON object per line with
keys: a, b, a_black, winner_a, plies), fits Bradley-Terry ratings via MLE,
converts to Elo, and prints a table + optional PNG curve.

Anchor convention: any model whose path contains '/anchors/' is pinned to the
anchor set; the numerically-smallest-iter anchor (or lexicographically first
if no iter number) is pinned to Elo 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ITER_RE = re.compile(r"(\d+)")


def parse_iter(path: str) -> int | None:
    """Extract a trailing iteration number from a model filename, if any."""
    name = Path(path).stem
    # Prefer the last numeric run in the stem (so 'anchor_iter_200' -> 200).
    matches = ITER_RE.findall(name)
    return int(matches[-1]) if matches else None


def load_games(path: Path) -> list[dict]:
    games = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[elo] skipping line {i}: {e}", file=sys.stderr)
    return games


def fit_bt(models: list[str], games: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fit Bradley-Terry ratings (natural-log scale) + std errors.

    Returns (ratings, std_errors) aligned with `models`. Rating[0] is pinned to
    0 (first model in list is the reference).
    """
    idx = {m: i for i, m in enumerate(models)}
    # Aggregate score into W/D/L counts per ordered (a, b) pair.
    pair = defaultdict(lambda: [0, 0, 0])  # (i, j) -> [a_wins, draws, b_wins]
    for g in games:
        i, j = idx[g["a"]], idx[g["b"]]
        w = g["winner_a"]
        if w > 0:
            pair[(i, j)][0] += 1
        elif w < 0:
            pair[(i, j)][2] += 1
        else:
            pair[(i, j)][1] += 1

    n = len(models)

    def neg_ll(r_free: np.ndarray) -> float:
        r = np.concatenate([[0.0], r_free])
        nll = 0.0
        for (i, j), (wa, d, wb) in pair.items():
            diff = r[i] - r[j]
            # log-sigmoid(diff) for a win; log-sigmoid(-diff) for b win;
            # draw counts as half of each (standard simplification).
            log_pa = -math.log1p(math.exp(-diff)) if diff > -50 else diff
            log_pb = -math.log1p(math.exp(diff)) if diff < 50 else -diff
            nll -= (wa + 0.5 * d) * log_pa + (wb + 0.5 * d) * log_pb
        # Weak L2 prior to keep unidentified models finite (rating 0.01 ~ 4 Elo).
        nll += 1e-4 * float(np.sum(r_free ** 2))
        return nll

    x0 = np.zeros(n - 1)
    res = minimize(neg_ll, x0, method="L-BFGS-B")
    r_free = res.x
    r_full = np.concatenate([[0.0], r_free])

    # Std errors via numerical Hessian at optimum (central differences).
    h = 1e-3
    hess = np.zeros((n - 1, n - 1))
    f0 = neg_ll(r_free)
    for i in range(n - 1):
        for j in range(i, n - 1):
            dij = np.zeros(n - 1)
            dij[i] += h
            dij[j] += h
            fpp = neg_ll(r_free + dij)
            dij[j] -= 2 * h
            fpm = neg_ll(r_free + dij)
            dij[i] -= 2 * h
            fmm = neg_ll(r_free + dij)
            dij[j] += 2 * h
            fmp = neg_ll(r_free + dij)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    try:
        cov = np.linalg.inv(hess)
        se_free = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se_free = np.full(n - 1, np.nan)
    se_full = np.concatenate([[0.0], se_free])
    return r_full, se_full


def pick_reference(models: list[str]) -> int:
    """Return the index of the model to pin at Elo 0."""
    anchors = [(i, m) for i, m in enumerate(models) if "/anchors/" in m or "\\anchors\\" in m]
    pool = anchors if anchors else list(enumerate(models))
    # Prefer anchor with smallest iteration number.
    def key(item):
        i, m = item
        it = parse_iter(m)
        return (it if it is not None else 10**9, m)
    return min(pool, key=key)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, type=Path, help="JSONL from gomoku_elo")
    ap.add_argument("--plot", type=Path, default=None, help="Output PNG path for Elo curve")
    ap.add_argument("--min-games", type=int, default=1,
                    help="Drop models with fewer than this many total games")
    args = ap.parse_args()

    games = load_games(args.games)
    if not games:
        print("[elo] no games found", file=sys.stderr)
        return 1

    # Collect models and per-model game counts.
    counts: dict[str, int] = defaultdict(int)
    for g in games:
        counts[g["a"]] += 1
        counts[g["b"]] += 1
    models = sorted(m for m, c in counts.items() if c >= args.min_games)
    if len(models) < 2:
        print("[elo] need >=2 models with games", file=sys.stderr)
        return 1

    # Reorder so reference model sits at index 0 (pinned to 0 in fit).
    ref_global = pick_reference(models)
    models = [models[ref_global]] + [m for i, m in enumerate(models) if i != ref_global]
    model_set = set(models)
    games = [g for g in games if g["a"] in model_set and g["b"] in model_set]

    r, se = fit_bt(models, games)
    # Convert natural log-odds to Elo (Elo = r * 400 / ln(10)).
    scale = 400.0 / math.log(10.0)
    elo = r * scale
    elo_se = se * scale

    # --- Table ---
    w = max(len(os.path.basename(m)) for m in models)
    print(f"{'model':<{w}}  {'games':>6}  {'Elo':>8}  {'±se':>6}  {'iter':>6}")
    rows = []
    for m, e, es in zip(models, elo, elo_se):
        rows.append((m, counts[m], e, es, parse_iter(m)))
    # Sort table by iteration (anchors first by their iter number too).
    rows.sort(key=lambda x: (x[4] if x[4] is not None else -1, x[0]))
    for m, c, e, es, it in rows:
        it_str = str(it) if it is not None else "-"
        tag = " [anchor]" if "/anchors/" in m else ""
        print(f"{os.path.basename(m):<{w}}  {c:>6}  {e:>+8.1f}  {es:>6.1f}  {it_str:>6}{tag}")

    # --- Plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[elo] matplotlib not installed; skipping plot", file=sys.stderr)
            return 0

        fig, ax = plt.subplots(figsize=(8, 5))
        trained = [(it, e, es, m) for m, c, e, es, it in rows
                   if it is not None and "/anchors/" not in m]
        trained.sort()
        if trained:
            its = [t[0] for t in trained]
            es_ = [t[1] for t in trained]
            err = [t[2] for t in trained]
            ax.errorbar(its, es_, yerr=err, fmt="-o", capsize=3, label="training run")
        # Anchors as horizontal dashed lines.
        for m, c, e, es, it in rows:
            if "/anchors/" not in m:
                continue
            ax.axhline(e, linestyle="--", alpha=0.6,
                       label=f"anchor {os.path.basename(m)} (Elo={e:+.0f})")
        ax.set_xlabel("training iteration")
        ax.set_ylabel("Elo (ref = 0)")
        ax.set_title("Model Elo vs training iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=120)
        print(f"[elo] wrote {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
