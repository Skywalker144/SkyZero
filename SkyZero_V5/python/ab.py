#!/usr/bin/env python
"""Head-to-head AB report from gomoku_ab JSONL match logs.

Reads game records produced by cpp/gomoku_ab (one JSON object per line with
keys: model, cfg_a, cfg_b, a_black, winner_a, plies), aggregates W-D-L per
model, computes head-to-head Elo difference (A minus B) with a Wilson 95%
CI, and prints a text table + optional PNG plot of Elo diff vs. checkpoint.
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

ITER_RE = re.compile(r"(\d+)")
ELO_CLAMP = 800.0


def parse_iter(path: str) -> int | None:
    matches = ITER_RE.findall(Path(path).stem)
    return int(matches[-1]) if matches else None


def score_to_elo(score: float) -> float:
    """Bradley-Terry score (in [0,1]) -> Elo difference (400-scale).

    Clamped to +/- ELO_CLAMP. The output clamp matters too — Wilson CI
    on N/N returns one ULP below 1.0 (not exactly 1.0), which evaluates
    to ~+6260 Elo without the output clamp.
    """
    if score <= 0.0:
        return -ELO_CLAMP
    if score >= 1.0:
        return ELO_CLAMP
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    return max(-ELO_CLAMP, min(ELO_CLAMP, elo))


def wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    `wins` may be fractional (we count draws as 0.5 wins). Returns (lo, hi)
    as a probability in [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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
                print(f"[ab] skipping line {i}: {e}", file=sys.stderr)
    return games


def aggregate(games: list[dict]) -> dict[str, dict]:
    """Group games by model. Each group has W/D/L overall and split by side."""
    out: dict[str, dict] = defaultdict(lambda: {
        "w": 0, "d": 0, "l": 0,
        "w_ab": 0, "d_ab": 0, "l_ab": 0,  # A as black
        "w_aw": 0, "d_aw": 0, "l_aw": 0,  # A as white
        "plies_sum": 0,
    })
    for g in games:
        m = g["model"]
        wa = g["winner_a"]
        a_black = bool(g["a_black"])
        suf = "ab" if a_black else "aw"
        if wa > 0:
            out[m]["w"] += 1; out[m][f"w_{suf}"] += 1
        elif wa < 0:
            out[m]["l"] += 1; out[m][f"l_{suf}"] += 1
        else:
            out[m]["d"] += 1; out[m][f"d_{suf}"] += 1
        out[m]["plies_sum"] += int(g.get("plies", 0))
    return out


def stats(w: int, d: int, l: int) -> tuple[int, float, float, float, float]:
    """Return (n, score, elo, elo_lo, elo_hi)."""
    n = w + d + l
    if n == 0:
        return (0, 0.5, 0.0, -ELO_CLAMP, ELO_CLAMP)
    score = (w + 0.5 * d) / n
    elo = score_to_elo(score)
    lo_p, hi_p = wilson_ci(w + 0.5 * d, n)
    return (n, score, elo, score_to_elo(lo_p), score_to_elo(hi_p))


def print_report(per_model: dict[str, dict], cfg_a: str, cfg_b: str) -> list[tuple]:
    """Print the text report. Returns [(model, n, elo, elo_lo, elo_hi)] for plotting."""
    width = max(len(os.path.basename(m)) for m in per_model) if per_model else 8
    width = max(width, 8)
    print(f"{'model':<{width}}  {'games':>6}  {'W-D-L':>14}  {'score':>6}  "
          f"{'elo_diff (95% CI)':>22}")

    rows = []
    pooled = {"w": 0, "d": 0, "l": 0,
              "w_ab": 0, "d_ab": 0, "l_ab": 0,
              "w_aw": 0, "d_aw": 0, "l_aw": 0,
              "plies_sum": 0}
    items = sorted(per_model.items(),
                   key=lambda kv: (parse_iter(kv[0]) if parse_iter(kv[0]) is not None else 10**9,
                                   kv[0]))
    for model, agg in items:
        n, score, elo, lo, hi = stats(agg["w"], agg["d"], agg["l"])
        wdl = f"{agg['w']}-{agg['d']}-{agg['l']}"
        ci = f"{elo:+.0f}  ({lo:+.0f}..{hi:+.0f})"
        print(f"{os.path.basename(model):<{width}}  {n:>6}  {wdl:>14}  "
              f"{score:>6.3f}  {ci:>22}")
        rows.append((model, n, elo, lo, hi))
        for k in pooled:
            pooled[k] += agg[k]

    n, score, elo, lo, hi = stats(pooled["w"], pooled["d"], pooled["l"])
    sep = "─" * (width + 60)
    print(sep)
    wdl = f"{pooled['w']}-{pooled['d']}-{pooled['l']}"
    ci = f"{elo:+.0f}  ({lo:+.0f}..{hi:+.0f})"
    print(f"{'pooled':<{width}}  {n:>6}  {wdl:>14}  {score:>6.3f}  {ci:>22}")

    # By-side pooled.
    n_b, sc_b, *_ = stats(pooled["w_ab"], pooled["d_ab"], pooled["l_ab"])
    n_w, sc_w, *_ = stats(pooled["w_aw"], pooled["d_aw"], pooled["l_aw"])
    print(f"by side (pooled): A-as-black {sc_b:.3f} "
          f"({pooled['w_ab']}-{pooled['d_ab']}-{pooled['l_ab']}) | "
          f"A-as-white {sc_w:.3f} "
          f"({pooled['w_aw']}-{pooled['d_aw']}-{pooled['l_aw']})")
    if n > 0:
        print(f"mean plies: {pooled['plies_sum'] / n:.1f}")
    print(f"cfg_a={cfg_a}  cfg_b={cfg_b}")

    return rows


def render_plot(rows: list[tuple], plot_path: Path, cfg_a: str, cfg_b: str,
                total_games: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ab] matplotlib not installed; skipping plot", file=sys.stderr)
        return

    rows = sorted(rows, key=lambda r: (parse_iter(r[0]) if parse_iter(r[0]) is not None else 10**9,
                                       r[0]))
    labels = [os.path.basename(m) for m, _, _, _, _ in rows]
    elos = [e for _, _, e, _, _ in rows]
    err_lo = [e - lo for _, _, e, lo, _ in rows]
    err_hi = [hi - e for _, _, e, _, hi in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(rows)))
    ax.errorbar(xs, elos, yerr=[err_lo, err_hi], fmt="-o", capsize=3,
                label="A − B Elo")
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("model checkpoint")
    ax.set_ylabel("Elo diff (A − B); positive ⇒ cfg_a stronger")
    ax.set_title(f"AB hyperparameter eval: {os.path.basename(cfg_a)} vs "
                 f"{os.path.basename(cfg_b)}  ({total_games} games)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[ab] wrote {plot_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, type=Path,
                    help="JSONL from gomoku_ab")
    ap.add_argument("--plot", type=Path, default=None,
                    help="Output PNG path for the Elo-diff curve")
    args = ap.parse_args()

    games = load_games(args.games)
    if not games:
        print("[ab] no games found", file=sys.stderr)
        return 1

    cfg_pairs = {(g["cfg_a"], g["cfg_b"]) for g in games}
    if len(cfg_pairs) > 1:
        print(f"[ab] warning: {len(cfg_pairs)} distinct (cfg_a, cfg_b) pairs in "
              f"{args.games} — report mixes them.", file=sys.stderr)
    cfg_a, cfg_b = next(iter(cfg_pairs))

    per_model = aggregate(games)
    rows = print_report(per_model, cfg_a, cfg_b)

    if args.plot:
        render_plot(rows, args.plot, cfg_a, cfg_b, total_games=len(games))

    return 0


if __name__ == "__main__":
    sys.exit(main())
