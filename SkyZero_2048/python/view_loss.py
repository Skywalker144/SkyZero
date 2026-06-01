"""Training-progress plots, V7.1 `view_loss.py`-style but for 2048.

Reads the per-iter TSV logs under <DATA_DIR>/ and renders PNGs into
<DATA_DIR>/logs/:

  loss.png      — per-network policy / value loss vs iter (nets/<net>/train.tsv)
  selfplay.png  — max-tile distribution (stacked), tile reach-rate curves, and
                  avg/best score, all from self-play (logs/selfplay_stats.tsv)
  eval.png      — periodic-eval reach rates + score curves (logs/eval.tsv)
  probe.png     — mcts_probe root-value trend vs iter (logs/probe.tsv)

Called once per iter by run.sh (best-effort; a plotting error never aborts
training), and runnable standalone:

    python view_loss.py --data-dir data2048_nbt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Tile milestones shown as reach-rate curves / eval columns.
MILESTONES = [256, 512, 1024, 2048, 4096, 8192]


def _read_tsv(path: pathlib.Path):
    """Return (header:list[str], rows:list[list[str]]) or (None, None)."""
    if not path.exists():
        return None, None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None, None
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def _cols(header, rows):
    """header/rows -> {name: [float|nan, ...]} (non-numeric -> nan)."""
    idx = {name: i for i, name in enumerate(header)}
    out = {}
    for name, i in idx.items():
        vals = []
        for r in rows:
            v = r[i] if i < len(r) else ""
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        out[name] = vals
    return out


def _finite_pairs(x, y):
    import math
    xs, ys = [], []
    for a, b in zip(x, y):
        if not (math.isnan(a) or math.isnan(b)):
            xs.append(a)
            ys.append(b)
    return xs, ys


def _plot_loss(data_dir, plt) -> None:
    """One curve per network (nets/<net>/train.tsv): policy + value loss vs iter."""
    series = []  # (net, iters, policy_loss, value_loss)
    for tsv in sorted((data_dir / "nets").glob("*/train.tsv")):
        header, rows = _read_tsv(tsv)
        if not header or not rows:
            continue
        c = _cols(header, rows)
        if "iter" in c and "policy_loss" in c and "value_loss" in c:
            series.append((tsv.parent.name, c["iter"], c["policy_loss"], c["value_loss"]))
    if not series:
        return
    fig, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for net, it, pl, vl in series:
        xs, ys = _finite_pairs(it, pl)
        if xs:
            ax_p.plot(xs, ys, "-", alpha=0.85, label=net)
        xs, ys = _finite_pairs(it, vl)
        if xs:
            ax_v.plot(xs, ys, "-", alpha=0.85, label=net)
    for ax, title in ((ax_p, "policy_loss"), (ax_v, "value_loss")):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    ax_v.set_xlabel("iter")
    fig.tight_layout()
    out = data_dir / "logs" / "loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_probe(data_dir, plt) -> None:
    """mcts_probe root value (+ raw NN values) vs iter, from logs/probe.tsv."""
    header, rows = _read_tsv(data_dir / "logs" / "probe.tsv")
    if not header or not rows:
        return
    c = _cols(header, rows)
    it = c.get("iter", [])
    if not it:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    for key, lab in (("root_value", "root value (MCTS)"), ("raw0", "raw v0"),
                     ("raw1", "raw v1"), ("raw2", "raw v2")):
        if key not in c:
            continue
        xs, ys = _finite_pairs(it, c[key])
        if xs:
            ax.plot(xs, ys, "-o", ms=3, alpha=0.85, label=lab)
    ax.set_xlabel("iter")
    ax.set_ylabel("value (raw points)")
    ax.set_title("mcts_probe value trend")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = data_dir / "logs" / "probe.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_selfplay(data_dir, plt) -> None:
    import numpy as np
    header, rows = _read_tsv(data_dir / "logs" / "selfplay_stats.tsv")
    if not rows:
        return
    c = _cols(header, rows)
    it = np.asarray(c.get("iter", []), dtype=float)
    if it.size == 0:
        return

    # exponent columns e1..eN -> per-iter max-tile histogram (counts).
    exp_keys = sorted((k for k in c if k.startswith("e") and k[1:].isdigit()),
                      key=lambda k: int(k[1:]))
    hist_all = np.nan_to_num(np.array([c[k] for k in exp_keys], dtype=float))  # (E, T)
    # Drop exponents never reached, so the stack legend stays uncluttered.
    nonzero = hist_all.sum(axis=1) > 0
    exps = [int(k[1:]) for k, nz in zip(exp_keys, nonzero) if nz]
    hist = hist_all[nonzero]
    if hist.size == 0:
        return
    totals = hist.sum(axis=0)
    totals[totals == 0] = 1.0
    frac = hist / totals  # fraction of games at each max-tile, per iter

    exps_arr = np.asarray(exps)
    tile_vals = (1 << exps_arr).astype(float)          # tile value per exponent row
    # per-iter best (max over games) and average max-tile.
    present = hist > 0
    max_tile = (1 << (present * exps_arr[:, None]).max(axis=0)).astype(float)
    avg_max_tile = (hist * tile_vals[:, None]).sum(axis=0) / totals

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(9, 16), sharex=True)

    # 1) Stacked max-tile distribution (fraction of self-play games).
    labels = [str(1 << e) for e in exps]
    ax1.stackplot(it, *frac, labels=labels, alpha=0.85)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("fraction of games")
    ax1.set_title("self-play max-tile distribution")
    ax1.legend(loc="upper left", ncol=4, fontsize=7, framealpha=0.6)
    ax1.grid(True, alpha=0.3)

    # 2) Reach-rate curves: fraction of games whose max tile >= milestone.
    for m in MILESTONES:
        mask = (1 << exps_arr) >= m
        if not mask.any():
            continue
        reach = (hist[mask].sum(axis=0) / totals)
        if reach.max() <= 0:
            continue
        ax2.plot(it, reach, "-", label=f"≥{m}", alpha=0.85)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("reach rate")
    ax2.set_title("self-play tile reach rates")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Per-iter max tile: best game (max over games) + average max tile.
    ax3.plot(it, max_tile, "-", color="C3", label="max (best game)")
    ax3.plot(it, avg_max_tile, "-", color="C0", alpha=0.7, label="avg max tile")
    ax3.set_yscale("log", base=2)
    # y ticks at the tile values actually spanned.
    lo = int(np.floor(np.log2(max(2.0, avg_max_tile.min()))))
    hi = int(np.ceil(np.log2(max(2.0, max_tile.max()))))
    ticks = [1 << e for e in range(lo, hi + 1)]
    ax3.set_yticks(ticks)
    ax3.set_yticklabels([str(t) for t in ticks])
    ax3.set_ylabel("max tile")
    ax3.set_title("self-play max tile per iter")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    # 4) avg / best score.
    if "avg_score" in c:
        xs, ys = _finite_pairs(it.tolist(), c["avg_score"])
        if xs:
            ax4.plot(xs, ys, "-", color="C0", label="avg score")
    if "best_score" in c:
        xs, ys = _finite_pairs(it.tolist(), c["best_score"])
        if xs:
            ax4.plot(xs, ys, "-", color="C1", alpha=0.6, label="best score")
    ax4.set_ylabel("score")
    ax4.set_title("self-play score")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5) Avg game length (moves/game) = new_rows / games. Longer games = stronger
    #    play (the agent survives more moves) — a clean single-agent progress signal.
    if "new_rows" in c and "games" in c:
        nr = np.asarray(c["new_rows"], dtype=float)
        gm = np.asarray(c["games"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            avglen = np.where(gm > 0, nr / gm, np.nan)
        xs, ys = _finite_pairs(it.tolist(), avglen.tolist())
        if xs:
            ax5.plot(xs, ys, "-", color="C2", label="avg game len")
    ax5.set_ylabel("moves / game")
    ax5.set_xlabel("iter")
    ax5.set_title("self-play avg game length")
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, alpha=0.3)

    fig.tight_layout()
    out = data_dir / "logs" / "selfplay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _plot_eval(data_dir, plt) -> None:
    header, rows = _read_tsv(data_dir / "logs" / "eval.tsv")
    if not rows:
        return
    c = _cols(header, rows)
    it = c.get("iter", [])
    if not it:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for m in MILESTONES:
        key = f"r{m}"
        if key not in c:
            continue
        xs, ys = _finite_pairs(it, c[key])
        if xs:
            ax1.plot(xs, ys, "-o", ms=3, label=f"≥{m}", alpha=0.85)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("reach rate")
    ax1.set_title("eval tile reach rates")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    for key, color, lab in (("avg_score", "C0", "avg"),
                            ("median_score", "C2", "median"),
                            ("max_score", "C1", "max")):
        if key not in c:
            continue
        xs, ys = _finite_pairs(it, c[key])
        if xs:
            ax2.plot(xs, ys, "-o", ms=3, color=color, label=lab, alpha=0.85)
    ax2.set_ylabel("score")
    ax2.set_xlabel("iter")
    ax2.set_title("eval score")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = data_dir / "logs" / "eval.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def make_plots(data_dir) -> None:
    """Render all PNGs. Best-effort: import/plot errors are swallowed so a
    plotting failure never aborts the training loop."""
    data_dir = pathlib.Path(data_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable: {e}", file=sys.stderr)
        return
    for fn in (_plot_loss, _plot_selfplay, _plot_eval, _plot_probe):
        try:
            fn(data_dir, plt)
        except Exception as e:  # noqa: BLE001
            print(f"[plot] {fn.__name__} failed: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data2048_nbt")
    args = ap.parse_args()
    make_plots(args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
