"""Quick text/plot view of training loss log at data/logs/train.tsv."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def _read_tsv(path: pathlib.Path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None, None
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return header, rows


def _plot_selfplay(data_dir: pathlib.Path, plt) -> None:
    log = data_dir / "logs" / "last_run.tsv"
    if not log.exists():
        print(f"no selfplay log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty selfplay log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    needed = ("iter", "min_len", "max_len", "avg_len",
              "black_win_rate", "white_win_rate", "draw_rate")
    if not all(k in idx for k in needed):
        print(f"last_run.tsv missing selfplay columns; skipping selfplay plot", file=sys.stderr)
        return

    def col(name):
        out = []
        for r in rows:
            v = r[idx[name]] if idx[name] < len(r) else ""
            out.append(float(v) if v else float("nan"))
        return out

    x = col("iter")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(x, col("min_len"), label="min")
    ax1.plot(x, col("avg_len"), label="avg")
    ax1.plot(x, col("max_len"), label="max")
    ax1.set_ylabel("game length")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x, col("black_win_rate"), label="black")
    ax2.plot(x, col("white_win_rate"), label="white")
    ax2.plot(x, col("draw_rate"), label="draw")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("rate")
    ax2.set_xlabel("iter")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out = data_dir / "logs" / "selfplay.png"
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    log = data_dir / "logs" / "train.tsv"
    if not log.exists():
        print(f"no log at {log}", file=sys.stderr)
        return 1

    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    if not lines:
        print("empty log", file=sys.stderr)
        return 1

    print(lines[0])
    for ln in lines[-args.tail:]:
        print(ln)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot", file=sys.stderr)
            return 0
        header = lines[0].split("\t")
        rows = [ln.split("\t") for ln in lines[1:]]
        cols = {name: [float(r[i]) for r in rows] for i, name in enumerate(header)}
        x = cols.get("global_step_samples", cols.get("iter", list(range(len(rows)))))
        xlabel = "samples" if "global_step_samples" in cols else "iter"
        keys = [k for k in ("policy_loss", "opp_policy_loss", "value_loss", "total_loss") if k in cols]
        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.5 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            ax.plot(x, cols[key])
            ax.set_yscale("log")
            ax.set_ylabel(key)
            ax.grid(True, which="both", alpha=0.3)
        axes[-1].set_xlabel(xlabel)
        fig.tight_layout()
        out = data_dir / "logs" / "loss.png"
        fig.savefig(out, dpi=200)
        print(f"saved plot to {out}")

        _plot_selfplay(data_dir, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
