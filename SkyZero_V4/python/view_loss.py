"""Quick text/plot view of training loss log at data/logs/train.tsv."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    log = pathlib.Path(args.data_dir) / "logs" / "train.tsv"
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
        out = pathlib.Path(args.data_dir) / "logs" / "loss.png"
        fig.savefig(out, dpi=200)
        print(f"saved plot to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
