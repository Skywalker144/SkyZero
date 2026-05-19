"""Quick text/plot view of training loss log at data/logs/train.tsv."""
from __future__ import annotations

import argparse
import math
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
    """3-panel selfplay plot from selfplay.tsv (main + daemon rows interleaved).

    Top:    cumulative rows by producer (stacked).
    Middle: main-pair avg game length per write event, colored by producer.
    Bottom: main-pair win rates per write event, colored by producer.
    """
    log = data_dir / "logs" / "selfplay.tsv"
    if not log.exists():
        print(f"no selfplay log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty selfplay log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    required = ("producer", "rows", "main_avg_len",
                "main_bwr", "main_wwr", "main_dwr")
    if not all(k in idx for k in required):
        print("selfplay.tsv missing expected columns; skipping selfplay plot",
              file=sys.stderr)
        return

    def col(name, cast=float):
        out = []
        for r in rows:
            v = r[idx[name]] if idx[name] < len(r) else ""
            if not v:
                out.append(float("nan") if cast is float else "")
                continue
            try:
                out.append(cast(v))
            except (ValueError, TypeError):
                out.append(float("nan") if cast is float else "")
        return out

    producer = col("producer", cast=str)
    rows_arr = col("rows")
    x = list(range(len(rows)))

    # Cumulative rows by producer over write index.
    cum_main, cum_daemon = [], []
    m_acc = d_acc = 0.0
    for p, r in zip(producer, rows_arr):
        rv = 0.0 if math.isnan(r) else r
        if p == "main":
            m_acc += rv
        elif p == "daemon":
            d_acc += rv
        cum_main.append(m_acc)
        cum_daemon.append(d_acc)

    main_size = os.environ.get("MAIN_BOARD_SIZE", "?")
    main_rule = os.environ.get("MAIN_RULE", "?")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax1.stackplot(x, cum_main, cum_daemon,
                  labels=["main", "daemon"], alpha=0.7,
                  colors=["C0", "C1"])
    ax1.set_ylabel("cumulative rows")
    ax1.set_title("selfplay production")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    main_avg = col("main_avg_len")
    for p, c in (("main", "C0"), ("daemon", "C1")):
        xs = [x[i] for i in range(len(x)) if producer[i] == p]
        ys = [main_avg[i] for i in range(len(x)) if producer[i] == p]
        if xs:
            ax2.plot(xs, ys, "o-", label=p, color=c, alpha=0.7, markersize=3)
    ax2.set_ylabel("avg game length")
    ax2.set_title(f"main pair ({main_size}×{main_rule}) game length")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    bwr, wwr, dwr = col("main_bwr"), col("main_wwr"), col("main_dwr")
    ax3.plot(x, bwr, "o-", label="black", color="C2", markersize=3, alpha=0.7)
    ax3.plot(x, wwr, "o-", label="white", color="C3", markersize=3, alpha=0.7)
    ax3.plot(x, dwr, "o-", label="draw", color="C7", markersize=3, alpha=0.7)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("rate")
    ax3.set_xlabel("tsv row index")
    ax3.set_title(f"main pair ({main_size}×{main_rule}) win rates")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = data_dir / "logs" / "selfplay.png"
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")


def _plot_probe(data_dir: pathlib.Path, plt) -> None:
    log = data_dir / "logs" / "probe.tsv"
    if not log.exists():
        print(f"no probe log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty probe log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    needed = ("iter", "gumbel_dist",
              "vmix_W", "vmix_L", "nn_W", "nn_L")
    if not all(k in idx for k in needed):
        print("probe.tsv missing expected columns; skipping probe plot",
              file=sys.stderr)
        return

    def col(name):
        out = []
        for r in rows:
            v = r[idx[name]] if idx[name] < len(r) else ""
            out.append(float(v) if v else float("nan"))
        return out

    x = col("iter")
    gumbel_dist = col("gumbel_dist")
    vmix_wl = [w - l for w, l in zip(col("vmix_W"), col("vmix_L"))]
    nn_wl = [w - l for w, l in zip(col("nn_W"), col("nn_L"))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(x, gumbel_dist, color="C0", label="gumbel")
    ax1.set_ylabel("euclid dist from center")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x, vmix_wl, color="C2", label="v_mix W-L")
    ax2.plot(x, nn_wl, color="C3", label="nn_value W-L")
    ax2.axhline(0.0, ls="--", color="gray", alpha=0.4)
    finite = [v for v in vmix_wl + nn_wl if not math.isnan(v)]
    if finite:
        lo = min(finite)
        hi = max(finite)
        pad = (hi - lo) * 0.1 if hi > lo else 0.05
        ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_ylabel("root value (W-L)")
    ax2.set_xlabel("iter")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out = data_dir / "logs" / "probe.png"
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
        keys = [k for k in ("total_loss", "policy_loss", "opp_policy_loss",
                              "intermediate_loss", "soft_policy_loss", "soft_opp_policy_loss",
                              "futurepos_loss", "value_loss", "td_value_loss") if k in cols]
        # Grid layout: square-ish (ncols ≈ √n) so many losses stay readable.
        # ≤3 panes => single column (vertical), keeping shared x-axis natural.
        n = len(keys)
        if n <= 3:
            ncols, nrows = 1, n
            figsize = (8, 2.5 * n)
        else:
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
            figsize = (5 * ncols, 2.8 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                                 squeeze=False)
        flat = axes.flatten()
        for ax, key in zip(flat, keys):
            ax.plot(x, cols[key])
            ax.set_yscale("log")
            ax.set_title(key)
            ax.grid(True, which="both", alpha=0.3)
        # Hide unused panes (when n is not a perfect rectangle).
        for ax in flat[n:]:
            ax.set_visible(False)
        # X-label only on the bottom row of visible panes.
        for ax in flat[max(0, n - ncols):n]:
            ax.set_xlabel(xlabel)
        fig.tight_layout()
        out = data_dir / "logs" / "loss.png"
        fig.savefig(out, dpi=200)
        print(f"saved plot to {out}")

        _plot_selfplay(data_dir, plt)
        _plot_probe(data_dir, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
