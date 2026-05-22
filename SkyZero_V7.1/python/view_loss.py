"""Quick text/plot view of training loss across all networks.

Each network writes to data/nets/<name>/train.tsv. This script aggregates
them onto a single loss.png (one line per network per loss key) and dumps
the tail of each network's TSV to stdout.
"""
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


def _list_networks(data_dir: pathlib.Path) -> list[str]:
    """Return the network list. Env NETWORKS first (preserves order); else
    fall back to scanning data/nets/* directories that contain a train.tsv."""
    env = os.environ.get("NETWORKS", "")
    if env.strip():
        return [t for t in env.replace(",", " ").split() if t]
    nets_dir = data_dir / "nets"
    if not nets_dir.is_dir():
        return []
    return sorted(p.name for p in nets_dir.iterdir()
                  if p.is_dir() and (p / "train.tsv").exists())


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
            ax2.plot(xs, ys, "-", label=p, color=c, alpha=0.7)
    ax2.set_ylabel("avg game length")
    ax2.set_title(f"main pair ({main_size}×{main_rule}) game length")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    bwr, wwr, dwr = col("main_bwr"), col("main_wwr"), col("main_dwr")
    ax3.plot(x, bwr, "-", label="black", color="C2", alpha=0.7)
    ax3.plot(x, wwr, "-", label="white", color="C3", alpha=0.7)
    ax3.plot(x, dwr, "-", label="draw", color="C7", alpha=0.7)
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


def _read_network_cols(net_dir: pathlib.Path) -> dict[str, list[float]] | None:
    log = net_dir / "train.tsv"
    if not log.exists():
        return None
    header, rows = _read_tsv(log)
    if not header or not rows:
        return None
    cols: dict[str, list[float]] = {}
    for i, name in enumerate(header):
        vs: list[float] = []
        for r in rows:
            v = r[i] if i < len(r) else ""
            try:
                vs.append(float(v))
            except (ValueError, TypeError):
                vs.append(float("nan"))
        cols[name] = vs
    return cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    networks = _list_networks(data_dir)
    if not networks:
        print(f"no networks with train logs under {data_dir}/nets/", file=sys.stderr)
        return 1

    # Tail print: one section per network.
    per_net_cols: dict[str, dict[str, list[float]]] = {}
    for net in networks:
        net_dir = data_dir / "nets" / net
        cols = _read_network_cols(net_dir)
        if cols is None:
            print(f"[{net}] no train.tsv yet", file=sys.stderr)
            continue
        per_net_cols[net] = cols
        log = net_dir / "train.tsv"
        lines = log.read_text().splitlines()
        print(f"=== {net} ({log}) ===")
        if lines:
            print(lines[0])
            for ln in lines[1:][-args.tail:]:
                print(ln)
        print()

    if not args.plot:
        return 0

    if not per_net_cols:
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return 0

    # Union of loss keys across networks (preserving canonical order below).
    canonical = ("total_loss", "policy_loss", "opp_policy_loss",
                 "intermediate_loss", "soft_policy_loss", "soft_opp_policy_loss",
                 "futurepos_loss", "value_loss", "td_value_loss")
    keys = [k for k in canonical if any(k in cols for cols in per_net_cols.values())]
    n = len(keys)
    if n <= 3:
        ncols, nrows = 1, n
        figsize = (8, 2.5 * n)
    else:
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        figsize = (5 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)
    flat = axes.flatten()

    color_cycle = [f"C{i}" for i in range(10)]
    for ax, key in zip(flat, keys):
        any_xlabel = "iter"
        for i, (net, cols) in enumerate(per_net_cols.items()):
            if key not in cols:
                continue
            x = cols.get("global_step_samples") or cols.get("iter") or list(range(len(cols[key])))
            if "global_step_samples" in cols:
                any_xlabel = "samples"
            ax.plot(x, cols[key], color=color_cycle[i % len(color_cycle)],
                    label=net, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(key)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(any_xlabel)
    # Hide unused panes.
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    out = data_dir / "logs" / "loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")

    _plot_selfplay(data_dir, plt)
    _plot_probe(data_dir, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
