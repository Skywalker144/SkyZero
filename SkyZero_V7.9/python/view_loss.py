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


def _load_schedule(data_dir: pathlib.Path):
    """Return (iters, cum_samples) parallel lists from logs/schedule.tsv.

    Returns None when the file is missing/empty or lacks required columns.
    Used to translate between iter and cumulative selfplay samples on plot axes.
    """
    log = data_dir / "logs" / "schedule.tsv"
    if not log.exists():
        return None
    header, rows = _read_tsv(log)
    if not header or not rows:
        return None
    idx = {name: i for i, name in enumerate(header)}
    if "iter" not in idx or "cum_samples" not in idx:
        return None
    pairs = []
    for r in rows:
        try:
            it = float(r[idx["iter"]])
            cs = float(r[idx["cum_samples"]])
        except (ValueError, IndexError):
            continue
        pairs.append((it, cs))
    if len(pairs) < 2:
        return None
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _add_iter_axis(ax, sched) -> None:
    """Add a top secondary x-axis showing iter, given primary x = cum samples."""
    if sched is None:
        return
    import numpy as np
    iters_a = np.asarray(sched[0], dtype=float)
    samples_a = np.asarray(sched[1], dtype=float)
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda s: np.interp(s, samples_a, iters_a),
                   lambda i: np.interp(i, iters_a, samples_a)),
    )
    import matplotlib.ticker as mticker
    secax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, prune="both"))
    secax.set_xlabel("iter")


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


def _bucket_by_games(xs, weights, series, min_games=500):
    """Merge consecutive points into buckets of >= min_games games, returning
    (bucket_xs, bucket_series) with games-weighted averages.

    selfplay.tsv rows are settle windows of wildly varying size (a 60s daemon
    window vs a 2s TERM flush when train re-drafts the card), so plotting them
    raw gives tiny-sample rows the same visual weight as large ones and the
    curves look like noise. Bucketing restores comparable statistical weight
    per point. The tail bucket is emitted even when short so the plot stays
    current (the last point may be noisier)."""
    bx, bys = [], [[] for _ in series]
    acc_w = 0.0
    acc_v = [0.0] * len(series)
    last_x = None
    for i, (xv, w) in enumerate(zip(xs, weights)):
        if not (math.isfinite(w) and w > 0
                and all(math.isfinite(s[i]) for s in series)):
            continue
        acc_w += w
        last_x = xv
        for k, s in enumerate(series):
            acc_v[k] += s[i] * w
        if acc_w >= min_games:
            bx.append(last_x)
            for k in range(len(series)):
                bys[k].append(acc_v[k] / acc_w)
            acc_w = 0.0
            acc_v = [0.0] * len(series)
    if acc_w > 0 and last_x is not None:
        bx.append(last_x)
        for k in range(len(series)):
            bys[k].append(acc_v[k] / acc_w)
    return bx, bys


def _plot_selfplay(data_dir: pathlib.Path, plt) -> None:
    """3-panel selfplay plot from selfplay.tsv (main + daemon rows interleaved).

    Top:    cumulative rows by producer (stacked).
    Middle: main-pair avg game length, >=500-game buckets, colored by producer.
    Bottom: main-pair win rates, >=500-game buckets.
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
    required = ("producer", "rows", "main_games", "main_avg_len",
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

    # Primary x = cumulative selfplay samples after this write event.
    x = [m + d for m, d in zip(cum_main, cum_daemon)]
    sched = _load_schedule(data_dir)

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
    main_games = col("main_games")
    for p, c in (("main", "C0"), ("daemon", "C1")):
        sel = [i for i in range(len(x)) if producer[i] == p]
        bx, (by,) = _bucket_by_games(
            [x[i] for i in sel], [main_games[i] for i in sel],
            [[main_avg[i] for i in sel]])
        if bx:
            ax2.plot(bx, by, "-", label=p, color=c, alpha=0.7)
    ax2.set_ylabel("avg game length")
    ax2.set_title(f"main pair ({main_size}×{main_rule}) game length "
                  f"(≥500-game buckets)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    bwr, wwr, dwr = col("main_bwr"), col("main_wwr"), col("main_dwr")
    bx, (bb, bw, bd) = _bucket_by_games(x, main_games, [bwr, wwr, dwr])
    ax3.plot(bx, bb, "-", label="black", color="C2", alpha=0.7)
    ax3.plot(bx, bw, "-", label="white", color="C3", alpha=0.7)
    ax3.plot(bx, bd, "-", label="draw", color="C7", alpha=0.7)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("rate")
    ax3.set_xlabel("cumulative selfplay samples")
    ax3.set_title(f"main pair ({main_size}×{main_rule}) win rates "
                  f"(≥500-game buckets)")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    _add_iter_axis(ax1, sched)

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

    iter_vals = col("iter")
    gumbel_dist = col("gumbel_dist")
    vmix_wl = [w - l for w, l in zip(col("vmix_W"), col("vmix_L"))]
    nn_wl = [w - l for w, l in zip(col("nn_W"), col("nn_L"))]

    sched = _load_schedule(data_dir)
    if sched is not None:
        import numpy as np
        x = list(np.interp(iter_vals, sched[0], sched[1]))
        xlabel = "cumulative selfplay samples"
    else:
        x = iter_vals
        xlabel = "iter"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(x, gumbel_dist, color="C0", label="gumbel")
    ax1.set_ylabel("euclid dist from center")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x, vmix_wl, color="C2", label="root value W-L")
    ax2.plot(x, nn_wl, color="C3", label="nn_value W-L")
    ax2.axhline(0.0, ls="--", color="gray", alpha=0.4)
    finite = [v for v in vmix_wl + nn_wl if not math.isnan(v)]
    if finite:
        lo = min(finite)
        hi = max(finite)
        pad = (hi - lo) * 0.1 if hi > lo else 0.05
        ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_ylabel("root value (W-L)")
    ax2.set_xlabel(xlabel)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    _add_iter_axis(ax1, sched)

    fig.tight_layout()
    out = data_dir / "logs" / "probe.png"
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")


def _plot_ratio(data_dir: pathlib.Path, plt) -> None:
    """Effective replay ratio per iter from logs/ratio.tsv (written by
    compute_selfplay_target.py), against the TARGET_REPLAY_RATIO soft bound."""
    log = data_dir / "logs" / "ratio.tsv"
    if not log.exists():
        print(f"no ratio log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty ratio log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    if not all(k in idx for k in ("iter", "cum_rows", "eff_ratio")):
        print("ratio.tsv missing expected columns; skipping ratio plot",
              file=sys.stderr)
        return

    # Keep the last row per iter (a resumed iter appends a duplicate row).
    by_iter: dict[int, list[str]] = {}
    for r in rows:
        try:
            by_iter[int(float(r[idx["iter"]]))] = r
        except (ValueError, IndexError):
            continue
    if not by_iter:
        return
    ordered = [by_iter[k] for k in sorted(by_iter)]

    def col(name):
        out = []
        for r in ordered:
            v = r[idx[name]] if idx[name] < len(r) else ""
            out.append(float(v) if v else float("nan"))
        return out

    x = col("cum_rows")
    ratio = col("eff_ratio")
    target = float(os.environ.get("TARGET_REPLAY_RATIO", "0") or 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, ratio, color="C4", label="effective ratio (trained/produced)")
    if target > 0:
        ax.axhline(target, ls="--", color="gray", alpha=0.6,
                   label=f"target {target:g}")
    ax.set_ylabel("replay ratio")
    ax.set_xlabel("cumulative selfplay samples")
    ax.set_title("replay ratio (target = soft upper bound)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    _add_iter_axis(ax, _load_schedule(data_dir))

    fig.tight_layout()
    out = data_dir / "logs" / "ratio.png"
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
    parser.add_argument("--only", default="",
                        help="comma list of plots to draw "
                             "(loss,selfplay,probe,ratio); empty = all")
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

    wanted = {t for t in args.only.replace(",", " ").split() if t} \
        or {"loss", "selfplay", "probe", "ratio"}

    if "loss" in wanted:
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

    if "selfplay" in wanted:
        _plot_selfplay(data_dir, plt)
    if "probe" in wanted:
        _plot_probe(data_dir, plt)
    if "ratio" in wanted:
        _plot_ratio(data_dir, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
