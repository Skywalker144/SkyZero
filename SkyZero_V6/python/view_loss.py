"""Text/plot view of training loss logs.

Discovers data/logs/train_<slot>.tsv files and overlays one curve per slot
in each loss-type pane of the saved plot. Tail print: per-slot section.
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


def _plot_selfplay_impl(data_dir: pathlib.Path, plt,
                        prefix: str, out_name: str, title_tag: str) -> None:
    """Render a 2-panel selfplay plot (gamelen + win rates).

    prefix=""        — full-set columns (min_len, avg_len, ...). Saved as selfplay.png.
    prefix="main_"   — main-pair columns. Saved as selfplay_main.png.
    """
    log = data_dir / "logs" / "last_run.tsv"
    if not log.exists():
        print(f"no selfplay log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty selfplay log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    needed = ("iter",
              f"{prefix}min_len", f"{prefix}max_len", f"{prefix}avg_len",
              f"{prefix}black_win_rate", f"{prefix}white_win_rate", f"{prefix}draw_rate")
    if not all(k in idx for k in needed):
        print(f"last_run.tsv missing columns for {out_name}; skipping", file=sys.stderr)
        return

    def col(name):
        out = []
        for r in rows:
            v = r[idx[name]] if idx[name] < len(r) else ""
            out.append(float(v) if v else float("nan"))
        return out

    x = col("iter")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(x, col(f"{prefix}min_len"), label="min")
    ax1.plot(x, col(f"{prefix}avg_len"), label="avg")
    ax1.plot(x, col(f"{prefix}max_len"), label="max")
    ax1.set_ylabel("game length")
    ax1.set_yscale("symlog", linthresh=1)
    ax1.set_title(f"selfplay game length ({title_tag})")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="best")

    ax2.plot(x, col(f"{prefix}black_win_rate"), label="black")
    ax2.plot(x, col(f"{prefix}white_win_rate"), label="white")
    ax2.plot(x, col(f"{prefix}draw_rate"), label="draw")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("rate")
    ax2.set_xlabel("iter")
    ax2.set_title(f"selfplay win rates ({title_tag})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out = data_dir / "logs" / out_name
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")


def _plot_selfplay(data_dir: pathlib.Path, plt) -> None:
    """Two plots: full set across all (size, rule), and MAIN_* pair only."""
    _plot_selfplay_impl(data_dir, plt,
                        prefix="", out_name="selfplay.png",
                        title_tag="all sizes / rules")
    main_size = os.environ.get("MAIN_BOARD_SIZE", "?")
    main_rule = os.environ.get("MAIN_RULE", "?")
    _plot_selfplay_impl(data_dir, plt,
                        prefix="main_", out_name="selfplay_main.png",
                        title_tag=f"main pair: {main_size}×{main_rule}")


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


def _read_train_log(path: pathlib.Path) -> tuple[list[str], dict[str, list[float]]] | None:
    """Read one train_<slot>.tsv. Returns (raw_lines, cols) or None if empty."""
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    cols: dict[str, list[float]] = {}
    for i, name in enumerate(header):
        cols[name] = [float(r[i]) if i < len(r) and r[i] else float("nan") for r in rows]
    return lines, cols


def _slot_logs(data_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    logs_dir = data_dir / "logs"
    out: list[tuple[str, pathlib.Path]] = []
    for p in sorted(logs_dir.glob("train_*.tsv")):
        slot = p.stem[len("train_"):]
        out.append((slot, p))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--slot", default=None,
                        help="restrict tail/plot to one slot (default: all)")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    slot_logs = _slot_logs(data_dir)
    if args.slot is not None:
        slot_logs = [sp for sp in slot_logs if sp[0] == args.slot]
    if not slot_logs:
        print(f"no train_<slot>.tsv files under {data_dir / 'logs'}", file=sys.stderr)
        return 1

    # Tail: per-slot section.
    parsed: list[tuple[str, list[str], dict[str, list[float]]]] = []
    for slot, path in slot_logs:
        loaded = _read_train_log(path)
        if loaded is None:
            continue
        lines, cols = loaded
        parsed.append((slot, lines, cols))
        print(f"=== {slot} ({path.name}) ===")
        print(lines[0])
        for ln in lines[1:][-args.tail:]:
            print(ln)
        print()

    if not args.plot:
        return 0
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return 0

    # Layout: one pane per loss key, one curve per slot overlaid.
    loss_keys = ("total_loss", "policy_loss", "opp_policy_loss",
                 "intermediate_loss", "soft_policy_loss", "soft_opp_policy_loss",
                 "futurepos_loss", "value_loss", "td_value_loss")
    keys = [k for k in loss_keys if any(k in cols for _, _, cols in parsed)]
    n = len(keys)
    if n == 0:
        print("no recognized loss columns; skipping loss plot", file=sys.stderr)
    else:
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
        # Pick x-axis: prefer global_step_samples, fall back to iter.
        # With multiple slots starting at different times, samples is the
        # cleanest common axis (each slot's own samples-seen).
        any_samples = any("global_step_samples" in cols for _, _, cols in parsed)
        xlabel = "samples" if any_samples else "iter"
        for ax, key in zip(flat, keys):
            for slot, _, cols in parsed:
                if key not in cols:
                    continue
                if any_samples and "global_step_samples" in cols:
                    x = cols["global_step_samples"]
                elif "iter" in cols:
                    x = cols["iter"]
                else:
                    x = list(range(len(cols[key])))
                ax.plot(x, cols[key], label=slot)
            ax.set_yscale("log")
            ax.set_title(key)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        for ax in flat[n:]:
            ax.set_visible(False)
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
