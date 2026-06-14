"""Fixed-train / adaptive-selfplay production ledger (V7.3, order model).

Train volume is FIXED (train_steps = TRAIN_SAMPLES_PER_EPOCH / BATCH_SIZE
every iter); selfplay is ORDERED per iter as a bounded game count.

Each iter, run.sh calls this module twice:

    --order  (BEFORE the batch selfplay phase)
        target_cum  += needed             (needed = TRAIN_SAMPLES_PER_EPOCH / RATIO)
        trained_cum += TRAIN_SAMPLES_PER_EPOCH
        deficit = target_cum - read_cum_rows(selfplay.tsv)
        games   = ceil(deficit / rows_per_game)    (0 when production ran ahead)
        Prints the integer game count on stdout; run.sh splits it across the
        train cards (selfplay_main --max-games, one batch per card). Spare
        cards keep producing via the persistent daemon throughout — their
        settled rows shrink the deficit before it is converted to games.

        rows_per_game = sum(rows)/sum(games) over the last RPG_WINDOW_ROWS
        ledger rows — NOT avg_len: soft-resign sample weighting writes fewer
        rows than the game has moves. Cold-start fallback = MAIN_BOARD_SIZE²
        (board area >= moves >= rows, so it deliberately OVER-estimates
        rows/game: the first order under-produces, shuffle skips that iter,
        and the next order re-aims with real ledger data — one lost iter,
        whereas under-estimating would over-produce several MIN_ROWS worth
        and freeze orders at 0 for dozens of iters). Later conversion error
        is harmless: the ledger is cumulative and never reset, so over/
        under-production rolls into the next iter's deficit.

    --log-row --batch-seconds S  (AFTER the batch barrier)
        Appends one row to data/logs/ratio.tsv (iter, cum_rows, trained_cum,
        target_cum, deficit, wait_s=S, eff_ratio) for view_loss.py's
        replay-ratio panel. Read-only w.r.t. the target state; wait_s now
        records the batch selfplay duration (the regime signal: >0 means
        production-bound, 0 means the daemon ran ahead).

Cold start (first --order): target_cum = max(MIN_ROWS, needed) — produce
enough for the first shuffle; shuffle skips below MIN_ROWS anyway.

cum_rows is read from selfplay.tsv (the single APPEND-ONLY cumulative truth
across all producers) — NOT by scanning data/selfplay/*.npz, which OOW-cleanup
keeps trimming to the in-window size (non-monotonic).

State persists in data/selfplay_target.json (consume-once family — resume must
NOT recompute it, or it desyncs from selfplay.tsv's cumulative rows):
    target_cum   — float, cumulative target rows
    trained_cum  — float, cumulative trained samples (for ratio reporting)
    last_iter    — int, iter of the last advance; re-running the same iter
                   (Ctrl+C between --order and train, then resume) does NOT
                   advance again — it re-orders only the remaining deficit
                   (an interrupted batch settles its partial rows on SIGTERM,
                   so the remainder is the true gap).

Without --order/--log-row this is a read-only report (no state change).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

from log_util import tag

TAG = tag("Target", sys.stderr)

RPG_WINDOW_ROWS = 20  # ledger rows used for the rows-per-game estimate


def read_cum_rows(selfplay_tsv: pathlib.Path) -> int:
    """Sum the rows column (col 3) across all producers."""
    if not selfplay_tsv.exists():
        return 0
    total = 0
    try:
        for ln in selfplay_tsv.read_text().splitlines():
            if not ln.strip() or ln.startswith("producer"):
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                continue
            try:
                total += int(float(parts[3]))
            except ValueError:
                continue
    except OSError:
        pass
    return total


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def load_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state_path: pathlib.Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


def append_ratio_row(path: pathlib.Path, iter_no: int, cum_rows: int,
                     trained_cum: float, target_cum: float, deficit: float,
                     wait_s: float, eff_ratio: float) -> None:
    """One row per iter for view_loss.py's ratio panel. Append-only; a
    resumed iter appends a duplicate iter row (the plot keeps the last)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a") as f:
        if write_header:
            f.write("iter\tcum_rows\ttrained_cum\ttarget_cum\tdeficit"
                    "\twait_s\teff_ratio\n")
        f.write(f"{iter_no}\t{cum_rows}\t{trained_cum:.0f}\t{target_cum:.0f}"
                f"\t{deficit:.0f}\t{wait_s:.1f}\t{eff_ratio:.4f}\n")


def recent_rows_per_game(selfplay_tsv: pathlib.Path, fallback: float) -> float:
    """sum(rows)/sum(games) over the last RPG_WINDOW_ROWS ledger rows (all
    producers). rows/games, not avg_len — soft-resign sample weighting writes
    fewer rows than the game has moves."""
    try:
        lines = selfplay_tsv.read_text().splitlines()
    except OSError:
        return fallback
    games = rows = seen = 0
    for ln in reversed(lines):
        if seen >= RPG_WINDOW_ROWS:
            break
        parts = ln.split("\t")
        if len(parts) < 4 or parts[0] == "producer":
            continue
        try:
            g = int(float(parts[2]))
            r = int(float(parts[3]))
        except ValueError:
            continue
        games += g
        rows += r
        seen += 1
    if games <= 0 or rows <= 0:
        return fallback
    return rows / games


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--order", action="store_true",
                   help="advance the target; print the games to order on stdout")
    p.add_argument("--log-row", action="store_true",
                   help="append a ratio.tsv row for --iter (no target advance)")
    p.add_argument("--batch-seconds", type=float, default=0.0,
                   help="batch selfplay duration, logged as wait_s with --log-row")
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    state_path = data_dir / "selfplay_target.json"
    selfplay_tsv = data_dir / "logs" / "selfplay.tsv"

    samples_per_epoch = _env_float("TRAIN_SAMPLES_PER_EPOCH", 128_000)
    ratio = _env_float("TARGET_REPLAY_RATIO", 6.0)
    min_rows = _env_float("MIN_ROWS", 250_000)
    needed = samples_per_epoch / ratio if ratio > 0 else samples_per_epoch

    state = load_state(state_path)
    cum_rows = read_cum_rows(selfplay_tsv)

    if args.log_row:
        # Post-batch report; read-only w.r.t. the target state (--order
        # already advanced it this iter).
        target_cum = float(state.get("target_cum", 0.0))
        trained_cum = float(state.get("trained_cum", 0.0))
        deficit = target_cum - cum_rows
        eff_ratio = (trained_cum / cum_rows) if cum_rows > 0 else 0.0
        if args.iter is not None:
            append_ratio_row(data_dir / "logs" / "ratio.tsv", args.iter,
                             cum_rows, trained_cum, target_cum, deficit,
                             args.batch_seconds, eff_ratio)
        print(f"{TAG} cum_rows={cum_rows} target_cum={target_cum:.0f} "
              f"batch_s={args.batch_seconds:.1f} eff_ratio={eff_ratio:.2f}",
              file=sys.stderr)
        return 0

    if not args.order:
        # Read-only report (ad-hoc inspection; no state change).
        target_cum = float(state.get("target_cum", 0.0))
        trained_cum = float(state.get("trained_cum", 0.0))
        eff_ratio = (trained_cum / cum_rows) if cum_rows > 0 else 0.0
        print(f"{TAG} cum_rows={cum_rows} target_cum={target_cum:.0f} "
              f"deficit={target_cum - cum_rows:.0f} eff_ratio={eff_ratio:.2f}",
              file=sys.stderr)
        return 0

    # --order: advance the cumulative target. First call: cold-start to
    # MIN_ROWS. Consume-once: a resumed re-run of the same iter — or of any
    # earlier iter (Ctrl+C in the pre-first-train phase resumes from iter 0
    # while last_iter is already ahead) — must not advance again, only
    # re-order against the unchanged target.
    if "target_cum" not in state:
        target_cum = max(min_rows, needed)
        trained_cum = samples_per_epoch
    elif args.iter is not None and int(state.get("last_iter", -1)) >= args.iter:
        target_cum = float(state["target_cum"])
        trained_cum = float(state.get("trained_cum", 0.0))
    else:
        target_cum = float(state["target_cum"]) + needed
        trained_cum = float(state.get("trained_cum", 0.0)) + samples_per_epoch

    state["target_cum"] = target_cum
    state["trained_cum"] = trained_cum
    if args.iter is not None:
        state["last_iter"] = args.iter
    # Persist BEFORE ordering: a Ctrl+C mid-iter resumes as a same-iter
    # replay (re-orders only what is still missing).
    save_state(state_path, state)

    deficit = target_cum - cum_rows
    # Fallback = board area: a guaranteed over-estimate of rows/game, so
    # a cold-start order under-produces (one shuffle-skip iter) instead
    # of over-producing (orders frozen at 0 for dozens of iters).
    # Cold-start rows/game estimate (used until selfplay.tsv has real data). The
    # fallback must OVER-estimate so iter 0 under-produces (one shuffle-skip
    # iter, safe) rather than over-produces. Board games end before filling the
    # board, so board area works there; 2048's afterstate games run hundreds of
    # moves, so it overrides via COLD_START_ROWS_PER_GAME in run.cfg.
    board = _env_float("MAIN_BOARD_SIZE", 15.0)
    rpg_fallback = _env_float("COLD_START_ROWS_PER_GAME", board * board)
    rpg = recent_rows_per_game(selfplay_tsv, fallback=rpg_fallback)
    games = math.ceil(deficit / rpg) if deficit > 0 else 0
    print(f"{TAG} cum_rows={cum_rows} target_cum={target_cum:.0f} "
          f"deficit={deficit:.0f} rows/game={rpg:.1f} -> order {games} games"
          + ("" if deficit > 0 else " (production ahead)"),
          file=sys.stderr)
    print(games)  # stdout: consumed by run.sh
    return 0


if __name__ == "__main__":
    sys.exit(main())
