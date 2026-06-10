"""Fixed-train / adaptive-selfplay production gate (V7.3).

Replaces the V7.2 token bucket (python/bucket.py). Train volume is FIXED
(train_steps = TRAIN_SAMPLES_PER_EPOCH / BATCH_SIZE every iter); selfplay has
no per-card orders — every card produces continuously (per-card + persistent
daemons, run.sh) and the C++ side settles stats into selfplay.tsv every
DAEMON_SETTLE_SECONDS. This module is the GATE between them.

Each iter (called by run.sh BEFORE shuffle, with --wait):
    target_cum  += needed                 (needed = TRAIN_SAMPLES_PER_EPOCH / RATIO)
    trained_cum += TRAIN_SAMPLES_PER_EPOCH
    block until read_cum_rows(selfplay.tsv) >= target_cum
        - instant when production already ran ahead (cards over-produced;
          ratio then floats BELOW the target — fresher data, by design);
        - otherwise poll while the cards close the deficit.

Cold start (first call): target_cum = max(MIN_ROWS, needed) — produce enough
for the first shuffle; shuffle skips below MIN_ROWS anyway.

cum_rows is read from selfplay.tsv (the single APPEND-ONLY cumulative truth
across all producers) — NOT by scanning data/selfplay/*.npz, which OOW-cleanup
keeps trimming to the in-window size (non-monotonic). The ledger is cumulative
and never reset: over-production rolls into the next iter's balance, so the
accounting self-corrects and there is no read-clear-write race.

State persists in data/selfplay_target.json (consume-once family — resume must
NOT recompute it, or it desyncs from selfplay.tsv's cumulative rows):
    target_cum   — float, cumulative target rows
    trained_cum  — float, cumulative trained samples (for ratio reporting)
    last_iter    — int, iter of the last advance; re-running the same iter
                   (Ctrl+C between this call and train, then resume) does NOT
                   advance again, only re-waits on the unchanged target.

Each call with --iter also appends a row to data/logs/ratio.tsv (iter,
cum_rows, trained_cum, target_cum, deficit, wait_s, eff_ratio) for
view_loss.py's replay-ratio panel.

CLI (one shot, called by run.sh):
    python compute_selfplay_target.py --data-dir DIR --iter N --wait
        Advances the target and blocks until the gate passes. Without --wait
        it only advances + reports (ad-hoc inspection).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from bucket import read_cum_rows  # reuse the exact tsv row-sum (col 3, all producers)
from log_util import tag

TAG = tag("Target", sys.stderr)

POLL_SECONDS = 5
PROGRESS_EVERY_SECONDS = 30
STALL_WARN_SECONDS = 300


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


def wait_for_rows(selfplay_tsv: pathlib.Path, target: float) -> tuple[int, float]:
    """Block until the settled ledger reaches target. Returns (cum_rows,
    seconds waited). Warns (but keeps waiting) when the ledger stops moving —
    daemons settle every DAEMON_SETTLE_SECONDS, so a long flat line means
    production died (check card_selfplay / daemon logs)."""
    t0 = time.monotonic()
    cum = read_cum_rows(selfplay_tsv)
    last_change = t0
    last_print = t0
    last_warn = t0
    while cum < target:
        time.sleep(POLL_SECONDS)
        new = read_cum_rows(selfplay_tsv)
        now = time.monotonic()
        if new > cum:
            last_change = now
        cum = new
        if now - last_print >= PROGRESS_EVERY_SECONDS:
            print(f"{TAG} gate: rows {cum}/{target:.0f} "
                  f"({100.0 * cum / target:.0f}%)", file=sys.stderr)
            last_print = now
        if now - last_change >= STALL_WARN_SECONDS and now - last_warn >= STALL_WARN_SECONDS:
            print(f"{TAG} WARNING: no settled production for "
                  f"{int(now - last_change)}s — selfplay daemons dead? "
                  f"(check card_selfplay / daemon logs)", file=sys.stderr)
            last_warn = now
    return cum, time.monotonic() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--wait", action="store_true")
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

    # Advance the cumulative target. First call: cold-start to MIN_ROWS.
    # Consume-once: a resumed re-run of the same iter must not advance again,
    # only re-wait against the unchanged target.
    if "target_cum" not in state:
        target_cum = max(min_rows, needed)
        trained_cum = samples_per_epoch
    elif args.iter is not None and state.get("last_iter") == args.iter:
        target_cum = float(state["target_cum"])
        trained_cum = float(state.get("trained_cum", 0.0))
    else:
        target_cum = float(state["target_cum"]) + needed
        trained_cum = float(state.get("trained_cum", 0.0)) + samples_per_epoch

    state["target_cum"] = target_cum
    state["trained_cum"] = trained_cum
    if args.iter is not None:
        state["last_iter"] = args.iter
    # Persist BEFORE waiting: a Ctrl+C mid-wait resumes as a same-iter replay.
    save_state(state_path, state)

    deficit = target_cum - cum_rows
    print(f"{TAG} cum_rows={cum_rows} target_cum={target_cum:.0f} "
          f"deficit={deficit:.0f}"
          + ("" if deficit > 0 else " (production ahead; gate passes)"),
          file=sys.stderr)

    wait_s = 0.0
    if args.wait and deficit > 0:
        cum_rows, wait_s = wait_for_rows(selfplay_tsv, target_cum)

    eff_ratio = (trained_cum / cum_rows) if cum_rows > 0 else 0.0
    if args.iter is not None:
        append_ratio_row(data_dir / "logs" / "ratio.tsv", args.iter, cum_rows,
                         trained_cum, target_cum, deficit, wait_s, eff_ratio)
    print(f"{TAG} gate passed: cum_rows={cum_rows} wait={wait_s:.1f}s "
          f"eff_ratio={eff_ratio:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
