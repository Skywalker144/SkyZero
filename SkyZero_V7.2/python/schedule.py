"""Selfplay-network schedule resolver.

Given NETWORKS and SELFPLAY_SCHEDULE (positionally matched), pick the
network whose threshold has been crossed by the current cumulative
selfplay sample count. Used by run.sh each iter to decide which network
the selfplay binary should load.

  NETWORKS="b5c128 b10c256 b15c384"
  SELFPLAY_SCHEDULE="0, 3e7, 6e7"

Active rule: pick the largest-index i where thresholds[i] <= cum_rows.

CLI:
  python schedule.py active --data-dir DIR
      Read cum rows from <DIR>/logs/selfplay.tsv, env NETWORKS and
      SELFPLAY_SCHEDULE; print the active network name.

  python schedule.py list
      Print "<network>\t<threshold>" for each (network, threshold) pair,
      sorted by threshold. For sanity-checking the config.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from bucket import read_cum_rows
from log_util import tag

TAG = tag("Schedule", sys.stderr)


def _sci(x: int) -> str:
    """Match run.sh's fmt_sci: 1200000 -> '1.2e6', 0 -> '0'."""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    if m.endswith(".0"):
        m = m[:-2]
    return f"{m}e{int(e)}"


def parse_networks(s: str) -> list[str]:
    return [t for t in s.replace(",", " ").split() if t]


def parse_schedule(s: str) -> list[float]:
    return [float(t) for t in s.replace(",", " ").split() if t]


def active_network(networks: list[str], thresholds: list[float], cum_rows: int) -> str:
    if len(networks) != len(thresholds):
        raise ValueError(
            f"NETWORKS ({len(networks)}) and SELFPLAY_SCHEDULE ({len(thresholds)}) "
            f"must have the same length"
        )
    if not networks:
        raise ValueError("NETWORKS is empty")
    # Sort by threshold to make the rule order-independent of how the user
    # wrote the schedule (defensive — we still expect ascending).
    pairs = sorted(zip(thresholds, networks))
    chosen = pairs[0][1]
    for thr, name in pairs:
        if cum_rows >= thr:
            chosen = name
        else:
            break
    return chosen


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_active = sub.add_parser("active", help="Print the active selfplay network name.")
    p_active.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))

    sub.add_parser("list", help="Print the (network, threshold) pairs.")

    args = p.parse_args()

    networks = parse_networks(os.environ.get("NETWORKS", ""))
    thresholds = parse_schedule(os.environ.get("SELFPLAY_SCHEDULE", ""))

    if args.cmd == "active":
        cum_rows = read_cum_rows(pathlib.Path(args.data_dir) / "logs" / "selfplay.tsv")
        name = active_network(networks, thresholds, cum_rows)
        print(f"{TAG} cum_rows={_sci(cum_rows)} -> active={name}", file=sys.stderr)
        print(name)
    elif args.cmd == "list":
        if len(networks) != len(thresholds):
            print(f"length mismatch: {len(networks)} networks vs {len(thresholds)} thresholds",
                  file=sys.stderr)
            return 1
        for thr, n in sorted(zip(thresholds, networks)):
            print(f"{n}\t{thr:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
