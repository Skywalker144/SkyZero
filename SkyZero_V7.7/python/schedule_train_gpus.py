"""LPT assignment of NETWORKS to GPUs for parallel training (V7.3, design §2.5).

Cost model: blocks * channels^2 — conv FLOPs dominate a residual tower and the
heads are a near-constant small term (board/batch/steps are fixed across nets).
Longest-processing-time greedy: place each network (cost descending) on the
currently-least-loaded card. With N >= M cards this is one network per card and
leaves N-M cards empty; with N < M some cards get several networks (trained
serially by run.sh).

CLI (called by run.sh once at startup; cost is fixed so the assignment is too):
    python schedule_train_gpus.py --gpus N [--networks "b5c128, b10c256, ..."]

Prints N lines, one per GPU 0..N-1: a space-separated network list, or BLANK
when no network is assigned to that card (run.sh gives those cards to the
selfplay daemon). Example (3 GPUs, nets b5c128/b10c256/b18c384):
    b18c384
    b10c256
    b5c128
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_NET_RE = re.compile(r"^b(\d+)c(\d+)$")


def cost(name: str) -> int:
    m = _NET_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad network name {name!r} (expected b<blocks>c<channels>)")
    blocks, channels = int(m.group(1)), int(m.group(2))
    return blocks * channels * channels


def lpt_assign(networks: list[str], n_gpus: int) -> list[list[str]]:
    assign: list[list[str]] = [[] for _ in range(n_gpus)]
    loads = [0] * n_gpus
    for net in sorted(networks, key=cost, reverse=True):
        i = min(range(n_gpus), key=lambda k: loads[k])
        assign[i].append(net)
        loads[i] += cost(net)
    return assign


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--networks", default=os.environ.get("NETWORKS", ""))
    args = p.parse_args()

    networks = [x for x in args.networks.replace(",", " ").split() if x]
    if not networks:
        print("schedule_train_gpus: empty NETWORKS", file=sys.stderr)
        return 1
    n = max(1, args.gpus)
    for cards in lpt_assign(networks, n):
        print(" ".join(cards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
