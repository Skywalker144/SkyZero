"""Shared list-driven warmup helper.

A warmup is parameterized by (1) a list of values, one per stage, and
(2) a window length in cumulative selfplay rows. Stages are evenly spaced:
with N values, the i-th value (0-indexed) applies for progress in
[i/N, (i+1)/N). Once progress >= 1.0, the last value is used (steady state).

Disabled (returns None) when len(stages) < 2 or warmup_samples <= 0; the
caller is expected to fall back to its existing steady-state value.
"""
from __future__ import annotations

from typing import Callable


def parse_stage_list(s: str | None, cast: Callable = float) -> list:
    """Parse "v1, v2, v3" -> [cast(v1), cast(v2), cast(v3)].

    Empty / whitespace-only strings yield []. Skips empty fields produced
    by trailing commas. Raises if a non-empty field fails to cast.
    """
    if not s:
        return []
    return [cast(p.strip()) for p in s.split(",") if p.strip()]


def staged_value(samples_seen: int, warmup_samples: int, stages: list):
    """Returns the stage value for samples_seen, or None if warmup disabled.

    Disabled when len(stages) < 2 or warmup_samples <= 0.
    """
    n = len(stages)
    if n < 2 or warmup_samples <= 0:
        return None
    progress = samples_seen / warmup_samples
    if progress >= 1.0:
        return stages[-1]
    if progress < 0.0:
        return stages[0]
    idx = min(n - 1, int(progress * n))
    return stages[idx]
