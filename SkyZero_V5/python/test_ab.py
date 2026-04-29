"""Smoke tests for the pure-math helpers in python/ab.py.

Run as: `python python/test_ab.py` from the repo root.
No pytest dependency — bare assertions, prints `ok` on success.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab import score_to_elo, wilson_ci  # noqa: E402


def test_score_to_elo_midpoint():
    assert abs(score_to_elo(0.5)) < 1e-9


def test_score_to_elo_known_value():
    # 75% winrate <-> ~191 Elo (standard Bradley-Terry / 400-scale).
    assert abs(score_to_elo(0.75) - 191.0) < 1.0


def test_score_to_elo_clamps_at_extremes():
    assert score_to_elo(0.0) == -800.0
    assert score_to_elo(1.0) == 800.0


def test_wilson_ci_brackets_score():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


def test_wilson_ci_tightens_with_n():
    _, hi_small = wilson_ci(5, 10)
    _, hi_big = wilson_ci(500, 1000)
    assert (hi_big - 0.5) < (hi_small - 0.5)


def test_wilson_ci_zero_wins():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0
    assert 0 < hi < 0.1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
