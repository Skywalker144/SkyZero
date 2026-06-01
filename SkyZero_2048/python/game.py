"""2048 environment (afterstate formulation), mirroring cpp/envs/game2048.h.

State = a length-16 numpy int8 array of EXPONENTS, row-major (loc = r*4 + c).
0 = empty; e>0 = a tile of value 2**e. So "2" is exponent 1, "2048" is 11.

Single-agent stochastic MDP:
  * action = one of 4 slide directions (0=up, 1=right, 2=down, 3=left),
  * apply_move is DETERMINISTIC -> (afterstate, reward, changed),
  * the environment then spawns one tile (2 w.p. 0.9, 4 w.p. 0.1) in a
    uniformly-random empty cell.

Hot paths (apply_move / legal / terminal) are table-driven: every possible
4-cell row-slide is precomputed once (a row depends only on its 4 exponents),
so a move is a handful of numpy gathers instead of Python loops. The pure-Python
`_slide_line` below is kept only to BUILD that table (and is the reference the
C++ port / unit tests agree with). numpy-only; no torch.
"""
from __future__ import annotations

import numpy as np

SIZE = 4
AREA = SIZE * SIZE
NUM_ACTIONS = 4          # 0=up, 1=right, 2=down, 3=left
NUM_PLANES = 16          # one-hot exponent planes 0..15 (plane 0 = empty)
PROB_2 = 0.9             # spawns exponent 1 (tile 2)
PROB_4 = 0.1             # spawns exponent 2 (tile 4)

_BASE = 18               # max indexable exponent+1 (covers up to tile 2**17)
_WEIGHTS = np.array([_BASE**0, _BASE**1, _BASE**2, _BASE**3], dtype=np.int64)


def _line_indices(action: int, line: int) -> list[int]:
    """The 4 board locs of `line`, ordered from the moving edge inward."""
    idx = []
    for i in range(SIZE):
        if action == 0:    r, c = i, line                  # up
        elif action == 1:  r, c = line, SIZE - 1 - i       # right
        elif action == 2:  r, c = SIZE - 1 - i, line       # down
        else:              r, c = line, i                  # left
        idx.append(r * SIZE + c)
    return idx


def _slide_line(vals: list[int]) -> tuple[list[int], int]:
    """Compress + single-merge toward index 0. Returns (out[4], reward).

    Reference implementation; used to build the lookup tables at import time.
    """
    packed = [v for v in vals if v != 0]
    out = []
    reward = 0
    i = 0
    while i < len(packed):
        if i + 1 < len(packed) and packed[i] == packed[i + 1]:
            merged = packed[i] + 1
            out.append(merged)
            reward += (1 << merged)
            i += 2
        else:
            out.append(packed[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return out, reward


def _build_tables():
    n = _BASE ** 4
    out = np.zeros((n, 4), dtype=np.int8)
    reward = np.zeros(n, dtype=np.int32)
    changed = np.zeros(n, dtype=bool)
    for idx in range(n):
        e0 = idx % _BASE
        e1 = (idx // _BASE) % _BASE
        e2 = (idx // (_BASE * _BASE)) % _BASE
        e3 = idx // (_BASE ** 3)
        line = [e0, e1, e2, e3]
        o, r = _slide_line(line)
        out[idx] = o
        reward[idx] = r
        changed[idx] = (o != line)
    return out, reward, changed


_SLIDE_OUT, _SLIDE_REWARD, _SLIDE_CHANGED = _build_tables()

# For each action: the 16 board locs grouped into 4 lines (line-major), so
# state[_READ[a]].reshape(4,4) gives the 4 lines with cell 0 = moving edge.
_READ = np.stack([
    np.array([loc for line in range(SIZE) for loc in _line_indices(a, line)], dtype=np.int64)
    for a in range(NUM_ACTIONS)
], axis=0)


def initial_state(rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(AREA, dtype=np.int8)
    _spawn_inplace(state, rng)
    _spawn_inplace(state, rng)
    return state


def _move_raw(state: np.ndarray, action: int):
    """Returns (afterstate int8(16), reward int, changed bool)."""
    read = _READ[action]
    lines = np.clip(state[read].astype(np.int64).reshape(4, 4), 0, _BASE - 1)
    idx = lines @ _WEIGHTS                       # (4,) one index per line
    outs = _SLIDE_OUT[idx]                       # (4,4)
    reward = int(_SLIDE_REWARD[idx].sum())
    changed = bool(_SLIDE_CHANGED[idx].any())
    after = np.empty(AREA, dtype=np.int8)
    after[read] = outs.reshape(AREA)
    return after, reward, changed


def apply_move(state: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    after, reward, changed = _move_raw(state, action)
    if not changed:
        return state.copy(), 0, False
    return after, reward, True


def legal_actions(state: np.ndarray) -> np.ndarray:
    legal = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    for a in range(NUM_ACTIONS):
        legal[a] = 1 if _move_raw(state, a)[2] else 0
    return legal


def is_terminal(state: np.ndarray) -> bool:
    return not any(_move_raw(state, a)[2] for a in range(NUM_ACTIONS))


def spawn_distribution(afterstate: np.ndarray) -> list[tuple[int, int, float]]:
    empties = [i for i in range(AREA) if afterstate[i] == 0]
    if not empties:
        return []
    per = 1.0 / len(empties)
    out = []
    for cell in empties:
        out.append((cell, 1, per * PROB_2))
        out.append((cell, 2, per * PROB_4))
    return out


def _spawn_inplace(state: np.ndarray, rng: np.random.Generator) -> None:
    empties = np.flatnonzero(state == 0)
    if empties.size == 0:
        return
    cell = int(rng.choice(empties))
    state[cell] = 2 if rng.random() < PROB_4 else 1


def spawn_random(afterstate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nxt = afterstate.copy()
    _spawn_inplace(nxt, rng)
    return nxt


def encode_state(state: np.ndarray) -> np.ndarray:
    """(NUM_PLANES, SIZE, SIZE) float32 one-hot exponent planes (clamped)."""
    exps = np.clip(state.astype(np.int64), 0, NUM_PLANES - 1)
    enc = np.zeros((NUM_PLANES, AREA), dtype=np.float32)
    enc[exps, np.arange(AREA)] = 1.0
    return enc.reshape(NUM_PLANES, SIZE, SIZE)


def encode_batch(states: list[np.ndarray]) -> np.ndarray:
    """(B, NUM_PLANES, SIZE, SIZE) float32."""
    arr = np.clip(np.asarray(states, dtype=np.int64), 0, NUM_PLANES - 1)  # (B,16)
    B = arr.shape[0]
    enc = np.zeros((B, NUM_PLANES, AREA), dtype=np.float32)
    rows = np.arange(B)[:, None]
    cols = np.arange(AREA)[None, :]
    enc[rows, arr, cols] = 1.0
    return enc.reshape(B, NUM_PLANES, SIZE, SIZE)


def max_tile_exp(state: np.ndarray) -> int:
    return int(state.max()) if state.size else 0


def render(state: np.ndarray) -> str:
    rows = []
    for r in range(SIZE):
        cells = []
        for c in range(SIZE):
            e = int(state[r * SIZE + c])
            cells.append(f"{(1 << e) if e else '.':>5}")
        rows.append(" ".join(cells))
    return "\n".join(rows)
