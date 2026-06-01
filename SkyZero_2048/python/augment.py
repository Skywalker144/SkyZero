"""D4 data augmentation for 2048.

The board has full dihedral symmetry, but — unlike Gomoku where only the board
planes move — rotating/flipping the board also PERMUTES the 4 slide directions.
We derive each transform's action permutation by brute force and assert the
equivariance  T(afterstate(s, a)) == afterstate(T(s), perm[a])  so a hand-sign
error can't slip through (this is the trap called out in the design notes).
"""
from __future__ import annotations

import numpy as np

import game as G


def _xf_grid(grid2d: np.ndarray, k: int, flip: bool) -> np.ndarray:
    g = np.rot90(grid2d, k)
    if flip:
        g = np.flip(g, axis=1)
    return np.ascontiguousarray(g)


def _xf_board(board16: np.ndarray, k: int, flip: bool) -> np.ndarray:
    return _xf_grid(board16.reshape(G.SIZE, G.SIZE), k, flip).reshape(G.AREA).astype(np.int8)


_TEST_BOARDS = [
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8),
    np.array([1, 1, 0, 2, 0, 3, 3, 0, 2, 0, 0, 2, 1, 0, 1, 0], dtype=np.int8),
    np.array([2, 0, 2, 0, 0, 3, 0, 3, 1, 1, 0, 0, 0, 0, 4, 4], dtype=np.int8),
]


def _derive_perm(k: int, flip: bool) -> list[int]:
    perm: list[int | None] = [None, None, None, None]
    for a in range(4):
        for ap in range(4):
            ok = True
            for s in _TEST_BOARDS:
                after_a, rew_a, ch_a = G.apply_move(s, a)
                s2 = _xf_board(s, k, flip)
                after_ap, rew_ap, ch_ap = G.apply_move(s2, ap)
                lhs = _xf_board(after_a, k, flip)
                if not np.array_equal(lhs, after_ap) or rew_a != rew_ap or ch_a != ch_ap:
                    ok = False
                    break
            if ok:
                perm[a] = ap
                break
        if perm[a] is None:
            raise RuntimeError(f"no action perm for transform k={k} flip={flip} action={a}")
    return perm  # type: ignore[return-value]


# (k_rot, flip, action_perm) for the 8 dihedral transforms.
TRANSFORMS: list[tuple[int, bool, list[int]]] = [
    (k, flip, _derive_perm(k, flip)) for flip in (False, True) for k in range(4)
]


def augment_batch(state: "np.ndarray | object", policy, transforms=TRANSFORMS):
    """Apply an independent random D4 transform to each sample (torch tensors).

    state:  (B, C, 4, 4) float tensor
    policy: (B, 4)       float tensor (improved-policy targets)
    Returns transformed (state, policy). Value targets are D4-invariant.
    """
    import torch

    B = state.shape[0]
    pick = torch.randint(0, len(transforms), (B,), device=state.device)
    out_s = state.clone()
    out_p = policy.clone()
    for t, (k, flip, perm) in enumerate(transforms):
        idx = (pick == t).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        s = state[idx]
        if k:
            s = torch.rot90(s, k, dims=(2, 3))
        if flip:
            s = torch.flip(s, dims=(3,))
        out_s[idx] = s
        # We need out_p[perm[a]] = policy[a]; a gather uses the INVERSE perm so
        # that out_p[:, j] = policy[:, inv[j]] gives out_p[:, perm[a]] = policy[:, a].
        inv = [0, 0, 0, 0]
        for a, p in enumerate(perm):
            inv[p] = a
        inv_t = torch.tensor(inv, device=policy.device)
        out_p[idx] = policy[idx][:, inv_t]
    return out_s, out_p
