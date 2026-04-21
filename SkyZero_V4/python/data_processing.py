"""NPZ I/O and D4 augmentation helpers shared by shuffle.py and train.py.

NPZ schema (written by C++ selfplay, read by Python):
    state:                  int8,   (N, num_planes, H, W)
    policy_target:          float32,(N, H*W)
    opponent_policy_target: float32,(N, H*W)
    opponent_policy_mask:   float32,(N,)
    value_target:           float32,(N, 3)    # WDL
    sample_weight:          float32,(N,)
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


NPZ_KEYS = (
    "state",
    "policy_target",
    "opponent_policy_target",
    "opponent_policy_mask",
    "value_target",
    "sample_weight",
)


@dataclass
class NpzBatch:
    state: np.ndarray
    policy_target: np.ndarray
    opponent_policy_target: np.ndarray
    opponent_policy_mask: np.ndarray
    value_target: np.ndarray
    sample_weight: np.ndarray

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def select(self, idx: np.ndarray) -> "NpzBatch":
        return NpzBatch(
            state=self.state[idx],
            policy_target=self.policy_target[idx],
            opponent_policy_target=self.opponent_policy_target[idx],
            opponent_policy_mask=self.opponent_policy_mask[idx],
            value_target=self.value_target[idx],
            sample_weight=self.sample_weight[idx],
        )


def load_npz(path: str | pathlib.Path) -> NpzBatch:
    with np.load(path) as f:
        return NpzBatch(
            state=np.asarray(f["state"], dtype=np.int8),
            policy_target=np.asarray(f["policy_target"], dtype=np.float32),
            opponent_policy_target=np.asarray(f["opponent_policy_target"], dtype=np.float32),
            opponent_policy_mask=np.asarray(f["opponent_policy_mask"], dtype=np.float32),
            value_target=np.asarray(f["value_target"], dtype=np.float32),
            sample_weight=np.asarray(f["sample_weight"], dtype=np.float32),
        )


def save_npz(path: str | pathlib.Path, batch: NpzBatch) -> None:
    np.savez(
        path,
        state=batch.state.astype(np.int8, copy=False),
        policy_target=batch.policy_target.astype(np.float32, copy=False),
        opponent_policy_target=batch.opponent_policy_target.astype(np.float32, copy=False),
        opponent_policy_mask=batch.opponent_policy_mask.astype(np.float32, copy=False),
        value_target=batch.value_target.astype(np.float32, copy=False),
        sample_weight=batch.sample_weight.astype(np.float32, copy=False),
    )


def concat_batches(batches: Iterable[NpzBatch]) -> NpzBatch:
    batches = list(batches)
    if not batches:
        raise ValueError("concat_batches: empty list")
    return NpzBatch(
        state=np.concatenate([b.state for b in batches], axis=0),
        policy_target=np.concatenate([b.policy_target for b in batches], axis=0),
        opponent_policy_target=np.concatenate([b.opponent_policy_target for b in batches], axis=0),
        opponent_policy_mask=np.concatenate([b.opponent_policy_mask for b in batches], axis=0),
        value_target=np.concatenate([b.value_target for b in batches], axis=0),
        sample_weight=np.concatenate([b.sample_weight for b in batches], axis=0),
    )


def count_rows(path: str | pathlib.Path) -> int:
    """Cheap row count — only reads the state array shape."""
    with np.load(path) as f:
        return int(f["state"].shape[0])


# ---------------------------------------------------------------------------
# D4 augmentation (torch, batched; used on-the-fly during training)
# ---------------------------------------------------------------------------

def _apply_rot_flip_2d(t: torch.Tensor, k: int, flip: bool, spatial_dims: tuple[int, int]) -> torch.Tensor:
    if k:
        t = torch.rot90(t, k, spatial_dims)
    if flip:
        t = torch.flip(t, dims=(spatial_dims[1],))
    return t


def random_d4_inplace(
    state: torch.Tensor,
    policy_target: torch.Tensor,
    opponent_policy_target: torch.Tensor,
    board_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply independent D4 transforms to each sample in the batch.

    state:                   (B, C, H, W)
    policy_target:           (B, H*W)
    opponent_policy_target:  (B, H*W)
    Returns transformed tensors (not in-place despite the name; caller should
    reassign).

    NOTE: value_target and sample_weight are invariant under D4 and left alone.
    """
    B = state.shape[0]
    H = board_size
    policy_target = policy_target.view(B, 1, H, H)
    opponent_policy_target = opponent_policy_target.view(B, 1, H, H)

    # Bucket samples by the 8 transforms so we can apply them with tensor ops.
    transforms = torch.randint(0, 8, (B,), device=state.device)

    out_state = state.clone()
    out_pol = policy_target.clone()
    out_opp = opponent_policy_target.clone()

    for t in range(8):
        mask = (transforms == t)
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        k = t % 4
        flip = t >= 4
        out_state[idx] = _apply_rot_flip_2d(state[idx], k, flip, (2, 3))
        out_pol[idx] = _apply_rot_flip_2d(policy_target[idx], k, flip, (2, 3))
        out_opp[idx] = _apply_rot_flip_2d(opponent_policy_target[idx], k, flip, (2, 3))

    return out_state, out_pol.view(B, H * H), out_opp.view(B, H * H)
