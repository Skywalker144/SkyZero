"""NPZ I/O helpers shared by shuffle.py and the training loop — the 2048
analogue of SkyZero_V7.1/python/data_processing.py, trimmed to the 2048 schema.

NPZ schema (written by the C++ self-play binary, read here):
    state:          int8,   (N, 16)   raw exponents, row-major loc = r*4 + c
    policy_target:  float32,(N, 4)    Gumbel improved policy over directions
    value_target:   float32,(N, 1)    discounted future score (raw points)
    sample_weight:  float32,(N,)      per-row weight (1.0 unless reweighted)
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

NPZ_KEYS = ("state", "policy_target", "value_target", "sample_weight")


@dataclass
class NpzBatch:
    state: np.ndarray          # (N, 16) int8
    policy_target: np.ndarray  # (N, 4)  float32
    value_target: np.ndarray   # (N, 1)  float32
    sample_weight: np.ndarray  # (N,)    float32

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def select(self, idx: np.ndarray) -> "NpzBatch":
        return NpzBatch(self.state[idx], self.policy_target[idx],
                        self.value_target[idx], self.sample_weight[idx])


def load_npz(path: str | pathlib.Path) -> NpzBatch:
    with np.load(path) as f:
        n = f["state"].shape[0]
        # sample_weight is optional in older shards — default to 1.0.
        sw = (np.asarray(f["sample_weight"], dtype=np.float32)
              if "sample_weight" in f.files else np.ones(n, dtype=np.float32))
        return NpzBatch(
            state=np.asarray(f["state"], dtype=np.int8),
            policy_target=np.asarray(f["policy_target"], dtype=np.float32),
            value_target=np.asarray(f["value_target"], dtype=np.float32).reshape(n, 1),
            sample_weight=sw,
        )


def save_npz(path: str | pathlib.Path, b: NpzBatch) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        state=b.state.astype(np.int8, copy=False),
        policy_target=b.policy_target.astype(np.float32, copy=False),
        value_target=b.value_target.astype(np.float32, copy=False),
        sample_weight=b.sample_weight.astype(np.float32, copy=False),
    )


def concat_batches(batches: Iterable[NpzBatch]) -> NpzBatch:
    batches = list(batches)
    if not batches:
        raise ValueError("concat_batches: empty list")
    return NpzBatch(
        state=np.concatenate([b.state for b in batches], axis=0),
        policy_target=np.concatenate([b.policy_target for b in batches], axis=0),
        value_target=np.concatenate([b.value_target for b in batches], axis=0),
        sample_weight=np.concatenate([b.sample_weight for b in batches], axis=0),
    )


def count_rows(path: str | pathlib.Path) -> int:
    with np.load(path) as f:
        return int(f["state"].shape[0])


def joint_shuffle_take_first_n(n: int, batch: NpzBatch, rng: np.random.Generator) -> NpzBatch:
    total = len(batch)
    perm = rng.permutation(total) if n >= total else rng.permutation(total)[:n]
    return batch.select(perm)
