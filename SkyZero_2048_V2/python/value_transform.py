"""Invertible value-scaling transform (MuZero / R2D2 `h`).

2048 value targets are discounted future score in raw points, reaching tens of
thousands, and the late-game spawn randomness gives the *large* targets a fat
tail. A plain linear `target / VALUE_SCALE` regression therefore chases a high-
variance, outlier-heavy signal. The MuZero scaling function

    h(x)  = sign(x) * (sqrt(|x| + 1) - 1) + eps * x          (eps = 1e-3)

compresses the large magnitudes (sqrt) far more than the small ones while the
`eps*x` term keeps it strictly monotone / invertible, so the value head regresses
a low-variance target and we decode back to raw points for the search. Its exact
inverse is

    h_inv(y) = sign(y) * ( ( (sqrt(1 + 4*eps*(|y| + 1 + eps)) - 1) / (2*eps) )^2 - 1 )

This module is the single source of truth; cpp/infer_server_2048.h mirrors the
same constants and formulas (keep them in sync). The pipeline composes h with
VALUE_SCALE: training target = h(raw) / VALUE_SCALE, decode = h_inv(out * SCALE).
The transform is ALWAYS applied (the old VALUE_TRANSFORM toggle was removed):
train.py wraps the target in to_h_torch, and net_evaluator / infer_server decode
with from_h / inv_value_h. VALUE_SCALE therefore lives in h-space (~30).
"""
from __future__ import annotations

EPS = 1e-3


def to_h_np(x):
    """raw points -> h-space (numpy / scalar)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + EPS * x


def from_h_np(y):
    """h-space -> raw points (numpy / scalar), exact inverse of to_h_np."""
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    z = (np.sqrt(1.0 + 4.0 * EPS * (np.abs(y) + 1.0 + EPS)) - 1.0) / (2.0 * EPS)
    return np.sign(y) * (z * z - 1.0)


def to_h_torch(x):
    """raw points -> h-space (torch tensor)."""
    import torch
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + EPS * x
