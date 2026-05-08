"""Zero-pad helpers for resuming pre-V6 checkpoints into V6 networks.

V6 ← V5 changes that break shape-strict load_state_dict:
- num_planes 5 → 6 (added my_only_loc one-hot)        → conv_spatial.weight in_channels grows
- num_global_features 12 → 14 (VCF + PDA dims)        → linear_global.weight in_features grows

Both growths are at the leading edge of the new dim (channel index 5,
global indices 12-13). Zero-padding the new slice is correct: the new
features carry no signal in the old checkpoint, and Phase A demonstrated
the network forward is stable when those columns are zero.

Both functions are idempotent: if the existing weight is already at
target_in (or larger), the state_dict is returned unchanged.
"""

from __future__ import annotations

import torch


def zero_pad_linear_global(state_dict: dict, target_in: int,
                           key: str = "linear_global.weight") -> dict:
    """Pad ``linear_global.weight`` along axis=1 from (C, old_in) to
    (C, target_in) with zeros. Used when V6's num_global_features (14) is
    larger than the checkpoint's (e.g. 12). Idempotent.
    """
    W = state_dict.get(key)
    if W is None or W.shape[1] >= target_in:
        return state_dict
    pad = torch.zeros(W.shape[0], target_in - W.shape[1],
                      dtype=W.dtype, device=W.device)
    state_dict[key] = torch.cat([W, pad], dim=1)
    return state_dict


def zero_pad_conv_spatial(state_dict: dict, target_in: int,
                          key: str = "conv_spatial.weight") -> dict:
    """Pad ``conv_spatial.weight`` along axis=1 (in_channels) from
    (out, old_in, k, k) to (out, target_in, k, k) with zeros. Used when V6's
    num_planes (6) is larger than the checkpoint's (e.g. 5). Idempotent.
    """
    W = state_dict.get(key)
    if W is None or W.shape[1] >= target_in:
        return state_dict
    pad = torch.zeros(W.shape[0], target_in - W.shape[1],
                      W.shape[2], W.shape[3],
                      dtype=W.dtype, device=W.device)
    state_dict[key] = torch.cat([W, pad], dim=1)
    return state_dict


def zero_pad_for_v6(state_dict: dict, num_planes: int,
                    num_global_features: int) -> dict:
    """Convenience: apply both pad helpers in one call."""
    state_dict = zero_pad_conv_spatial(state_dict, num_planes)
    state_dict = zero_pad_linear_global(state_dict, num_global_features)
    return state_dict
