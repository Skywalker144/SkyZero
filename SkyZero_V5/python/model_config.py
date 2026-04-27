"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.

Defaults match a b12c128 KataGo v15 model. The Phase A KataGo v15 fields
(c_mid, c_gpool, c_p1, c_g1, c_v1, c_v2, intermediate_head_blocks) auto-
derive from num_channels / num_blocks unless explicitly overridden, so
users can scale the model just by setting num_blocks / num_channels.

Legacy nets.py reads board_size, num_planes, num_blocks, num_channels
plus the four properties below — auto-derived fields are ignored by it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 5  # mask, own, opp, forbidden_black, forbidden_white
    num_blocks: int = 12
    num_channels: int = 128

    # ----- Phase A (KataGo v15) extensions -----
    num_global_features: int = 12
    internal_length: int = 2
    has_intermediate_head: bool = True
    activation: str = "mish"
    version: int = 15

    # Auto-derived from num_channels / num_blocks unless explicitly set.
    c_mid: Optional[int] = None
    c_gpool: Optional[int] = None
    c_p1: Optional[int] = None
    c_g1: Optional[int] = None
    c_v1: Optional[int] = None
    c_v2: Optional[int] = None
    intermediate_head_blocks: Optional[int] = None

    def __post_init__(self) -> None:
        if self.c_mid is None:
            self.c_mid = max(16, self.num_channels // 2)
        if self.c_gpool is None:
            self.c_gpool = max(16, self.num_channels // 8)
        if self.c_p1 is None:
            self.c_p1 = max(16, self.num_channels // 4)
        if self.c_g1 is None:
            self.c_g1 = max(16, self.num_channels // 4)
        if self.c_v1 is None:
            self.c_v1 = max(16, self.num_channels // 4)
        if self.c_v2 is None:
            self.c_v2 = max(32, self.num_channels // 4 + 16)
        if self.intermediate_head_blocks is None:
            self.intermediate_head_blocks = max(1, self.num_blocks * 2 // 3)

    # ----- Legacy properties (used by nets.py only) -----
    @property
    def mid_channels(self) -> int:
        return max(16, self.num_channels // 2)

    @property
    def policy_head_channels(self) -> int:
        return self.num_channels // 2

    @property
    def value_head_channels(self) -> int:
        return self.num_channels // 4

    @property
    def value_fc_channels(self) -> int:
        return self.num_channels // 2


def net_config_from_env() -> NetConfig:
    """Read env vars set by scripts/run.cfg (sourced in bash then exported).

    Reads the basic dimensions (NUM_BLOCKS, NUM_CHANNELS, NUM_PLANES,
    BOARD_SIZE, NUM_GLOBAL_FEATURES) plus optional explicit overrides for
    derived fields (C_MID, C_GPOOL, INTERMEDIATE_HEAD_BLOCKS, ...).
    Constructor's __post_init__ fills any field left as None.
    """
    import os
    kwargs: dict = {}
    if (v := os.environ.get("BOARD_SIZE")):
        kwargs["board_size"] = int(v)
    if (v := os.environ.get("NUM_PLANES")):
        kwargs["num_planes"] = int(v)
    if (v := os.environ.get("NUM_BLOCKS")):
        kwargs["num_blocks"] = int(v)
    if (v := os.environ.get("NUM_CHANNELS")):
        kwargs["num_channels"] = int(v)
    if (v := os.environ.get("NUM_GLOBAL_FEATURES")):
        kwargs["num_global_features"] = int(v)
    if (v := os.environ.get("C_MID")):
        kwargs["c_mid"] = int(v)
    if (v := os.environ.get("C_GPOOL")):
        kwargs["c_gpool"] = int(v)
    if (v := os.environ.get("INTERMEDIATE_HEAD_BLOCKS")):
        kwargs["intermediate_head_blocks"] = int(v)
    if (v := os.environ.get("C_P1")):
        kwargs["c_p1"] = int(v)
    if (v := os.environ.get("C_G1")):
        kwargs["c_g1"] = int(v)
    if (v := os.environ.get("C_V1")):
        kwargs["c_v1"] = int(v)
    if (v := os.environ.get("C_V2")):
        kwargs["c_v2"] = int(v)
    if (v := os.environ.get("ACTIVATION")):
        kwargs["activation"] = v
    return NetConfig(**kwargs)
