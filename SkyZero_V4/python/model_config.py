"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.

Default values target the legacy nets.py shape (b12c128 trunk, 4 input
planes, no global features). Phase A KataGo v15 model uses the extended
fields (c_mid, c_gpool, num_global_features, has_intermediate_head, ...)
which legacy nets.py ignores.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 4  # own, opp, forbidden_black, forbidden_white
    num_blocks: int = 12
    num_channels: int = 128

    # ----- Phase A (KataGo v15) extensions -----
    # Legacy nets.py ignores these fields; nets_v2 reads them.
    num_global_features: int = 12
    c_mid: int = 64                     # bottleneck mid channels
    c_gpool: int = 16                   # ConvAndGPool branch channels
    internal_length: int = 2
    has_intermediate_head: bool = True
    intermediate_head_blocks: int = 8
    c_p1: int = 32                      # PolicyHead conv1p
    c_g1: int = 32                      # PolicyHead conv1g
    c_v1: int = 32                      # ValueHead conv1
    c_v2: int = 48                      # ValueHead linear2
    activation: str = "mish"
    version: int = 15

    # ----- Legacy fields (used by nets.py only) -----
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
    """Read env vars set by scripts/run.cfg (sourced in bash then exported)."""
    import os
    cfg = NetConfig()
    if (v := os.environ.get("BOARD_SIZE")):
        cfg.board_size = int(v)
    if (v := os.environ.get("NUM_PLANES")):
        cfg.num_planes = int(v)
    if (v := os.environ.get("NUM_BLOCKS")):
        cfg.num_blocks = int(v)
    if (v := os.environ.get("NUM_CHANNELS")):
        cfg.num_channels = int(v)
    if (v := os.environ.get("NUM_GLOBAL_FEATURES")):
        cfg.num_global_features = int(v)
    if (v := os.environ.get("C_MID")):
        cfg.c_mid = int(v)
    if (v := os.environ.get("C_GPOOL")):
        cfg.c_gpool = int(v)
    if (v := os.environ.get("INTERMEDIATE_HEAD_BLOCKS")):
        cfg.intermediate_head_blocks = int(v)
    return cfg
