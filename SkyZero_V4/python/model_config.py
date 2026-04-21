"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 4  # own, opp, forbidden_black, forbidden_white
    num_blocks: int = 6
    num_channels: int = 96

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
    return cfg
