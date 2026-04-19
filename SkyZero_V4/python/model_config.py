"""Model configuration for SkyZero_V4 (aligned with V2.1 ResNet shape)."""

from typing import Dict, Any

ModelConfig = Dict[str, Any]

SKYZERO_B6C96 = {"num_blocks": 6, "num_channels": 96}
SKYZERO_B4C32 = {"num_blocks": 4, "num_channels": 32}
SKYZERO_B10C128 = {"num_blocks": 10, "num_channels": 128}

CONFIG_BY_NAME = {
    "b6c96": SKYZERO_B6C96,
    "b4c32": SKYZERO_B4C32,
    "b10c128": SKYZERO_B10C128,
}
