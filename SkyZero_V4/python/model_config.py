"""
Model configuration for SkyZero_V4.
Adapted from KataGomo-Gom2024/python/modelconfigs.py.
"""

from typing import Dict, Any

ModelConfig = Dict[str, Any]

SKYZERO_B6C96 = {
    "norm_kind": "bnorm",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 96,
    "mid_num_channels": 96,
    "gpool_num_channels": 32,
    "use_attention_pool": False,
    "block_kind": [
        ["rconv1", "regular"],
        ["rconv2", "regular"],
        ["rconv3", "regulargpool"],
        ["rconv4", "regular"],
        ["rconv5", "regulargpool"],
        ["rconv6", "regular"],
    ],
    "p1_num_channels": 32,
    "g1_num_channels": 32,
    "v1_num_channels": 32,
    "v2_size": 64,
    "activation": "relu",
}

SKYZERO_B4C32 = {
    "norm_kind": "bnorm",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 32,
    "mid_num_channels": 32,
    "gpool_num_channels": 16,
    "use_attention_pool": False,
    "block_kind": [
        ["rconv1", "regular"],
        ["rconv2", "regular"],
        ["rconv3", "regulargpool"],
        ["rconv4", "regular"],
    ],
    "p1_num_channels": 12,
    "g1_num_channels": 12,
    "v1_num_channels": 12,
    "v2_size": 24,
    "activation": "relu",
}

SKYZERO_B10C128 = {
    "norm_kind": "bnorm",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels": 128,
    "mid_num_channels": 128,
    "gpool_num_channels": 32,
    "use_attention_pool": False,
    "block_kind": [
        ["rconv1", "regular"],
        ["rconv2", "regular"],
        ["rconv3", "regular"],
        ["rconv4", "regular"],
        ["rconv5", "regulargpool"],
        ["rconv6", "regular"],
        ["rconv7", "regular"],
        ["rconv8", "regulargpool"],
        ["rconv9", "regular"],
        ["rconv10", "regular"],
    ],
    "p1_num_channels": 32,
    "g1_num_channels": 32,
    "v1_num_channels": 32,
    "v2_size": 80,
    "activation": "relu",
}

CONFIG_BY_NAME = {
    "b6c96": SKYZERO_B6C96,
    "b4c32": SKYZERO_B4C32,
    "b10c128": SKYZERO_B10C128,
}
