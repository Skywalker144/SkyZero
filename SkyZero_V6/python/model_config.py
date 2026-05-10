"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.

Defaults match a b12c128 KataGo v15 model. The KataGo v15 derived fields
(c_mid, c_gpool, c_p1, c_g1, c_v1, c_v2, intermediate_head_blocks) auto-
derive from num_channels / num_blocks unless explicitly overridden, so
users can scale the model just by setting num_blocks / num_channels.

Multi-slot training: num_blocks / num_channels come from the slot table
(MODEL_BLOCKS / MODEL_CHANNELS in run.cfg, parsed by slots.py).
net_config_for_slot(name) is the single entry point — train.py /
init_model.py / export_model.py all go through it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 6  # mask, own, opp, forbidden_black, forbidden_white, my_only_loc
    num_blocks: int = 12
    num_channels: int = 128

    # ----- KataGo v15 fields -----
    num_global_features: int = 14
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


def net_config_for_slot(slot_name: str) -> NetConfig:
    """Build a NetConfig for the named slot.

    num_blocks / num_channels come from the slot table (slots.py). All other
    fields (board canvas, planes, optional dim overrides like C_MID) come
    from process env — these are shared across every slot.

    NetConfig.board_size (model canvas) is read from env MAX_BOARD_SIZE,
    which scripts/run.cfg owns. cpp/CMakeLists.txt parses the same line and
    bakes it into the C++ binary as SKYZERO_MAX_BOARD_SIZE (consumed by
    cpp/envs/gomoku.h's Gomoku::MAX_BOARD_SIZE). To change canvas size: edit
    MAX_BOARD_SIZE in run.cfg, then `cmake --build cpp/build` (auto-detects
    the change) and re-trace via init_model.py.
    """
    import os
    from slots import get_slot
    slot = get_slot(slot_name)
    kwargs: dict = {
        "num_blocks": slot.num_blocks,
        "num_channels": slot.num_channels,
    }
    if (v := os.environ.get("MAX_BOARD_SIZE")):
        kwargs["board_size"] = int(v)
    if (v := os.environ.get("NUM_PLANES")):
        kwargs["num_planes"] = int(v)
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
