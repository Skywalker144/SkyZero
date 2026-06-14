"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.

KataGo-style scaling (V7.7): give just `num_blocks` (b1..b18) and everything
derives — num_channels ≈ num_blocks*20 snapped to the nearest power-of-2 or
multiple-of-8, then the head widths from num_channels:

    c_mid   = C/2          (KataGo 唯一通用比例)
    c_gpool ≈ C/6   (< c_mid)
    c_p1=c_g1 ≈ C/8
    c_v1    ≈ C/4
    c_v2    ≈ C/4 + 32
    intermediate_head_blocks = num_blocks*2//3   (保留中间监督)

KataGo 各官方配置的 head 宽度其实是逐个手调的(无统一公式),所以这套是
「KataGo 风格」的缩放,不是某个具体 KataGo 配置的精确复刻。任何字段显式给值即覆盖推导。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


def _round8(x: float) -> int:
    """就近取 8 的倍数(下限 8)。"""
    return max(8, int(x / 8 + 0.5) * 8)


def _nearest_pow2_or_mult8(target: int) -> int:
    """返回与 target 最近的「2 的幂 或 8 的倍数」(并列时取 8 的倍数)。"""
    m8 = int(target / 8 + 0.5) * 8
    lo = 2 ** int(math.floor(math.log2(target)))
    hi = 2 ** int(math.ceil(math.log2(target)))
    p2 = lo if abs(target - lo) <= abs(target - hi) else hi
    return p2 if abs(target - p2) < abs(target - m8) else m8


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 5  # mask, own, opp, forbidden_black, forbidden_white
    num_blocks: int = 12
    num_channels: Optional[int] = None  # None → 由 num_blocks 推导 (≈ num_blocks*20, 取最近的 2 幂或 8 倍数)

    # ----- KataGo v15 fields -----
    num_global_features: int = 12
    internal_length: int = 2
    has_intermediate_head: bool = True
    activation: str = "mish"
    version: int = 15
    # fixscaleonenorm 的 gamma 约定: True=中心 0 (forward 用 (gamma+1)*scale, 当代 KataGo / b40 /
    # 自家从零训练默认); False=中心 1 (gamma*scale, 旧约定 — config 无 gamma_weight_decay_center_1,
    # 如官方 b28c512nbt)。仅在加载对应约定的预训练权重时才需设 False。
    gamma_center_one: bool = True

    # Auto-derived from num_channels / num_blocks unless explicitly set.
    c_mid: Optional[int] = None
    c_gpool: Optional[int] = None
    c_p1: Optional[int] = None
    c_g1: Optional[int] = None
    c_v1: Optional[int] = None
    c_v2: Optional[int] = None
    intermediate_head_blocks: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_channels is None:
            self.num_channels = _nearest_pow2_or_mult8(self.num_blocks * 20)
        c = self.num_channels
        if self.c_mid is None:
            self.c_mid = _round8(c / 2)
        if self.c_gpool is None:
            self.c_gpool = max(8, min(_round8(c / 6), self.c_mid - 8))
        if self.c_p1 is None:
            self.c_p1 = max(8, _round8(c / 8))
        if self.c_g1 is None:
            self.c_g1 = max(8, _round8(c / 8))
        if self.c_v1 is None:
            self.c_v1 = max(8, _round8(c / 4))
        if self.c_v2 is None:
            self.c_v2 = max(32, _round8(c / 4 + 32))
        if self.intermediate_head_blocks is None:
            self.intermediate_head_blocks = max(1, self.num_blocks * 2 // 3)


def net_config_from_env() -> NetConfig:
    """Read env vars set by scripts/run.cfg (sourced in bash then exported).

    Reads the basic dimensions (NUM_BLOCKS, NUM_CHANNELS, NUM_PLANES,
    NUM_GLOBAL_FEATURES) plus optional explicit overrides for derived
    fields (C_MID, C_GPOOL, INTERMEDIATE_HEAD_BLOCKS, ...).
    Constructor's __post_init__ fills any field left as None.

    NetConfig.board_size (model canvas) is read from env MAX_BOARD_SIZE,
    which scripts/run.cfg owns. cpp/CMakeLists.txt parses the same line and
    bakes it into the C++ binary as SKYZERO_MAX_BOARD_SIZE (consumed by
    cpp/envs/gomoku.h's Gomoku::MAX_BOARD_SIZE). To change canvas size: edit
    MAX_BOARD_SIZE in run.cfg, then `cmake --build cpp/build` (auto-detects
    the change) and re-trace via init_model.py.
    """
    import os
    kwargs: dict = {}
    if (v := os.environ.get("MAX_BOARD_SIZE")):
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
    if (v := os.environ.get("GAMMA_CENTER_ONE")):
        kwargs["gamma_center_one"] = v.strip().lower() in ("1", "true", "yes", "on")
    return NetConfig(**kwargs)


_NET_NAME_RE = re.compile(r"^b(\d+)(?:c(\d+))?$")


def net_config_from_name(name: str) -> NetConfig:
    """Parse a network name into a NetConfig.

    Format: 'b<blocks>' (channels auto-derived ≈ blocks*20) or
    'b<blocks>c<channels>' (channels explicit). All other fields (board_size,
    num_planes, num_global_features, derived c_* widths, etc.) come from
    env vars via the same path as net_config_from_env(): we read those
    overrides first, then apply num_blocks / num_channels from the name.
    """
    m = _NET_NAME_RE.match(name)
    if not m:
        raise ValueError(
            f"bad network name {name!r}: expected format b<blocks> or "
            "b<blocks>c<channels>, e.g. b7, b5c128, b10c256"
        )
    cfg = net_config_from_env()
    cfg.num_blocks = int(m.group(1))
    import os
    if m.group(2):
        cfg.num_channels = int(m.group(2))
    elif not os.environ.get("NUM_CHANNELS"):
        cfg.num_channels = None  # 让 __post_init__ 按 num_blocks 推导
    # Re-derive auto fields (c_mid, c_gpool, ...) that depend on the
    # block/channel count we just overrode. Anything explicitly set via
    # env var (C_MID, etc.) survives because we only reset fields that
    # net_config_from_env didn't populate.
    for field, attr in (
        ("C_MID", "c_mid"), ("C_GPOOL", "c_gpool"),
        ("C_P1", "c_p1"), ("C_G1", "c_g1"),
        ("C_V1", "c_v1"), ("C_V2", "c_v2"),
        ("INTERMEDIATE_HEAD_BLOCKS", "intermediate_head_blocks"),
    ):
        if not os.environ.get(field):
            setattr(cfg, attr, None)
    cfg.__post_init__()
    return cfg
