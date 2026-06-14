"""Faithful KataGo nested-bottleneck trunk for 4x4 2048 (scalar value head).

This is the opt-in "strong" network (FULL_NET=1); the default is the compact
BatchNorm net in nets.py. It reuses KataGo's exact building blocks — fixscale
("fixscaleonenorm") normalization, RepVGG-style fixup init, the 3-statistic
global pool with a board-size factor, and the nested-bottleneck residual block —
but specialized to 2048:

  - input: (B, NUM_PLANES, 4, 4), no global-feature vector, no variable board
    size (the 4x4 board is always full → mask is all-ones, kept only because the
    KataGo blocks take a mask argument);
  - heads: a 4-direction policy head and a single scalar value head
    (softplus >= 0, in h(raw)/value_scale units), forward(x) -> (policy[B,4],
    value[B]) — byte-identical I/O contract to nets.Net2048 so the C++ infer
    server and TorchScript trace are unchanged.

The WDL / multi-horizon-TD / ownership / futurepos / intermediate / opponent
heads of the mainline KataGoNet are intentionally dropped (2048 is a single-agent
scalar-value MDP).

Traps preserved from the KataGo port (NOTES.md §3):
  1. NestedBottleneckResBlock.forward returns only the residual; the caller does
     `out = out + block(out, ...)`.
  2. FixscaleNorm gamma is a delta — forward uses (gamma + 1) (gamma_center_one).
  3. FixscaleNorm/BiasMask .scale is a plain Python float, NOT in state_dict:
     call initialize() (fresh) or set_norm_scales() (after load_state_dict).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import Config


# ============================================================
# 归一化层 (fixscaleonenorm)
# ============================================================

class FixscaleNorm(nn.Module):
    """fixscaleonenorm 下非-last 归一化层.

    forward: out = (x * (gamma + 1) * scale + beta) [* mask]
    scale 由 set_scale() 设置 (None 时省略); gamma 初始化为 0 (gamma_center_one,
    weight decay 推向 0 → (gamma+1) 中心 1).
    """

    def __init__(self, num_channels: int, use_gamma: bool = True) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        if use_gamma:
            self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.gamma = None
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gamma is not None:
            g = self.gamma + 1.0
            if self.scale is not None:
                out = x * (g * self.scale) + self.beta
            else:
                out = x * g + self.beta
        else:
            if self.scale is not None:
                out = x * self.scale + self.beta
            else:
                out = x + self.beta
        if mask is not None:
            out = out * mask
        return out


class BiasMask(nn.Module):
    """trunk_normless=True 下 trunk-final / head 的 normless-bias 层.

    forward: out = (x * scale + beta) [* mask]  (scale 默认 None → 仅 bias)
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scale is not None:
            out = x * self.scale + self.beta
        else:
            out = x + self.beta
        if mask is not None:
            out = out * mask
        return out


# ============================================================
# 池化模块
# ============================================================

class KataGPool(nn.Module):
    """3 个统计量: mean, mean * board_factor, max.

    board_factor = (sqrt(mask_sum_hw) - 14) / 10. 对固定 4x4 (mask_sum=16) 退化为
    常数 -1.0 — 保留以与 KataGo 块对齐 (网络可吸收该常数偏置).
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True) / mask_sum_hw
        layer_max = (x + (mask - 1.0)).amax(dim=(2, 3), keepdim=True)
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_max,
        ), dim=1)


class KataValueHeadGPool(nn.Module):
    """Value head 专用 GPool: 3 个 mean 统计量 (无 max), 二阶 board factor."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True) / mask_sum_hw
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_mean * ((sqrt_off * sqrt_off) / 100.0 - 0.1),
        ), dim=1)


# ============================================================
# RepVGG init helpers
# ============================================================

def compute_gain(activation: str) -> float:
    """Per KataGo `compute_gain`. Mish gain matches lightvector master."""
    if activation in ("relu", "hardswish"):
        return math.sqrt(2.0)
    if activation == "elu":
        return math.sqrt(1.55052)
    if activation == "mish":
        return math.sqrt(2.210277)
    if activation == "silu":
        return math.sqrt(2.0)
    if activation == "gelu":
        return math.sqrt(2.351718)
    if activation == "identity":
        return 1.0
    raise ValueError(f"Unknown activation: {activation}")


_TRUNC_CORRECTION = 0.87962566103423978   # std correction for trunc_normal a=-2,b=2


def init_weights(tensor: torch.Tensor, activation: str, scale: float,
                 fan_tensor: Optional[torch.Tensor] = None) -> None:
    """KataGo's truncated-normal init: std = scale * gain / sqrt(fan_in)."""
    gain = compute_gain(activation)
    src = fan_tensor if fan_tensor is not None else tensor
    if src.dim() < 2:
        src = src.view(-1, 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(src)
    target_std = scale * gain / math.sqrt(fan_in)
    std = target_std / _TRUNC_CORRECTION
    with torch.no_grad():
        if std < 1e-10:
            tensor.fill_(0.0)
        else:
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


# ============================================================
# 卷积 + 全局池化模块
# ============================================================

class ConvAndGPool(nn.Module):
    """合并到 ResBlock 内层第一个 conv 的位置 (不是独立残差块).

    out = conv1r(x) + linear_g(gpool(act(normg(conv1g(x)))))   [as conv bias]
    """

    def __init__(self, c_in: int, c_out: int, c_gpool: int, activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.normg = FixscaleNorm(c_gpool, use_gamma=True)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale: float) -> None:
        # KataGo master 538-549 (fixscaleonenorm path)
        r_scale, g_scale = 0.8, 0.6
        init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
        init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
        init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_r = self.conv1r(x)
        out_g = self.conv1g(x)
        out_g = self.normg(out_g, mask)
        out_g = self.actg(out_g)
        out_g = self.gpool(out_g, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)
        return out_r + out_g


# ============================================================
# norm → act → (conv | conv-and-gpool) 组合
# ============================================================

class NormActConv(nn.Module):
    """norm → act → (conv | conv-and-gpool)."""

    def __init__(self, c_in: int, c_out: int, activation: str = "mish",
                 kernel_size: int = 3, c_gpool: Optional[int] = None,
                 fixup_use_gamma: bool = True,
                 use_repvgg_init: bool = True) -> None:
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.norm = FixscaleNorm(c_in, use_gamma=fixup_use_gamma)
        self.act = nn.Mish() if activation == "mish" else nn.ReLU()
        self.use_repvgg_init = use_repvgg_init and kernel_size > 1
        if c_gpool is not None:
            self.convpool = ConvAndGPool(c_in, c_out, c_gpool, activation)
            self.conv = None
        else:
            padding = kernel_size // 2
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                                  padding=padding, bias=False)
            self.convpool = None

    def initialize(self, scale: float, norm_scale: Optional[float] = None) -> None:
        self.norm.set_scale(norm_scale)
        if self.convpool is not None:
            self.convpool.initialize(scale=scale)
        else:
            if self.use_repvgg_init:
                init_weights(self.conv.weight, self.activation, scale=scale * 0.8)
                w = self.conv.weight
                center_bonus = w.new_zeros((w.shape[0], w.shape[1]), requires_grad=False)
                init_weights(center_bonus, self.activation, scale=scale * 0.6)
                with torch.no_grad():
                    self.conv.weight[:, :, 1, 1] += center_bonus
            else:
                init_weights(self.conv.weight, self.activation, scale=scale)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.norm(x, mask)
        out = self.act(out)
        if self.convpool is not None:
            return self.convpool(out, mask, mask_sum_hw)
        return self.conv(out)


# ============================================================
# 残差块 (ResBlock 只返残差; 调用方负责加回主干)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, c_mid: int, c_gpool: Optional[int] = None,
                 activation: str = "mish") -> None:
        super().__init__()
        c_out1 = c_mid - (0 if c_gpool is None else c_gpool)
        self.normactconv1 = NormActConv(c_mid, c_out1, activation,
                                        kernel_size=3, c_gpool=c_gpool,
                                        fixup_use_gamma=True)
        self.normactconv2 = NormActConv(c_out1, c_mid, activation,
                                        kernel_size=3, c_gpool=None,
                                        fixup_use_gamma=True)

    def initialize(self, fixup_scale: float) -> None:
        self.normactconv1.initialize(scale=1.0, norm_scale=fixup_scale)
        self.normactconv2.initialize(scale=1.0, norm_scale=None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.normactconv1(x, mask, mask_sum_hw)
        out = self.normactconv2(out, mask, mask_sum_hw)
        return out


# ============================================================
# 嵌套瓶颈残差块 (NestedBottleneckResBlock)
# ============================================================

class NestedBottleneckResBlock(nn.Module):
    """1×1 (c_main → c_mid) → N 个内层 ResBlock → 1×1 (c_mid → c_main).

    ⚠️ forward 只返残差; 调用方必须 `out = out + block(out, mask, mask_sum_hw)`.
    """

    def __init__(self, internal_length: int, c_main: int, c_mid: int,
                 c_gpool: Optional[int] = None, activation: str = "mish") -> None:
        super().__init__()
        self.internal_length = internal_length
        self.normactconvp = NormActConv(c_main, c_mid, activation,
                                        kernel_size=1, fixup_use_gamma=True)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            use_gpool = c_gpool if i == 0 else None
            self.blockstack.append(ResBlock(c_mid, c_gpool=use_gpool, activation=activation))
        self.normactconvq = NormActConv(c_mid, c_main, activation,
                                        kernel_size=1, fixup_use_gamma=True)

    def initialize(self, fixup_scale: float) -> None:
        self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
        for j, block in enumerate(self.blockstack):
            block.initialize(fixup_scale=1.0 / math.sqrt(j + 1.0))
        self.normactconvq.initialize(scale=1.0,
                                     norm_scale=1.0 / math.sqrt(self.internal_length + 1.0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.normactconvp(x, mask, mask_sum_hw)
        for block in self.blockstack:
            out = out + block(out, mask, mask_sum_hw)
        out = self.normactconvq(out, mask, mask_sum_hw)
        return out


# ============================================================
# 2048 标量 nbt 网络 (4x4, 无 global / 无变 board)
# ============================================================

def _derive_head_widths(c_main: int, value_hidden: int) -> tuple[int, int, int]:
    """KataGo-ish head-width ratios off the trunk channel count (p1≈C/8,
    v1≈C/4); value c_v2 reuses Config.value_hidden."""
    c_p1 = max(16, c_main // 8)
    c_v1 = max(16, c_main // 4)
    c_v2 = value_hidden
    return c_p1, c_v1, c_v2


class Net2048NBT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.activation = "mish"
        c = cfg.channels
        c_mid = cfg.c_mid
        c_gp = cfg.c_gpool
        self.num_blocks = cfg.blocks
        c_p1, c_v1, c_v2 = _derive_head_widths(c, cfg.value_hidden)

        # 输入投影 (无 global-feature 向量).
        self.stem = nn.Conv2d(cfg.num_planes, c, 3, padding=1, bias=False)

        # 主干 — gpool 仅在 i % 3 == 2 的块 (KataGo 默认).
        self.blocks = nn.ModuleList([
            NestedBottleneckResBlock(
                internal_length=cfg.internal_length, c_main=c, c_mid=c_mid,
                c_gpool=(c_gp if (i % 3 == 2) else None), activation=self.activation,
            ) for i in range(cfg.blocks)
        ])
        self.norm_trunkfinal = BiasMask(c)
        self.act_trunkfinal = nn.Mish()

        # policy head: 1x1 -> bias -> act -> flatten(16 cells) -> 4 directions.
        self.p_conv = nn.Conv2d(c, c_p1, 1, bias=False)
        self.p_bias = BiasMask(c_p1)
        self.act_p = nn.Mish()
        self.p_fc = nn.Linear(c_p1 * 16, 4, bias=False)

        # value head: 1x1 -> bias -> act -> value gpool -> MLP -> 1 (softplus >=0).
        self.v_conv = nn.Conv2d(c, c_v1, 1, bias=False)
        self.v_bias = BiasMask(c_v1)
        self.act_v = nn.Mish()
        self.v_gpool = KataValueHeadGPool()
        self.v_fc1 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act_v2 = nn.Mish()
        self.v_fc2 = nn.Linear(c_v2, 1, bias=True)

    def initialize(self) -> None:
        """RepVGG-style fixscaleonenorm init + set all norm scales. Call once
        before training from scratch (mirrors KataGoNet.initialize)."""
        with torch.no_grad():
            init_weights(self.stem.weight, self.activation, scale=0.8)
            for i, block in enumerate(self.blocks):
                block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))
            self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))
            init_weights(self.p_conv.weight, self.activation, scale=0.8)
            init_weights(self.p_fc.weight, "identity", scale=0.3)
            init_weights(self.v_conv.weight, self.activation, scale=1.0)
            init_weights(self.v_fc1.weight, self.activation, scale=1.0)
            init_weights(self.v_fc1.bias, self.activation, scale=0.2, fan_tensor=self.v_fc1.weight)
            init_weights(self.v_fc2.weight, "identity", scale=1.0)
            init_weights(self.v_fc2.bias, "identity", scale=0.2, fan_tensor=self.v_fc2.weight)

    def set_norm_scales(self) -> None:
        """Re-set the (non-state_dict) fixscale norm scales after load_state_dict."""
        for i, block in enumerate(self.blocks):
            block.normactconvp.norm.set_scale(1.0 / math.sqrt(i + 1.0))
            block.normactconvq.norm.set_scale(1.0 / math.sqrt(block.internal_length + 1.0))
            for j, inner in enumerate(block.blockstack):
                inner.normactconv1.norm.set_scale(1.0 / math.sqrt(j + 1.0))
                inner.normactconv2.norm.set_scale(None)
        self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 4x4 board is always full → all-ones mask (dtype/device follow x so AMP
        # / half inference stay consistent). mask_sum_hw is the constant 16.
        mask = x.new_ones((x.shape[0], 1, 4, 4))
        mask_sum_hw = x.new_full((x.shape[0], 1, 1, 1), 16.0)

        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h, mask, mask_sum_hw)
        h = self.act_trunkfinal(self.norm_trunkfinal(h, mask))

        p = self.act_p(self.p_bias(self.p_conv(h), mask))
        p = self.p_fc(p.flatten(1))                          # (B, 4) logits

        v = self.act_v(self.v_bias(self.v_conv(h), mask))
        v = self.v_gpool(v, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        v = self.act_v2(self.v_fc1(v))
        v = F.softplus(self.v_fc2(v)).squeeze(-1)            # (B,) scaled units
        return p, v


def build_full_net(cfg: Config) -> Net2048NBT:
    return Net2048NBT(cfg)
