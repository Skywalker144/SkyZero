"""KataGo b28c512nbt v15 网络模块 — SkyZero_V4 适配版.

参考: KataGoModel/model.py (lightvector/KataGo master 对齐)
基础设计: norm_kind=fixscaleonenorm, trunk_normless=True, bnorm_use_gamma=True,
         gamma_weight_decay_center_1=True, use_repvgg_init=True, version=15.

陷阱（来自 NOTES.md §3）:
1. NestedBottleneckResBlock.forward 只返残差; 调用方负责 out + block(out) 加回主干
2. gamma 张量是 delta, forward 必须用 (gamma + 1)
3. NormMask.scale 是 plain Python float, 不在 state_dict
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 归一化层
# ============================================================

class FixscaleNorm(nn.Module):
    """fixscaleonenorm 下非-last 归一化层.

    forward: out = (x * (gamma + 1) * scale + beta) [* mask]
    其中 scale 由 set_scale() 设置 (None 时省略).
    gamma 初始化为 0 (gamma_weight_decay_center_1=True, weight decay 推向 0).
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
    """trunk_normless=True 下 trunk-final 的 normless-bias 层.

    forward: out = (x * scale + beta) [* mask]
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


class LastBatchNorm(nn.Module):
    """fixscaleonenorm 全网唯一的 batchnorm: 用在 intermediate head 入口.

    训练: mask-aware mini-batch mean/std + running 统计量 EMA 更新.
    推理: running stats.
    """

    def __init__(self, num_channels: int, eps: float = 1e-4, momentum: float = 0.001) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_std", torch.ones(num_channels))
        self.eps = eps
        self.momentum = momentum
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if self.training:
            mask_sum = mask.sum()
            mean = (x * mask).sum(dim=(0, 2, 3), keepdim=True) / mask_sum
            zeromean_x = x - mean
            var = ((zeromean_x * mask) ** 2).sum(dim=(0, 2, 3), keepdim=True) / mask_sum
            std = (var + self.eps).sqrt()
            with torch.no_grad():
                self.running_mean.add_(self.momentum * (mean.view(c).detach() - self.running_mean))
                self.running_std.add_(self.momentum * (std.view(c).detach() - self.running_std))
            normed = zeromean_x / std
        else:
            normed = (x - self.running_mean.view(1, c, 1, 1)) / self.running_std.view(1, c, 1, 1)
        g = self.gamma + 1.0
        if self.scale is not None:
            out = normed * (g * self.scale) + self.beta
        else:
            out = normed * g + self.beta
        return out * mask


# ============================================================
# 池化模块
# ============================================================

class KataGPool(nn.Module):
    """3 个统计量: mean, mean * board_factor, max.

    board_factor = (sqrt(mask_sum_hw) - 14) / 10 — 让网络对棋盘大小有显式感知.
    15×15 时退化为 0.1 (常数).
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        layer_max = (x + (mask - 1.0)).to(torch.float32).amax(dim=(2, 3), keepdim=True)
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
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
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
    """KataGo's truncated-normal init: std = scale * gain / sqrt(fan_in).

    For 1D tensors, reshape to (n, 1) to get fan_in=1 via PyTorch's fan calculation.
    """
    gain = compute_gain(activation)
    src = fan_tensor if fan_tensor is not None else tensor

    # Handle 1D tensors (e.g., bias) which don't work with _calculate_fan_in_and_fan_out
    # Reshape to (n, 1) so that _calculate_fan_in_and_fan_out gives fan_in=1
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
