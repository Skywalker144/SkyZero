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
