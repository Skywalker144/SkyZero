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
        # 1×1 conv => 不走 RepVGG (在 NormActConv 内部 use_repvgg_init=ks>1)
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
        # ⚠️ 只返残差, 调用方负责 out + block(out, mask)
        out = self.normactconvp(x, mask, mask_sum_hw)
        for block in self.blockstack:
            out = out + block(out, mask, mask_sum_hw)        # 内层加法
        out = self.normactconvq(out, mask, mask_sum_hw)
        return out


# ============================================================
# Policy Head (v15, no pass — Gomoku 不需要)
# ============================================================

class PolicyHead(nn.Module):
    """v15 PolicyHead 6 输出 (main/aux/soft/soft_aux/opt/opp), 删 pass 路径.

    KataGo 原版: forward 输出 cat([spatial logits, pass logit]) → (B, 6, H*W+1).
    SkyZero 改: 只输出 (B, 6, H*W) — Gomoku 无 pass 着法.
    """

    def __init__(self, c_in: int, c_p1: int, c_g1: int,
                 activation: str = "mish", version: int = 15) -> None:
        super().__init__()
        self.activation = activation
        self.version = version
        # v15 → 6 outputs (我们固定 v15)
        if version <= 11:
            self.num_policy_outputs = 4
        elif version <= 15:
            self.num_policy_outputs = 6
        else:
            self.num_policy_outputs = 8

        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = BiasMask(c_g1)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = BiasMask(c_p1)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.conv2p = nn.Conv2d(c_p1, self.num_policy_outputs, kernel_size=1, bias=False)

    def initialize(self) -> None:
        # KataGo master 2416-2432
        p_scale, g_scale, scale_output = 0.8, 0.6, 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        outp = self.conv1p(x)
        outg = self.conv1g(x)
        outg = self.biasg(outg, mask)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = outp + outg
        outp = self.bias2(outp, mask)
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        outp = outp - (1.0 - mask) * 5000.0   # mask 外 cell push 到极小
        return outp.view(outp.shape[0], outp.shape[1], -1)


# ============================================================
# Value Head (Gomoku-only outputs)
# ============================================================

class ValueHead(nn.Module):
    """简化的 ValueHead — 删去围棋特化的 score-belief / scoring / seki.

    输出 6 元 tuple:
        wdl:                 (B, 3)         W/L/draw logits
        td_value:            (B, 9)         3 horizons × 3 wdl (long/mid/short × W/L/draw)
        shortterm_error:     (B, 1)         pretanh, 短 horizon Q 预测误差幅度
        variance_time:       (B, 1)         残留 KataGo 输出 (Gomoku 可选, 占位)
        ownership_pretanh:   (B, 1, H, W)   终局每格占有 ±1
        futurepos_pretanh:   (B, 2, H, W)   +N / +2N 步占据
    """

    def __init__(self, c_in: int, c_v1: int, c_v2: int, activation: str = "mish",
                 pos_len: int = 15) -> None:
        super().__init__()
        self.activation = activation
        self.pos_len = pos_len

        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.bias1 = BiasMask(c_v1)
        self.act1 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataValueHeadGPool()

        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()

        # WDL 输出 (KataGo: linear_valuehead 输出 3, 我们保留)
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        # td_value 多 horizon: 3 horizons × 3 wdl = 9 输出
        self.linear_miscvaluehead = nn.Linear(c_v2, 9, bias=True)
        # shortterm error + variance_time (合并 2 维)
        self.linear_moremiscvaluehead = nn.Linear(c_v2, 2, bias=True)

        # 空间 head
        self.conv_ownership = nn.Conv2d(c_v1, 1, kernel_size=1, bias=False)
        self.conv_futurepos = nn.Conv2d(c_in, 2, kernel_size=1, bias=False)

    def initialize(self) -> None:
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale,
                     fan_tensor=self.linear2.weight)

        for lin in (self.linear_valuehead, self.linear_miscvaluehead, self.linear_moremiscvaluehead):
            init_weights(lin.weight, "identity", scale=1.0)
            init_weights(lin.bias, "identity", scale=bias_scale, fan_tensor=lin.weight)

        aux_scale = 0.2
        for c in (self.conv_ownership, self.conv_futurepos):
            init_weights(c.weight, "identity", scale=aux_scale)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor, torch.Tensor]:
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask)
        outv1 = self.act1(outv1)

        outpooled = self.gpool(outv1, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outv2 = self.act2(self.linear2(outpooled))

        wdl = self.linear_valuehead(outv2)                          # (B, 3)
        td_value = self.linear_miscvaluehead(outv2)                 # (B, 9) = 3 horizons × 3
        more_misc = self.linear_moremiscvaluehead(outv2)            # (B, 2)
        st_error = more_misc[:, 0:1]
        var_time = more_misc[:, 1:2]

        ownership = self.conv_ownership(outv1) * mask               # (B, 1, H, W)
        futurepos = self.conv_futurepos(x) * mask                   # (B, 2, H, W)

        return wdl, td_value, st_error, var_time, ownership, futurepos


# ============================================================
# 完整模型
# ============================================================

class KataGoNet(nn.Module):
    def __init__(
        self,
        num_blocks: int = 8,
        c_main: int = 96,
        c_mid: int = 48,
        c_gpool: int = 16,
        internal_length: int = 2,
        num_in_channels: int = 4,
        num_global_features: int = 12,
        activation: str = "mish",
        version: int = 15,
        has_intermediate_head: bool = True,
        intermediate_head_blocks: int = 5,
        c_p1: int = 24,
        c_g1: int = 24,
        c_v1: int = 24,
        c_v2: int = 32,
        pos_len: int = 15,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.version = version
        self.num_blocks = num_blocks
        self.pos_len = pos_len
        self.has_intermediate_head = has_intermediate_head
        self.intermediate_head_blocks = intermediate_head_blocks
        self.num_in_channels = num_in_channels
        self.num_global_features = num_global_features

        # 输入投影
        self.conv_spatial = nn.Conv2d(num_in_channels, c_main, kernel_size=3,
                                      padding=1, bias=False)
        self.linear_global = nn.Linear(num_global_features, c_main, bias=False)

        # 主干 — gpool 仅在 i % 3 == 2 的块上
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = c_gpool if (i % 3 == 2) else None
            self.blocks.append(NestedBottleneckResBlock(
                internal_length=internal_length, c_main=c_main, c_mid=c_mid,
                c_gpool=use_gpool, activation=activation,
            ))

        self.norm_trunkfinal = BiasMask(c_main)
        self.act_trunkfinal = nn.Mish() if activation == "mish" else nn.ReLU()

        if has_intermediate_head:
            self.norm_intermediate_trunkfinal = LastBatchNorm(c_main)
            self.act_intermediate_trunkfinal = nn.Mish() if activation == "mish" else nn.ReLU()
            self.intermediate_policy_head = PolicyHead(c_main, c_p1, c_g1,
                                                       activation=activation, version=version)
            self.intermediate_value_head = ValueHead(c_main, c_v1, c_v2,
                                                     activation=activation, pos_len=pos_len)

        self.policy_head = PolicyHead(c_main, c_p1, c_g1,
                                      activation=activation, version=version)
        self.value_head = ValueHead(c_main, c_v1, c_v2,
                                    activation=activation, pos_len=pos_len)

    def initialize(self) -> None:
        """KataGo RepVGG-style fixscaleonenorm 初始化 + 设置所有 NormMask.scale.

        从零训前调用一次.
        """
        with torch.no_grad():
            init_weights(self.conv_spatial.weight, self.activation, scale=0.8)
            init_weights(self.linear_global.weight, self.activation, scale=0.6)

            for i, block in enumerate(self.blocks):
                block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))

            self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

            self.policy_head.initialize()
            self.value_head.initialize()
            if self.has_intermediate_head:
                self.intermediate_policy_head.initialize()
                self.intermediate_value_head.initialize()

    def set_norm_scales(self) -> None:
        """加载预训练 ckpt 时调用: 仅设置 scale, 不重置权重.

        scale 不在 state_dict (plain attribute), load_state_dict 后必须手动调.
        """
        for i, block in enumerate(self.blocks):
            block.normactconvp.norm.set_scale(1.0 / math.sqrt(i + 1.0))
            block.normactconvq.norm.set_scale(1.0 / math.sqrt(block.internal_length + 1.0))
            for j, inner in enumerate(block.blockstack):
                inner.normactconv1.norm.set_scale(1.0 / math.sqrt(j + 1.0))
                inner.normactconv2.norm.set_scale(None)
        self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

    def forward(self, input_spatial: torch.Tensor,
                input_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        # SkyZero 固定 15×15: mask 在网络内部硬编码为 ones, mask_sum_hw=225
        B = input_spatial.shape[0]
        mask = torch.ones(B, 1, self.pos_len, self.pos_len,
                          dtype=input_spatial.dtype, device=input_spatial.device)
        mask_sum_hw = torch.full((B, 1, 1, 1), float(self.pos_len * self.pos_len),
                                 dtype=input_spatial.dtype, device=input_spatial.device)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_spatial + x_global

        iout_policy: Optional[torch.Tensor] = None
        iout_value = None

        if self.has_intermediate_head:
            for block in self.blocks[: self.intermediate_head_blocks]:
                out = out + block(out, mask, mask_sum_hw)

            iout = self.norm_intermediate_trunkfinal(out, mask)
            iout = self.act_intermediate_trunkfinal(iout)
            iout_policy = self.intermediate_policy_head(iout, mask, mask_sum_hw)
            iout_value = self.intermediate_value_head(iout, mask, mask_sum_hw)

            for block in self.blocks[self.intermediate_head_blocks:]:
                out = out + block(out, mask, mask_sum_hw)
        else:
            for block in self.blocks:
                out = out + block(out, mask, mask_sum_hw)

        out = self.norm_trunkfinal(out, mask)
        out = self.act_trunkfinal(out)

        out_policy = self.policy_head(out, mask, mask_sum_hw)
        out_value = self.value_head(out, mask, mask_sum_hw)

        return {
            "policy": out_policy,
            "value": out_value,
            "intermediate_policy": iout_policy,
            "intermediate_value": iout_value,
        }


# ============================================================
# 工厂函数 — 与 spec §3.1 表对齐
# ============================================================

def build_b8c96(activation: str = "mish") -> KataGoNet:
    """初版测试规模: 8 块 × 96 trunk × 48 mid × 16 gpool, ~1.5-2M 参数."""
    return KataGoNet(
        num_blocks=8, c_main=96, c_mid=48, c_gpool=16,
        internal_length=2,
        num_in_channels=4, num_global_features=12,
        activation=activation, version=15,
        has_intermediate_head=True, intermediate_head_blocks=5,
        c_p1=24, c_g1=24, c_v1=24, c_v2=32,
        pos_len=15,
    )


def build_b12c128(activation: str = "mish") -> KataGoNet:
    """生产规模: 12 块 × 128 trunk × 64 mid × 16 gpool, ~4-5M 参数."""
    return KataGoNet(
        num_blocks=12, c_main=128, c_mid=64, c_gpool=16,
        internal_length=2,
        num_in_channels=4, num_global_features=12,
        activation=activation, version=15,
        has_intermediate_head=True, intermediate_head_blocks=8,
        c_p1=32, c_g1=32, c_v1=32, c_v2=48,
        pos_len=15,
    )
