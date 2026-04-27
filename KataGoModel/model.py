"""
KataGo b28c512nbt / b40c768nbt 模型结构 (PyTorch)

对齐 lightvector/KataGo master `python/katago/train/model_pytorch.py`。
配置: norm_kind=fixscaleonenorm, trunk_normless=True, bnorm_use_gamma=True,
       gamma_weight_decay_center_1=True, use_repvgg_init=True, activation=mish, version=15.

要点:
- gamma 张量保存的是「实际 scale 减 1 后的 delta」, 前向必须用 (gamma+1)。
- 每个 NormMask 还有一个不属于 state_dict 的 `scale` 系数, 由 initialize() 设置;
  ckpt 训练时是带着这个 scale 跑的, 推理时必须把它乘回去。
- NestedBottleneckResBlock.forward 只返回残差; 由调用方决定是否 `out + block(out)`。
- LastBatchNorm 训练时按 mask-aware mini-batch stats 工作并更新 running 统计量。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RepVGG init helpers (master line 78-109)
# ============================================================

def compute_gain(activation: str) -> float:
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


_TRUNC_CORRECTION = 0.87962566103423978  # std correction for trunc_normal a=-2,b=2


def init_weights(tensor: torch.Tensor, activation: str, scale: float,
                 fan_tensor: Optional[torch.Tensor] = None) -> None:
    """KataGo 的 truncated-normal 初始化, std = scale * gain / sqrt(fan_in)。

    fan_tensor 用于 bias 共享主 weight 的 fan_in (master 调 init_weights 给 bias 时这样用)。
    scale 极小 (例如残差最后一层的 0.0) 时直接置 0。
    """
    gain = compute_gain(activation)
    src = fan_tensor if fan_tensor is not None else tensor
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(src)
    target_std = scale * gain / math.sqrt(fan_in)
    std = target_std / _TRUNC_CORRECTION
    with torch.no_grad():
        if std < 1e-10:
            tensor.fill_(0.0)
        else:
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


# ============================================================
# 归一化层
# ============================================================

class FixscaleNorm(nn.Module):
    """fixscaleonenorm 下的非-last 归一化层。

    前向: out = (x * (gamma + 1) * scale + beta) * mask
    其中 scale 由 set_scale() 设置 (None 时省略)。 gamma 初始为 0
    (因为 gamma_weight_decay_center_1=True, 训练以 0 为 weight-decay 中心)。
    """

    def __init__(self, num_channels: int, use_gamma: bool = True):
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
            out = x * (g * self.scale) + self.beta if self.scale is not None else x * g + self.beta
        else:
            out = x * self.scale + self.beta if self.scale is not None else x + self.beta
        if mask is not None:
            out = out * mask
        return out


class BiasMask(nn.Module):
    """trunk_normless=True 下 trunk-final 的 normless-bias 层。"""

    def __init__(self, num_channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x * self.scale + self.beta if self.scale is not None else x + self.beta
        if mask is not None:
            out = out * mask
        return out


class LastBatchNorm(nn.Module):
    """fixscaleonenorm 全网唯一的 batchnorm: 用在 intermediate head 入口。

    训练: mask-aware mini-batch mean/std + running 统计量 EMA 更新 (momentum=0.001)。
    推理: running stats。
    """

    def __init__(self, num_channels: int,
                 eps: float = 1e-4, momentum: float = 0.001):
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
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        layer_max = (x + (mask - 1.0)).to(torch.float32).amax(dim=(2, 3), keepdim=True)
        return torch.cat((layer_mean,
                          layer_mean * (sqrt_off / 10.0),
                          layer_max), dim=1)


class KataValueHeadGPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        return torch.cat((layer_mean,
                          layer_mean * (sqrt_off / 10.0),
                          layer_mean * ((sqrt_off * sqrt_off) / 100.0 - 0.1)), dim=1)


# ============================================================
# 卷积 + 全局池化模块
# ============================================================

class ConvAndGPool(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_gpool: int, activation: str = "mish"):
        super().__init__()
        self.activation = activation
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding="same", bias=False)
        self.normg = FixscaleNorm(c_gpool, use_gamma=True)
        self.actg = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale: float) -> None:
        # master 538-549 fixscaleonenorm path
        r_scale, g_scale = 0.8, 0.6
        init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
        init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
        init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))

    def forward(self, x, mask, mask_sum_hw=None):
        out_r = self.conv1r(x)
        out_g = self.conv1g(x)
        out_g = self.normg(out_g, mask)
        out_g = self.actg(out_g)
        out_g = self.gpool(out_g, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)
        return out_r + out_g


# ============================================================
# Norm-Act-Conv
# ============================================================

class NormActConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, activation: str = "mish",
                 kernel_size: int = 3, c_gpool: Optional[int] = None,
                 fixup_use_gamma: bool = True,
                 use_repvgg_init: bool = True):
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.norm = FixscaleNorm(c_in, use_gamma=fixup_use_gamma)
        self.act = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
        self.use_repvgg_init = use_repvgg_init and kernel_size > 1  # master 611
        if c_gpool is not None:
            self.convpool = ConvAndGPool(c_in, c_out, c_gpool, activation)
            self.conv = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                                  padding="same" if kernel_size > 1 else 0, bias=False)
            self.convpool = None

    def initialize(self, scale: float, norm_scale: Optional[float] = None) -> None:
        # master 624-639
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

    def forward(self, x, mask=None, mask_sum_hw=None):
        out = self.norm(x, mask)
        out = self.act(out)
        if self.convpool is not None:
            return self.convpool(out, mask, mask_sum_hw)
        return self.conv(out)


# ============================================================
# 残差块
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, c_mid: int, c_gpool: Optional[int] = None,
                 activation: str = "mish"):
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

    def forward(self, x, mask=None, mask_sum_hw=None):
        out = self.normactconv1(x, mask, mask_sum_hw)
        out = self.normactconv2(out, mask, mask_sum_hw)
        return out


class NestedBottleneckResBlock(nn.Module):
    def __init__(self, internal_length: int, c_main: int, c_mid: int,
                 c_gpool: Optional[int] = None, activation: str = "mish"):
        super().__init__()
        self.internal_length = internal_length
        # 1×1 conv => 不走 RepVGG
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

    def forward(self, x, mask=None, mask_sum_hw=None):
        # 只返回残差; 调用方 `out + block(out)` 加回主干
        out = self.normactconvp(x, mask, mask_sum_hw)
        for block in self.blockstack:
            out = out + block(out, mask, mask_sum_hw)
        out = self.normactconvq(out, mask, mask_sum_hw)
        return out


# ============================================================
# Policy Head
# ============================================================

class PolicyHead(nn.Module):
    def __init__(self, c_in: int, c_p1: int, c_g1: int,
                 activation: str = "mish", version: int = 15):
        super().__init__()
        self.activation = activation
        self.version = version
        self.num_policy_outputs = 4 if version <= 11 else 6 if version <= 15 else 8

        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, padding="same", bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, padding="same", bias=False)
        self.biasg = BiasMask(c_g1)
        self.actg = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)

        if version <= 14:
            self.linear_pass = nn.Linear(3 * c_g1, self.num_policy_outputs, bias=False)
        else:
            self.linear_pass = nn.Linear(3 * c_g1, c_p1, bias=True)
            self.act_pass = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
            self.linear_pass2 = nn.Linear(c_p1, self.num_policy_outputs, bias=False)

        self.bias2 = BiasMask(c_p1)
        self.act2 = nn.Mish(inplace=False) if activation == "mish" else nn.ReLU(inplace=False)
        self.conv2p = nn.Conv2d(c_p1, self.num_policy_outputs, kernel_size=1, padding="same", bias=False)

    def initialize(self) -> None:
        # master 2416-2432
        p_scale, g_scale, bias_scale, scale_output = 0.8, 0.6, 0.2, 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        if self.version <= 14:
            init_weights(self.linear_pass.weight, "identity", scale=scale_output)
        else:
            init_weights(self.linear_pass.weight, self.activation, scale=1.0)
            init_weights(self.linear_pass.bias, self.activation, scale=bias_scale,
                         fan_tensor=self.linear_pass.weight)
            init_weights(self.linear_pass2.weight, "identity", scale=scale_output)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def forward(self, x, mask, mask_sum_hw=None):
        outp = self.conv1p(x)
        outg = self.conv1g(x)
        outg = self.biasg(outg, mask)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask, mask_sum_hw).squeeze(-1).squeeze(-1)

        if self.version <= 14:
            outpass = self.linear_pass(outg)
        else:
            outpass = self.linear_pass(outg)
            outpass = self.act_pass(outpass)
            outpass = self.linear_pass2(outpass)

        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = outp + outg
        outp = self.bias2(outp, mask)
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        outp = outp - (1.0 - mask) * 5000.0
        return torch.cat((outp.view(outp.shape[0], outp.shape[1], -1),
                          outpass.unsqueeze(-1)), dim=2)


# ============================================================
# Value Head
# ============================================================

EXTRA_SCORE_DISTR_RADIUS = 60


class ValueHead(nn.Module):
    def __init__(self, c_in: int, c_v1: int, c_v2: int, c_sv2: int,
                 num_scorebeliefs: int, activation: str = "mish", pos_len: int = 19):
        super().__init__()
        self.activation = activation
        self.c_sv2 = c_sv2
        self.num_scorebeliefs = num_scorebeliefs
        self.pos_len = pos_len
        self.scorebelief_mid = pos_len * pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2

        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, padding="same", bias=False)
        self.bias1 = BiasMask(c_v1)
        self.act1 = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
        self.gpool = KataValueHeadGPool()

        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        # act2 在 outv2 / outsv2 两条不同张量上各调用一次 -> 必须非 inplace
        self.act2 = nn.Mish(inplace=False) if activation == "mish" else nn.ReLU(inplace=False)

        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = nn.Linear(c_v2, 10, bias=True)
        self.linear_moremiscvaluehead = nn.Linear(c_v2, 8, bias=True)

        self.conv_ownership = nn.Conv2d(c_v1, 1, kernel_size=1, padding="same", bias=False)
        self.conv_scoring = nn.Conv2d(c_v1, 1, kernel_size=1, padding="same", bias=False)
        self.conv_futurepos = nn.Conv2d(c_in, 2, kernel_size=1, padding="same", bias=False)
        self.conv_seki = nn.Conv2d(c_in, 4, kernel_size=1, padding="same", bias=False)

        self.linear_s2 = nn.Linear(3 * c_v1, c_sv2, bias=True)
        self.linear_s2off = nn.Linear(1, c_sv2, bias=False)
        self.linear_s2par = nn.Linear(1, c_sv2, bias=False)
        self.linear_s3 = nn.Linear(c_sv2, num_scorebeliefs, bias=True)
        self.linear_smix = nn.Linear(3 * c_v1, num_scorebeliefs, bias=True)

        self.register_buffer("score_belief_offset_vector", torch.tensor(
            [(float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
            [0.05 * (float(i - self.scorebelief_mid) + 0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer("score_belief_parity_vector", torch.tensor(
            [0.5 - float((i - self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
            dtype=torch.float32, requires_grad=False), persistent=False)

    def initialize(self) -> None:
        # master 2537-2567
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale,
                     fan_tensor=self.linear2.weight)

        for lin in (self.linear_valuehead, self.linear_miscvaluehead, self.linear_moremiscvaluehead):
            init_weights(lin.weight, "identity", scale=1.0)
            init_weights(lin.bias, "identity", scale=bias_scale, fan_tensor=lin.weight)

        aux_scale = 0.2
        for c in (self.conv_ownership, self.conv_scoring, self.conv_futurepos, self.conv_seki):
            init_weights(c.weight, "identity", scale=aux_scale)

        init_weights(self.linear_s2.weight, self.activation, scale=1.0)
        init_weights(self.linear_s2.bias, self.activation, scale=1.0,
                     fan_tensor=self.linear_s2.weight)
        init_weights(self.linear_s2off.weight, self.activation, scale=1.0,
                     fan_tensor=self.linear_s2.weight)
        init_weights(self.linear_s2par.weight, self.activation, scale=1.0,
                     fan_tensor=self.linear_s2.weight)

        sb_out = 0.5
        init_weights(self.linear_s3.weight, "identity", scale=sb_out)
        init_weights(self.linear_s3.bias, "identity", scale=sb_out * bias_scale,
                     fan_tensor=self.linear_s3.weight)
        init_weights(self.linear_smix.weight, "identity", scale=1.0)
        init_weights(self.linear_smix.bias, "identity", scale=bias_scale,
                     fan_tensor=self.linear_smix.weight)

    def forward(self, x, mask, mask_sum_hw=None, input_global=None):
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask)
        outv1 = self.act1(outv1)

        outpooled = self.gpool(outv1, mask, mask_sum_hw).squeeze(-1).squeeze(-1)

        outv2 = self.linear2(outpooled)
        outv2 = self.act2(outv2)

        out_value = self.linear_valuehead(outv2)
        out_miscvalue = self.linear_miscvaluehead(outv2)
        out_moremiscvalue = self.linear_moremiscvaluehead(outv2)
        out_ownership = self.conv_ownership(outv1) * mask
        out_scoring = self.conv_scoring(outv1) * mask
        out_futurepos = self.conv_futurepos(x) * mask
        out_seki = self.conv_seki(x) * mask

        N = x.shape[0]
        outsv2 = (
            self.linear_s2(outpooled).view(N, 1, self.c_sv2)
            + self.linear_s2off(self.score_belief_offset_bias_vector.view(1, self.scorebelief_len, 1))
            + self.linear_s2par(
                (self.score_belief_parity_vector.view(1, self.scorebelief_len) * input_global[:, -1:])
                .view(N, self.scorebelief_len, 1)
            )
        )
        outsv2 = self.act2(outsv2)
        outsv3 = self.linear_s3(outsv2)

        outsmix = self.linear_smix(outpooled)
        outsmix_logweights = F.log_softmax(outsmix, dim=1)
        out_scorebelief_logprobs = F.log_softmax(outsv3, dim=1)
        out_scorebelief_logprobs = torch.logsumexp(
            out_scorebelief_logprobs + outsmix_logweights.view(-1, 1, self.num_scorebeliefs),
            dim=2,
        )

        return (
            out_value, out_miscvalue, out_moremiscvalue,
            out_ownership, out_scoring, out_futurepos, out_seki,
            out_scorebelief_logprobs,
        )


# ============================================================
# 完整模型
# ============================================================

class KataGoModel(nn.Module):
    def __init__(
        self,
        num_blocks: int = 28,
        c_main: int = 512,
        c_mid: int = 256,
        c_gpool: int = 64,
        internal_length: int = 2,
        num_in_channels: int = 22,
        num_global_features: int = 19,
        activation: str = "mish",
        version: int = 15,
        has_intermediate_head: bool = True,
        intermediate_head_blocks: int = 28,
        c_p1: int = 64,
        c_g1: int = 64,
        c_v1: int = 128,
        c_v2: int = 144,
        c_sv2: int = 128,
        num_scorebeliefs: int = 8,
        pos_len: int = 19,
    ):
        super().__init__()
        self.activation = activation
        self.version = version
        self.num_blocks = num_blocks
        self.has_intermediate_head = has_intermediate_head
        self.intermediate_head_blocks = intermediate_head_blocks

        # 输入投影
        self.conv_spatial = nn.Conv2d(num_in_channels, c_main, kernel_size=3,
                                      padding="same", bias=False)
        self.linear_global = nn.Linear(num_global_features, c_main, bias=False)

        # 主干
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = c_gpool if (i % 3 == 2) else None
            self.blocks.append(NestedBottleneckResBlock(
                internal_length=internal_length, c_main=c_main, c_mid=c_mid,
                c_gpool=use_gpool, activation=activation,
            ))

        self.norm_trunkfinal = BiasMask(c_main)
        self.act_trunkfinal = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)

        if has_intermediate_head:
            self.norm_intermediate_trunkfinal = LastBatchNorm(c_main)
            self.act_intermediate_trunkfinal = nn.Mish(inplace=True) if activation == "mish" else nn.ReLU(inplace=True)
            self.intermediate_policy_head = PolicyHead(c_main, c_p1, c_g1,
                                                       activation=activation, version=version)
            self.intermediate_value_head = ValueHead(c_main, c_v1, c_v2, c_sv2,
                                                    num_scorebeliefs, activation=activation,
                                                    pos_len=pos_len)

        self.policy_head = PolicyHead(c_main, c_p1, c_g1,
                                      activation=activation, version=version)
        self.value_head = ValueHead(c_main, c_v1, c_v2, c_sv2,
                                    num_scorebeliefs, activation=activation,
                                    pos_len=pos_len)

    def initialize(self) -> None:
        """KataGo RepVGG-style fixscaleonenorm 初始化 + 设置所有 NormMask.scale。
        从零训前调用一次。
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
        """加载预训练 ckpt 时调用: 仅设置 scale, 不重置权重。
        scale 不在 state_dict 里 (plain attribute), 所以 load_state_dict 后必须手动调用。
        """
        for i, block in enumerate(self.blocks):
            block.normactconvp.norm.set_scale(1.0 / math.sqrt(i + 1.0))
            block.normactconvq.norm.set_scale(1.0 / math.sqrt(block.internal_length + 1.0))
            for j, inner in enumerate(block.blockstack):
                inner.normactconv1.norm.set_scale(1.0 / math.sqrt(j + 1.0))
                inner.normactconv2.norm.set_scale(None)
        self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor) -> dict:
        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_spatial + x_global

        if self.has_intermediate_head:
            for block in self.blocks[: self.intermediate_head_blocks]:
                out = out + block(out, mask, mask_sum_hw)

            iout = self.norm_intermediate_trunkfinal(out, mask)
            iout = self.act_intermediate_trunkfinal(iout)
            iout_policy = self.intermediate_policy_head(iout, mask, mask_sum_hw)
            iout_value = self.intermediate_value_head(iout, mask, mask_sum_hw, input_global)

            for block in self.blocks[self.intermediate_head_blocks:]:
                out = out + block(out, mask, mask_sum_hw)
        else:
            for block in self.blocks:
                out = out + block(out, mask, mask_sum_hw)
            iout_policy = None
            iout_value = None

        out = self.norm_trunkfinal(out, mask)
        out = self.act_trunkfinal(out)

        out_policy = self.policy_head(out, mask, mask_sum_hw)
        out_value = self.value_head(out, mask, mask_sum_hw, input_global)

        return {
            "policy": out_policy,
            "value": out_value,
            "intermediate_policy": iout_policy,
            "intermediate_value": iout_value,
            "trunk": out,
            "mask": mask,
        }


# ============================================================
# 工厂函数
# ============================================================

def b28c512nbt(activation: str = "mish") -> KataGoModel:
    return KataGoModel(
        num_blocks=28, c_main=512, c_mid=256, c_gpool=64,
        internal_length=2, activation=activation,
        has_intermediate_head=True, intermediate_head_blocks=28,
    )


def b40c768nbt(activation: str = "mish") -> KataGoModel:
    return KataGoModel(
        num_blocks=40, c_main=768, c_mid=384, c_gpool=128,
        internal_length=2, activation=activation,
        has_intermediate_head=True, intermediate_head_blocks=40,
        c_p1=128, c_g1=128, c_v1=256, c_v2=256, c_sv2=256,
    )


# ============================================================
# 自检
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os
    print("=" * 70)
    print("b28c512nbt 自检 (from-scratch initialize)")
    print("=" * 70)
    model = b28c512nbt()
    model.initialize()
    print(f"Params: {count_parameters(model):,}")

    model.eval()
    x_spatial = torch.zeros(1, 22, 19, 19)
    x_spatial[:, 0] = 1.0
    x_global = torch.zeros(1, 19)
    with torch.no_grad():
        out = model(x_spatial, x_global)
    print(f"trunk shape: {list(out['trunk'].shape)}, |x| mean = {out['trunk'].abs().mean():.4f}")
    print(f"value W/L/draw logits = {out['value'][0]}")
    print(f"value W/L/draw probs  = {torch.softmax(out['value'][0], dim=1)}")

    ckpt_path = os.path.join(os.path.dirname(__file__), "kata1-zhizi-b28c512nbt-muonfd2.ckpt")
    if os.path.exists(ckpt_path):
        print()
        print("=" * 70)
        print("加载预训练 ckpt + set_norm_scales()")
        print("=" * 70)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        model.set_norm_scales()  # 必须!
        model.eval()
        with torch.no_grad():
            out = model(x_spatial, x_global)
        print(f"trunk |x| mean = {out['trunk'].abs().mean():.4f}")
        print(f"value W/L/draw logits = {out['value'][0]}")
        print(f"value W/L/draw probs  = {torch.softmax(out['value'][0], dim=1)}")
