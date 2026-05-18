"""KataGo-style nested-bottleneck network for SkyZero V1.

Trunk aligned with lightvector/KataGo master and SkyZero V5's slim KataGoNet:

  stem (3×3 conv) → N × NestedBottleneckResBlock → BiasMask + Mish → heads

Each block: 1×1 down (c_main → c_mid) → `internal_length` × ResBlock(c_mid)
with KataGPool injected into every 3rd block's first ResBlock → 1×1 up
(c_mid → c_main), wrapped in a single external residual.

  - FixscaleNorm with (gamma+1) semantics, scale = 1/√(block_idx+1)
  - Mish activation, gain(mish) = √2.21
  - RepVGG init: 3×3 conv weight = trunc_normal(scale 0.8) + 1×1 center
    bonus(scale 0.6) added at kernel center
  - KataGPool: cat([mean, mean × board_factor, max]) per channel
  - PolicyHead: spatial 1×1 + gpool branch summed, then 1×1 → 1 logit channel
  - ValueHead: 1×1 → KataValueHeadGPool → MLP → WDL (3 logits)

Differences from V5 (intentional — V1 doesn't need / can't train these yet):
  - No mask (V1 棋盘永远满)
  - No global feature stream (V1 没有 global state — 比如 komi/规则参数)
  - No intermediate head (训练不监督中间层)
  - No aux heads (opp / soft policy / TD value / futurepos — V1 没生成 target)

API contract:
    ResNet(game, num_blocks=N, num_channels=C)
    forward(x: (B, num_planes, H, W)) → (
        policy_logits: (B, 4, H*W),   # main / opp / soft_main / soft_opp
        value_logits:  (B, 3)         # WDL
    )

Callers consuming a single policy distribution (MCTS search) take channel 0:
    policy_logits[:, 0, :].

NormMask trap (KataGoModel NOTES.md §3.3) avoided: scale is a plain Python
float computed at __init__ from block index. It's not in state_dict, but the
model is always re-constructed before load_state_dict, so __init__ re-sets it.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Init helpers (KataGo master python/katago/train/model_pytorch.py:78-109)
# ============================================================

_TRUNC_CORRECTION = 0.87962566103423978  # std correction for trunc_normal a=-2, b=2


def compute_gain(activation: str) -> float:
    if activation in ("relu", "hardswish"):
        return math.sqrt(2.0)
    if activation == "mish":
        return math.sqrt(2.210277)
    if activation == "identity":
        return 1.0
    raise ValueError(f"unknown activation: {activation!r}")


def init_weights(tensor: torch.Tensor, activation: str, scale: float,
                 fan_tensor: Optional[torch.Tensor] = None) -> None:
    """Truncated-normal init: std = scale * gain(activation) / sqrt(fan_in)."""
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
# Normalization layers
# ============================================================

class FixscaleNorm(nn.Module):
    """KataGo fixscaleonenorm. forward: x*(gamma+1)*scale + beta.

    gamma 张量是 delta (gamma_weight_decay_center_1=True 训练目标是把 gamma
    推向 0), 实际缩放因子是 gamma+1. scale 是 1/√(layer_idx+1), 由构造方
    在 __init__ 时传入. scale=None 表示不做缩放.
    """

    def __init__(self, num_channels: int, scale: Optional[float] = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gamma + 1.0
        if self.scale is not None:
            return x * (g * self.scale) + self.beta
        return x * g + self.beta


class BiasMask(nn.Module):
    """trunk_normless=True 下用在 trunk-final 与 head 内的 normless bias.

    forward: x*scale + beta. (scale 在 trunk-final 为 1/√(num_blocks+1).)
    """

    def __init__(self, num_channels: int, scale: Optional[float] = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            return x * self.scale + self.beta
        return x + self.beta


# ============================================================
# Global pooling
# ============================================================

class KataGPool(nn.Module):
    """3 statistics per channel: mean, mean × board_factor, max.

    board_factor = (sqrt(H*W) - 14) / 10 — 让网络对棋盘大小有感知.
    Returns (B, 3*C, 1, 1).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = float(x.shape[2] * x.shape[3])
        sqrt_off = math.sqrt(hw) - 14.0
        layer_mean = x.mean(dim=(2, 3), keepdim=True)
        layer_max = x.amax(dim=(2, 3), keepdim=True)
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_max,
        ), dim=1)


class KataValueHeadGPool(nn.Module):
    """ValueHead 专用 GPool: 3 个 mean 统计量 (无 max), 二阶 board factor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hw = float(x.shape[2] * x.shape[3])
        sqrt_off = math.sqrt(hw) - 14.0
        layer_mean = x.mean(dim=(2, 3), keepdim=True)
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_mean * ((sqrt_off * sqrt_off) / 100.0 - 0.1),
        ), dim=1)


# ============================================================
# Conv-with-gpool branch (merged into the first ResBlock of gpool blocks)
# ============================================================

class ConvAndGPool(nn.Module):
    """out = conv1r(x) + (linear_g · gpool · act · norm · conv1g)(x).

    Used as the first conv inside a ResBlock when c_gpool is not None.
    """

    def __init__(self, c_in: int, c_out: int, c_gpool: int,
                 activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.normg = FixscaleNorm(c_gpool, scale=None)  # gpool branch: no fixup scale
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale: float) -> None:
        r_scale, g_scale = 0.8, 0.6
        init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
        init_weights(self.conv1g.weight, self.activation,
                     scale=math.sqrt(scale) * math.sqrt(g_scale))
        init_weights(self.linear_g.weight, self.activation,
                     scale=math.sqrt(scale) * math.sqrt(g_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_r = self.conv1r(x)
        out_g = self.normg(self.conv1g(x))
        out_g = self.actg(out_g)
        out_g = self.gpool(out_g).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)
        return out_r + out_g


# ============================================================
# NormActConv: pre-act conv (norm → act → conv, optionally with gpool)
# ============================================================

class NormActConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 activation: str = "mish", c_gpool: Optional[int] = None,
                 norm_scale: Optional[float] = 1.0,
                 use_repvgg_init: bool = True) -> None:
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.norm = FixscaleNorm(c_in, scale=norm_scale)
        self.act = nn.Mish() if activation == "mish" else nn.ReLU()
        self.use_repvgg_init = use_repvgg_init and kernel_size > 1
        if c_gpool is not None:
            self.convpool = ConvAndGPool(c_in, c_out, c_gpool, activation)
            self.conv = None
        else:
            self.convpool = None
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)

    def initialize(self, scale: float) -> None:
        if self.convpool is not None:
            self.convpool.initialize(scale=scale)
        elif self.use_repvgg_init:
            # 3×3 main + 1×1 center bonus → equivalent to RepVGG train-time branches.
            init_weights(self.conv.weight, self.activation, scale=scale * 0.8)
            w = self.conv.weight
            center_bonus = w.new_zeros((w.shape[0], w.shape[1]))
            init_weights(center_bonus, self.activation, scale=scale * 0.6)
            with torch.no_grad():
                self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2] += center_bonus
        else:
            init_weights(self.conv.weight, self.activation, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(x))
        if self.convpool is not None:
            return self.convpool(out)
        return self.conv(out)


# ============================================================
# Residual blocks
# ============================================================

class ResBlock(nn.Module):
    """Inner ResBlock: norm-act-conv ×2 with optional gpool in conv1."""

    def __init__(self, c_mid: int, c_gpool: Optional[int] = None,
                 activation: str = "mish",
                 norm_scale1: Optional[float] = 1.0) -> None:
        super().__init__()
        # gpool merges into conv1; reduce c_out by c_gpool channels so total stays c_mid.
        c_out1 = c_mid - (0 if c_gpool is None else c_gpool)
        self.normactconv1 = NormActConv(c_mid, c_out1, kernel_size=3,
                                        activation=activation, c_gpool=c_gpool,
                                        norm_scale=norm_scale1)
        # 2nd conv has no fixup scale (KataGo: norm_scale=None on inner ResBlock 2nd norm).
        self.normactconv2 = NormActConv(c_out1, c_mid, kernel_size=3,
                                        activation=activation, c_gpool=None,
                                        norm_scale=None)

    def initialize(self) -> None:
        self.normactconv1.initialize(scale=1.0)
        self.normactconv2.initialize(scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normactconv1(x)
        out = self.normactconv2(out)
        return out


class NestedBottleneckResBlock(nn.Module):
    """1×1 down (c_main → c_mid) → N × ResBlock(c_mid) → 1×1 up (c_mid → c_main).

    forward returns only the residual; caller does `x = x + block(x)`.
    """

    def __init__(self, internal_length: int, c_main: int, c_mid: int,
                 c_gpool: Optional[int] = None, activation: str = "mish",
                 outer_norm_scale: float = 1.0) -> None:
        super().__init__()
        self.internal_length = internal_length
        # 1×1 conv => skip RepVGG (NormActConv enforces ks>1 anyway).
        self.normactconvp = NormActConv(c_main, c_mid, kernel_size=1,
                                        activation=activation,
                                        norm_scale=outer_norm_scale)
        self.blockstack = nn.ModuleList()
        for j in range(internal_length):
            use_gpool = c_gpool if j == 0 else None
            self.blockstack.append(ResBlock(
                c_mid=c_mid, c_gpool=use_gpool, activation=activation,
                norm_scale1=1.0 / math.sqrt(j + 1.0),
            ))
        self.normactconvq = NormActConv(c_mid, c_main, kernel_size=1,
                                        activation=activation,
                                        norm_scale=1.0 / math.sqrt(internal_length + 1.0))

    def initialize(self) -> None:
        self.normactconvp.initialize(scale=1.0)
        for block in self.blockstack:
            block.initialize()
        self.normactconvq.initialize(scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normactconvp(x)
        for block in self.blockstack:
            out = out + block(out)  # inner add
        out = self.normactconvq(out)
        return out


# ============================================================
# Heads (slim — single policy output, WDL value)
# ============================================================

class PolicyHead(nn.Module):
    """KataGo PolicyHead with 4 output channels (slim, matching V5 nets.py).

    Channel layout (training-time indices, see _train_step head_losses):
        0: main_policy        target = MCTS visits at t
        1: opp_policy         target = opponent's MCTS visits at t+1
        2: soft_main_policy   target = (visits + 1e-7)^0.25 / Z, masked
        3: soft_opp_policy    target = same on opponent visits

    Only channel 0 is consumed by MCTS during search; the rest are auxiliary
    supervision signals at training time.
    """

    NUM_POLICY_OUTPUTS = 4

    def __init__(self, c_in: int, c_p1: int, c_g1: int,
                 activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = BiasMask(c_g1, scale=None)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = BiasMask(c_p1, scale=None)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.conv2p = nn.Conv2d(c_p1, self.NUM_POLICY_OUTPUTS, kernel_size=1, bias=False)

    def initialize(self) -> None:
        p_scale, g_scale, scale_output = 0.8, 0.6, 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outp = self.conv1p(x)
        outg = self.actg(self.biasg(self.conv1g(x)))
        outg = self.gpool(outg).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = self.act2(self.bias2(outp + outg))
        outp = self.conv2p(outp)
        return outp.view(outp.shape[0], outp.shape[1], -1)  # (B, 4, H*W)


class ValueHead(nn.Module):
    """KataGo-style ValueHead — gpool → MLP → WDL logits."""

    def __init__(self, c_in: int, c_v1: int, c_v2: int,
                 activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.bias1 = BiasMask(c_v1, scale=None)
        self.act1 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataValueHeadGPool()
        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)

    def initialize(self) -> None:
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale,
                     fan_tensor=self.linear2.weight)
        init_weights(self.linear_valuehead.weight, "identity", scale=1.0)
        init_weights(self.linear_valuehead.bias, "identity", scale=bias_scale,
                     fan_tensor=self.linear_valuehead.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outv = self.act1(self.bias1(self.conv1(x)))
        outv = self.gpool(outv).squeeze(-1).squeeze(-1)
        outv = self.act2(self.linear2(outv))
        return self.linear_valuehead(outv)  # (B, 3)


# ============================================================
# Top-level model (backward-compatible name & signature)
# ============================================================

class ResNet(nn.Module):
    """KataGo-style nested-bottleneck network, named ResNet for V1 compat.

    Args:
        game:           env exposing .board_size and .num_planes
        num_blocks:     number of NestedBottleneckResBlocks
        num_channels:   trunk width (c_main)
        c_mid:          bottleneck mid-width. Default num_channels // 2.
        c_gpool:        gpool channels in every 3rd block. Default num_channels // 8 (≥4).
        internal_length: inner ResBlocks per nested block. Default 2.
        activation:     'mish' (KataGo default) or 'relu'.
    """

    def __init__(self, game, num_blocks: int = 10, num_channels: int = 96,
                 c_mid: Optional[int] = None,
                 c_gpool: Optional[int] = None,
                 internal_length: int = 2,
                 activation: str = "mish") -> None:
        super().__init__()
        self.board_size = game.board_size
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        c_mid = c_mid if c_mid is not None else num_channels // 2
        c_gpool = c_gpool if c_gpool is not None else max(4, num_channels // 8)
        # Slim head sizes — V5 picks c_p1 ≈ c_g1 ≈ c_v1 ≈ num_channels // 4.
        c_p1 = max(8, num_channels // 4)
        c_g1 = max(8, num_channels // 4)
        c_v1 = max(8, num_channels // 4)
        c_v2 = max(16, num_channels // 3)

        # Stem: 3×3 conv, no global stream (V1 没有 global state).
        self.conv_spatial = nn.Conv2d(game.num_planes, num_channels,
                                      kernel_size=3, padding=1, bias=False)

        # Trunk: N nested-bottleneck blocks with gpool on every 3rd (i % 3 == 2).
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = c_gpool if (i % 3 == 2) else None
            self.blocks.append(NestedBottleneckResBlock(
                internal_length=internal_length, c_main=num_channels, c_mid=c_mid,
                c_gpool=use_gpool, activation=activation,
                outer_norm_scale=1.0 / math.sqrt(i + 1.0),
            ))

        self.norm_trunkfinal = BiasMask(num_channels,
                                        scale=1.0 / math.sqrt(num_blocks + 1.0))
        self.act_trunkfinal = nn.Mish() if activation == "mish" else nn.ReLU()

        self.policy_head = PolicyHead(num_channels, c_p1, c_g1, activation=activation)
        self.value_head = ValueHead(num_channels, c_v1, c_v2, activation=activation)

        # Initialize all weights once at construction (no separate initialize() call
        # required by callers — preserves V1's `ResNet(...)` usage pattern).
        self._initialize(activation)

    def _initialize(self, activation: str) -> None:
        with torch.no_grad():
            init_weights(self.conv_spatial.weight, activation, scale=0.8)
            for block in self.blocks:
                block.initialize()
            self.policy_head.initialize()
            self.value_head.initialize()

    def forward(self, x: torch.Tensor):
        out = self.conv_spatial(x)
        for block in self.blocks:
            out = out + block(out)
        out = self.act_trunkfinal(self.norm_trunkfinal(out))
        policy_logits = self.policy_head(out)  # (B, H*W)
        value_logits = self.value_head(out)    # (B, 3)
        return policy_logits, value_logits
