"""Network for 4x4 2048 — simplified KataGo-style NestedBottleneck ResNet.

Input : (B, NUM_PLANES, 4, 4) one-hot exponent planes.
Output: policy_logits (B, 4) over slide directions, value (B,) in SCALED units
        (raw expected discounted score / value_scale).

Trunk = `blocks` NestedBottleneck blocks (1x1 reduce -> internal ResBlocks at
c_mid -> 1x1 expand, residual), with global-pooling injected into a subset so
the network sees board-global context (crucial for 2048's value). Simplified vs
V7.1's full_nets.py: BatchNorm instead of FixscaleNorm, no mask (the 4x4 board
is always full), mean+max global pool (no variable-size scaling), single scalar
value head. Pre-activation order (norm -> act -> conv), mish activation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import Config


def _act(x: torch.Tensor) -> torch.Tensor:
    return F.mish(x)


class GPool(nn.Module):
    """Global pool over space -> [mean, max] per channel (B, 2C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class NormActConv(nn.Module):
    """Pre-activation BN -> mish -> conv."""

    def __init__(self, c_in: int, c_out: int, k: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_act(self.bn(x)))


class ConvAndGPool(nn.Module):
    """Spatial conv + a global-pooled bias broadcast over space (KataGo style)."""

    def __init__(self, c_in: int, c_out: int, c_gpool: int) -> None:
        super().__init__()
        self.conv_r = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.conv_g = nn.Conv2d(c_in, c_gpool, 3, padding=1, bias=False)
        self.bn_g = nn.BatchNorm2d(c_gpool)
        self.gpool = GPool()
        self.lin_g = nn.Linear(2 * c_gpool, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.conv_r(x)
        g = _act(self.bn_g(self.conv_g(x)))
        bias = self.lin_g(self.gpool(g))            # (B, c_out)
        return s + bias[:, :, None, None]


class NormActConvGPool(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_gpool: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.cgp = ConvAndGPool(c_in, c_out, c_gpool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cgp(_act(self.bn(x)))


class InnerRes(nn.Module):
    """c_mid residual block; first conv optionally global-pooling-augmented."""

    def __init__(self, c_mid: int, c_gpool: int | None) -> None:
        super().__init__()
        self.nac1 = (NormActConvGPool(c_mid, c_mid, c_gpool) if c_gpool
                     else NormActConv(c_mid, c_mid, 3))
        self.nac2 = NormActConv(c_mid, c_mid, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.nac2(self.nac1(x))


class NestedBottleneck(nn.Module):
    """1x1 (c_main->c_mid) -> internal_length InnerRes -> 1x1 (c_mid->c_main), residual."""

    def __init__(self, c_main: int, c_mid: int, internal_length: int, c_gpool: int | None) -> None:
        super().__init__()
        self.pre = NormActConv(c_main, c_mid, 1)
        self.inner = nn.ModuleList(
            [InnerRes(c_mid, c_gpool if i == 0 else None) for i in range(internal_length)]
        )
        self.post = NormActConv(c_mid, c_main, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        for blk in self.inner:
            h = blk(h)
        return x + self.post(h)


class Net2048(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        c_mid = cfg.c_mid
        c_gp = cfg.c_gpool

        self.stem = nn.Conv2d(cfg.num_planes, c, 3, padding=1, bias=False)
        # Global pooling on every other block (indices 1,3,5,...).
        self.trunk = nn.ModuleList([
            NestedBottleneck(c, c_mid, cfg.internal_length, c_gp if (i % 2 == 1) else None)
            for i in range(cfg.blocks)
        ])
        self.trunk_bn = nn.BatchNorm2d(c)

        # policy head: 1x1 -> BN -> act -> FC over flattened spatial -> 4
        self.p_conv = nn.Conv2d(c, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 16, 4)

        # value head: 1x1 -> BN -> act -> global pool -> MLP -> 1 (scalar, >=0)
        self.v_conv = nn.Conv2d(c, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_gpool = GPool()
        self.v_fc1 = nn.Linear(2 * 32, cfg.value_hidden)
        self.v_fc2 = nn.Linear(cfg.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.trunk:
            h = blk(h)
        h = _act(self.trunk_bn(h))

        p = _act(self.p_bn(self.p_conv(h)))
        p = self.p_fc(p.flatten(1))                       # (B, 4) logits

        v = _act(self.v_bn(self.v_conv(h)))
        v = _act(self.v_fc1(self.v_gpool(v)))             # (B, value_hidden)
        v = F.softplus(self.v_fc2(v)).squeeze(-1)         # (B,) scaled units
        return p, v


def build_net(cfg: Config) -> Net2048:
    return Net2048(cfg)
