"""Slim KataGo v15 network — only the heads SkyZero V5 actually trains.

Drops vs `full_nets.KataGoNet` (matching KataGomo Gomoku training schedule;
see KataGomo metrics_pytorch.py:483-487, 553/590/697/690 — those heads are
multiplied by `target_weight_ownership=0` for Gomoku so they're never trained):

    policy idx 1 (aux)              ← removed
    policy idx 4 (opt)              ← removed
    value_st_error                  ← removed
    value_var_time                  ← removed
    value_ownership                 ← removed

Kept (= what we have selfplay targets + losses for):

    policy: 4 outputs reordered → idx 0 main / 1 opp / 2 soft_main / 3 soft_opp
    value:  WDL (3) + td_value (9 = 3 horizons × WLD) + futurepos (2×H×W)

The trunk (stem + NestedBottleneckResBlocks + final BiasMask + intermediate
LastBatchNorm) is identical to `full_nets`; we reuse the primitives from
that module so there's a single source of truth for the shared parts.

Use:
    from nets import build_model, build_b4c64, build_b8c96, build_b12c128

forward signature: model(input_spatial, input_global) → Dict[str, Tensor]
returned keys (always present):
    policy:                     (B, 4, H*W)
    value_wdl:                  (B, 3)
    value_td:                   (B, 9)
    value_futurepos:            (B, 2, H, W)
returned keys (when has_intermediate_head=True):
    intermediate_policy:        (B, 4, H*W)
    intermediate_value_wdl:     (B, 3)
    intermediate_value_td:      (B, 9)
    intermediate_value_futurepos: (B, 2, H, W)

NormMask.scale traps still apply — see full_nets module docstring §3.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from full_nets import (
    FixscaleNorm,
    BiasMask,
    LastBatchNorm,
    KataGPool,
    KataValueHeadGPool,
    ConvAndGPool,
    NormActConv,
    ResBlock,
    NestedBottleneckResBlock,
    init_weights,
    compute_gain,
    TD_LONG_SLICE,
    TD_MID_SLICE,
    TD_SHORT_SLICE,
)


# ============================================================
# Slim Policy Head (4 outputs: main / opp / soft_main / soft_opp)
# ============================================================

class PolicyHead(nn.Module):
    """v15-style PolicyHead, slimmed to 4 outputs.

    Channel layout (training-time idx, see train.py head_losses):
        0: main_policy       (target = MCTS visits at t)
        1: opp_policy        (target = opponent MCTS visits at t+1)
        2: soft_main_policy  (target = (visits + 1e-7)^0.25 / Z)
        3: soft_opp_policy   (target = same on opponent visits)

    Layers identical to full_nets.PolicyHead except `conv2p` outputs 4
    channels instead of 6 (saves c_p1*2 weights — negligible but cleaner).
    """

    NUM_POLICY_OUTPUTS = 4

    def __init__(self, c_in: int, c_p1: int, c_g1: int,
                 activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = BiasMask(c_g1)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = BiasMask(c_p1)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.conv2p = nn.Conv2d(c_p1, self.NUM_POLICY_OUTPUTS, kernel_size=1, bias=False)

    def initialize(self) -> None:
        # KataGo master 2416-2432 (same scales as full_nets).
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
        outp = outp - (1.0 - mask) * 5000.0
        return outp.view(outp.shape[0], outp.shape[1], -1)


# ============================================================
# Slim Value Head (WDL + td_value + futurepos only)
# ============================================================

class ValueHead(nn.Module):
    """Slim ValueHead — drops linear_moremiscvaluehead and conv_ownership.

    Outputs (3-tuple):
        wdl:                 (B, 3)         W/L/draw logits
        td_value:            (B, 9)         3 horizons × WLD (long/mid/short)
        futurepos_pretanh:   (B, 2, H, W)   +8 / +32 step occupancy
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

        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = nn.Linear(c_v2, 9, bias=True)
        self.conv_futurepos = nn.Conv2d(c_in, 2, kernel_size=1, bias=False)

    def initialize(self) -> None:
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale,
                     fan_tensor=self.linear2.weight)

        for lin in (self.linear_valuehead, self.linear_miscvaluehead):
            init_weights(lin.weight, "identity", scale=1.0)
            init_weights(lin.bias, "identity", scale=bias_scale, fan_tensor=lin.weight)

        init_weights(self.conv_futurepos.weight, "identity", scale=0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask)
        outv1 = self.act1(outv1)

        outpooled = self.gpool(outv1, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outv2 = self.act2(self.linear2(outpooled))

        wdl = self.linear_valuehead(outv2)                          # (B, 3)
        td_value = self.linear_miscvaluehead(outv2)                 # (B, 9)
        futurepos = self.conv_futurepos(x) * mask                   # (B, 2, H, W)
        return wdl, td_value, futurepos


# ============================================================
# Slim full model
# ============================================================

class KataGoNet(nn.Module):
    """Slim KataGoNet: same trunk/stem/intermediate as full_nets, slim heads."""

    def __init__(
        self,
        num_blocks: int = 8,
        c_main: int = 96,
        c_mid: int = 48,
        c_gpool: int = 16,
        internal_length: int = 2,
        num_in_channels: int = 5,
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

        self.conv_spatial = nn.Conv2d(num_in_channels, c_main, kernel_size=3,
                                      padding=1, bias=False)
        self.linear_global = nn.Linear(num_global_features, c_main, bias=False)

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
            self.intermediate_policy_head = PolicyHead(c_main, c_p1, c_g1, activation=activation)
            self.intermediate_value_head = ValueHead(c_main, c_v1, c_v2,
                                                     activation=activation, pos_len=pos_len)

        self.policy_head = PolicyHead(c_main, c_p1, c_g1, activation=activation)
        self.value_head = ValueHead(c_main, c_v1, c_v2, activation=activation, pos_len=pos_len)

    def initialize(self) -> None:
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
        for i, block in enumerate(self.blocks):
            block.normactconvp.norm.set_scale(1.0 / math.sqrt(i + 1.0))
            block.normactconvq.norm.set_scale(1.0 / math.sqrt(block.internal_length + 1.0))
            for j, inner in enumerate(block.blockstack):
                inner.normactconv1.norm.set_scale(1.0 / math.sqrt(j + 1.0))
                inner.normactconv2.norm.set_scale(None)
        self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

    def forward(self, input_spatial: torch.Tensor,
                input_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_spatial + x_global

        result: Dict[str, torch.Tensor] = {}

        if self.has_intermediate_head:
            for block in self.blocks[: self.intermediate_head_blocks]:
                out = out + block(out, mask, mask_sum_hw)

            iout = self.norm_intermediate_trunkfinal(out, mask)
            iout = self.act_intermediate_trunkfinal(iout)
            iout_policy = self.intermediate_policy_head(iout, mask, mask_sum_hw)
            iout_value = self.intermediate_value_head(iout, mask, mask_sum_hw)
            result["intermediate_policy"] = iout_policy
            result["intermediate_value_wdl"] = iout_value[0]
            result["intermediate_value_td"] = iout_value[1]
            result["intermediate_value_futurepos"] = iout_value[2]

            for block in self.blocks[self.intermediate_head_blocks:]:
                out = out + block(out, mask, mask_sum_hw)
        else:
            for block in self.blocks:
                out = out + block(out, mask, mask_sum_hw)

        out = self.norm_trunkfinal(out, mask)
        out = self.act_trunkfinal(out)

        out_policy = self.policy_head(out, mask, mask_sum_hw)
        out_value = self.value_head(out, mask, mask_sum_hw)

        result["policy"] = out_policy
        result["value_wdl"] = out_value[0]
        result["value_td"] = out_value[1]
        result["value_futurepos"] = out_value[2]

        return result


# ============================================================
# Factories
# ============================================================

def build_model(cfg) -> KataGoNet:
    """Build a slim KataGoNet from a NetConfig (model_config.NetConfig).

    NOTE: caller must run `model.initialize()` before training, OR call
    `model.set_norm_scales()` after `load_state_dict` (NormMask.scale is
    not in state_dict — see full_nets module docstring §3 trap 3).
    """
    return KataGoNet(
        num_blocks=cfg.num_blocks,
        c_main=cfg.num_channels,
        c_mid=cfg.c_mid,
        c_gpool=cfg.c_gpool,
        internal_length=cfg.internal_length,
        num_in_channels=cfg.num_planes,
        num_global_features=cfg.num_global_features,
        activation=cfg.activation,
        version=cfg.version,
        has_intermediate_head=cfg.has_intermediate_head,
        intermediate_head_blocks=cfg.intermediate_head_blocks,
        c_p1=cfg.c_p1,
        c_g1=cfg.c_g1,
        c_v1=cfg.c_v1,
        c_v2=cfg.c_v2,
        pos_len=cfg.board_size,
    )


def build_b4c64(activation: str = "mish") -> KataGoNet:
    """Smoke / pipeline-validation size: 4 blocks × 64 trunk, ~120K params."""
    from model_config import NetConfig
    return build_model(NetConfig(num_blocks=4, num_channels=64, activation=activation))


def build_b8c96(activation: str = "mish") -> KataGoNet:
    """Test-run size: 8 blocks × 96 trunk × 48 mid × 16 gpool, ~700K params."""
    from model_config import NetConfig
    return build_model(NetConfig(num_blocks=8, num_channels=96, activation=activation))


def build_b12c128(activation: str = "mish") -> KataGoNet:
    """Production size: 12 blocks × 128 trunk × 64 mid × 16 gpool, ~2M params."""
    from model_config import NetConfig
    return build_model(NetConfig(num_blocks=12, num_channels=128, activation=activation))
