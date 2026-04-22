"""ResNet policy/value network.

Topology ported from CSkyZero_V3/nets.h. The forward pass returns a tuple
``(policy_logits, opponent_policy_logits, value_logits)`` so the TorchScript
export can be consumed from C++ via ``torch::jit::script::Module::forward``.

Shapes:
    input:                    (B, num_planes, H, W)
    policy_logits:            (B, 1, H, W)
    opponent_policy_logits:   (B, 1, H, W)
    value_logits:             (B, 3)  — WDL
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from model_config import NetConfig


class NormActConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.bn = nn.BatchNorm2d(c_in)
        self.act = nn.Mish()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.bn(x)))


class KataGPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_mean = x.mean(dim=(2, 3))
        layer_max = torch.amax(x, dim=(2, 3))
        return torch.cat([layer_mean, layer_max], dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normactconv1 = NormActConv(channels, channels, 3)
        self.normactconv2 = NormActConv(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normactconv1(x)
        out = self.normactconv2(out)
        return x + out


class GlobalPoolingResidualBlock(nn.Module):
    def __init__(self, channels: int, gpool_channels: int = -1) -> None:
        super().__init__()
        if gpool_channels <= 0:
            gpool_channels = channels
        self.pre_bn = nn.BatchNorm2d(channels)
        self.pre_act = nn.Mish()
        self.regular_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, gpool_channels, 3, padding=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(gpool_channels)
        self.gpool_act = nn.Mish()
        self.gpool = KataGPool()
        self.gpool_to_bias = nn.Linear(gpool_channels * 2, channels, bias=False)
        self.normactconv2 = NormActConv(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pre_act(self.pre_bn(x))
        regular = self.regular_conv(out)
        g = self.gpool_act(self.gpool_bn(self.gpool_conv(out)))
        bias = self.gpool_to_bias(self.gpool(g)).unsqueeze(-1).unsqueeze(-1)
        regular = regular + bias
        regular = self.normactconv2(regular)
        return x + regular


class NestedBottleneckResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mid_channels: int,
        internal_length: int = 2,
        use_gpool: bool = False,
    ) -> None:
        super().__init__()
        self.normactconvp = NormActConv(channels, mid_channels, 1)
        blocks: list[nn.Module] = []
        for i in range(internal_length):
            if use_gpool and i == 0:
                blocks.append(GlobalPoolingResidualBlock(mid_channels))
            else:
                blocks.append(ResBlock(mid_channels))
        self.blockstack = nn.ModuleList(blocks)
        self.normactconvq = NormActConv(mid_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normactconvp(x)
        for block in self.blockstack:
            out = block(out)
        out = self.normactconvq(out)
        return x + out


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, board_size: int, head_channels: int = 64) -> None:
        super().__init__()
        self.board_size = board_size
        self.conv_p = nn.Conv2d(in_channels, head_channels, 1, bias=False)
        self.conv_g = nn.Conv2d(in_channels, head_channels, 1, bias=False)
        self.g_bn = nn.BatchNorm2d(head_channels)
        self.g_act = nn.Mish()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(head_channels * 2, head_channels, bias=False)
        self.p_bn = nn.BatchNorm2d(head_channels)
        self.p_act = nn.Mish()
        self.conv_final = nn.Conv2d(head_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.conv_p(x)
        g = self.g_act(self.g_bn(self.conv_g(x)))
        g = self.linear_g(self.gpool(g)).unsqueeze(-1).unsqueeze(-1)
        p = p + g
        p = self.p_act(self.p_bn(p))
        return self.conv_final(p)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3, head_channels: int = 32, value_channels: int = 64) -> None:
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, head_channels, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(head_channels)
        self.v_act = nn.Mish()
        self.gpool = KataGPool()
        self.fc1 = nn.Linear(head_channels * 2, value_channels)
        self.act2 = nn.Mish()
        self.fc_value = nn.Linear(value_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_act(self.v_bn(self.conv_v(x)))
        v_pooled = self.gpool(v)
        out = self.act2(self.fc1(v_pooled))
        return self.fc_value(out)


class ResNet(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.board_size = cfg.board_size
        self.num_planes = cfg.num_planes
        self.num_blocks = cfg.num_blocks
        self.num_channels = cfg.num_channels

        self.start_layer = nn.Sequential(
            nn.Conv2d(cfg.num_planes, cfg.num_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cfg.num_channels),
            nn.Mish(),
        )

        trunk: list[nn.Module] = []
        for i in range(cfg.num_blocks):
            use_gpool = ((i + 2) % 3 == 0)
            trunk.append(
                NestedBottleneckResBlock(
                    cfg.num_channels,
                    cfg.mid_channels,
                    internal_length=2,
                    use_gpool=use_gpool,
                )
            )
        self.trunk_blocks = nn.ModuleList(trunk)
        self.trunk_tip_bn = nn.BatchNorm2d(cfg.num_channels)
        self.trunk_tip_act = nn.Mish()

        self.total_policy_head = PolicyHead(
            cfg.num_channels, out_channels=2,
            board_size=cfg.board_size, head_channels=cfg.policy_head_channels,
        )
        self.value_head = ValueHead(
            cfg.num_channels, out_channels=3,
            head_channels=cfg.value_head_channels,
            value_channels=cfg.value_fc_channels,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        act_gain = math.sqrt(2.422)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.weight.size(0) * m.weight.size(2) * m.weight.size(3)
                stdv = act_gain / math.sqrt(fan_out)
                nn.init.normal_(m.weight, 0.0, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        fixup_scale = 1.0 / math.sqrt(max(self.num_blocks, 1))
        for block in self.trunk_blocks:
            if isinstance(block, NestedBottleneckResBlock):
                nn.init.normal_(block.normactconvq.conv.weight, 0.0, fixup_scale * 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.start_layer(x)
        for block in self.trunk_blocks:
            out = block(out)
        out = self.trunk_tip_act(self.trunk_tip_bn(out))

        total_policy_logits = self.total_policy_head(out)        # (B, 2, H, W)
        policy_logits = total_policy_logits[:, 0:1, :, :]        # (B, 1, H, W)
        opponent_policy_logits = total_policy_logits[:, 1:2, :, :]
        value_logits = self.value_head(out)                       # (B, 3)
        return policy_logits, opponent_policy_logits, value_logits


def build_model(cfg: NetConfig | None = None) -> ResNet:
    if cfg is None:
        cfg = NetConfig()
    return ResNet(cfg)


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report ResNet parameter count.")
    parser.add_argument("--num_blocks", type=int, required=True)
    parser.add_argument("--num_channels", type=int, required=True)
    args = parser.parse_args()

    cfg = NetConfig()
    cfg.num_blocks = args.num_blocks
    cfg.num_channels = args.num_channels
    model = ResNet(cfg)

    total, trainable = _count_params(model)
    print(
        f"num_blocks={cfg.num_blocks} num_channels={cfg.num_channels} "
        f"params={total:,} ({total / 1e6:.2f}M) trainable={trainable:,}"
    )


if __name__ == "__main__":
    main()
