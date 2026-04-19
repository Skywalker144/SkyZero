"""
SkyZero_V4 Neural Network, aligned with SkyZero_V2.1 architecture.

Trunk: SiLU + BatchNorm2d + NestedBottleneckResBlock (same as V2.1).
Heads (kept from V4): 2-channel policy (policy + opponent_policy) and
value head with an extra shortterm value-error prediction.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


SHORTTERM_VALUE_ERROR_MULTIPLIER = 0.25


class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """Softplus with a floor on the backward-pass gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_floor: float, square: bool):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor
        if square:
            return torch.square(torch.nn.functional.softplus(0.5 * x))
        else:
            return torch.nn.functional.softplus(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_floor = ctx.grad_floor
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, None, None


# ---------------------------------------------------------------------------
# Building blocks (ported from SkyZero_V2.1/nets.py)
# ---------------------------------------------------------------------------

class NormActConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.act = nn.SiLU(inplace=True)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return self.conv(x)


class KataGPool(nn.Module):
    def forward(self, x):
        layer_mean = torch.mean(x, dim=(2, 3))
        layer_max = torch.amax(x, dim=(2, 3))
        return torch.cat((layer_mean, layer_max), dim=1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.normactconv1 = NormActConv(channels, channels, kernel_size=3)
        self.normactconv2 = NormActConv(channels, channels, kernel_size=3)

    def forward(self, x):
        out = self.normactconv1(x)
        out = self.normactconv2(out)
        return x + out


class GlobalPoolingResidualBlock(nn.Module):
    def __init__(self, channels, gpool_channels=None):
        super().__init__()
        if gpool_channels is None:
            gpool_channels = channels
        self.pre_bn = nn.BatchNorm2d(channels)
        self.pre_act = nn.SiLU(inplace=True)
        self.regular_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gpool_conv = nn.Conv2d(channels, gpool_channels, kernel_size=3, padding=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(gpool_channels)
        self.gpool_act = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.gpool_to_bias = nn.Linear(gpool_channels * 2, channels, bias=False)
        self.normactconv2 = NormActConv(channels, channels, kernel_size=3)

    def forward(self, x):
        out = self.pre_bn(x)
        out = self.pre_act(out)

        regular = self.regular_conv(out)
        gpool = self.gpool_conv(out)
        gpool = self.gpool_bn(gpool)
        gpool = self.gpool_act(gpool)

        bias = self.gpool_to_bias(self.gpool(gpool)).unsqueeze(-1).unsqueeze(-1)
        regular = regular + bias

        regular = self.normactconv2(regular)
        return x + regular


class NestedBottleneckResBlock(nn.Module):
    def __init__(self, channels, mid_channels, internal_length=2, use_gpool=False):
        super().__init__()
        self.normactconvp = NormActConv(channels, mid_channels, kernel_size=1)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            if use_gpool and i == 0:
                self.blockstack.append(GlobalPoolingResidualBlock(mid_channels))
            else:
                self.blockstack.append(ResBlock(mid_channels))
        self.normactconvq = NormActConv(mid_channels, channels, kernel_size=1)

    def forward(self, x):
        out = self.normactconvp(x)
        for block in self.blockstack:
            out = block(out)
        out = self.normactconvq(out)
        return x + out


# ---------------------------------------------------------------------------
# Heads (V2.1 trunk style, V4 output shape)
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Two-channel policy head: channel 0 = policy, channel 1 = opponent_policy."""

    def __init__(self, in_channels, head_channels, out_channels=2):
        super().__init__()
        self.conv_p = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.conv_g = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.g_bn = nn.BatchNorm2d(head_channels)
        self.g_act = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(head_channels * 2, head_channels, bias=False)
        self.p_bn = nn.BatchNorm2d(head_channels)
        self.p_act = nn.SiLU(inplace=True)
        self.conv_final = nn.Conv2d(head_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        p = self.conv_p(x)
        g = self.conv_g(x)
        g = self.g_bn(g)
        g = self.g_act(g)
        g = self.gpool(g)
        g = self.linear_g(g).unsqueeze(-1).unsqueeze(-1)
        p = p + g
        p = self.p_bn(p)
        p = self.p_act(p)
        return self.conv_final(p)


class ValueHead(nn.Module):
    """V2.1-style WDL value head, plus a shortterm value-error prediction head."""

    def __init__(self, in_channels, head_channels, value_channels, out_channels=3):
        super().__init__()
        self.conv_v = nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(head_channels)
        self.v_act = nn.SiLU(inplace=True)
        self.gpool = KataGPool()
        self.fc1 = nn.Linear(head_channels * 2, value_channels, bias=True)
        self.act2 = nn.SiLU(inplace=True)
        self.fc_value = nn.Linear(value_channels, out_channels, bias=True)
        # Shortterm value error head: predicts (utility_pred - actual_outcome)^2.
        self.linear_value_error = nn.Linear(value_channels, 1, bias=True)

    def forward(self, x):
        v = self.conv_v(x)
        v = self.v_bn(v)
        v = self.v_act(v)

        v_pooled = self.gpool(v)
        out = self.act2(self.fc1(v_pooled))

        value_logits = self.fc_value(out)
        value_error_raw = self.linear_value_error(out).squeeze(-1)
        if self.training:
            value_error_pred = SoftPlusWithGradientFloorFunction.apply(
                value_error_raw, 0.05, False
            ) * SHORTTERM_VALUE_ERROR_MULTIPLIER
        else:
            # torch.jit.trace can't capture autograd.Function; softplus is
            # mathematically identical here (square=False).
            value_error_pred = F.softplus(value_error_raw) * SHORTTERM_VALUE_ERROR_MULTIPLIER
        return value_logits, value_error_pred.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, config: Dict, pos_len: int, num_input_planes: int):
        super().__init__()
        self.config = config
        self.pos_len = pos_len
        self.num_input_planes = num_input_planes

        num_blocks = config["num_blocks"]
        num_channels = config["num_channels"]
        mid_channels = max(16, num_channels // 2)

        self.start_layer = nn.Sequential(
            nn.Conv2d(num_input_planes, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(inplace=True),
        )

        self.trunk_blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = (i + 2) % 3 == 0
            self.trunk_blocks.append(
                NestedBottleneckResBlock(
                    channels=num_channels,
                    mid_channels=mid_channels,
                    internal_length=2,
                    use_gpool=use_gpool,
                )
            )
        self.trunk_tip_bn = nn.BatchNorm2d(num_channels)
        self.trunk_tip_act = nn.SiLU(inplace=True)

        self.policy_head = PolicyHead(
            in_channels=num_channels,
            head_channels=max(1, num_channels // 2),
            out_channels=2,
        )
        self.value_head = ValueHead(
            in_channels=num_channels,
            head_channels=max(1, num_channels // 4),
            value_channels=max(1, num_channels // 2),
            out_channels=3,
        )

    def initialize(self):
        """V2.1-style weight init: normal conv/linear + fixup scaling on block exits."""
        silu_gain = math.sqrt(2.35)
        num_blocks = len(self.trunk_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                std = silu_gain / math.sqrt(fan_out)
                nn.init.normal_(m.weight, 0.0, std)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        fixup_scale = 1.0 / math.sqrt(max(num_blocks, 1))
        for block in self.trunk_blocks:
            if hasattr(block, "normactconvq"):
                nn.init.normal_(block.normactconvq.conv.weight, 0.0, fixup_scale * 0.01)

        # Near-zero uncertainty prior at init: 0.25 * softplus(-2.25) ~= 0.025.
        nn.init.constant_(self.value_head.linear_value_error.bias, -2.25)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        """Populate param groups used by train.py. Policy/value head params go
        to `output`/`output_noreg` (halved LR + reduced WD in train.py); the
        rest go to `normal`/`noreg`. `normal_gamma` stays empty under plain BN.
        """
        reg_dict.setdefault("normal", [])
        reg_dict.setdefault("normal_gamma", [])
        reg_dict.setdefault("output", [])
        reg_dict.setdefault("noreg", [])
        reg_dict.setdefault("output_noreg", [])

        # Classify each parameter by the type of its owning module.
        param_to_group = {}
        for mod_name, mod in self.named_modules():
            is_head = mod_name.startswith(("policy_head", "value_head"))
            if isinstance(mod, nn.BatchNorm2d):
                for pname, p in mod.named_parameters(recurse=False):
                    param_to_group[id(p)] = "output_noreg" if is_head else "noreg"
            elif isinstance(mod, (nn.Conv2d, nn.Linear)):
                for pname, p in mod.named_parameters(recurse=False):
                    if pname == "bias":
                        param_to_group[id(p)] = "output_noreg" if is_head else "noreg"
                    else:
                        param_to_group[id(p)] = "output" if is_head else "normal"

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            group = param_to_group.get(id(p), "normal")
            reg_dict[group].append(p)

    def forward(self, x):
        out = self.start_layer(x)
        for block in self.trunk_blocks:
            out = block(out)
        out = self.trunk_tip_bn(out)
        out = self.trunk_tip_act(out)

        policy_out = self.policy_head(out)  # [B, 2, H, W]
        value_logits, value_error_pred = self.value_head(out)

        return {
            "policy_logits": policy_out[:, 0:1, :, :],
            "opponent_policy_logits": policy_out[:, 1:2, :, :],
            "value_logits": value_logits,
            "value_error_pred": value_error_pred,
        }


# ---------------------------------------------------------------------------
# ExportWrapper — TorchScript tuple output
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return (
            out["policy_logits"],
            out["opponent_policy_logits"],
            out["value_logits"],
            out["value_error_pred"],
        )
