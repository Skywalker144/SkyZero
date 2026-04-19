"""
SkyZero_V4 Neural Network.
Ported from KataGomo-Gom2024/python/model_pytorch.py with simplifications:
- Single spatial input only (no global features)
- Board mask = all-ones (gomoku: all positions valid)
- Only 3 output heads: policy, opponent_policy, value
- bnorm (BatchNorm) normalization
- No pass move, no attention pool, no RepVGG
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from model_config import ModelConfig


# ---------------------------------------------------------------------------
# Helper functions (from KataGomo model_pytorch.py)
# ---------------------------------------------------------------------------

SHORTTERM_VALUE_ERROR_MULTIPLIER = 0.25


class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """
    Same as softplus, except on backward pass, we never let the gradient decrease below grad_floor.
    Equivalent to having a dynamic learning rate depending on stop_grad(x) where x is the input.
    If square, then also squares the result while halving the input, and still also keeping the same gradient.
    """
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
        grad_grad_floor = None
        grad_square = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, grad_grad_floor, grad_square


def act(activation, inplace=False):
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == "elu":
        return nn.ELU(inplace=inplace)
    if activation == "mish":
        return nn.Mish(inplace=inplace)
    if activation == "gelu":
        return nn.GELU()
    assert False, f"Unknown activation: {activation}"


def compute_gain(activation):
    if activation == "relu":
        return math.sqrt(2.0)
    if activation == "elu":
        return math.sqrt(1.55052)
    if activation == "mish":
        return math.sqrt(2.210277)
    if activation == "gelu":
        return math.sqrt(2.351718)
    if activation == "identity":
        return 1.0
    assert False, f"Unknown activation: {activation}"


def init_weights(tensor, activation, scale, fan_tensor=None):
    gain = compute_gain(activation)
    if fan_tensor is not None:
        (fan_in, _) = nn.init._calculate_fan_in_and_fan_out(fan_tensor)
    else:
        (fan_in, _) = nn.init._calculate_fan_in_and_fan_out(tensor)
    target_std = scale * gain / math.sqrt(fan_in)
    std = target_std / 0.87962566103423978
    if std < 1e-10:
        tensor.fill_(0.0)
    else:
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


# ---------------------------------------------------------------------------
# BiasMask
# ---------------------------------------------------------------------------

class BiasMask(nn.Module):
    def __init__(self, c_in, config: ModelConfig, is_after_batchnorm: bool = False):
        super().__init__()
        self.c_in = c_in
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.is_after_batchnorm = is_after_batchnorm
        self.scale = None

    def set_scale(self, scale: Optional[float]):
        self.scale = scale

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        if self.is_after_batchnorm:
            reg_dict["output_noreg"].append(self.beta)
        else:
            reg_dict["noreg"].append(self.beta)

    def forward(self, x, mask, mask_sum: float):
        if self.scale is not None:
            return (x * self.scale + self.beta) * mask
        else:
            return (x + self.beta) * mask


# ---------------------------------------------------------------------------
# NormMask — supports fixup, fixscale, bnorm
# ---------------------------------------------------------------------------

class NormMask(nn.Module):
    def __init__(
        self,
        c_in: int,
        config: ModelConfig,
        fixup_use_gamma: bool,
        force_use_gamma: bool = False,
        is_last_batchnorm: bool = False,
    ):
        super().__init__()
        self.norm_kind = config["norm_kind"]
        self.epsilon = config["bnorm_epsilon"]
        self.running_avg_momentum = config["bnorm_running_avg_momentum"]
        self.fixup_use_gamma = fixup_use_gamma
        self.is_last_batchnorm = is_last_batchnorm
        self.use_gamma = (
            ("bnorm_use_gamma" in config and config["bnorm_use_gamma"])
            or ((self.norm_kind in ("fixup", "fixscale")) and fixup_use_gamma)
            or force_use_gamma
        )
        self.c_in = c_in
        self.scale = None
        self.gamma = None

        if self.norm_kind == "bnorm":
            self.is_using_batchnorm = True
            if self.use_gamma:
                self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
            self.register_buffer("running_mean", torch.zeros(c_in))
            self.register_buffer("running_std", torch.ones(c_in))

        elif self.norm_kind in ("fixup", "fixscale"):
            self.is_using_batchnorm = False
            self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
            if self.use_gamma:
                self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        else:
            assert False, f"Unimplemented norm_kind: {self.norm_kind}"

    def set_scale(self, scale: Optional[float]):
        self.scale = scale

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        if self.is_last_batchnorm:
            if self.gamma is not None:
                reg_dict["output"].append(self.gamma)
            reg_dict["output_noreg"].append(self.beta)
        else:
            if self.gamma is not None:
                reg_dict["normal_gamma"].append(self.gamma)
            reg_dict["noreg"].append(self.beta)

    def _compute_bnorm_values(self, x, mask, mask_sum: float):
        mean = torch.sum(x * mask, dim=(0, 2, 3), keepdim=True) / mask_sum
        zeromean_x = x - mean
        var = torch.sum(torch.square(zeromean_x * mask), dim=(0, 2, 3), keepdim=True) / mask_sum
        std = torch.sqrt(var + self.epsilon)
        return zeromean_x, mean, std

    def apply_gamma_beta_scale_mask(self, x, mask):
        if self.scale is not None:
            if self.gamma is not None:
                return (x * (self.gamma * self.scale) + self.beta) * mask
            else:
                return (x * self.scale + self.beta) * mask
        else:
            if self.gamma is not None:
                return (x * self.gamma + self.beta) * mask
            else:
                return (x + self.beta) * mask

    def forward(self, x, mask, mask_sum: float):
        if self.norm_kind == "bnorm":
            if self.training:
                zeromean_x, mean, std = self._compute_bnorm_values(x, mask, mask_sum)
                detached_mean = mean.view(self.c_in).detach()
                detached_std = std.view(self.c_in).detach()
                with torch.no_grad():
                    self.running_mean += self.running_avg_momentum * (detached_mean - self.running_mean)
                    self.running_std += self.running_avg_momentum * (detached_std - self.running_std)
                return self.apply_gamma_beta_scale_mask(zeromean_x / std, mask)
            else:
                return self.apply_gamma_beta_scale_mask(
                    (x - self.running_mean.view(1, self.c_in, 1, 1)) / self.running_std.view(1, self.c_in, 1, 1),
                    mask,
                )

        elif self.norm_kind in ("fixup", "fixscale"):
            return self.apply_gamma_beta_scale_mask(x, mask)

        else:
            assert False


# ---------------------------------------------------------------------------
# Global Pooling
# ---------------------------------------------------------------------------

class KataGPool(nn.Module):
    def forward(self, x, mask, mask_sum_hw):
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        (layer_max, _) = torch.max(
            (x + (mask - 1.0)).view(x.shape[0], x.shape[1], -1).to(torch.float32), dim=2
        )
        layer_max = layer_max.view(x.shape[0], x.shape[1], 1, 1)
        out = torch.cat((layer_mean, layer_mean * (mask_sum_hw_sqrt_offset / 10.0), layer_max), dim=1)
        return out.type_as(x)


class KataValueHeadGPool(nn.Module):
    def forward(self, x, mask, mask_sum_hw):
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        out = torch.cat((
            layer_mean,
            layer_mean * (mask_sum_hw_sqrt_offset / 10.0),
            layer_mean * ((mask_sum_hw_sqrt_offset * mask_sum_hw_sqrt_offset) / 100.0 - 0.1),
        ), dim=1)
        return out.type_as(x)


# ---------------------------------------------------------------------------
# KataConvAndGPool — conv + global pooling branch
# ---------------------------------------------------------------------------

class KataConvAndGPool(nn.Module):
    def __init__(self, c_in, c_out, c_gpool, config, activation):
        super().__init__()
        self.norm_kind = config["norm_kind"]
        self.activation = activation
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.normg = NormMask(c_gpool, config=config, fixup_use_gamma=False)
        self.actg = act(self.activation, inplace=True)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale):
        r_scale = 0.8
        g_scale = 0.6
        if self.norm_kind in ("fixup", "fixscale"):
            init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
            init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
        else:
            init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * 1.0)
            init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * g_scale)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["normal"].append(self.conv1r.weight)
        reg_dict["normal"].append(self.conv1g.weight)
        self.normg.add_reg_dict(reg_dict)
        reg_dict["normal"].append(self.linear_g.weight)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        outr = self.conv1r(x)
        outg = self.conv1g(x)
        outg = self.normg(outg, mask=mask, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        return outr + outg


# ---------------------------------------------------------------------------
# NormActConv — Norm -> Activation -> Conv (optionally with gpool)
# ---------------------------------------------------------------------------

class NormActConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_gpool: Optional[int],
        config: ModelConfig,
        activation: str,
        kernel_size: int,
        fixup_use_gamma: bool,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_gpool = c_gpool
        self.norm = NormMask(c_in, config=config, fixup_use_gamma=fixup_use_gamma)
        self.activation = activation
        self.act = act(activation, inplace=True)

        if c_gpool is not None:
            self.convpool = KataConvAndGPool(c_in=c_in, c_out=c_out, c_gpool=c_gpool, config=config, activation=activation)
            self.conv = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.convpool = None

    def initialize(self, scale, norm_scale=None):
        self.norm.set_scale(norm_scale)
        if self.convpool is not None:
            self.convpool.initialize(scale=scale)
        else:
            init_weights(self.conv.weight, self.activation, scale=scale)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        self.norm.add_reg_dict(reg_dict)
        if self.convpool is not None:
            self.convpool.add_reg_dict(reg_dict)
        else:
            reg_dict["normal"].append(self.conv.weight)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        out = self.norm(x, mask=mask, mask_sum=mask_sum)
        out = self.act(out)
        if self.convpool is not None:
            out = self.convpool(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        else:
            out = self.conv(out)
        return out


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, name: str, c_main: int, c_mid: int, c_gpool: Optional[int], config: ModelConfig, activation: str):
        super().__init__()
        self.norm_kind = config["norm_kind"]
        self.normactconv1 = NormActConv(
            c_in=c_main,
            c_out=c_mid - (0 if c_gpool is None else c_gpool),
            c_gpool=c_gpool,
            config=config,
            activation=activation,
            kernel_size=3,
            fixup_use_gamma=False,
        )
        self.normactconv2 = NormActConv(
            c_in=c_mid - (0 if c_gpool is None else c_gpool),
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=3,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconv1.initialize(scale=fixup_scale)
            self.normactconv2.initialize(scale=0.0)
        elif self.norm_kind == "fixscale":
            self.normactconv1.initialize(scale=1.0, norm_scale=fixup_scale)
            self.normactconv2.initialize(scale=1.0)
        else:
            self.normactconv1.initialize(scale=1.0)
            self.normactconv2.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        self.normactconv1.add_reg_dict(reg_dict)
        self.normactconv2.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        out = self.normactconv1(x, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        out = self.normactconv2(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        return x + out


# ---------------------------------------------------------------------------
# PolicyHead — simplified: 2 policy outputs (player + opponent), no pass move
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    def __init__(self, c_in, c_p1, c_g1, config, activation):
        super().__init__()
        self.activation = activation
        self.num_policy_outputs = 2  # policy + opponent_policy

        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = BiasMask(c_g1, config=config, is_after_batchnorm=True)
        self.actg = act(self.activation)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = BiasMask(c_p1, config=config, is_after_batchnorm=True)
        self.act2 = act(activation)
        self.conv2p = nn.Conv2d(c_p1, self.num_policy_outputs, kernel_size=1, bias=False)

    def initialize(self):
        p_scale = 0.8
        g_scale = 0.6
        scale_output = 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["output"].append(self.conv1p.weight)
        reg_dict["output"].append(self.conv1g.weight)
        reg_dict["output"].append(self.linear_g.weight)
        reg_dict["output"].append(self.conv2p.weight)
        self.biasg.add_reg_dict(reg_dict)
        self.bias2.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        outp = self.conv1p(x)
        outg = self.conv1g(x)
        outg = self.biasg(outg, mask=mask, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = outp + outg
        outp = self.bias2(outp, mask=mask, mask_sum=mask_sum)
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        # Mask out off-board positions (for gomoku 15x15 this is a no-op)
        outp = outp - (1.0 - mask) * 5000.0
        return outp  # [B, 2, H, W]


# ---------------------------------------------------------------------------
# ValueHead — simplified: only WDL value output [B, 3]
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    def __init__(self, c_in, c_v1, c_v2, config, activation):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.bias1 = BiasMask(c_v1, config=config, is_after_batchnorm=True)
        self.act1 = act(activation)
        self.gpool = KataValueHeadGPool()
        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = act(activation)
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        # Shortterm value error head: predicts (utility_pred - actual_outcome)^2.
        # Forward applies softplus-with-gradient-floor and multiplier (0.25) like KataGo.
        self.linear_value_error = nn.Linear(c_v2, 1, bias=True)

    def initialize(self):
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale, fan_tensor=self.linear2.weight)
        init_weights(self.linear_valuehead.weight, "identity", scale=1.0)
        init_weights(self.linear_valuehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_valuehead.weight)
        init_weights(self.linear_value_error.weight, "identity", scale=1.0)
        # 0.25 * softplus(-2.25) ~= 0.025, a near-zero uncertainty prior at init.
        nn.init.constant_(self.linear_value_error.bias, -2.25)

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["output"].append(self.conv1.weight)
        reg_dict["output"].append(self.linear2.weight)
        reg_dict["output_noreg"].append(self.linear2.bias)
        reg_dict["output"].append(self.linear_valuehead.weight)
        reg_dict["output_noreg"].append(self.linear_valuehead.bias)
        reg_dict["output"].append(self.linear_value_error.weight)
        reg_dict["output_noreg"].append(self.linear_value_error.bias)
        self.bias1.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask=mask, mask_sum=mask_sum)
        outv1 = self.act1(outv1)
        outpooled = self.gpool(outv1, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outv2 = self.linear2(outpooled)
        outv2 = self.act2(outv2)
        value_logits = self.linear_valuehead(outv2)          # [B, 3]
        value_error_raw = self.linear_value_error(outv2).squeeze(-1)  # [B]
        if self.training:
            value_error_pred = SoftPlusWithGradientFloorFunction.apply(
                value_error_raw, 0.05, False
            ) * SHORTTERM_VALUE_ERROR_MULTIPLIER                      # [B]
        else:
            # Inference/trace path: autograd.Function can't be exported by jit.trace.
            # Forward is mathematically identical to F.softplus when square=False.
            value_error_pred = F.softplus(value_error_raw) * SHORTTERM_VALUE_ERROR_MULTIPLIER  # [B]
        return value_logits, value_error_pred.unsqueeze(-1)           # [B, 1]


# ---------------------------------------------------------------------------
# Model — main class
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, config: ModelConfig, pos_len: int, num_input_planes: int):
        super().__init__()
        self.config = config
        self.norm_kind = config["norm_kind"]
        self.block_kind = config["block_kind"]
        self.c_trunk = config["trunk_num_channels"]
        self.c_mid = config["mid_num_channels"]
        self.c_gpool = config["gpool_num_channels"]
        self.c_p1 = config["p1_num_channels"]
        self.c_g1 = config["g1_num_channels"]
        self.c_v1 = config["v1_num_channels"]
        self.c_v2 = config["v2_size"]
        self.num_total_blocks = len(self.block_kind)
        self.pos_len = pos_len
        self.num_input_planes = num_input_planes
        self.activation = config.get("activation", "relu")

        # Initial spatial convolution
        if config.get("initial_conv_1x1", False):
            self.conv_spatial = nn.Conv2d(num_input_planes, self.c_trunk, kernel_size=1, bias=False)
        else:
            self.conv_spatial = nn.Conv2d(num_input_planes, self.c_trunk, kernel_size=3, padding=1, bias=False)

        # Trunk blocks
        self.blocks = nn.ModuleList()
        for block_config in self.block_kind:
            block_name = block_config[0]
            block_type = block_config[1]
            use_gpool = block_type.endswith("gpool")
            if use_gpool:
                block_type = block_type[:-5]

            if block_type == "regular":
                self.blocks.append(ResBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool else None),
                    config=self.config,
                    activation=self.activation,
                ))
            else:
                assert False, f"Unknown block kind: {block_config[1]}"

        # Final trunk norm
        self.norm_trunkfinal = NormMask(self.c_trunk, self.config, fixup_use_gamma=False, is_last_batchnorm=True)
        self.act_trunkfinal = act(self.activation)

        # Heads
        self.policy_head = PolicyHead(self.c_trunk, self.c_p1, self.c_g1, self.config, self.activation)
        self.value_head = ValueHead(self.c_trunk, self.c_v1, self.c_v2, self.config, self.activation)

    def initialize(self):
        with torch.no_grad():
            init_weights(self.conv_spatial.weight, self.activation, scale=0.8)

            if self.norm_kind == "fixup":
                fixup_scale = 1.0 / math.sqrt(self.num_total_blocks)
                for block in self.blocks:
                    block.initialize(fixup_scale=fixup_scale)
            elif self.norm_kind == "fixscale":
                for i, block in enumerate(self.blocks):
                    block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))
                self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_total_blocks + 1.0))
            else:
                for block in self.blocks:
                    block.initialize(fixup_scale=1.0)

            self.policy_head.initialize()
            self.value_head.initialize()

    def add_reg_dict(self, reg_dict: Dict[str, List]):
        reg_dict["normal"] = []
        reg_dict["normal_gamma"] = []
        reg_dict["output"] = []
        reg_dict["noreg"] = []
        reg_dict["output_noreg"] = []

        reg_dict["normal"].append(self.conv_spatial.weight)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        self.norm_trunkfinal.add_reg_dict(reg_dict)
        self.policy_head.add_reg_dict(reg_dict)
        self.value_head.add_reg_dict(reg_dict)

    def forward(self, x):
        # Construct all-ones mask for gomoku (all positions valid)
        mask = torch.ones_like(x[:, 0:1, :, :])
        mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
        mask_sum = float(x.shape[0] * x.shape[2] * x.shape[3])

        out = self.conv_spatial(x)

        for block in self.blocks:
            out = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)

        out = self.norm_trunkfinal(out, mask=mask, mask_sum=mask_sum)
        out = self.act_trunkfinal(out)

        policy_out = self.policy_head(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)  # [B, 2, H, W]
        value_out, value_error_pred = self.value_head(
            out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum
        )

        return {
            "policy_logits": policy_out[:, 0:1, :, :],            # [B, 1, H, W]
            "opponent_policy_logits": policy_out[:, 1:2, :, :],   # [B, 1, H, W]
            "value_logits": value_out,                             # [B, 3]
            "value_error_pred": value_error_pred,                  # [B, 1]
        }


# ---------------------------------------------------------------------------
# ExportWrapper — for TorchScript export (dict -> tuple)
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
