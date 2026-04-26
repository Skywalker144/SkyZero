import pytest
import torch
from nets_v2 import ConvAndGPool, NormActConv


def test_conv_and_gpool_forward_shape():
    """Input (B, c_in, H, W) → output (B, c_out, H, W)."""
    m = ConvAndGPool(c_in=32, c_out=24, c_gpool=8, activation="mish")
    x = torch.randn(2, 32, 15, 15)
    mask = torch.ones(2, 1, 15, 15)
    out = m(x, mask)
    assert out.shape == (2, 24, 15, 15)


def test_conv_and_gpool_components_exist():
    m = ConvAndGPool(c_in=16, c_out=12, c_gpool=4)
    assert hasattr(m, "conv1r")    # 主分支 3x3
    assert hasattr(m, "conv1g")    # gpool 分支 3x3
    assert hasattr(m, "normg")
    assert hasattr(m, "actg")
    assert hasattr(m, "gpool")
    assert hasattr(m, "linear_g")
    assert m.linear_g.in_features == 3 * 4   # 3 stats × c_gpool


def test_conv_and_gpool_initialize_runs():
    m = ConvAndGPool(c_in=8, c_out=6, c_gpool=2)
    m.initialize(scale=1.0)
    assert m.conv1r.weight.abs().max() > 0   # 不再是 zeros
    assert m.conv1g.weight.abs().max() > 0


def test_conv_and_gpool_with_mask_sum_hw():
    """传入 mask_sum_hw 应该结果一致."""
    m = ConvAndGPool(c_in=8, c_out=4, c_gpool=2)
    x = torch.randn(1, 8, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
    out_implicit = m(x, mask)
    out_explicit = m(x, mask, mask_sum_hw=mask_sum_hw)
    assert torch.allclose(out_implicit, out_explicit)


def test_norm_act_conv_basic():
    m = NormActConv(c_in=16, c_out=12, activation="mish", kernel_size=3,
                    c_gpool=None, fixup_use_gamma=True)
    x = torch.randn(1, 16, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = m(x, mask)
    assert out.shape == (1, 12, 15, 15)
    assert m.conv is not None
    assert m.convpool is None


def test_norm_act_conv_with_gpool():
    """When c_gpool is given, uses ConvAndGPool inside."""
    m = NormActConv(c_in=16, c_out=12, activation="mish", kernel_size=3,
                    c_gpool=4, fixup_use_gamma=True)
    x = torch.randn(1, 16, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = m(x, mask)
    assert out.shape == (1, 12, 15, 15)
    assert m.conv is None
    assert m.convpool is not None


def test_norm_act_conv_kernel_size_1_skips_repvgg():
    """kernel_size=1 (bottleneck pre/post) shouldn't use RepVGG init."""
    m = NormActConv(c_in=16, c_out=8, kernel_size=1)
    assert m.use_repvgg_init is False


def test_norm_act_conv_initialize_sets_norm_scale():
    m = NormActConv(c_in=8, c_out=4, kernel_size=3)
    m.initialize(scale=1.0, norm_scale=0.5)
    assert m.norm.scale == 0.5
