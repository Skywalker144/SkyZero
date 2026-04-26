import pytest
import torch
from nets_v2 import ConvAndGPool


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
