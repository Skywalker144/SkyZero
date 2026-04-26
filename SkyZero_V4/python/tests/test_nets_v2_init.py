import pytest
import torch
import math
from nets_v2 import compute_gain, init_weights


def test_compute_gain_mish():
    assert compute_gain("mish") == pytest.approx(math.sqrt(2.210277), abs=1e-6)


def test_compute_gain_relu():
    assert compute_gain("relu") == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_compute_gain_identity():
    assert compute_gain("identity") == 1.0


def test_compute_gain_unknown_raises():
    with pytest.raises(ValueError):
        compute_gain("foobar")


def test_init_weights_zero_when_scale_too_small():
    """When scale * gain / sqrt(fan_in) < 1e-10, zero out the tensor."""
    w = torch.randn(8, 8, 3, 3)
    init_weights(w, "mish", scale=1e-15)
    assert torch.all(w == 0.0)


def test_init_weights_truncated_normal():
    """Default path: trunc-normal with target_std = scale * gain / sqrt(fan_in)."""
    torch.manual_seed(42)
    w = torch.zeros(64, 32, 3, 3)
    init_weights(w, "mish", scale=1.0)
    # fan_in = 32*3*3 = 288; target_std = 1.0 * sqrt(2.21) / sqrt(288) ≈ 0.0876
    expected_std = math.sqrt(2.210277) / math.sqrt(288)
    actual_std = w.std().item()
    assert abs(actual_std - expected_std) / expected_std < 0.15
    assert w.abs().max() <= 4 * expected_std + 1e-3   # 截断在 ~2*std/0.88


def test_init_weights_uses_fan_tensor():
    """fan_tensor 用于 bias 初始化时取主 weight 的 fan_in."""
    torch.manual_seed(0)
    w_main = torch.zeros(16, 32)   # Linear weight, fan_in=32
    bias = torch.zeros(16)
    # 不传 fan_tensor → fan_in = 1 (bias 是 1D)
    init_weights(bias, "mish", scale=1.0)
    std_no_fan = bias.std().item()
    bias.fill_(0.0)
    init_weights(bias, "mish", scale=1.0, fan_tensor=w_main)   # fan_in=32
    std_with_fan = bias.std().item()
    assert std_with_fan < std_no_fan / 4.5   # fan_in=32 vs 1 → std 缩 sqrt(32)x, allow for random variation
