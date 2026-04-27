import pytest
import torch
from nets_v2 import KataGPool, KataValueHeadGPool


def test_kata_gpool_output_shape():
    """3 统计量, 每个 (B, C, 1, 1) → cat → (B, 3*C, 1, 1)."""
    p = KataGPool()
    x = torch.randn(2, 8, 15, 15)
    mask = torch.ones(2, 1, 15, 15)
    out = p(x, mask)
    assert out.shape == (2, 24, 1, 1)   # 3 * 8


def test_kata_gpool_15x15_constants():
    """15×15 mask: mask_sum_hw=225, sqrt_off=sqrt(225)-14=1, board_factor=0.1."""
    p = KataGPool()
    x = torch.ones(1, 4, 15, 15) * 2.0   # 全 2
    mask = torch.ones(1, 1, 15, 15)
    out = p(x, mask)
    # ch 0-3: layer_mean = 2.0
    assert torch.allclose(out[0, 0:4], torch.full((4, 1, 1), 2.0))
    # ch 4-7: layer_mean * 0.1 = 0.2
    assert torch.allclose(out[0, 4:8], torch.full((4, 1, 1), 0.2))
    # ch 8-11: layer_max = 2.0
    assert torch.allclose(out[0, 8:12], torch.full((4, 1, 1), 2.0))


def test_kata_gpool_max_respects_mask():
    """Mask=0 cells should be excluded from max computation."""
    p = KataGPool()
    x = torch.ones(1, 1, 3, 3) * 1.0
    x[0, 0, 0, 0] = 99.0
    mask = torch.ones(1, 1, 3, 3)
    mask[0, 0, 0, 0] = 0.0   # 屏蔽掉那个 99
    out = p(x, mask)
    # ch 2 是 max, 应该是 1.0 (99 被 push 到 -∞ 等价)
    assert out[0, 2, 0, 0].item() < 99.0


def test_kata_gpool_with_explicit_mask_sum():
    """允许调用方传入 precomputed mask_sum_hw 避免重复 sum."""
    p = KataGPool()
    x = torch.ones(1, 2, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    mask_sum = torch.tensor([[[[225.0]]]])
    out = p(x, mask, mask_sum_hw=mask_sum)
    assert torch.allclose(out[0, 0:2], torch.ones(2, 1, 1))   # layer_mean=1.0


def test_kata_value_head_gpool_output_shape():
    p = KataValueHeadGPool()
    x = torch.randn(2, 8, 15, 15)
    mask = torch.ones(2, 1, 15, 15)
    out = p(x, mask)
    assert out.shape == (2, 24, 1, 1)


def test_kata_value_head_gpool_15x15_constants():
    """3 mean 统计量: mean, mean * (sqrt_off/10), mean * (sqrt_off²/100 - 0.1).

    15×15: sqrt_off=1 → channels = mean * [1, 0.1, 1/100 - 0.1] = mean * [1, 0.1, -0.09]
    """
    p = KataValueHeadGPool()
    x = torch.ones(1, 4, 15, 15) * 2.0
    mask = torch.ones(1, 1, 15, 15)
    out = p(x, mask)
    assert torch.allclose(out[0, 0:4], torch.full((4, 1, 1), 2.0))
    assert torch.allclose(out[0, 4:8], torch.full((4, 1, 1), 0.2))
    assert torch.allclose(out[0, 8:12], torch.full((4, 1, 1), 2.0 * (1.0/100.0 - 0.1)))
