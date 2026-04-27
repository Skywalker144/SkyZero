import pytest
import torch
from nets_v2 import FixscaleNorm


def test_fixscale_norm_init_state():
    """gamma init = 0 (delta), beta init = 0, scale = None."""
    n = FixscaleNorm(num_channels=8, use_gamma=True)
    assert torch.all(n.gamma == 0.0)
    assert torch.all(n.beta == 0.0)
    assert n.scale is None


def test_fixscale_norm_forward_default_when_scale_none():
    """When scale is None, forward = x * (gamma + 1) + beta = x (since gamma=0, beta=0)."""
    n = FixscaleNorm(num_channels=4)
    x = torch.randn(2, 4, 5, 5)
    out = n(x)
    assert torch.allclose(out, x)


def test_fixscale_norm_uses_gamma_plus_one():
    """KataGo trap 2: gamma is delta, so forward must use (gamma + 1)."""
    n = FixscaleNorm(num_channels=4)
    n.gamma.data.fill_(0.5)        # 真实 scale 应该是 1.5
    n.beta.data.fill_(0.1)
    x = torch.ones(1, 4, 3, 3)
    out = n(x)
    expected = torch.full_like(x, 1.5 * 1.0 + 0.1)
    assert torch.allclose(out, expected)


def test_fixscale_norm_with_scale():
    """When scale is set, forward = x * (gamma + 1) * scale + beta."""
    n = FixscaleNorm(num_channels=4)
    n.set_scale(0.5)
    x = torch.ones(1, 4, 3, 3) * 2.0   # x = 2
    # gamma=0 → (gamma+1)=1; out = 2*1*0.5 + 0 = 1.0
    out = n(x)
    assert torch.allclose(out, torch.ones_like(x))


def test_fixscale_norm_mask():
    """mask=0 cells should be zeroed in output."""
    n = FixscaleNorm(num_channels=2)
    x = torch.ones(1, 2, 3, 3)
    mask = torch.ones(1, 1, 3, 3)
    mask[:, :, 0, 0] = 0.0
    out = n(x, mask=mask)
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 1, 1] == 1.0   # gamma=0+1=1, beta=0


def test_fixscale_norm_no_gamma():
    """use_gamma=False should drop gamma parameter."""
    n = FixscaleNorm(num_channels=4, use_gamma=False)
    assert n.gamma is None
    x = torch.ones(1, 4, 3, 3)
    out = n(x)
    assert torch.allclose(out, x)   # beta=0


from nets_v2 import BiasMask


def test_bias_mask_init():
    b = BiasMask(num_channels=4)
    assert torch.all(b.beta == 0.0)
    assert b.scale is None


def test_bias_mask_forward_default():
    b = BiasMask(num_channels=4)
    x = torch.randn(1, 4, 3, 3)
    out = b(x)
    assert torch.allclose(out, x)   # beta=0, scale=None


def test_bias_mask_with_scale():
    b = BiasMask(num_channels=4)
    b.set_scale(0.5)
    b.beta.data.fill_(0.1)
    x = torch.ones(1, 4, 3, 3) * 2.0
    out = b(x)
    expected = torch.full_like(x, 2.0 * 0.5 + 0.1)
    assert torch.allclose(out, expected)


def test_bias_mask_mask():
    b = BiasMask(num_channels=2)
    b.beta.data.fill_(0.5)
    x = torch.ones(1, 2, 3, 3)
    mask = torch.ones(1, 1, 3, 3)
    mask[:, :, 0, 0] = 0.0
    out = b(x, mask=mask)
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 1, 1] == 1.5


from nets_v2 import LastBatchNorm


def test_last_batch_norm_init():
    bn = LastBatchNorm(num_channels=4)
    assert torch.all(bn.gamma == 0.0)   # delta init
    assert torch.all(bn.beta == 0.0)
    assert torch.all(bn.running_mean == 0.0)
    assert torch.all(bn.running_std == 1.0)
    assert bn.scale is None


def test_last_batch_norm_train_mode_uses_batch_stats():
    """In train mode, normalize by mask-aware mini-batch mean/std."""
    bn = LastBatchNorm(num_channels=2)
    bn.train()
    # 构造 mask 全 1, 数据均值 5, std 2 (近似)
    x = torch.randn(8, 2, 3, 3) * 2.0 + 5.0
    mask = torch.ones(8, 1, 3, 3)
    out = bn(x, mask)
    # 标准化后均值应该接近 0 (×(0+1)*scale=None)
    out_mean = out.mean(dim=(0, 2, 3))
    assert torch.allclose(out_mean, torch.zeros(2), atol=1e-3)


def test_last_batch_norm_eval_mode_uses_running_stats():
    """In eval mode, use running_mean/running_std."""
    bn = LastBatchNorm(num_channels=2)
    bn.running_mean.fill_(10.0)
    bn.running_std.fill_(2.0)
    bn.eval()
    x = torch.full((1, 2, 3, 3), 14.0)
    mask = torch.ones(1, 1, 3, 3)
    out = bn(x, mask)
    # (14 - 10) / 2 = 2; gamma=0 → (gamma+1)=1; * mask
    assert torch.allclose(out, torch.full((1, 2, 3, 3), 2.0))


def test_last_batch_norm_running_stats_update():
    """Train mode should EMA-update running stats with momentum=0.001."""
    bn = LastBatchNorm(num_channels=1, momentum=0.5)   # 大 momentum 加速测试
    bn.train()
    x = torch.full((4, 1, 3, 3), 10.0)
    mask = torch.ones(4, 1, 3, 3)
    bn(x, mask)
    # mean=10, EMA: 0 + 0.5 * (10 - 0) = 5
    assert torch.allclose(bn.running_mean, torch.tensor([5.0]), atol=1e-4)
