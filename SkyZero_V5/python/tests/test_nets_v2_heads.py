import pytest
import torch
from nets_v2 import PolicyHead, ValueHead


def test_policy_head_v15_6_outputs():
    """v15 → 6 policy outputs (main/aux/soft/soft_aux/opt/opp)."""
    h = PolicyHead(c_in=128, c_p1=32, c_g1=32, activation="mish", version=15)
    assert h.num_policy_outputs == 6


def test_policy_head_no_pass_path():
    """SkyZero Gomoku 不需要 pass: 不应该有 linear_pass / linear_pass2 / act_pass."""
    h = PolicyHead(c_in=64, c_p1=24, c_g1=24, version=15)
    assert not hasattr(h, "linear_pass")
    assert not hasattr(h, "linear_pass2")
    assert not hasattr(h, "act_pass")


def test_policy_head_forward_shape():
    """Output shape: (B, 6, H*W) — 不再 cat outpass."""
    h = PolicyHead(c_in=64, c_p1=24, c_g1=24, version=15)
    x = torch.randn(2, 64, 15, 15)
    mask = torch.ones(2, 1, 15, 15)
    out = h(x, mask)
    assert out.shape == (2, 6, 225)


def test_policy_head_masks_invalid_logits():
    """Output - 5000.0 * (1 - mask) so masked cells get very negative logits."""
    h = PolicyHead(c_in=8, c_p1=4, c_g1=4, version=15)
    x = torch.randn(1, 8, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    mask[0, 0, 0, 0] = 0.0   # 屏蔽 (0,0)
    out = h(x, mask)
    # mask 外 cell (idx 0) 应该 ≤ -4000
    assert out[0, 0, 0].item() < -4000.0


def test_policy_head_initialize_runs():
    h = PolicyHead(c_in=16, c_p1=8, c_g1=8, version=15)
    h.initialize()
    assert h.conv2p.weight.abs().max() > 0


def test_value_head_no_go_specific_modules():
    """ValueHead 不应该有 score-belief / scoring / seki 模块."""
    h = ValueHead(c_in=64, c_v1=32, c_v2=48, pos_len=15)
    # 删去的 (KataGo 围棋特化):
    assert not hasattr(h, "linear_s2")
    assert not hasattr(h, "linear_s2off")
    assert not hasattr(h, "linear_s2par")
    assert not hasattr(h, "linear_s3")
    assert not hasattr(h, "linear_smix")
    assert not hasattr(h, "conv_scoring")
    assert not hasattr(h, "conv_seki")
    assert not hasattr(h, "score_belief_offset_vector")
    # 保留的 (Gomoku 适用):
    assert hasattr(h, "linear_valuehead")          # WDL
    assert hasattr(h, "linear_miscvaluehead")      # td_value 多 horizon
    assert hasattr(h, "conv_ownership")
    assert hasattr(h, "conv_futurepos")


def test_value_head_forward_returns_6_tuple():
    """返回 6 元 tuple: (wdl, td_value, shortterm_error, variance_time, ownership, futurepos)."""
    h = ValueHead(c_in=64, c_v1=32, c_v2=48, pos_len=15)
    x = torch.randn(2, 64, 15, 15)
    mask = torch.ones(2, 1, 15, 15)
    out = h(x, mask)
    assert isinstance(out, tuple) and len(out) == 6
    wdl, td_value, st_err, var_t, ownership, futurepos = out
    assert wdl.shape == (2, 3)
    assert td_value.shape == (2, 9)              # 3 horizons × 3 wdl
    assert st_err.shape == (2, 1)
    assert var_t.shape == (2, 1)
    assert ownership.shape == (2, 1, 15, 15)
    assert futurepos.shape == (2, 2, 15, 15)


def test_value_head_initialize_runs():
    h = ValueHead(c_in=16, c_v1=8, c_v2=12, pos_len=15)
    h.initialize()
    assert h.linear_valuehead.weight.abs().max() > 0
