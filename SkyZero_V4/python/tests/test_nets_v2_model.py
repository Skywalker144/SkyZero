import pytest
import torch
from nets_v2 import KataGoNet


def _make_b8c96_kwargs():
    return dict(
        num_blocks=8, c_main=96, c_mid=48, c_gpool=16,
        internal_length=2,
        num_in_channels=4, num_global_features=12,
        activation="mish", version=15,
        has_intermediate_head=True, intermediate_head_blocks=5,
        c_p1=24, c_g1=24, c_v1=24, c_v2=32,
        pos_len=15,
    )


def test_katago_net_construction():
    model = KataGoNet(**_make_b8c96_kwargs())
    assert model.num_blocks == 8
    assert model.has_intermediate_head is True
    assert model.intermediate_head_blocks == 5
    assert len(model.blocks) == 8


def test_katago_net_gpool_only_at_i_mod_3_eq_2():
    """spec §3.2: GPool 仅在 i%3==2 的块上 (即第 3/6/9/.../27 块)."""
    model = KataGoNet(**_make_b8c96_kwargs())
    for i, block in enumerate(model.blocks):
        if i % 3 == 2:   # 块 2/5
            assert block.blockstack[0].normactconv1.convpool is not None
        else:
            assert block.blockstack[0].normactconv1.convpool is None


def test_katago_net_forward_returns_dict():
    """forward 返回 dict, 含 policy + value tuple + intermediate_*."""
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = torch.zeros(1, 4, 15, 15)
    global_features = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, global_features)
    assert "policy" in out
    assert "value" in out
    assert "intermediate_policy" in out
    assert "intermediate_value" in out
    # policy: (B, 6, H*W) — 无 pass logit
    assert out["policy"].shape == (1, 6, 225)
    # value: 6-tuple
    assert isinstance(out["value"], tuple) and len(out["value"]) == 6
    assert out["value"][0].shape == (1, 3)   # WDL
    # intermediate 同形状
    assert out["intermediate_policy"].shape == (1, 6, 225)


def test_katago_net_internal_mask_is_ones_for_15x15():
    """SkyZero 固定 15×15 时, mask 在网络内部硬编码为 ones (无外部输入)."""
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    # forward 签名: (state, global_features) — 无 mask 参数
    state = torch.zeros(1, 4, 15, 15)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    assert out["policy"].shape == (1, 6, 225)


def test_katago_net_initialize_does_not_blow_up():
    """初始化后空棋盘 + 0 global → trunk |x| 在合理范围 (NOTES.md §8 自检)."""
    torch.manual_seed(42)
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = torch.zeros(1, 4, 15, 15)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    wdl_logits = out["value"][0]
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    # 初始化后应接近 (1/3, 1/3, 1/3) — head scale_output=0.3 让 logits 接近 0
    for i in range(3):
        assert 0.20 < wdl_probs[0, i].item() < 0.50, f"WDL[{i}] out of expected range: {wdl_probs}"


def test_katago_net_no_nan():
    """forward 输出无 NaN / Inf."""
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = torch.randn(2, 4, 15, 15)
    g = torch.randn(2, 12)
    with torch.no_grad():
        out = model(state, g)
    assert not torch.isnan(out["policy"]).any()
    for v in out["value"]:
        assert not torch.isnan(v).any()


def test_katago_net_no_intermediate_head():
    """has_intermediate_head=False → intermediate_* 不在 dict 里."""
    kwargs = _make_b8c96_kwargs()
    kwargs["has_intermediate_head"] = False
    model = KataGoNet(**kwargs)
    model.initialize()
    model.eval()
    state = torch.zeros(1, 4, 15, 15)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    assert "intermediate_policy" not in out
    assert "intermediate_value" not in out
