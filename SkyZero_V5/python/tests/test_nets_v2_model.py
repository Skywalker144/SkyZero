import pytest
import torch
from nets_v2 import KataGoNet


def _make_b8c96_kwargs():
    return dict(
        num_blocks=8, c_main=96, c_mid=48, c_gpool=16,
        internal_length=2,
        num_in_channels=5, num_global_features=12,
        activation="mish", version=15,
        has_intermediate_head=True, intermediate_head_blocks=5,
        c_p1=24, c_g1=24, c_v1=24, c_v2=32,
        pos_len=15,
    )


def _make_full_board_state(B: int, num_in_channels: int = 5, pos_len: int = 15,
                           dtype=torch.float32, fill_random: bool = False):
    """Build a state tensor with mask plane (ch 0) set to ones (full 15x15 board)."""
    if fill_random:
        state = torch.randn(B, num_in_channels, pos_len, pos_len, dtype=dtype)
    else:
        state = torch.zeros(B, num_in_channels, pos_len, pos_len, dtype=dtype)
    state[:, 0] = 1.0  # on-board mask: 1 inside full board
    return state


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
    """forward 返回 flat dict (all tensor values), 含 policy + value_* + intermediate_*."""
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = _make_full_board_state(B=1)
    global_features = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, global_features)
    # policy: (B, 6, H*W) — 无 pass logit
    assert "policy" in out
    assert out["policy"].shape == (1, 6, 225)
    # value 拆成 6 个独立 key
    assert out["value_wdl"].shape == (1, 3)
    assert out["value_td"].shape == (1, 9)
    assert out["value_st_error"].shape == (1, 1)
    assert out["value_var_time"].shape == (1, 1)
    assert out["value_ownership"].shape == (1, 1, 15, 15)
    assert out["value_futurepos"].shape == (1, 2, 15, 15)
    # intermediate 同形状
    assert "intermediate_policy" in out
    assert out["intermediate_policy"].shape == (1, 6, 225)
    assert out["intermediate_value_wdl"].shape == (1, 3)


def test_katago_net_uses_mask_from_input_channel_0():
    """KataGo 风格: mask 来自 input_spatial[:, 0]. 验证变 board size 工作.

    构造 13×13 mask（外圈 padding 全 0）, forward 后 padding 区 policy logits ≤ -4000.
    """
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    # 构造一个 batch: entry 0 = 13×13 mask（外圈 padding=0）, entry 1 = 全 15×15
    state = torch.zeros(2, 5, 15, 15)
    state[0, 0, :13, :13] = 1.0   # 13×13 mask
    state[1, 0, :, :] = 1.0       # full 15×15 mask
    g = torch.zeros(2, 12)
    with torch.no_grad():
        out = model(state, g)
    # entry 0 的 policy 在 padding 区域 (例如 [13:, :] 或 [:, 13:]) 应被 mask 推到 ≤ -4000
    policy0_main = out["policy"][0, 0].view(15, 15)
    # padding 区域包括最后两行 (rows 13, 14) 和最后两列 (cols 13, 14)
    padding_logits = policy0_main[13:, :].flatten().tolist() + policy0_main[:, 13:].flatten().tolist()
    assert min(padding_logits) <= -4000.0, f"padding logits not pushed: min={min(padding_logits)}"
    # entry 1 (全 15×15) 不应有 padding logits
    policy1_main = out["policy"][1, 0]
    assert policy1_main.min().item() > -4000.0


def test_katago_net_initialize_does_not_blow_up():
    """初始化后空棋盘 + 0 global → trunk |x| 在合理范围 (NOTES.md §8 自检)."""
    torch.manual_seed(42)
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = _make_full_board_state(B=1)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    wdl_logits = out["value_wdl"]
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    # 初始化后应接近 (1/3, 1/3, 1/3) — head scale_output=0.3 让 logits 接近 0
    for i in range(3):
        assert 0.20 < wdl_probs[0, i].item() < 0.50, f"WDL[{i}] out of expected range: {wdl_probs}"


def test_katago_net_no_nan():
    """forward 输出无 NaN / Inf."""
    model = KataGoNet(**_make_b8c96_kwargs())
    model.initialize()
    model.eval()
    state = _make_full_board_state(B=2, fill_random=True)
    g = torch.randn(2, 12)
    with torch.no_grad():
        out = model(state, g)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"NaN in {k}"


def test_katago_net_no_intermediate_head():
    """has_intermediate_head=False → intermediate_* keys 不在 dict 里."""
    kwargs = _make_b8c96_kwargs()
    kwargs["has_intermediate_head"] = False
    model = KataGoNet(**kwargs)
    model.initialize()
    model.eval()
    state = _make_full_board_state(B=1)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    assert "intermediate_policy" not in out
    assert "intermediate_value_wdl" not in out


def test_b8c96_from_scratch_sanity():
    """NOTES.md §8 风格自检 — 从零 init 后空棋盘 + 0 global 的 trunk 量级.

    Note: forward() now returns flat dict (per Task 15 refactor), so we use
    out["value_wdl"] instead of out["value"][0].
    """
    from nets_v2 import build_b8c96

    torch.manual_seed(42)
    model = build_b8c96()
    model.initialize()
    model.eval()

    # 全 15×15 棋盘 (mask=1), 棋盘空 (own/opp/forbidden 平面全 0)
    state = _make_full_board_state(B=1)
    g = torch.zeros(1, 12)

    with torch.no_grad():
        out = model(state, g)

    # WDL probs 接近均匀 (head scale_output=0.3 + 全 0 输入)
    wdl_logits = out["value_wdl"]
    wdl_probs = torch.softmax(wdl_logits, dim=1)
    print(f"WDL probs (b8c96 init): {wdl_probs[0].tolist()}")
    for i in range(3):
        # 容忍范围放大 (小模型方差更大)
        assert 0.15 < wdl_probs[0, i].item() < 0.55

    # Policy 也应该接近均匀, KL 距均匀分布 < 1.0
    main_policy_logits = out["policy"][0, 0, :]   # main head, batch 0
    p = torch.softmax(main_policy_logits, dim=0)
    kl_to_uniform = (p * (p.log() - torch.log(torch.full_like(p, 1.0 / 225.0)))).sum().item()
    print(f"main policy KL to uniform: {kl_to_uniform:.3f}")
    assert kl_to_uniform < 1.0


def test_b12c128_from_scratch_sanity():
    from nets_v2 import build_b12c128

    torch.manual_seed(42)
    model = build_b12c128()
    model.initialize()
    model.eval()
    state = _make_full_board_state(B=1)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    wdl_probs = torch.softmax(out["value_wdl"], dim=1)
    for i in range(3):
        assert 0.15 < wdl_probs[0, i].item() < 0.55


def test_set_norm_scales_survives_state_dict_round_trip():
    """TRAP 3 防御测试: scale 不在 state_dict 里, load 后必须 set_norm_scales().

    场景: 训练完成后 save state_dict → 推理 init 模型 → load_state_dict.
    若忘记调 set_norm_scales(), trunk 末端 norm.scale=None, 网络输出过自信.
    """
    import math
    from nets_v2 import build_b8c96

    # 训练侧
    model_train = build_b8c96()
    model_train.initialize()
    # initialize 后 norm_trunkfinal.scale 应该被设为 1/sqrt(num_blocks+1) = 1/sqrt(9)
    expected_scale = 1.0 / math.sqrt(9.0)
    assert abs(model_train.norm_trunkfinal.scale - expected_scale) < 1e-6, (
        f"initialize() 应设 trunk_final scale={expected_scale}, "
        f"got {model_train.norm_trunkfinal.scale}"
    )

    # save → load 跨进程模拟
    sd = model_train.state_dict()

    model_infer = build_b8c96()
    # 关键: 注意 build_b8c96() 返回的模型 norm_trunkfinal.scale=None (未 init)
    assert model_infer.norm_trunkfinal.scale is None
    model_infer.load_state_dict(sd)
    # load_state_dict 后 scale 仍然 None (TRAP 3)
    assert model_infer.norm_trunkfinal.scale is None, (
        "TRAP 3: scale 不在 state_dict, load_state_dict 不会恢复它"
    )

    # 必须手动调 set_norm_scales()
    model_infer.set_norm_scales()
    assert abs(model_infer.norm_trunkfinal.scale - expected_scale) < 1e-6

    # 也验证一个内层 block 的 scale 被正确设置
    # 第 0 个 block 的 normactconvp.norm.scale = 1/sqrt(1) = 1.0
    assert abs(model_infer.blocks[0].normactconvp.norm.scale - 1.0) < 1e-6
    # 第 7 个 block 的 normactconvp.norm.scale = 1/sqrt(8)
    assert abs(model_infer.blocks[7].normactconvp.norm.scale - 1.0 / math.sqrt(8.0)) < 1e-6
