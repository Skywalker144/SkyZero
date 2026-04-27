import pytest
import torch
from nets_v2 import build_b8c96, build_b12c128, build_model


def test_build_b8c96_param_count():
    model = build_b8c96()
    n = sum(p.numel() for p in model.parameters())
    assert 600_000 < n < 1_000_000, f"b8c96 param count {n} out of expected range"


def test_build_b12c128_param_count():
    model = build_b12c128()
    n = sum(p.numel() for p in model.parameters())
    assert 1_500_000 < n < 2_500_000, f"b12c128 param count {n} out of expected range"


def test_factories_produce_runnable_models():
    for build_fn in (build_b8c96, build_b12c128):
        model = build_fn()
        model.initialize()
        model.eval()
        state = torch.zeros(1, 4, 15, 15)
        g = torch.zeros(1, 12)
        with torch.no_grad():
            out = model(state, g)
        assert out["policy"].shape == (1, 6, 225)


def test_b8c96_has_8_blocks():
    model = build_b8c96()
    assert model.num_blocks == 8
    assert len(model.blocks) == 8


def test_b12c128_has_12_blocks():
    model = build_b12c128()
    assert model.num_blocks == 12
    assert len(model.blocks) == 12


def test_build_model_from_config():
    """build_model(cfg) is the generic configurable factory."""
    from model_config import NetConfig

    cfg = NetConfig(num_blocks=8, num_channels=96)
    model = build_model(cfg)
    assert model.num_blocks == 8
    n = sum(p.numel() for p in model.parameters())
    assert n > 100_000   # sanity: should be a real model


def test_netconfig_auto_derive_matches_b12c128_defaults():
    """When user sets num_blocks=12 num_channels=128 (defaults), all derived
    fields should equal the originally-hardcoded b12c128 values."""
    from model_config import NetConfig

    cfg = NetConfig()
    assert cfg.num_blocks == 12
    assert cfg.num_channels == 128
    assert cfg.c_mid == 64
    assert cfg.c_gpool == 16
    assert cfg.c_p1 == 32
    assert cfg.c_g1 == 32
    assert cfg.c_v1 == 32
    assert cfg.c_v2 == 48
    assert cfg.intermediate_head_blocks == 8


def test_netconfig_auto_derive_b8c96():
    """num_channels=96 should auto-derive sensible Phase A fields."""
    from model_config import NetConfig

    cfg = NetConfig(num_blocks=8, num_channels=96)
    assert cfg.c_mid == 48
    assert cfg.c_gpool == 16
    assert cfg.c_p1 == 24
    assert cfg.c_g1 == 24
    assert cfg.c_v1 == 24
    assert cfg.c_v2 == 40            # 24 + 16 (consistent with b12c128 ratio)
    assert cfg.intermediate_head_blocks == 5


def test_netconfig_explicit_overrides_honored():
    """User-passed values should not be overridden by __post_init__."""
    from model_config import NetConfig

    cfg = NetConfig(num_blocks=8, num_channels=96, c_v2=32, c_mid=24)
    assert cfg.c_v2 == 32
    assert cfg.c_mid == 24
