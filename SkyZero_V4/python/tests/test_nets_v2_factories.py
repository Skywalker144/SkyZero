import pytest
import torch
from nets_v2 import build_b8c96, build_b12c128


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
