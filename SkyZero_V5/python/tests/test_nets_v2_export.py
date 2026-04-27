import pytest
import torch
from nets_v2 import build_b8c96


def test_torchscript_trace_b8c96():
    """Trace KataGoNet 用 (state, global_features) 双输入。"""
    model = build_b8c96()
    model.initialize()
    model.eval()
    example_state = torch.zeros(1, 5, 15, 15, dtype=torch.float32)
    example_state[:, 0] = 1.0   # mask plane
    example_global = torch.zeros(1, 12, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)
    # Trace 后再调一次, 输出形状应一致
    out_eager = model(example_state, example_global)
    out_scripted = scripted(example_state, example_global)
    assert out_eager["policy"].shape == out_scripted["policy"].shape
    assert torch.allclose(out_eager["policy"], out_scripted["policy"], atol=1e-5)


def test_torchscript_trace_save_and_load(tmp_path):
    """Save → load round-trip."""
    model = build_b8c96()
    model.initialize()
    model.eval()
    example_state = torch.zeros(1, 5, 15, 15)
    example_state[:, 0] = 1.0   # mask plane
    example_global = torch.zeros(1, 12)
    with torch.no_grad():
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)
    save_path = tmp_path / "model.pt"
    scripted.save(str(save_path))
    loaded = torch.jit.load(str(save_path))
    out_loaded = loaded(example_state, example_global)
    assert out_loaded["policy"].shape == (1, 6, 225)
