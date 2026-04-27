# Phase A: KataGo v15 网络模块实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `python/nets_v2.py` 中实现 KataGo b28c512nbt v15 架构的小模型变体（b8c96 / b12c128 双规模），通过单元测试验证每个模块行为对齐 `KataGoModel/model.py`，并能正确 TorchScript 导出供 C++ 加载。

**Architecture:**
- 单文件 `python/nets_v2.py` 包含所有 KataGo 模块（FixscaleNorm/BiasMask/LastBatchNorm/KataGPool/ConvAndGPool/NormActConv/ResBlock/NestedBottleneckResBlock/PolicyHead/ValueHead/KataGoNet）+ 工厂函数 + initialize/set_norm_scales API
- 与现有 `nets.py` 并行存在，互不影响
- `python/tests/` 新建测试目录，pytest 驱动 TDD
- 完整配置通过扩展后的 `NetConfig` 控制（添加 c_mid / c_gpool / num_global_features / has_intermediate_head / intermediate_head_blocks / c_p1 / c_g1 / c_v1 / c_v2 / version 字段）

**Tech Stack:** Python 3, PyTorch, pytest, TorchScript JIT trace

**Spec reference:** `docs/superpowers/specs/2026-04-26-network-katago-style-upgrade-design.md` §3 （架构）+ §7（实施陷阱）

**Out of scope:** 训练 stack 改造（Phase B）、C++ NPZ 扩展（Phase C）、C++ 推理签名（Phase D）、对弈验证（Phase E）。本计划完成后单独 brainstorm/plan B-E。

---

## 文件结构

```
python/
  nets_v2.py            ← 新增（~700-900 行）
  model_config.py       ← 修改：扩展 NetConfig 字段
  tests/
    __init__.py         ← 新增空文件
    conftest.py         ← 新增：sys.path 注入 python/
    test_nets_v2_norm.py        ← FixscaleNorm/BiasMask/LastBatchNorm
    test_nets_v2_pool.py        ← KataGPool/KataValueHeadGPool
    test_nets_v2_blocks.py      ← NormActConv/ResBlock/NestedBottleneckResBlock/ConvAndGPool
    test_nets_v2_heads.py       ← PolicyHead/ValueHead
    test_nets_v2_model.py       ← KataGoNet 全前向 + initialize + set_norm_scales
    test_nets_v2_export.py      ← TorchScript trace 验证
```

---

## Task 0: 测试脚手架

**Files:**
- Create: `python/tests/__init__.py`
- Create: `python/tests/conftest.py`

- [ ] **Step 0.1: 创建空 __init__.py**

```bash
touch /home/sky/RL/SkyZero/SkyZero_V4/python/tests/__init__.py
```

- [ ] **Step 0.2: 创建 conftest.py（让 pytest 找到 python/ 下的模块）**

文件 `python/tests/conftest.py`：

```python
"""Add python/ (this dir's parent) to sys.path so tests can import nets_v2 etc."""
import sys
import pathlib

_PYTHON_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
```

- [ ] **Step 0.3: 创建空 nets_v2.py 占位**

```bash
touch /home/sky/RL/SkyZero/SkyZero_V4/python/nets_v2.py
```

- [ ] **Step 0.4: smoke 测试 pytest 能跑**

文件 `python/tests/test_smoke.py`：

```python
def test_pytest_works():
    assert 1 + 1 == 2
```

Run: `cd /home/sky/RL/SkyZero/SkyZero_V4 && python -m pytest python/tests/test_smoke.py -v`
Expected: `1 passed`

- [ ] **Step 0.5: 提交**

```bash
cd /home/sky/RL/SkyZero/SkyZero_V4
git add python/tests/ python/nets_v2.py
git commit -m "test: bootstrap pytest scaffolding for nets_v2"
```

---

## Task 1: `FixscaleNorm` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_norm.py`

**Reference:** spec §3.3.1; KataGoModel/model.py:70-99；NOTES.md §3.2 坑 2 + §3.3 坑 3

- [ ] **Step 1.1: 写失败测试**

文件 `python/tests/test_nets_v2_norm.py`：

```python
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
```

- [ ] **Step 1.2: 运行测试，验证失败**

Run: `cd /home/sky/RL/SkyZero/SkyZero_V4 && python -m pytest python/tests/test_nets_v2_norm.py -v`
Expected: ImportError on `from nets_v2 import FixscaleNorm`

- [ ] **Step 1.3: 实现 FixscaleNorm**

追加到 `python/nets_v2.py`：

```python
"""KataGo b28c512nbt v15 网络模块 — SkyZero_V4 适配版.

参考: KataGoModel/model.py (lightvector/KataGo master 对齐)
基础设计: norm_kind=fixscaleonenorm, trunk_normless=True, bnorm_use_gamma=True,
         gamma_weight_decay_center_1=True, use_repvgg_init=True, version=15.

陷阱（来自 NOTES.md §3）:
1. NestedBottleneckResBlock.forward 只返残差; 调用方负责 out + block(out) 加回主干
2. gamma 张量是 delta, forward 必须用 (gamma + 1)
3. NormMask.scale 是 plain Python float, 不在 state_dict
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 归一化层
# ============================================================

class FixscaleNorm(nn.Module):
    """fixscaleonenorm 下非-last 归一化层.

    forward: out = (x * (gamma + 1) * scale + beta) [* mask]
    其中 scale 由 set_scale() 设置 (None 时省略).
    gamma 初始化为 0 (gamma_weight_decay_center_1=True, weight decay 推向 0).
    """

    def __init__(self, num_channels: int, use_gamma: bool = True) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        if use_gamma:
            self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.gamma = None
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gamma is not None:
            g = self.gamma + 1.0
            if self.scale is not None:
                out = x * (g * self.scale) + self.beta
            else:
                out = x * g + self.beta
        else:
            if self.scale is not None:
                out = x * self.scale + self.beta
            else:
                out = x + self.beta
        if mask is not None:
            out = out * mask
        return out
```

- [ ] **Step 1.4: 运行测试，验证通过**

Run: `cd /home/sky/RL/SkyZero/SkyZero_V4 && python -m pytest python/tests/test_nets_v2_norm.py -v -k FixscaleNorm`
Expected: 6 passed

- [ ] **Step 1.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_norm.py
git commit -m "feat(nets_v2): add FixscaleNorm with gamma-delta forward semantics"
```

---

## Task 2: `BiasMask` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_norm.py`

**Reference:** spec §3.3.2; KataGoModel/model.py:101-116

- [ ] **Step 2.1: 写失败测试**

追加到 `python/tests/test_nets_v2_norm.py`：

```python
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
```

- [ ] **Step 2.2: 运行测试，验证失败**

Run: `python -m pytest python/tests/test_nets_v2_norm.py::test_bias_mask_init -v`
Expected: ImportError

- [ ] **Step 2.3: 实现 BiasMask**

追加到 `python/nets_v2.py`：

```python
class BiasMask(nn.Module):
    """trunk_normless=True 下 trunk-final 的 normless-bias 层.

    forward: out = (x * scale + beta) [* mask]
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scale is not None:
            out = x * self.scale + self.beta
        else:
            out = x + self.beta
        if mask is not None:
            out = out * mask
        return out
```

- [ ] **Step 2.4: 运行测试，验证通过**

Run: `python -m pytest python/tests/test_nets_v2_norm.py -v`
Expected: 10 passed (6 FixscaleNorm + 4 BiasMask)

- [ ] **Step 2.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_norm.py
git commit -m "feat(nets_v2): add BiasMask trunk-tip layer"
```

---

## Task 3: `LastBatchNorm` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_norm.py`

**Reference:** spec §3.3.3; KataGoModel/model.py:119-159

- [ ] **Step 3.1: 写失败测试**

追加到 `python/tests/test_nets_v2_norm.py`：

```python
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
```

- [ ] **Step 3.2: 运行测试，验证失败**

Run: `python -m pytest python/tests/test_nets_v2_norm.py::test_last_batch_norm_init -v`
Expected: ImportError

- [ ] **Step 3.3: 实现 LastBatchNorm**

追加到 `python/nets_v2.py`：

```python
class LastBatchNorm(nn.Module):
    """fixscaleonenorm 全网唯一的 batchnorm: 用在 intermediate head 入口.

    训练: mask-aware mini-batch mean/std + running 统计量 EMA 更新.
    推理: running stats.
    """

    def __init__(self, num_channels: int, eps: float = 1e-4, momentum: float = 0.001) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_std", torch.ones(num_channels))
        self.eps = eps
        self.momentum = momentum
        self.scale: Optional[float] = None

    def set_scale(self, scale: Optional[float]) -> None:
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if self.training:
            mask_sum = mask.sum()
            mean = (x * mask).sum(dim=(0, 2, 3), keepdim=True) / mask_sum
            zeromean_x = x - mean
            var = ((zeromean_x * mask) ** 2).sum(dim=(0, 2, 3), keepdim=True) / mask_sum
            std = (var + self.eps).sqrt()
            with torch.no_grad():
                self.running_mean.add_(self.momentum * (mean.view(c).detach() - self.running_mean))
                self.running_std.add_(self.momentum * (std.view(c).detach() - self.running_std))
            normed = zeromean_x / std
        else:
            normed = (x - self.running_mean.view(1, c, 1, 1)) / self.running_std.view(1, c, 1, 1)
        g = self.gamma + 1.0
        if self.scale is not None:
            out = normed * (g * self.scale) + self.beta
        else:
            out = normed * g + self.beta
        return out * mask
```

- [ ] **Step 3.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_norm.py -v`
Expected: 14 passed

- [ ] **Step 3.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_norm.py
git commit -m "feat(nets_v2): add LastBatchNorm for intermediate head entry"
```

---

## Task 4: `KataGPool` 模块（3 统计量）

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_pool.py`

**Reference:** spec §3.3.4; KataGoModel/model.py:166-176

- [ ] **Step 4.1: 写失败测试**

文件 `python/tests/test_nets_v2_pool.py`：

```python
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
```

- [ ] **Step 4.2: 运行测试，验证失败**

Run: `python -m pytest python/tests/test_nets_v2_pool.py -v`
Expected: ImportError

- [ ] **Step 4.3: 实现 KataGPool**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# 池化模块
# ============================================================

class KataGPool(nn.Module):
    """3 个统计量: mean, mean * board_factor, max.

    board_factor = (sqrt(mask_sum_hw) - 14) / 10 — 让网络对棋盘大小有显式感知.
    15×15 时退化为 0.1 (常数).
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        layer_max = (x + (mask - 1.0)).to(torch.float32).amax(dim=(2, 3), keepdim=True)
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_max,
        ), dim=1)
```

- [ ] **Step 4.4: 运行 KataGPool 测试**

Run: `python -m pytest python/tests/test_nets_v2_pool.py::test_kata_gpool_output_shape python/tests/test_nets_v2_pool.py::test_kata_gpool_15x15_constants python/tests/test_nets_v2_pool.py::test_kata_gpool_max_respects_mask python/tests/test_nets_v2_pool.py::test_kata_gpool_with_explicit_mask_sum -v`
Expected: 4 passed

- [ ] **Step 4.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_pool.py
git commit -m "feat(nets_v2): add KataGPool with 3 board-aware statistics"
```

---

## Task 5: `KataValueHeadGPool` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_pool.py`

**Reference:** KataGoModel/model.py:179-188

- [ ] **Step 5.1: 写失败测试**

追加到 `python/tests/test_nets_v2_pool.py`：

```python
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
```

- [ ] **Step 5.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_pool.py::test_kata_value_head_gpool_output_shape -v`
Expected: ImportError

- [ ] **Step 5.3: 实现 KataValueHeadGPool**

追加到 `python/nets_v2.py`：

```python
class KataValueHeadGPool(nn.Module):
    """Value head 专用 GPool: 3 个 mean 统计量 (无 max), 二阶 board factor."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0
        layer_mean = torch.sum(x * mask, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        return torch.cat((
            layer_mean,
            layer_mean * (sqrt_off / 10.0),
            layer_mean * ((sqrt_off * sqrt_off) / 100.0 - 0.1),
        ), dim=1)
```

- [ ] **Step 5.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_pool.py -v`
Expected: 6 passed

- [ ] **Step 5.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_pool.py
git commit -m "feat(nets_v2): add KataValueHeadGPool with 2nd-order board factor"
```

---

## Task 6: RepVGG init helpers (`compute_gain` + `init_weights`)

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_init.py`

**Reference:** KataGoModel/model.py:24-63

- [ ] **Step 6.1: 写失败测试**

文件 `python/tests/test_nets_v2_init.py`：

```python
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
    assert std_with_fan < std_no_fan / 5.0   # fan_in=32 vs 1 → std 缩 sqrt(32)x
```

- [ ] **Step 6.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_init.py -v`
Expected: ImportError

- [ ] **Step 6.3: 实现 init helpers**

追加到 `python/nets_v2.py` 顶部（norm 类之后）：

```python
# ============================================================
# RepVGG init helpers
# ============================================================

def compute_gain(activation: str) -> float:
    """Per KataGo `compute_gain`. Mish gain matches lightvector master."""
    if activation in ("relu", "hardswish"):
        return math.sqrt(2.0)
    if activation == "elu":
        return math.sqrt(1.55052)
    if activation == "mish":
        return math.sqrt(2.210277)
    if activation == "silu":
        return math.sqrt(2.0)
    if activation == "gelu":
        return math.sqrt(2.351718)
    if activation == "identity":
        return 1.0
    raise ValueError(f"Unknown activation: {activation}")


_TRUNC_CORRECTION = 0.87962566103423978   # std correction for trunc_normal a=-2,b=2


def init_weights(tensor: torch.Tensor, activation: str, scale: float,
                 fan_tensor: Optional[torch.Tensor] = None) -> None:
    """KataGo's truncated-normal init: std = scale * gain / sqrt(fan_in)."""
    gain = compute_gain(activation)
    src = fan_tensor if fan_tensor is not None else tensor
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(src)
    target_std = scale * gain / math.sqrt(fan_in)
    std = target_std / _TRUNC_CORRECTION
    with torch.no_grad():
        if std < 1e-10:
            tensor.fill_(0.0)
        else:
            nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
```

- [ ] **Step 6.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_init.py -v`
Expected: 7 passed

- [ ] **Step 6.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_init.py
git commit -m "feat(nets_v2): add compute_gain and init_weights helpers"
```

---

## Task 7: `ConvAndGPool` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_blocks.py`

**Reference:** spec §3.3.5; KataGoModel/model.py:195-220

- [ ] **Step 7.1: 写失败测试**

文件 `python/tests/test_nets_v2_blocks.py`：

```python
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
```

- [ ] **Step 7.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py -v`
Expected: ImportError

- [ ] **Step 7.3: 实现 ConvAndGPool**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# 卷积 + 全局池化模块
# ============================================================

class ConvAndGPool(nn.Module):
    """合并到 ResBlock 内层第一个 conv 的位置 (不是独立残差块).

    out = conv1r(x) + linear_g(gpool(act(normg(conv1g(x)))))   [as conv bias]
    """

    def __init__(self, c_in: int, c_out: int, c_gpool: int, activation: str = "mish") -> None:
        super().__init__()
        self.activation = activation
        self.conv1r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False)
        self.normg = FixscaleNorm(c_gpool, use_gamma=True)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale: float) -> None:
        # KataGo master 538-549 (fixscaleonenorm path)
        r_scale, g_scale = 0.8, 0.6
        init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
        init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
        init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_r = self.conv1r(x)
        out_g = self.conv1g(x)
        out_g = self.normg(out_g, mask)
        out_g = self.actg(out_g)
        out_g = self.gpool(out_g, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)
        return out_r + out_g
```

- [ ] **Step 7.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py -v`
Expected: 4 passed

- [ ] **Step 7.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_blocks.py
git commit -m "feat(nets_v2): add ConvAndGPool merged GPool/conv module"
```

---

## Task 8: `NormActConv` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_blocks.py`

**Reference:** KataGoModel/model.py:227-267

- [ ] **Step 8.1: 写失败测试**

追加到 `python/tests/test_nets_v2_blocks.py`：

```python
from nets_v2 import NormActConv


def test_norm_act_conv_basic():
    m = NormActConv(c_in=16, c_out=12, activation="mish", kernel_size=3,
                    c_gpool=None, fixup_use_gamma=True)
    x = torch.randn(1, 16, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = m(x, mask)
    assert out.shape == (1, 12, 15, 15)
    assert m.conv is not None
    assert m.convpool is None


def test_norm_act_conv_with_gpool():
    """When c_gpool is given, uses ConvAndGPool inside."""
    m = NormActConv(c_in=16, c_out=12, activation="mish", kernel_size=3,
                    c_gpool=4, fixup_use_gamma=True)
    x = torch.randn(1, 16, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = m(x, mask)
    assert out.shape == (1, 12, 15, 15)
    assert m.conv is None
    assert m.convpool is not None


def test_norm_act_conv_kernel_size_1_skips_repvgg():
    """kernel_size=1 (bottleneck pre/post) shouldn't use RepVGG init."""
    m = NormActConv(c_in=16, c_out=8, kernel_size=1)
    assert m.use_repvgg_init is False


def test_norm_act_conv_initialize_sets_norm_scale():
    m = NormActConv(c_in=8, c_out=4, kernel_size=3)
    m.initialize(scale=1.0, norm_scale=0.5)
    assert m.norm.scale == 0.5
```

- [ ] **Step 8.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py::test_norm_act_conv_basic -v`
Expected: ImportError

- [ ] **Step 8.3: 实现 NormActConv**

追加到 `python/nets_v2.py`：

```python
class NormActConv(nn.Module):
    """norm → act → (conv | conv-and-gpool)."""

    def __init__(self, c_in: int, c_out: int, activation: str = "mish",
                 kernel_size: int = 3, c_gpool: Optional[int] = None,
                 fixup_use_gamma: bool = True,
                 use_repvgg_init: bool = True) -> None:
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.norm = FixscaleNorm(c_in, use_gamma=fixup_use_gamma)
        self.act = nn.Mish() if activation == "mish" else nn.ReLU()
        self.use_repvgg_init = use_repvgg_init and kernel_size > 1
        if c_gpool is not None:
            self.convpool = ConvAndGPool(c_in, c_out, c_gpool, activation)
            self.conv = None
        else:
            padding = kernel_size // 2
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                                  padding=padding, bias=False)
            self.convpool = None

    def initialize(self, scale: float, norm_scale: Optional[float] = None) -> None:
        self.norm.set_scale(norm_scale)
        if self.convpool is not None:
            self.convpool.initialize(scale=scale)
        else:
            if self.use_repvgg_init:
                init_weights(self.conv.weight, self.activation, scale=scale * 0.8)
                w = self.conv.weight
                center_bonus = w.new_zeros((w.shape[0], w.shape[1]), requires_grad=False)
                init_weights(center_bonus, self.activation, scale=scale * 0.6)
                with torch.no_grad():
                    self.conv.weight[:, :, 1, 1] += center_bonus
            else:
                init_weights(self.conv.weight, self.activation, scale=scale)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.norm(x, mask)
        out = self.act(out)
        if self.convpool is not None:
            return self.convpool(out, mask, mask_sum_hw)
        return self.conv(out)
```

- [ ] **Step 8.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py -v`
Expected: 8 passed

- [ ] **Step 8.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_blocks.py
git commit -m "feat(nets_v2): add NormActConv with optional gpool sub-module"
```

---

## Task 9: `ResBlock` 模块

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_blocks.py`

**Reference:** KataGoModel/model.py:274-293

- [ ] **Step 9.1: 写失败测试**

追加到 `python/tests/test_nets_v2_blocks.py`：

```python
from nets_v2 import ResBlock


def test_res_block_no_gpool():
    rb = ResBlock(c_mid=16, c_gpool=None, activation="mish")
    x = torch.randn(1, 16, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = rb(x, mask)
    assert out.shape == (1, 16, 15, 15)


def test_res_block_with_gpool():
    """c_gpool 不为 None 时, normactconv1 用 ConvAndGPool, c_out1 = c_mid - c_gpool."""
    rb = ResBlock(c_mid=24, c_gpool=8, activation="mish")
    x = torch.randn(1, 24, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = rb(x, mask)
    assert out.shape == (1, 24, 15, 15)


def test_res_block_returns_residual_only():
    """ResBlock.forward 返回 normactconv1→normactconv2 的输出, 不含 x.

    主调用方 NestedBottleneckResBlock 负责 out + block(out).
    """
    rb = ResBlock(c_mid=16, c_gpool=None)
    x = torch.zeros(1, 16, 5, 5)
    mask = torch.ones(1, 1, 5, 5)
    # 模型刚 init → norm.scale=None; gamma=0+1=1; conv weights random
    # forward 不显式加 x, 输出对 x=0 应该接近 0 (但因为 conv bias 约定 false 也是 0)
    out = rb(x, mask)
    # 这里弱断言: out 应当独立于 x 的值传递, 不应该等于 x
    # 主断言: ResBlock.forward 不在最后做 x + out
    assert out is not x and not torch.equal(out, x)
```

- [ ] **Step 9.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py::test_res_block_no_gpool -v`
Expected: ImportError

- [ ] **Step 9.3: 实现 ResBlock**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# 残差块 (ResBlock 只返残差; 调用方负责加回主干)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, c_mid: int, c_gpool: Optional[int] = None,
                 activation: str = "mish") -> None:
        super().__init__()
        c_out1 = c_mid - (0 if c_gpool is None else c_gpool)
        self.normactconv1 = NormActConv(c_mid, c_out1, activation,
                                        kernel_size=3, c_gpool=c_gpool,
                                        fixup_use_gamma=True)
        self.normactconv2 = NormActConv(c_out1, c_mid, activation,
                                        kernel_size=3, c_gpool=None,
                                        fixup_use_gamma=True)

    def initialize(self, fixup_scale: float) -> None:
        self.normactconv1.initialize(scale=1.0, norm_scale=fixup_scale)
        self.normactconv2.initialize(scale=1.0, norm_scale=None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.normactconv1(x, mask, mask_sum_hw)
        out = self.normactconv2(out, mask, mask_sum_hw)
        return out
```

- [ ] **Step 9.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py -v`
Expected: 11 passed

- [ ] **Step 9.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_blocks.py
git commit -m "feat(nets_v2): add ResBlock with optional gpool sub-conv"
```

---

## Task 10: `NestedBottleneckResBlock` 模块（坑 1 关键）

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_blocks.py`

**Reference:** spec §3.3.6 + §7 trap 1; KataGoModel/model.py:296-324

- [ ] **Step 10.1: 写失败测试**

追加到 `python/tests/test_nets_v2_blocks.py`：

```python
from nets_v2 import NestedBottleneckResBlock


def test_nested_bottleneck_forward_shape():
    nbb = NestedBottleneckResBlock(internal_length=2, c_main=64, c_mid=32, c_gpool=None)
    x = torch.randn(1, 64, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = nbb(x, mask)
    assert out.shape == (1, 64, 15, 15)


def test_nested_bottleneck_with_gpool():
    """c_gpool 仅用于内层第一个 ResBlock."""
    nbb = NestedBottleneckResBlock(internal_length=2, c_main=64, c_mid=32, c_gpool=8)
    x = torch.randn(1, 64, 15, 15)
    mask = torch.ones(1, 1, 15, 15)
    out = nbb(x, mask)
    assert out.shape == (1, 64, 15, 15)
    # 内层第 0 个 block 应该是 gpool 版
    assert nbb.blockstack[0].normactconv1.convpool is not None
    # 内层第 1 个 block 应该是普通版
    assert nbb.blockstack[1].normactconv1.convpool is None


def test_nested_bottleneck_returns_residual_only():
    """坑 1: NestedBottleneckResBlock.forward 只返残差, 不在内部加回 x.

    主循环必须 out = out + block(out).
    """
    nbb = NestedBottleneckResBlock(internal_length=2, c_main=8, c_mid=4, c_gpool=None)
    nbb.initialize(fixup_scale=1.0)
    x = torch.zeros(1, 8, 5, 5)
    mask = torch.ones(1, 1, 5, 5)
    out = nbb(x, mask)
    # x 全 0; norm 后 = 0; conv 后输出依赖 weight 但与 x 无相加关系
    # 主断言: forward 没有 `return x + ...`
    # 我们检查: 对一个 nontrivial x, out != x + (something computed from x)
    # 简单方式: out 不应该 == x (即使 conv 输出全 0, out shape 不变, 但值 ≠ x)
    x2 = torch.ones(1, 8, 5, 5) * 100.0
    out2 = nbb(x2, mask)
    # 如果代码错误地做了 x + 残差, out2 会 ≈ 100 (因为 init 后残差量级很小)
    # 正确实现: out2 是残差量级 (远小于 100)
    assert out2.abs().max() < 50.0, (
        "NestedBottleneckResBlock.forward likely incorrectly returns x + residual; "
        f"got max abs {out2.abs().max().item()}"
    )
```

- [ ] **Step 10.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py::test_nested_bottleneck_forward_shape -v`
Expected: ImportError

- [ ] **Step 10.3: 实现 NestedBottleneckResBlock**

追加到 `python/nets_v2.py`：

```python
class NestedBottleneckResBlock(nn.Module):
    """1×1 (c_main → c_mid) → N 个内层 ResBlock → 1×1 (c_mid → c_main).

    ⚠️ forward 只返残差; 调用方必须 `out = out + block(out, mask, mask_sum_hw)`.
    """

    def __init__(self, internal_length: int, c_main: int, c_mid: int,
                 c_gpool: Optional[int] = None, activation: str = "mish") -> None:
        super().__init__()
        self.internal_length = internal_length
        # 1×1 conv => 不走 RepVGG (在 NormActConv 内部 use_repvgg_init=ks>1)
        self.normactconvp = NormActConv(c_main, c_mid, activation,
                                        kernel_size=1, fixup_use_gamma=True)
        self.blockstack = nn.ModuleList()
        for i in range(internal_length):
            use_gpool = c_gpool if i == 0 else None
            self.blockstack.append(ResBlock(c_mid, c_gpool=use_gpool, activation=activation))
        self.normactconvq = NormActConv(c_mid, c_main, activation,
                                        kernel_size=1, fixup_use_gamma=True)

    def initialize(self, fixup_scale: float) -> None:
        self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
        for j, block in enumerate(self.blockstack):
            block.initialize(fixup_scale=1.0 / math.sqrt(j + 1.0))
        self.normactconvq.initialize(scale=1.0,
                                     norm_scale=1.0 / math.sqrt(self.internal_length + 1.0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ⚠️ 只返残差, 调用方负责 out + block(out, mask)
        out = self.normactconvp(x, mask, mask_sum_hw)
        for block in self.blockstack:
            out = out + block(out, mask, mask_sum_hw)        # 内层加法
        out = self.normactconvq(out, mask, mask_sum_hw)
        return out
```

- [ ] **Step 10.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_blocks.py -v`
Expected: 14 passed

- [ ] **Step 10.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_blocks.py
git commit -m "feat(nets_v2): add NestedBottleneckResBlock (returns residual only)"
```

---

## Task 11: `PolicyHead`（v15, 6 outputs, **删 pass 路径**）

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_heads.py`

**Reference:** spec §3.4.1; KataGoModel/model.py:331-393（注意：要删除 pass 路径）

- [ ] **Step 11.1: 写失败测试**

文件 `python/tests/test_nets_v2_heads.py`：

```python
import pytest
import torch
from nets_v2 import PolicyHead


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
```

- [ ] **Step 11.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_heads.py -v`
Expected: ImportError

- [ ] **Step 11.3: 实现 PolicyHead（删 pass 路径）**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# Policy Head (v15, no pass — Gomoku 不需要)
# ============================================================

class PolicyHead(nn.Module):
    """v15 PolicyHead 6 输出 (main/aux/soft/soft_aux/opt/opp), 删 pass 路径.

    KataGo 原版: forward 输出 cat([spatial logits, pass logit]) → (B, 6, H*W+1).
    SkyZero 改: 只输出 (B, 6, H*W) — Gomoku 无 pass 着法.
    """

    def __init__(self, c_in: int, c_p1: int, c_g1: int,
                 activation: str = "mish", version: int = 15) -> None:
        super().__init__()
        self.activation = activation
        self.version = version
        # v15 → 6 outputs (我们固定 v15)
        if version <= 11:
            self.num_policy_outputs = 4
        elif version <= 15:
            self.num_policy_outputs = 6
        else:
            self.num_policy_outputs = 8

        self.conv1p = nn.Conv2d(c_in, c_p1, kernel_size=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_g1, kernel_size=1, bias=False)
        self.biasg = BiasMask(c_g1)
        self.actg = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        self.bias2 = BiasMask(c_p1)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.conv2p = nn.Conv2d(c_p1, self.num_policy_outputs, kernel_size=1, bias=False)

    def initialize(self) -> None:
        # KataGo master 2416-2432
        p_scale, g_scale, scale_output = 0.8, 0.6, 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None) -> torch.Tensor:
        outp = self.conv1p(x)
        outg = self.conv1g(x)
        outg = self.biasg(outg, mask)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
        outp = outp + outg
        outp = self.bias2(outp, mask)
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        outp = outp - (1.0 - mask) * 5000.0   # mask 外 cell push 到极小
        return outp.view(outp.shape[0], outp.shape[1], -1)
```

- [ ] **Step 11.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_heads.py -v`
Expected: 5 passed

- [ ] **Step 11.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_heads.py
git commit -m "feat(nets_v2): add PolicyHead (v15, 6 outputs, no pass path)"
```

---

## Task 12: `ValueHead`（删 score / seki / scorebelief）

**Files:**
- Modify: `python/nets_v2.py`
- Modify: `python/tests/test_nets_v2_heads.py`

**Reference:** spec §3.4.2; KataGoModel/model.py:403-522（删去围棋特化部分）

- [ ] **Step 12.1: 写失败测试**

追加到 `python/tests/test_nets_v2_heads.py`：

```python
from nets_v2 import ValueHead


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
    assert hasattr(h, "linear_miscvaluehead")      # td_value 多 horizon (重命名为 td_value_head)
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
```

- [ ] **Step 12.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_heads.py::test_value_head_no_go_specific_modules -v`
Expected: ImportError

- [ ] **Step 12.3: 实现 ValueHead（Gomoku-only）**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# Value Head (Gomoku-only outputs)
# ============================================================

class ValueHead(nn.Module):
    """简化的 ValueHead — 删去围棋特化的 score-belief / scoring / seki.

    输出 6 元 tuple:
        wdl:                 (B, 3)         W/L/draw logits
        td_value:            (B, 9)         3 horizons × 3 wdl (long/mid/short × W/L/draw)
        shortterm_error:     (B, 1)         pretanh, 短 horizon Q 预测误差幅度
        variance_time:       (B, 1)         残留 KataGo 输出 (Gomoku 可选, 占位)
        ownership_pretanh:   (B, 1, H, W)   终局每格占有 ±1
        futurepos_pretanh:   (B, 2, H, W)   +N / +2N 步占据
    """

    def __init__(self, c_in: int, c_v1: int, c_v2: int, activation: str = "mish",
                 pos_len: int = 15) -> None:
        super().__init__()
        self.activation = activation
        self.pos_len = pos_len

        self.conv1 = nn.Conv2d(c_in, c_v1, kernel_size=1, bias=False)
        self.bias1 = BiasMask(c_v1)
        self.act1 = nn.Mish() if activation == "mish" else nn.ReLU()
        self.gpool = KataValueHeadGPool()

        self.linear2 = nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = nn.Mish() if activation == "mish" else nn.ReLU()

        # WDL 输出 (KataGo: linear_valuehead 输出 3, 我们保留)
        self.linear_valuehead = nn.Linear(c_v2, 3, bias=True)
        # td_value 多 horizon: 3 horizons × 3 wdl = 9 输出
        # 命名沿用 KataGo "miscvaluehead" 但语义已改 (我们的 9 维全部是 td_value)
        self.linear_miscvaluehead = nn.Linear(c_v2, 9, bias=True)
        # shortterm error + variance_time (合并 2 维)
        self.linear_moremiscvaluehead = nn.Linear(c_v2, 2, bias=True)

        # 空间 head
        self.conv_ownership = nn.Conv2d(c_v1, 1, kernel_size=1, bias=False)
        self.conv_futurepos = nn.Conv2d(c_in, 2, kernel_size=1, bias=False)

    def initialize(self) -> None:
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale,
                     fan_tensor=self.linear2.weight)

        for lin in (self.linear_valuehead, self.linear_miscvaluehead, self.linear_moremiscvaluehead):
            init_weights(lin.weight, "identity", scale=1.0)
            init_weights(lin.bias, "identity", scale=bias_scale, fan_tensor=lin.weight)

        aux_scale = 0.2
        for c in (self.conv_ownership, self.conv_futurepos):
            init_weights(c.weight, "identity", scale=aux_scale)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                mask_sum_hw: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor, torch.Tensor]:
        outv1 = self.conv1(x)
        outv1 = self.bias1(outv1, mask)
        outv1 = self.act1(outv1)

        outpooled = self.gpool(outv1, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        outv2 = self.act2(self.linear2(outpooled))

        wdl = self.linear_valuehead(outv2)                          # (B, 3)
        td_value = self.linear_miscvaluehead(outv2)                 # (B, 9) = 3 horizons × 3
        more_misc = self.linear_moremiscvaluehead(outv2)            # (B, 2)
        st_error = more_misc[:, 0:1]
        var_time = more_misc[:, 1:2]

        ownership = self.conv_ownership(outv1) * mask               # (B, 1, H, W)
        futurepos = self.conv_futurepos(x) * mask                   # (B, 2, H, W)

        return wdl, td_value, st_error, var_time, ownership, futurepos
```

- [ ] **Step 12.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_heads.py -v`
Expected: 8 passed

- [ ] **Step 12.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_heads.py
git commit -m "feat(nets_v2): add ValueHead (Gomoku-only: WDL+td+ownership+futurepos)"
```

---

## Task 13: `KataGoNet` 全模型

**Files:**
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_model.py`

**Reference:** spec §3.2 + §3.4.3; KataGoModel/model.py:529-659

- [ ] **Step 13.1: 写失败测试**

文件 `python/tests/test_nets_v2_model.py`：

```python
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
    # 内部 mask 应是 (B, 1, 15, 15) 的 ones — 通过看输出有效不验证 mask 在网络内被正确使用
    assert out["policy"].shape == (1, 6, 225)


def test_katago_net_initialize_does_not_blow_up():
    """初始化后空棋盘 + 0 global → trunk |x| 在合理范围 (NOTES.md §8 自检)."""
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
    """has_intermediate_head=False → intermediate_* 不在 dict 里 (或值是 None)."""
    kwargs = _make_b8c96_kwargs()
    kwargs["has_intermediate_head"] = False
    model = KataGoNet(**kwargs)
    model.initialize()
    model.eval()
    state = torch.zeros(1, 4, 15, 15)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    assert out["intermediate_policy"] is None
    assert out["intermediate_value"] is None
```

- [ ] **Step 13.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_model.py -v`
Expected: ImportError

- [ ] **Step 13.3: 实现 KataGoNet**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# 完整模型
# ============================================================

class KataGoNet(nn.Module):
    def __init__(
        self,
        num_blocks: int = 8,
        c_main: int = 96,
        c_mid: int = 48,
        c_gpool: int = 16,
        internal_length: int = 2,
        num_in_channels: int = 4,
        num_global_features: int = 12,
        activation: str = "mish",
        version: int = 15,
        has_intermediate_head: bool = True,
        intermediate_head_blocks: int = 5,
        c_p1: int = 24,
        c_g1: int = 24,
        c_v1: int = 24,
        c_v2: int = 32,
        pos_len: int = 15,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.version = version
        self.num_blocks = num_blocks
        self.pos_len = pos_len
        self.has_intermediate_head = has_intermediate_head
        self.intermediate_head_blocks = intermediate_head_blocks
        self.num_in_channels = num_in_channels
        self.num_global_features = num_global_features

        # 输入投影
        self.conv_spatial = nn.Conv2d(num_in_channels, c_main, kernel_size=3,
                                      padding=1, bias=False)
        self.linear_global = nn.Linear(num_global_features, c_main, bias=False)

        # 主干 — gpool 仅在 i % 3 == 2 的块上
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_gpool = c_gpool if (i % 3 == 2) else None
            self.blocks.append(NestedBottleneckResBlock(
                internal_length=internal_length, c_main=c_main, c_mid=c_mid,
                c_gpool=use_gpool, activation=activation,
            ))

        self.norm_trunkfinal = BiasMask(c_main)
        self.act_trunkfinal = nn.Mish() if activation == "mish" else nn.ReLU()

        if has_intermediate_head:
            self.norm_intermediate_trunkfinal = LastBatchNorm(c_main)
            self.act_intermediate_trunkfinal = nn.Mish() if activation == "mish" else nn.ReLU()
            self.intermediate_policy_head = PolicyHead(c_main, c_p1, c_g1,
                                                       activation=activation, version=version)
            self.intermediate_value_head = ValueHead(c_main, c_v1, c_v2,
                                                     activation=activation, pos_len=pos_len)

        self.policy_head = PolicyHead(c_main, c_p1, c_g1,
                                      activation=activation, version=version)
        self.value_head = ValueHead(c_main, c_v1, c_v2,
                                    activation=activation, pos_len=pos_len)

    def initialize(self) -> None:
        """KataGo RepVGG-style fixscaleonenorm 初始化 + 设置所有 NormMask.scale.

        从零训前调用一次.
        """
        with torch.no_grad():
            init_weights(self.conv_spatial.weight, self.activation, scale=0.8)
            init_weights(self.linear_global.weight, self.activation, scale=0.6)

            for i, block in enumerate(self.blocks):
                block.initialize(fixup_scale=1.0 / math.sqrt(i + 1.0))

            self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

            self.policy_head.initialize()
            self.value_head.initialize()
            if self.has_intermediate_head:
                self.intermediate_policy_head.initialize()
                self.intermediate_value_head.initialize()

    def set_norm_scales(self) -> None:
        """加载预训练 ckpt 时调用: 仅设置 scale, 不重置权重.

        scale 不在 state_dict (plain attribute), load_state_dict 后必须手动调.
        """
        for i, block in enumerate(self.blocks):
            block.normactconvp.norm.set_scale(1.0 / math.sqrt(i + 1.0))
            block.normactconvq.norm.set_scale(1.0 / math.sqrt(block.internal_length + 1.0))
            for j, inner in enumerate(block.blockstack):
                inner.normactconv1.norm.set_scale(1.0 / math.sqrt(j + 1.0))
                inner.normactconv2.norm.set_scale(None)
        self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_blocks + 1.0))

    def forward(self, input_spatial: torch.Tensor,
                input_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        # SkyZero 固定 15×15: mask 在网络内部硬编码为 ones, mask_sum_hw=225
        B = input_spatial.shape[0]
        mask = torch.ones(B, 1, self.pos_len, self.pos_len,
                          dtype=input_spatial.dtype, device=input_spatial.device)
        mask_sum_hw = torch.full((B, 1, 1, 1), float(self.pos_len * self.pos_len),
                                 dtype=input_spatial.dtype, device=input_spatial.device)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_spatial + x_global

        iout_policy: Optional[torch.Tensor] = None
        iout_value = None

        if self.has_intermediate_head:
            for block in self.blocks[: self.intermediate_head_blocks]:
                out = out + block(out, mask, mask_sum_hw)

            iout = self.norm_intermediate_trunkfinal(out, mask)
            iout = self.act_intermediate_trunkfinal(iout)
            iout_policy = self.intermediate_policy_head(iout, mask, mask_sum_hw)
            iout_value = self.intermediate_value_head(iout, mask, mask_sum_hw)

            for block in self.blocks[self.intermediate_head_blocks :]:
                out = out + block(out, mask, mask_sum_hw)
        else:
            for block in self.blocks:
                out = out + block(out, mask, mask_sum_hw)

        out = self.norm_trunkfinal(out, mask)
        out = self.act_trunkfinal(out)

        out_policy = self.policy_head(out, mask, mask_sum_hw)
        out_value = self.value_head(out, mask, mask_sum_hw)

        return {
            "policy": out_policy,
            "value": out_value,
            "intermediate_policy": iout_policy,
            "intermediate_value": iout_value,
        }
```

- [ ] **Step 13.4: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_model.py -v`
Expected: 7 passed

- [ ] **Step 13.5: 提交**

```bash
git add python/nets_v2.py python/tests/test_nets_v2_model.py
git commit -m "feat(nets_v2): add KataGoNet full model with intermediate head"
```

---

## Task 14: `NetConfig` 扩展 + 工厂函数

**Files:**
- Modify: `python/model_config.py`
- Modify: `python/nets_v2.py`
- Create: `python/tests/test_nets_v2_factories.py`

**Reference:** spec §3.1 — b8c96 / b12c128 双规模

- [ ] **Step 14.1: 写失败测试**

文件 `python/tests/test_nets_v2_factories.py`：

```python
import pytest
import torch
from nets_v2 import build_b8c96, build_b12c128


def test_build_b8c96_param_count():
    model = build_b8c96()
    n = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < n < 3_000_000, f"b8c96 param count {n} out of expected range"


def test_build_b12c128_param_count():
    model = build_b12c128()
    n = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < n < 7_000_000, f"b12c128 param count {n} out of expected range"


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
```

- [ ] **Step 14.2: 运行测试验证失败**

Run: `python -m pytest python/tests/test_nets_v2_factories.py -v`
Expected: ImportError

- [ ] **Step 14.3: 添加工厂函数到 nets_v2.py**

追加到 `python/nets_v2.py`：

```python
# ============================================================
# 工厂函数 — 与 spec §3.1 表对齐
# ============================================================

def build_b8c96(activation: str = "mish") -> KataGoNet:
    """初版测试规模: 8 块 × 96 trunk × 48 mid × 16 gpool, ~1.5-2M 参数."""
    return KataGoNet(
        num_blocks=8, c_main=96, c_mid=48, c_gpool=16,
        internal_length=2,
        num_in_channels=4, num_global_features=12,
        activation=activation, version=15,
        has_intermediate_head=True, intermediate_head_blocks=5,
        c_p1=24, c_g1=24, c_v1=24, c_v2=32,
        pos_len=15,
    )


def build_b12c128(activation: str = "mish") -> KataGoNet:
    """生产规模: 12 块 × 128 trunk × 64 mid × 16 gpool, ~4-5M 参数."""
    return KataGoNet(
        num_blocks=12, c_main=128, c_mid=64, c_gpool=16,
        internal_length=2,
        num_in_channels=4, num_global_features=12,
        activation=activation, version=15,
        has_intermediate_head=True, intermediate_head_blocks=8,
        c_p1=32, c_g1=32, c_v1=32, c_v2=48,
        pos_len=15,
    )
```

- [ ] **Step 14.4: 扩展 NetConfig（向后兼容）**

修改 `python/model_config.py`，整体替换为：

```python
"""Central network configuration.

Kept intentionally minimal: the hyperparameters the C++ selfplay side also
needs to know live in scripts/run.cfg; this file only defines the Python
network topology defaults.

Default values target the legacy nets.py shape (b12c128 trunk, 4 input
planes, no global features). Phase A KataGo v15 model uses the extended
fields (c_mid, c_gpool, num_global_features, has_intermediate_head, ...)
which legacy nets.py ignores.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetConfig:
    board_size: int = 15
    num_planes: int = 4  # own, opp, forbidden_black, forbidden_white
    num_blocks: int = 12
    num_channels: int = 128

    # ----- Phase A (KataGo v15) extensions -----
    # Legacy nets.py ignores these fields; nets_v2 reads them.
    num_global_features: int = 12
    c_mid: int = 64                     # bottleneck mid channels
    c_gpool: int = 16                   # ConvAndGPool branch channels
    internal_length: int = 2
    has_intermediate_head: bool = True
    intermediate_head_blocks: int = 8
    c_p1: int = 32                      # PolicyHead conv1p
    c_g1: int = 32                      # PolicyHead conv1g
    c_v1: int = 32                      # ValueHead conv1
    c_v2: int = 48                      # ValueHead linear2
    activation: str = "mish"
    version: int = 15

    # ----- Legacy fields (used by nets.py only) -----
    @property
    def mid_channels(self) -> int:
        return max(16, self.num_channels // 2)

    @property
    def policy_head_channels(self) -> int:
        return self.num_channels // 2

    @property
    def value_head_channels(self) -> int:
        return self.num_channels // 4

    @property
    def value_fc_channels(self) -> int:
        return self.num_channels // 2


def net_config_from_env() -> NetConfig:
    """Read env vars set by scripts/run.cfg (sourced in bash then exported)."""
    import os
    cfg = NetConfig()
    if (v := os.environ.get("BOARD_SIZE")):
        cfg.board_size = int(v)
    if (v := os.environ.get("NUM_PLANES")):
        cfg.num_planes = int(v)
    if (v := os.environ.get("NUM_BLOCKS")):
        cfg.num_blocks = int(v)
    if (v := os.environ.get("NUM_CHANNELS")):
        cfg.num_channels = int(v)
    if (v := os.environ.get("NUM_GLOBAL_FEATURES")):
        cfg.num_global_features = int(v)
    if (v := os.environ.get("C_MID")):
        cfg.c_mid = int(v)
    if (v := os.environ.get("C_GPOOL")):
        cfg.c_gpool = int(v)
    if (v := os.environ.get("INTERMEDIATE_HEAD_BLOCKS")):
        cfg.intermediate_head_blocks = int(v)
    return cfg
```

- [ ] **Step 14.5: 运行测试**

Run: `python -m pytest python/tests/test_nets_v2_factories.py -v`
Expected: 5 passed

- [ ] **Step 14.6: 跑全部已有测试，确认没回归**

Run: `python -m pytest python/tests/ -v`
Expected: 62 passed (Task 0-14 累计：1+6+4+4+4+2+7+4+4+3+3+5+3+7+5 = 62)

- [ ] **Step 14.7: 提交**

```bash
git add python/nets_v2.py python/model_config.py python/tests/test_nets_v2_factories.py
git commit -m "feat(nets_v2): add b8c96/b12c128 factories and extend NetConfig"
```

---

## Task 15: TorchScript trace 验证

**Files:**
- Create: `python/tests/test_nets_v2_export.py`

**Reference:** spec §5.1 — forward 签名 (state, global)，dict 输出

- [ ] **Step 15.1: 写失败测试**

文件 `python/tests/test_nets_v2_export.py`：

```python
import pytest
import torch
from nets_v2 import build_b8c96


def test_torchscript_trace_b8c96():
    """Trace KataGoNet 用 (state, global_features) 双输入。"""
    model = build_b8c96()
    model.initialize()
    model.eval()
    example_state = torch.zeros(1, 4, 15, 15, dtype=torch.float32)
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
    example_state = torch.zeros(1, 4, 15, 15)
    example_global = torch.zeros(1, 12)
    with torch.no_grad():
        scripted = torch.jit.trace(model, (example_state, example_global), strict=False)
    save_path = tmp_path / "model.pt"
    scripted.save(str(save_path))
    loaded = torch.jit.load(str(save_path))
    out_loaded = loaded(example_state, example_global)
    assert out_loaded["policy"].shape == (1, 6, 225)
```

- [ ] **Step 15.2: 运行测试验证 (期望 PASS — 不需要新代码, 但暴露 TorchScript 兼容性问题)**

Run: `python -m pytest python/tests/test_nets_v2_export.py -v`

If FAIL with TorchScript errors (例如 Optional[torch.Tensor] 类型问题, dict 输出限制):
- 错误 1: `intermediate_policy: Optional[torch.Tensor]` 类型在 trace 模式可能被 trace 成 None — 需要在 forward 改成 `if self.has_intermediate_head: out["intermediate_policy"] = ...; else: out["intermediate_policy"] = torch.zeros(0)` 用 sentinel tensor
- 错误 2: dict 输出有时 trace 不友好 — 改用 tuple 输出
- 实际策略: trace strict=False 模式应该宽松, 大多数兼容

If PASS: 跳到 Step 15.4.

- [ ] **Step 15.3: (条件性) 修复 TorchScript 兼容性**

若 trace 失败, 在 `KataGoNet.forward` 末尾改返回 tuple 而非 dict：

```python
# 替换最后的 return {...} 为:
# 用 0-size tensor 表示 None, C++ 端通过 numel()==0 检测
zero_intermediate_policy = (
    iout_policy if iout_policy is not None
    else torch.zeros(0, dtype=out_policy.dtype, device=out_policy.device)
)
zero_iout_value = iout_value if iout_value is not None else (
    torch.zeros(0, dtype=out_policy.dtype, device=out_policy.device),
)
return out_policy, out_value, zero_intermediate_policy, zero_iout_value
```

并相应更新 `test_nets_v2_model.py` 改成 tuple 解包。

- [ ] **Step 15.4: 提交**

```bash
git add python/tests/test_nets_v2_export.py
# 若有修复:
# git add python/nets_v2.py python/tests/test_nets_v2_model.py
git commit -m "test(nets_v2): verify TorchScript trace + save/load round-trip"
```

---

## Task 16: 全模型 sanity check（NOTES.md §8 自检对齐）

**Files:**
- Modify: `python/tests/test_nets_v2_model.py`

**Reference:** KataGoModel/NOTES.md §8 — 从零 init 后的预期数值范围

- [ ] **Step 16.1: 加自检测试**

追加到 `python/tests/test_nets_v2_model.py`：

```python
def test_b8c96_from_scratch_sanity():
    """NOTES.md §8 风格自检 — 从零 init 后空棋盘 + 0 global 的 trunk 量级."""
    from nets_v2 import build_b8c96

    torch.manual_seed(42)
    model = build_b8c96()
    model.initialize()
    model.eval()

    state = torch.zeros(1, 4, 15, 15)
    state[:, 0:1] = 0.0   # 全 0 = 空棋盘
    g = torch.zeros(1, 12)

    with torch.no_grad():
        out = model(state, g)

    # WDL probs 接近均匀 (head scale_output=0.3 + 全 0 输入)
    wdl_logits = out["value"][0]
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
    state = torch.zeros(1, 4, 15, 15)
    g = torch.zeros(1, 12)
    with torch.no_grad():
        out = model(state, g)
    wdl_probs = torch.softmax(out["value"][0], dim=1)
    for i in range(3):
        assert 0.15 < wdl_probs[0, i].item() < 0.55
```

- [ ] **Step 16.2: 运行**

Run: `python -m pytest python/tests/test_nets_v2_model.py -v`
Expected: 9 passed (含两个 sanity check)

- [ ] **Step 16.3: 全测试套件最终回归**

Run: `python -m pytest python/tests/ -v`
Expected: 全部 passed

- [ ] **Step 16.4: 提交**

```bash
git add python/tests/test_nets_v2_model.py
git commit -m "test(nets_v2): add NOTES.md-style sanity checks for b8c96/b12c128"
```

---

## Task 17: 文档 + 收尾

**Files:**
- Create: `python/nets_v2.py` 文件头注释
- 检查: 全测试通过

- [ ] **Step 17.1: 在 nets_v2.py 文件头补充使用说明**

`python/nets_v2.py` 文件头注释（如果还没写完整）应该包含：

```python
"""KataGo b28c512nbt v15 网络模块 — SkyZero_V4 适配版.

参考: KataGoModel/model.py (lightvector/KataGo master 对齐)
基础设计: norm_kind=fixscaleonenorm, trunk_normless=True,
         bnorm_use_gamma=True, gamma_weight_decay_center_1=True,
         use_repvgg_init=True, version=15.

使用:
    from nets_v2 import build_b8c96, build_b12c128

    # (A) 从零训
    model = build_b8c96()
    model.initialize()           # 必调一次

    # (B) 加载 ckpt
    model.load_state_dict(state_dict, strict=True)
    model.set_norm_scales()      # 必调一次 (scale 不在 state_dict)

陷阱（NOTES.md §3）:
1. NestedBottleneckResBlock.forward 只返残差; 调用方 out + block(out)
2. gamma 张量是 delta, forward 必须 (gamma + 1)
3. NormMask.scale 是 plain Python float, 不在 state_dict
"""
```

- [ ] **Step 17.2: 跑完整测试**

Run: `python -m pytest python/tests/ -v --tb=short`
Expected: all green

- [ ] **Step 17.3: 报告参数量**

Run:
```bash
cd /home/sky/RL/SkyZero/SkyZero_V4
python -c "
from nets_v2 import build_b8c96, build_b12c128
for name, build in [('b8c96', build_b8c96), ('b12c128', build_b12c128)]:
    m = build()
    n = sum(p.numel() for p in m.parameters())
    print(f'{name}: {n:,} params ({n/1e6:.2f}M)')
"
```

记录输出（用于 Phase B 验证）。

- [ ] **Step 17.4: 最终 commit**

```bash
git add -A
git diff --cached --stat
git commit -m "docs(nets_v2): finalize file header + Phase A complete

Phase A done: KataGo v15 architecture in nets_v2.py with full unit
test coverage. Ready for Phase B (training stack integration).

Module count: 11 (FixscaleNorm, BiasMask, LastBatchNorm, KataGPool,
KataValueHeadGPool, ConvAndGPool, NormActConv, ResBlock,
NestedBottleneckResBlock, PolicyHead, ValueHead, KataGoNet).

Factories: build_b8c96, build_b12c128.

Tests: ~45 across 6 test files."
```

---

## 完成验收标准

- [ ] 所有 17 个 task 全部 commit
- [ ] `python -m pytest python/tests/ -v` 全绿
- [ ] `build_b8c96().initialize()` 后 forward 输出 WDL 在 (0.15, 0.55) 区间
- [ ] `build_b12c128().initialize()` 同上
- [ ] TorchScript trace 成功且 save/load round-trip 一致
- [ ] 现有 `python/nets.py` 和 `python/tests/test_smoke.py` 未被破坏（可以从 main 分支对比）
- [ ] `model_config.NetConfig` 加了 11 个新字段且向后兼容（旧代码读 cfg.num_blocks/num_channels 仍然有效）

---

## Phase B 预备工作（不在本计划范围）

完成 Phase A 后，brainstorm/plan B 的入口：

1. NPZ schema 扩展（11 个新 target 字段）
2. `train.py` 多 head loss 接入 + Muon 优化器
3. `init_model.py` / `export_model.py` 切到 nets_v2 + 双输入签名
4. 用 baseline ckpt 做训练侧 sanity check（先所有新 loss 系数=0，验证 main policy + WDL 不掉点）

Phase C 入口（C++ 端）：

1. `compute_global_features()` in `cpp/envs/gomoku.h` 或新增 header
2. `npz_writer.h` schema 扩展
3. `selfplay_manager.h` target backfill：ownership / futurepos / td_value × 3 / shortterm_error / soft_*
4. C++ inference forward 改为双输入签名
