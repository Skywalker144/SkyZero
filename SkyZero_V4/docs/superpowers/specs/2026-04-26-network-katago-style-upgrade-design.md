# SkyZero_V4 网络升级到 KataGo b28c512nbt v15 架构（小模型版）

日期：2026-04-26
状态：Design

## 1. 目标

把 `python/nets.py` 升级为 `KataGoModel/model.py`（KataGo v15 / b28c512nbt 实现）的小模型变体，并把所有不与围棋特化的多任务 head、训练/推理特性都接入 SkyZero_V4 的训练-自对弈流水线。

**规模**：保持 `b12c128`（12 块 × 512 trunk... 不，128 trunk × 64 mid × 16 gpool 通道；约 5-8M 参数）。规模放大留作后续实验，本次只做架构对齐。

**非目标**：
- 不做迁移学习（不加载 `kata1-zhizi-b28c512nbt-muonfd2.ckpt`）
- 不实现 VCF solver（占位 6 维 global，未来再写入真实值）
- 不动 MCTS / Gumbel 选择规则（IMPROVEMENTS.md §1-15 的 #1/#3/#6/#7 等都是后续 sprint）

## 2. 输入设计

### 2.1 Spatial 输入：4 通道（与现状一致）

| ch | 含义 |
|----|------|
| 0 | own stones |
| 1 | opp stones |
| 2 | forbidden_black |
| 3 | forbidden_white |

无 on-board mask 通道（15×15 固定，mask 在网络内部以 `torch.ones` 常量参与 GPool/Norm，便于未来切到变 board size 时只改一处）。

### 2.2 Global 输入：12 维（6 用，6 VCF 占位）

| dim | 含义 | 现在写入？|
|-----|------|---------|
| 0 | rule_freestyle（one-hot）| ✅ |
| 1 | rule_standard | ✅ |
| 2 | rule_renju | ✅ |
| 3 | renju_color_sign（Renju+black=-1, Renju+white=+1, 其它=0）| ✅ |
| 4 | has_forbidden（rule_renju AND useForbidden）| ✅ |
| 5 | ply / 225（归一化步数）| ✅ |
| 6 | VCF: own_can_win | ❌ 占位 0 |
| 7 | VCF: own_cannot_vcf | ❌ 占位 0 |
| 8 | VCF: own_no_short_vcf | ❌ 占位 0 |
| 9 | VCF: opp_can_vcf | ❌ 占位 0 |
| 10 | VCF: opp_cannot_vcf | ❌ 占位 0 |
| 11 | VCF: opp_no_short_vcf | ❌ 占位 0 |

## 3. 网络架构（对齐 KataGoModel/model.py）

### 3.1 模型规格

| 项 | 值 | 备注 |
|---|---|---|
| `num_blocks` | 12 | |
| `c_main` | 128 | trunk 主通道 |
| `c_mid` | 64 | bottleneck 中通道 |
| `c_gpool` | 16 | ConvAndGPool 分支通道 |
| `internal_length` | 2 | NestedBottleneckResBlock 内层 ResBlock 数 |
| `num_in_channels` | 4 | spatial 通道 |
| `num_global_features` | 12 | global 通道 |
| `pos_len` | 15 | 棋盘边长 |
| `activation` | mish | 门 = √2.210277 |
| `version` | 15 | 决定 PolicyHead 输出数 = 6 |
| `has_intermediate_head` | True | |
| `intermediate_head_blocks` | 8 | 在 12 块中的第 8 块后接 intermediate head（中段监督，比 b28 ckpt 的"末尾两套 norm"更经典）|
| `c_p1, c_g1` | 32, 32 | PolicyHead 通道（按比例缩小）|
| `c_v1, c_v2` | 32, 48 | ValueHead 通道 |

### 3.2 Trunk 结构（与 model.py 1:1 对齐，删 score-belief / pass / scoring / seki 路径）

```
input_spatial (B,4,15,15) ──┬─→ conv_spatial (3×3, 128 ch, no bias)─┐
                            │                                         │
input_global  (B,12)    ──┴─→ linear_global (12→128, no bias)──┐  │
                                                                  ↓ ↓
                                                                  add
                                                                  │
              for i in range(12):                                  │
                use_gpool = (i % 3 == 2)  # 块 2/5/8/11 含 gpool   │
                out = out + NestedBottleneckResBlock_i(...)       │
                # 第 8 块后存档供 intermediate head            │
              ↓                                                    │
              norm_intermediate_trunkfinal = LastBatchNorm(128)   │
              act_intermediate_trunkfinal = Mish                  │
              ↓                                                    │
              intermediate_policy_head + intermediate_value_head ─┘
              （中间监督, weight 系数较小; 详见 §4.5）

最后 4 块继续：
              ↓
              norm_trunkfinal = BiasMask(128)  scale=1/√(num_blocks+1)=1/√13
              act_trunkfinal = Mish
              ↓
              policy_head + value_head
```

### 3.3 关键模块（必须照抄 KataGoModel 的语义，避开 NOTES.md §3 三个坑）

#### 3.3.1 `FixscaleNorm`（替换现有 `FixScaleNorm`）

```python
class FixscaleNorm(nn.Module):
    def __init__(self, num_channels, use_gamma=True):
        self.beta  = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))   # ⚠️ 初始 0（delta 表示）
        self.scale: Optional[float] = None                    # ⚠️ plain float, 不在 state_dict

    def forward(self, x, mask=None):
        g = self.gamma + 1.0                                  # ⚠️ 必须 +1
        if self.scale is not None:
            out = x * (g * self.scale) + self.beta
        else:
            out = x * g + self.beta
        return out * mask if mask is not None else out
```

变更点（相对当前 nets.py）：
- `gamma` 初值 `1 → 0`
- forward `x * gamma → x * (gamma + 1)`
- `fixed_scale` 从 `register_buffer` 改为 plain Python float
- 加 `set_scale(s)` 方法，初始化时由 `model.initialize()` 调用
- weight decay 仍然推向 0（语义现在是推向"真正的 1"）

#### 3.3.2 `BiasMask`（trunk tip 用）

替代当前的 `GroupNorm(1, C)`：
```python
class BiasMask(nn.Module):
    def forward(self, x, mask=None):
        out = x * self.scale + self.beta if self.scale is not None else x + self.beta
        return out * mask if mask is not None else out
```

trunk tip 设 `scale = 1/√(num_blocks + 1) = 1/√13`。

#### 3.3.3 `LastBatchNorm`（intermediate head 入口唯一一处 BN）

mask-aware mini-batch BN with running stats。要求 batch_size ≥ 32（SkyZero 当前 ≥ 256，OK）。

#### 3.3.4 `KataGPool`（3 个统计量）

```python
class KataGPool(nn.Module):
    def forward(self, x, mask, mask_sum_hw=None):
        if mask_sum_hw is None:
            mask_sum_hw = mask.sum(dim=(2,3), keepdim=True)   # 15×15 时常数 225
        sqrt_off = torch.sqrt(mask_sum_hw) - 14.0             # 15×15 时常数 1.0
        layer_mean = (x * mask).sum(dim=(2,3), keepdim=True) / mask_sum_hw
        layer_max  = (x + (mask - 1.0)).amax(dim=(2,3), keepdim=True)
        return torch.cat([layer_mean,
                          layer_mean * (sqrt_off / 10.0),     # 15×15 时退化为 mean*0.1
                          layer_max], dim=1)
```

`KataValueHeadGPool` 同样 3 个统计量但都是 mean 系列（无 max）。

#### 3.3.5 `ConvAndGPool`（替代现有 `GlobalPoolingResidualBlock`）

```python
class ConvAndGPool(nn.Module):
    """合并到 ResBlock 内层第一个 conv 的位置, 不是独立残差块。"""
    def forward(self, x, mask, mask_sum_hw=None):
        out_r = self.conv1r(x)                                       # 主分支 3×3
        out_g = self.actg(self.normg(self.conv1g(x), mask))           # gpool 分支 3×3
        out_g = self.gpool(out_g, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        out_g = self.linear_g(out_g).unsqueeze(-1).unsqueeze(-1)      # 3*c_gpool→c_out
        return out_r + out_g                                          # 当作 conv 的 bias 注入
```

集成到 `NestedBottleneckResBlock` 的内层 ResBlock：
- `i % 3 == 2` 的外层块的内层 ResBlock 0 用 `c_gpool=16`
- 其他都用普通 `c_gpool=None`

#### 3.3.6 `NestedBottleneckResBlock`（坑 1：只返残差）

```python
def forward(self, x, mask=None, mask_sum_hw=None):
    out = self.normactconvp(x, mask, mask_sum_hw)
    for block in self.blockstack:
        out = out + block(out, mask, mask_sum_hw)         # 内层残差
    out = self.normactconvq(out, mask, mask_sum_hw)
    return out                                            # ⚠️ 只返回残差
```

调用方负责 `out = out + block(out, mask, mask_sum_hw)` 加回主干。

### 3.4 Output Heads

#### 3.4.1 `PolicyHead`（v15，6 输出，**删 pass 路径**）

```python
self.num_policy_outputs = 6   # main / soft / aux / soft_aux / opt / opp
# 删: linear_pass, linear_pass2, act_pass
# forward 里: outp = outp - (1.0 - mask) * 5000.0
# 输出 (B, 6, H*W) — 不再 cat outpass
```

6 个输出对应 SkyZero training target：

| idx | 含义 | Target 公式 |
|-----|------|-----------|
| 0 | main | softmax(visits) ← 现有 |
| 1 | soft | softmax(visits / T_soft), T_soft=4 |
| 2 | aux | 下一步对手实际落子的 one-hot（多步预测）|
| 3 | soft_aux | softmax(下一步对手 MCTS visits / T_soft) |
| 4 | opt | optimistic target（误差加权） — Sayuri 公式，详见 §4.4 |
| 5 | opp | 对手视角 policy ← 现有 |

#### 3.4.2 `ValueHead`（删 score / seki / scorebelief，保留 WDL + ownership + futurepos + miscvalue）

输出 6 元 tuple（KataGo 是 8 元，删 idx 4 scoring 和 idx 6 seki 和 idx 7 scorebelief）：

| idx | shape | 含义 | SkyZero target |
|-----|-------|------|---------------|
| 0 | (B, 3) | WDL logits | 现有 outcome WDL |
| 1 | (B, 4) | miscvalue 子集：td-value-long{w,l,nr} + variance-time | 多 horizon Q（详见 §4.3）|
| 2 | (B, 4) | moremiscvalue 子集：shortterm-value-error + td-value-short{w,l,nr} | shortterm error + 短 horizon Q |
| 3 | (B, 1, H, W) | ownership pretanh | 终局每格占有 ±1 |
| 4 | (B, 2, H, W) | futurepos +N / +2N 步占据 | 未来步占据 ±1 |

Loss 全是 mask-aware（在每格上乘 mask）。

注意：删 KataGo `linear_s2/s2off/s2par/s3/smix` + 3 个 score-belief buffer + `conv_scoring` + `conv_seki`。

#### 3.4.3 `intermediate_policy_head` + `intermediate_value_head`

与 main head 同样结构，接在第 8 块后（`intermediate_head_blocks=8`）。loss 权重为 main head 的 ~30%。

### 3.5 RepVGG 初始化

照抄 `KataGoModel/model.py` 的 `init_weights()` + `initialize()`：
- kernel_size > 1 的 conv：3×3 主分支 scale=0.8 + 1×1 中心融合 scale=0.6
- 1×1 conv（bottleneck 入/出，head 中间）：普通 init, scale=1.0
- head 输出 conv/linear：activation='identity', scale=0.3
- 所有 NormMask 的 scale 由 `initialize()` 一次性设置（NOTES.md §3.3 表）

## 4. Training Pipeline 改动

### 4.1 NPZ Schema（C++ 写，Python 读）

新增字段（数组维度按 `H=W=15`、`P=H*W=225`、`O=ownership=225`、`F=futurepos=2*225=450`）：

| key | dtype | shape | 含义 | 来源 |
|-----|-------|-------|------|------|
| `state` | int8 | (N, 4, 15, 15) | spatial（不变）|
| `global_features` | float32 | (N, 12) | **新增** | 见 §2.2 |
| `policy_target` | float32 | (N, P) | main MCTS visit dist（不变）|
| `soft_policy_target` | float32 | (N, P) | **新增** | softmax(visits / 4) |
| `aux_policy_target` | float32 | (N, P) | **新增** | 下一步对手 one-hot |
| `soft_aux_policy_target` | float32 | (N, P) | **新增** | softmax(next_visits / 4) |
| `opt_policy_target` | float32 | (N, P) | **新增** | 见 §4.4，初版用 main 占位 |
| `opponent_policy_target` | float32 | (N, P) | opp policy（不变）|
| `opponent_policy_mask` | float32 | (N,) | （不变）|
| `value_target` | float32 | (N, 3) | WDL（不变）|
| `td_value_long_target` | float32 | (N, 3) | **新增** | 长 horizon (5 步) Q WDL |
| `td_value_short_target` | float32 | (N, 3) | **新增** | 短 horizon (1 步) Q WDL |
| `shortterm_error_target` | float32 | (N,) | **新增** | abs(value_pred - value_target_short) |
| `ownership_target` | float32 | (N, 15, 15) | **新增** | 终局 ±1 / 0 |
| `futurepos_target` | float32 | (N, 2, 15, 15) | **新增** | +5 / +10 步占据 |
| `intermediate_aux_mask` | float32 | (N,) | **新增** | intermediate head 是否参与 loss（早期局面 = 1）|
| `sample_weight` | float32 | (N,) | （不变）|

### 4.2 Loss 权重（KataGo 默认值的子集）

```python
LOSS_WEIGHTS = {
    "policy_main":      1.0,
    "policy_soft":      0.5,
    "policy_aux":       0.15,
    "policy_soft_aux":  0.075,
    "policy_opt":       0.15,
    "policy_opp":       0.15,
    "value_wdl":        1.20,
    "td_value_long":    0.6,
    "td_value_short":   0.6,
    "shortterm_error":  2.0,
    "ownership":        1.5,
    "futurepos":        0.25,
    # intermediate head: 上述每条 × 0.3
    "intermediate_scale": 0.3,
}
```

### 4.3 多 Horizon Q Target 生成

C++ 端在 `selfplay_manager.h:545-565` 附近的 backfill 段扩展：

```cpp
// 假设 game.samples 是按下棋顺序排序的 SamplePack 数组
for (int i = 0; i < n; ++i) {
    auto &s = game.samples[i];
    // 长 horizon: 5 步后的 Q（如果到游戏结尾不够 5 步, 用终局 outcome）
    int j_long = std::min(i + 5, n - 1);
    s.td_value_long  = (j_long == n - 1) ? game.outcome
                                          : game.samples[j_long].mcts_root_q_wdl;
    // 短 horizon: 1 步
    int j_short = std::min(i + 1, n - 1);
    s.td_value_short = (j_short == n - 1) ? game.outcome
                                           : game.samples[j_short].mcts_root_q_wdl;
    // shortterm error: |NN value pred at this state - actual short-horizon outcome|
    // 需要 NN 在这一步的 value pred, 由 inference server 写回 SamplePack 时记录
    s.shortterm_error = std::abs(s.nn_value_wdl_w - s.td_value_short_w_class);
}
```

需要 SamplePack 增加 `mcts_root_q_wdl`（已有）和 `nn_value_wdl_w`（要从 inference 路径写回）字段。

### 4.4 Optimistic Policy Target

Sayuri 的简化公式（`network.py:1280-1290` 风格）：

```python
# C++ 端 (selfplay_manager.h):
# opt = main_policy + alpha * (main_policy - prior_policy)
# alpha = clip(value_pred_error / 0.1, 0, 1) — 不确定时弱化乐观
# 退化为 main_policy 当 alpha=0
opt_target = main_policy + alpha * (main_policy - nn_prior)
opt_target = renormalize(opt_target.clip_min(0))
```

**初版实现**：直接令 `opt_target = main_policy`（占位），等 §4.3 的 `shortterm_error` 字段稳定后再写入真实 alpha。这样网络结构不变，loss 收敛到 0（因为预测 = target），后续上线无需改网络。

### 4.5 Intermediate Head Target

KataGo 的做法：intermediate head 的 target 与 main head **完全相同**，但 loss 权重 ×0.3。
- 这迫使 trunk 中段已经能产生有意义的 policy/value 表示
- 实现极简：在 `train.py` 算 main head loss 后，对 intermediate head 输出再算同样的 loss 然后乘 0.3

### 4.6 Ownership / Futurepos Target

- **Ownership**：游戏结束时回填，每格 +1 (己方) / -1 (对方) / 0 (空)。`mcts_policy` 视角：以"当前下棋方"为 +1。
- **Futurepos**：从样本 `i` 出发往后看 5/10 步的棋盘状态（每格被己/对/空）。如果到结尾不够步数，用终局棋盘填充。每格 +1/-1/0。

### 4.7 Optimizer：Muon + AdamW 混合

下载 `muon_kissin.py`（见 NOTES.md §6.1）：
```bash
curl -L -O https://raw.githubusercontent.com/hzyhhzy/KataGomo/kata1_training_tests/python/muon_kissin.py
# 放到 SkyZero_V4/python/muon_kissin.py
```

替换 `train.py:277-287` 的 AdamW 构建：
```python
from muon_kissin import MuonWithAuxAdamKimi

def get_param_groups(model):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        # 矩阵权重 (2D conv weight, Linear weight) → Muon
        # bias / norm gamma+beta / 1D / running stats → AdamW
        if p.ndim >= 2 and "norm" not in name and "bias" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return [
        dict(params=muon_params, use_muon=True,  lr=lr_muon, momentum=0.95),
        dict(params=adam_params, use_muon=False, lr=lr_adam, betas=(0.9, 0.99), eps=1e-6),
    ]

optimizer = MuonWithAuxAdamKimi(get_param_groups(model), muon_lr_multiplier=8.0)
```

`lr_adam = lr_muon / 8`。具体 `lr_muon` 起始值：从 `current_AdamW_lr × 8` 开始扫描。

SWA 保留：包在 Muon 外层（先做 Muon step，再更新 SWA EMA）。

## 5. C++ Inference 接口改动

### 5.1 TorchScript Forward 签名

```cpp
// 旧: tensor forward(tensor state)  → (policy, opp_policy, value)
// 新: tuple<tensor, tensor> forward(tensor state, tensor global_features)
//     → dict{policy, value, intermediate_*, ...}
```

C++ 端 `nn_inference.h` / `nn_inference.cpp` 需要：
- 把每条 query 同时传入 `(state[B,4,15,15], global[B,12])`
- 解包 dict 输出，只取 `policy[:, 0:1]`（main policy）和 `value[0]`（WDL logits）做 MCTS 用
- 其他 head 输出**只在训练时用**，inference 时丢弃（或设 `model.eval_inference_only()` 模式跳过那些 head）

### 5.2 `feature_planes.h` 扩展

新增 `compute_global_features()`，每次推理前调一次（也用于 NPZ 写入）：

```cpp
struct GlobalFeatures {
    static constexpr int DIM = 12;
    float data[DIM];
};

GlobalFeatures compute_global_features(const GameState& s, const Rules& r) {
    GlobalFeatures g{};
    g.data[0] = (r.basic == FREESTYLE);
    g.data[1] = (r.basic == STANDARD);
    g.data[2] = (r.basic == RENJU);
    g.data[3] = (r.basic == RENJU) ? (s.to_move == BLACK ? -1.0f : +1.0f) : 0.0f;
    g.data[4] = (r.basic == RENJU && r.use_forbidden) ? 1.0f : 0.0f;
    g.data[5] = float(s.move_count) / 225.0f;
    // dim 6-11: VCF 占位, 全 0
    return g;
}
```

## 6. 实施阶段（每阶段独立可验证）

### Phase A — 网络重写（隔离）

- [ ] 创建 `python/nets_v2.py`（不动现有 `nets.py`）
- [ ] 在 `nets_v2.py` 实现：FixscaleNorm、BiasMask、LastBatchNorm、KataGPool、KataValueHeadGPool、ConvAndGPool、NestedBottleneckResBlock（坑 1 行为）、PolicyHead（v15, 删 pass）、ValueHead（删 score/seki/scorebelief）、IntermediatePolicyHead/IntermediateValueHead、RepVGG init
- [ ] 单元测试 `python/tests/test_nets_v2.py`：
  - 前向 shape 正确（policy (B,6,P), value 5-tuple）
  - `initialize()` 后空棋盘 W/L/draw probs ≈ (0.34, 0.34, 0.32)
  - 无 NaN，trunk |x| 在合理范围
  - mask-aware 路径 test（mask=0 区域输出可忽略）

### Phase B — 训练 stack 改造

- [ ] 扩展 `data_processing.py` NPZ schema（多加 11 个字段，先全填 0/默认值，让 train.py 不崩）
- [ ] `train.py` 接 `nets_v2`，改 forward 签名（state + global），实现新 loss（先所有新 loss 系数设 0，验证 main policy + WDL 仍然能学）
- [ ] 引入 Muon 优化器（`muon_kissin.py`），lr 扫描

### Phase C — C++ NPZ 扩展

- [ ] `cpp/feature_planes.h`：实现 `compute_global_features()`
- [ ] `cpp/npz_writer.h`：扩展 schema 写出新字段
- [ ] `cpp/selfplay_manager.h`：
  - SamplePack 字段：mcts_root_q_wdl（如果还没有）、nn_value_wdl_w
  - 游戏结束 backfill：ownership / futurepos / td_value_{long,short} / shortterm_error
  - soft / soft_aux / aux 现场计算
  - opt = main 占位
- [ ] `cpp/inference/`（用到的接口文件）：forward 签名改为 (state, global)，dict 解包

### Phase D — 验证

- [ ] 用 nets_v2 跑一个完整训练 round（1000 步），对比 baseline 模型 200 局：目标 ≥ baseline -10 Elo（先做到不掉棋力）
- [ ] 各 loss 权重逐个开（0 → KataGo 默认值），分阶段确认收敛
- [ ] Intermediate head 接入 loss
- [ ] Muon lr 扫描（vs AdamW baseline）

成功验收标准：
- 与现有 `nets.py` baseline 对弈 200 局 ≥ +30 Elo（多任务正则化收益）
- self-play 速度损失 < 30%（多 head + global 通路开销）
- 训练 loss 各项稳定下降

### Phase E — 切换

- [ ] `nets.py` ← `nets_v2.py` 重命名替换
- [ ] 删旧 `FixScaleNorm` / `GlobalPoolingResidualBlock` 代码
- [ ] 提交，归档 baseline checkpoint

## 7. 实施陷阱清单（必读）

来自 `KataGoModel/NOTES.md §3`：

1. **`NestedBottleneckResBlock.forward` 只返残差**——调用方负责 `out + block(out)`。别在 block 里再 `return x + out`。
2. **`gamma + 1.0`**——gamma 存的是 delta，前向必须 `(gamma + 1) * x`。
3. **`NormMask.scale` 不在 state_dict**——`initialize()` 或 `set_norm_scales()` 必调。

来自本设计：

4. **多 horizon Q 视角**——td_value 是从"当前下棋方"的视角，每跳一步要 flip WDL（`flip_wdl(...)`，参考现有 `selfplay_manager.h:564`）。
5. **opt policy 占位**——初版 = main_policy，loss 立即收敛到 0；不要把它当成 bug。
6. **shortterm_error 需要 NN value pred**——必须在 inference 路径把 value 写回 SamplePack，否则 target 缺失。
7. **mask-aware 一致性**——所有 `FixscaleNorm` / `BiasMask` / `LastBatchNorm` / `KataGPool` 在 forward 都要支持 mask；当前 fixed 15×15 时传 `torch.ones(B,1,15,15)`，未来变 board 不用改网络。
8. **训练 batch ≥ 32**——`LastBatchNorm` 要求；SkyZero 当前 ≥ 256，OK，但单元测试要用 ≥ 32。

## 8. 文件改动清单

```
python/
  nets.py                        ← 全部重写（~600-800 行）
  model_config.py                ← 加 c_mid / c_gpool / num_global_features / has_intermediate_head 等字段
  data_processing.py             ← NPZ schema 扩展
  train.py                       ← 多 head loss + Muon + global features 接入
  muon_kissin.py                 ← 新增（curl 下载，~200 行）
  init_model.py                  ← 调用 model.initialize()
  export_model.py                ← TorchScript 导出新 forward 签名
  shuffle.py                     ← 透传新 NPZ 字段
  tests/test_nets_v2.py          ← 新增

cpp/
  envs/gomoku.h（或新增 cpp/feature_planes.h）  ← compute_global_features()
  npz_writer.h                   ← schema 扩展
  selfplay_manager.h             ← target backfill (ownership / futurepos / td_value / shortterm_error / soft_*)
  alphazero.h / alphazero_parallel.h / alphazero_tree_parallel.h
                                 ← TorchScript forward 调用改为 (state, global)，dict 解包
  selfplay_main.cpp / gomoku_eval_main.cpp / gomoku_play_main.cpp
                                 ← 模型加载入口，传新输入

scripts/
  run.cfg                        ← 加 NUM_GLOBAL_FEATURES, C_MID, C_GPOOL, INTERMEDIATE_HEAD_BLOCKS

docs/
  superpowers/specs/2026-04-26-network-katago-style-upgrade-design.md  ← 本文档
```

预计 LOC：Python ~1500 新增/修改，C++ ~600 新增/修改。

## 9. 后续 Sprint（不在本次范围）

- IMPROVEMENTS.md §1 LCB 终着选择
- §3 Tree reuse
- §4 cpuct KataGo 形式
- §6 Subtree value bias
- §7 Forced playouts (non-root)
- §13 Graph search
- §14 VCF solver（一旦上线，本次预留的 6 维 global 占位即可填入）
