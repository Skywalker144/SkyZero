# KataGo b28c512nbt 模型 —— 使用说明

本文件夹内容：
- `model.py` —— PyTorch 模型实现，对齐 lightvector/KataGo master。
- `kata1-zhizi-b28c512nbt-muonfd2.ckpt` —— 智子（hzyhhzy）社区训练的预训练权重，约 73.16M 参数。
- `katago_official/` —— 从 master 拉的 5 个权威源文件，用于对照。

---

## 1. 模型架构速览

| 项 | 值 |
|---|---|
| 残差块数 | 28 |
| trunk 通道 | 512 |
| bottleneck mid 通道 | 256 |
| gpool 通道 | 64（仅出现在 i % 3 == 2 的块上，即第 3/6/9/…/27 个块） |
| internal_length（每个 nested-bottleneck 内部 ResBlock 数） | 2 |
| 输入 spatial 通道 | 22 |
| 输入 global 特征 | 19 |
| 激活 | Mish |
| 总参数 | 73,162,378 |

**block 结构**：每个 NestedBottleneckResBlock = `1×1 conv (512→256) → 2 个 inner ResBlock (256→256, 3×3+3×3) → 1×1 conv (256→512)`，外层是大残差。

**两个输出头**：每个头都是 PolicyHead + ValueHead 的组合：
- `intermediate_*_head` 接在第 28 个块之后（即所有块之后；这个权重 `intermediate_head_blocks=28`，等于 `num_blocks`，所以 intermediate 实际接在最后一个块之后）。
- `*_head` 接在 trunk-final-norm 之后。

> **注**：本权重的 `intermediate_head_blocks=28` 让 intermediate head 和 main head 拿到几乎相同的 trunk，差别只在 `LastBatchNorm` vs `BiasMask` 的归一化方式不同。一般训练时 `intermediate_head_blocks` 会比 `num_blocks` 小（如 14），让中间监督更早接入。

---

## 2. 三个**不能少**的 API

```python
from model import b28c512nbt

# (A) 从零训自己的任务
model = b28c512nbt()
model.initialize()          # ← 必调一次。RepVGG init + 设置所有 NormMask.scale
# ... 训练循环

# (B) 加载预训练 ckpt 做推理或微调
model = b28c512nbt()
ckpt = torch.load("kata1-zhizi-b28c512nbt-muonfd2.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=True)
model.set_norm_scales()     # ← 必调一次。scale 不在 state_dict 里, load_state_dict 不会设它
model.eval()
```

**忘记调 `set_norm_scales()` 的后果**：每个 norm 的 scale 默认是 None，于是前向退化成 `(gamma+1)*x + beta`，丢失了 `1/sqrt(block_idx+1)` 这样的尺度因子。预测结果会 **过自信**（W/L logits 量级会从 ~2 变成 ~25），表现上像是「模型坏了」。

---

## 3. 三处坑（已修复，但别在改代码时踩回去）

### 3.1 `NestedBottleneckResBlock.forward` 只返回残差

```python
def forward(self, x, mask, mask_sum_hw=None):
    out = self.normactconvp(x, mask, mask_sum_hw)
    for block in self.blockstack:
        out = out + block(out, mask, mask_sum_hw)
    out = self.normactconvq(out, mask, mask_sum_hw)
    return out                        # ← 只返回残差
```

主循环负责加回主干：`out = out + block(out, mask, mask_sum_hw)`。**别在 block 里再 `return x + out`**——会变成主干每过一个块翻倍，28 个块后放大 ~2.7×10⁸ 倍。

### 3.2 gamma 用 `(gamma + 1)`

`gamma_weight_decay_center_1=True` 表示训练时 weight decay 把 gamma 推向 0（而不是 1），即保存的 gamma 实际上是「真正缩放因子 - 1」的 delta 表示。本权重里 178 个 gamma 张量，平均 mean ≈ +0.12，min = -1.0，max = +2.76——明确是 delta。

所以 `FixscaleNorm` 和 `LastBatchNorm` 的前向都是：

```python
out = x * (self.gamma + 1.0) + self.beta    # 而不是 x * self.gamma + self.beta
```

### 3.3 NormMask 有个 `scale` 属性，**不在 state_dict**

`scale` 是 plain Python float（不是 `nn.Parameter` 也不是 `register_buffer`），通过 `set_scale(value)` 设置。各模块的 scale 取值（fixscaleonenorm 路径）：

| Norm 位置 | scale |
|---|---|
| 第 i 个 block 的 `normactconvp.norm` | `1/sqrt(i+1)` |
| 第 i 个 block 的 `normactconvq.norm` | `1/sqrt(internal_length+1) = 1/sqrt(3)` |
| 第 i 个 block 内第 j 个 ResBlock 的 `normactconv1.norm` | `1/sqrt(j+1)` |
| 第 i 个 block 内第 j 个 ResBlock 的 `normactconv2.norm` | `None`（不应用 scale） |
| 含 gpool 的 ResBlock 的 `convpool.normg` | `None` |
| `norm_trunkfinal`（BiasMask） | `1/sqrt(num_blocks+1) = 1/sqrt(29)` |
| `norm_intermediate_trunkfinal`（LastBatchNorm） | `None` |

`KataGoModel.initialize()` 和 `set_norm_scales()` 已经把这些都串好了。

---

## 4. RepVGG init 怎么工作

`use_repvgg_init=True` 仅对 **kernel_size > 1 的 conv** 生效。核心代码在 `NormActConv.initialize`：

```python
init_weights(conv.weight, activation, scale=scale*0.8)              # 3×3 主分支
center_bonus = zeros(c_out, c_in)
init_weights(center_bonus, activation, scale=scale*0.6)             # 1×1 中心分支
conv.weight[:, :, 1, 1] += center_bonus                             # 把 1×1 加到 3×3 中心
```

等价于 RepVGG 的「训练用 3×3+1×1 并联，推理融合成单个 3×3」的初始化方差。

`init_weights(tensor, activation, scale)` 是 truncated-normal：

```python
target_std = scale * gain(activation) / sqrt(fan_in)
std = target_std / 0.87962566103423978   # trunc 修正
trunc_normal_(tensor, mean=0, std=std, a=-2*std, b=2*std)
```

`gain('mish') = sqrt(2.210277)`，类似 ReLU 的 sqrt(2)。

**1×1 conv（bottleneck 入/出）和所有 head 的最终输出 conv 不走 RepVGG**：
- 1×1 用普通 `init_weights(conv.weight, activation, scale=1.0)`
- head 输出 conv/linear 用 `activation='identity'`、`scale=0.3` 或 `0.2` —— 让初始预测接近均匀分布。

---

## 5. 输入/输出语义

### 5.1 spatial 输入（22 通道，N×22×19×19）

| ch | 含义 |
|---|---|
| 0 | on-board mask（棋盘有效格子=1，padding=0） |
| 1 | 己方棋子 |
| 2 | 对方棋子 |
| 3-5 | 棋串气数 = 1 / 2 / 3 |
| 6 | simple ko 点 |
| 7-8 | encore ko-prohibition（python 端置 0） |
| 9-13 | 历史走子 m-1 / m-2 / m-3 / m-4 / m-5（己/对方交替） |
| 14-17 | ladder 信息（4 通道） |
| 18 | 己方 area / pass-alive |
| 19 | 对方 area |
| 20-21 | encore-2 起始棋子 |

### 5.2 global 输入（19 维，N×19）

| idx | 含义 |
|---|---|
| 0-4 | prev1..prev5 是否 pass |
| 5 | selfKomi / 20 |
| 6 | ko 规则非简单 |
| 7 | ko 规则正负号（+0.5 positional / -0.5 situational） |
| 8 | 多子自杀合法 |
| 9 | scoring=territory |
| 10 | tax=seki-or-all |
| 11 | tax=all |
| 12-13 | encorePhase>0, >1 |
| 14 | passWouldEndPhase |
| 15-16 | asymPowersOfTwo（异位 sign 与原值） |
| 17 | hasButton |
| 18 | komi-wave（亚整数贴目的 parity 特征，**ValueHead score-belief 用** `input_global[:, -1:]`） |

### 5.3 输出

`forward` 返回 dict：

| 键 | 形状 | 含义 |
|---|---|---|
| `policy` | `(N, 6, 362)` | 6 个 policy 输出（player/opponent/short-term/optimistic-soft/optimistic-hard 等），最后一维 = 19²+1（包含 pass） |
| `value` | tuple of 8 | 见下表 |
| `intermediate_policy` | `(N, 6, 362)` | 中间监督的 policy |
| `intermediate_value` | tuple of 8 | 中间监督的 value |
| `trunk` | `(N, 512, 19, 19)` | trunk 末端表示 |
| `mask` | `(N, 1, 19, 19)` | 板面 mask（直接来自 `input_spatial[:,0:1]`） |

`value` tuple 8 元素：

| idx | 形状 | 含义 |
|---|---|---|
| 0 | `(N, 3)` | W / L / draw 的 logits（softmax 取胜率） |
| 1 | `(N, 10)` | misc：scoremean*mult, scorestdev (softplus), lead, variance-time, td-value-long{w,l,nr}, td-value-mid{w,l,nr} |
| 2 | `(N, 8)` | more-misc：shortterm-value-error, shortterm-score-error, td-value-short{w,l,nr}, td-score{long,mid,short} |
| 3 | `(N, 1, 19, 19)` | per-cell ownership pretanh |
| 4 | `(N, 1, 19, 19)` | per-cell scoring 贡献 |
| 5 | `(N, 2, 19, 19)` | future-pos（+N 与 +2N 步预测） |
| 6 | `(N, 4, 19, 19)` | seki 4-class logits |
| 7 | `(N, 842)` | score-belief log-probs，长度 = `2*(19² + 60) = 842`，offset = `i - 421 + 0.5`，即范围 `[-420.5, +420.5]` 目分 |

---

## 6. 训练自己任务的清单

如果你的任务**不是围棋**：

1. **改输入维度**：`KataGoModel(num_in_channels=??, num_global_features=??)`。`input_spatial[:, 0]` 必须是有效区域 mask。
2. **砍冗余 head**：你大概率只需要 policy + value（W/L），不需要 ownership / scoring / futurepos / seki / scorebelief。直接删 `ValueHead` 里对应的 4 个 conv 和 score-belief 一整套（`linear_s2/s2off/s2par/s3/smix` 与 3 个 buffer）。然后在 forward 里也只 return 你要的部分。
3. **改 PolicyHead 输出数**：当前 6 个 policy logit 通道都是围棋专用语义；你只需要 1 个就够了。把 `num_policy_outputs` 改成 1，并把 `linear_pass*` 那条独立路径删掉（你的任务可能没有「pass」动作）。
4. **改 pos_len**：默认 19。如果你的任务棋盘/网格不是方形或者尺寸不同，要么改 pos_len，要么把所有 spatial-shape 假设抽出来。

如果你的任务**就是围棋**：直接用 `b28c512nbt()` + `model.initialize()` 即可。

### 6.1 优化器建议

- **简单稳妥**：AdamW，`betas=(0.9, 0.99)`、`eps=1e-6`、`weight_decay=0.01`。
- **追求 SOTA**：Muon（智子用的就是这个）。代码：
  ```bash
  curl -L -O https://raw.githubusercontent.com/hzyhhzy/KataGomo/kata1_training_tests/python/muon_kissin.py
  ```
  用法：`MuonWithAuxAdamKimi(get_param_groups(model), muon_momentum=0.95)`，矩阵权重走 Muon，bias/norm 走 AdamW。`muon_lr_multiplier=8.0`（Adam 学习率 = Muon LR / 8）。

### 6.2 Loss 默认权重（围棋任务，metrics_pytorch.py）

policy=1.0、opponent-policy=0.15、value-CE=1.20、TD-value=1.20、ownership=1.5、scoring=1.0、futurepos=0.25、scorebelief CDF/PDF 各 0.020、shortterm-value-error=2.0……

非围棋任务 **直接把权重设为 0** 让对应 head 不参与。

### 6.3 注意 LastBatchNorm

`norm_intermediate_trunkfinal` 是 LastBatchNorm，训练模式下做 mask-aware mini-batch 统计。**batch size 不能太小**（建议 ≥32），否则 BN 统计噪声会很大；如果 batch=1，请改用 BiasMask 替代。

---

## 7. 参考资料

- **lightvector/KataGo master**（架构本体的权威源）  
  `https://github.com/lightvector/KataGo`，关键文件 `python/katago/train/model_pytorch.py`、`metrics_pytorch.py`、`modelconfigs.py` 已下载在 `katago_official/`。
- **hzyhhzy/KataGomo**（智子分布式训练的 fork，Muon 优化器在这）  
  分支 `kata1_training_tests`：`https://github.com/hzyhhzy/KataGomo/tree/kata1_training_tests`
- **Muon 优化器**（Keller Jordan, 2024）  
  `https://kellerjordan.github.io/posts/muon/`
- **本权重命名解析**：`kata1-zhizi-b28c512nbt-muonfd2.ckpt`
  - `kata1` = KataGo 训练 run 名
  - `zhizi` = 「智子」中文社区 run（**非** lightvector 官方权重）
  - `b28c512nbt` = 28 块 / 512 通道 / nested bottleneck
  - `muonfd2` = Muon 优化器 + 某种 LR 调度变体（"fd2"/"fixed_decay" 不是代码里的 flag，应是 release naming）

---

## 8. 自检命令

```bash
cd KataGoModel
python model.py
```

预期输出：
- 从零 init 后：trunk |x| ≈ 0.04，W/L/draw probs ≈ (0.34, 0.34, 0.32)（接近均匀）。
- 加载 ckpt 后：trunk |x| ≈ 0.37，W/L/draw 在空棋盘上 ≈ (0.91, 0.09, 0.001)。

若数值偏离这个范围 1 个数量级以上，多半是 `set_norm_scales()` 没调，或者改代码时踩回了第 3 节里的某个坑。
