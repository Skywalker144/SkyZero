# 网络规模设置 (num_blocks / num_channels)

V0 用的是简单 ResBlock(`nets.py:ResBlock`),每 block 含 **2 个 3×3 卷积**。深度与宽度的经验比值:

| Block 类型 | 来源 | 经验比值 `c / b` | 例子 |
|---|---|---|---|
| **Simple ResBlock(V0 适用)** | AlphaZero 论文 / KataGo 早期 | **~8–13** | b10c128 / b15c192 / b20c256 / b19c256 / b39c256 |
| Nested bottleneck(V6 / KataGo 现代) | KataGo 现代 | ~10–20 | b8c96 / b12c128 / b28c512 / b40c768 |

## 选 V0 配置时的原则

1. `c ≈ 10 · b` 起步,围绕这个比值微调。
2. 想加容量,优先**同步加深加宽**,别只单向加。
3. **不要把现代 KataGo 的 `c ≈ 20·b` 直接套到 V0** —— bottleneck 把每 block 参数压到 ~10·c²,simple ResBlock 是 18·c²,直接套 20× 会得到"又胖又浅"、单位算力下深度不够的网络。

## V0 vs V6 block 结构对比(为什么比值不同)

- **V0 block**: 2 × (3×3 conv,c→c)。每 block ≈ 18·c² 参数。
- **V6 block** (`NestedBottleneckResBlock`,`internal_length=2`):
  - 1×1 下投影 c_main → c_mid (≈ c_main/2)
  - 2 个内层 ResBlock(每个 2 × 3×3 conv,在 c_mid 空间)
  - 1×1 上投影 c_mid → c_main
  - 共 ~6 个卷积,每 block ≈ 10·c_main² 参数
- 同 num_blocks 下,V6 总卷积层数 ≈ V0 的 3 倍;同 num_channels 下,V0 单 block 参数量 ≈ V6 的 1.8 倍。

## 当前 V0 默认值参考

- `gomoku/gomoku_train.py`: b4c128(比值 32,偏胖)→ 可考虑改为 b10c128 或 b8c96。
- `tictactoe/tictactoe_train.py`: b2c32(比值 16)→ 小棋盘小网够用,不需要动。
