# SkyZero

SkyZero 是一个基于强化学习算法 **Gumbel AlphaZero** 的棋类 AI 框架。它实现了高效的并行训练、多游戏环境支持（井字棋、五子棋）以及 Web 端部署。

## Gumbel AlphaZero 原理简介

Gumbel AlphaZero 是对原始 AlphaZero 的重要改进（源自 ICLR 2022 论文 *Policy Improvement by Planning with Gumbel*）。其核心思想是通过更具理论基础的采样和搜索机制替代原有的启发式方法：

1.  **Gumbel-Top-k 采样**：在根节点使用 Gumbel-Max trick 进行无放回采样，选择初始候选动作。
2.  **Sequential Halving (分段折半搜索)**：在搜索过程中，将模拟预算分阶段分配。每一阶段结束后，剔除一半评估值较低的动作，将预算集中在更有潜力的动作上。
3.  **策略改进保证**：相比原始 AlphaZero 依赖 Dirichlet 噪声进行探索，Gumbel AlphaZero 在理论上保证了搜索出的策略（Target Policy）在期望上不差于原始策略网络，即使在模拟次数（Simulations）极少的情况下也能有效学习。
4.  **无偏更新**：通过完成 Q 值的计算和特定的损失函数，使得策略网络能从搜索结果中学习到更丰富的信息。

## 项目特性

- **算法实现**：完整实现 Gumbel AlphaZero 逻辑，包括 Sequential Halving 和策略改进转换。
- **并行化**：支持多进程并行自对弈（Self-play），显著提升数据收集效率。
- **多环境支持**：
    - Tic-Tac-Toe (3x3)
    - Gomoku (五子棋，支持不同棋盘尺寸)
- **跨平台部署**：支持将训练好的模型导出为 ONNX 格式，并可在 Web 端通过 ONNX Runtime 运行。
- **可视化**：包含训练损失和胜率的自动绘图功能。

## 快速开始

### 环境配置
```bash
pip install torch numpy matplotlib tqdm onnx onnxruntime
```

### 训练模型
- **井字棋**：
  ```bash
  python -m tictactoe.tictactoe_train
  ```
- **五子棋**：
  ```bash
  python -m gomoku.gomoku_train
  ```

### 与 AI 对弈
- **井字棋**：
  ```bash
  python -m tictactoe.tictactoe_play
  ```
- **五子棋**：
  ```bash
  python -m gomoku.gomoku_play
  ```

### 模型导出
```bash
python export_onnx.py --model_path path/to/your/model.pth
```

## 参考资料
- [Policy Improvement by Planning with Gumbel (ICLR 2022)](https://arxiv.org/abs/2111.09794)

## 开源协议
本项目采用 [MIT License](LICENSE) 开源。
