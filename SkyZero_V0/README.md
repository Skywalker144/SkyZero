# SkyZero: AlphaZero Implementation in PyTorch

SkyZero 是一个简洁且高效的 AlphaZero 算法实现项目，支持井字棋 (Tic-Tac-Toe) 和五子棋 (Gomoku) 等环境。

## 1. AlphaZero 核心原理

AlphaZero 的核心思想是**蒙特卡洛树搜索 (MCTS)** 与**深度神经网络 (DNN)** 的结合。

*   **双头网络 (Dual-head Network)**: 使用一个 ResNet 结构的神经网络，同时输出：
    *   **策略 (Policy, $p$)**: 在当前状态下各合法动作的预测概率。
    *   **价值 (Value, $v$)**: 对当前状态胜率的估计（范围 -1 到 1）。
*   **零知识学习 (Zero Knowledge)**: 不依赖人类棋谱或启发式评估函数，仅通过游戏规则，从随机权重开始进行自我博弈。

## 2. 算法流程

AlphaZero 的训练是一个不断迭代的过程，主要包含三个步骤：

1.  **自我博弈 (Self-Play)**:
    *   智能体通过 MCTS 选择动作。
    *   MCTS 使用当前神经网络来指导搜索过程（减少搜索宽度并进行状态评估）。
    *   记录博弈过程中的状态、MCTS 产生的概率分布以及最终胜负结果。
2.  **数据收集 (Data Collection)**:
    *   将自我博弈产生的经验数据存入经验回放池 (Replay Buffer)。
3.  **网络优化 (Optimization)**:
    *   从回放池中抽取样本，训练神经网络。
    *   **目标函数**: 使得预测策略 $p$ 接近 MCTS 概率，预测价值 $v$ 接近真实胜负。
    *   $L = (z - v)^2 - \pi^\top \log p + c \|\theta\|^2$

## 3. 项目结构

*   `alphazero.py` & `alphazero_parallel.py`: AlphaZero 训练逻辑（支持并行化）。
*   `nets.py`: ResNet 网络定义。
*   `envs/`: 游戏环境定义。
*   `tictactoe/` & `gomoku/`: 针对特定游戏的训练脚本和对战脚本。
*   `playgame.py`: 通用的游戏对战工具。

## 4. 快速开始

### 训练模型
以井字棋为例：
```bash
python tictactoe/tictactoe_train.py
```

### 与模型对弈
```bash
python tictactoe/tictactoe_play.py
```

## 5. 依赖项

*   Python 3.x
*   PyTorch
*   NumPy
*   tqdm

## 6. 许可证

本项目基于 [MIT License](LICENSE) 协议。
