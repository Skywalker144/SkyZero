import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import ResNet
from envs.tictactoe import TicTacToe
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
import numpy as np
import torch.optim as optim

train_args = {

    "num_iterations": 100,  # 总共训练多少个iteration。设置为0即一直跑，CtrlC停止。
    
    "num_blocks": 3,       # 网络block数量
    "num_channels": 32,    # 网络channels
    "lr": 0.001,           # 学习率
    "weight_decay": 3e-5,  # 权重衰减系数

    "num_simulations": 50,      # 模拟次数
    "dirichlet_alpha": 1,     # 控制Dirichlet噪声的锐度，不用动
    "dirichlet_epsilon": 0.25,  # 控制Dirichlet噪声在根节点的占比，不用动

    "batch_size": 128,                # 训练时的batch_size
    "train_steps_per_iteration": 50,  # train阶段训多少个batch
    "target_ReplayRatio": 4,          # 样本回放率，即每个样本会被训练的次数。

    "move_temperature": 1.0,   # 落子温度
    "half_life": 6,            # 每一局经过这个步数之后改为argmax落子

    "min_buffer_size": 6400,        # 最小多少样本开始训（需 >= batch_size * train_steps_per_iteration）
    "max_buffer_size": 1e7,      # 最大容纳多少样本（受内存限制）
    "window_exponent": 0.65,        # 不用动
    "window_expand_per_row": 0.4,   # 不用动
    
    "save_interval": 10,           # 隔几个iteration保存一次checkpoint
    "data_dir": "data/tictactoe_",
    "device": "mps",

    "num_workers": 12,     # 并行worker数量
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe()
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
