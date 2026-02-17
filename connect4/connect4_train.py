"""
Connect4 训练脚本
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from connect4 import Connect4
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    game = Connect4(history_step=3)
    model = ResNet(game, num_blocks=2, num_channels=128).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    args = {
        'mode': 'train',
        'num_simulations': 600,
        'fast_simulations': 100,
        'full_search_prob': 0.25,
        'c_puct': 1.5,
        'root_temperature_init': 1.25,
        'root_temperature_final': 1.1,
        'move_temperature_init': 0.8,
        'move_temperature_final': 0.2,
        'total_dirichlet_alpha': 10.83,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 100000,
        'batch_size': 1024,
        'min_buffer_size': 5000,
        'buffer_size_k': 1.0,

        'train_steps_per_generation': 5,
        'target_ReplayRatio': 8,

        'forced_playout_coeff': 2.0,

        # 'Q_norm_bounds': [-1, 1],
        'Q_norm_bounds': None,

        'psw_baseline_ratio': 0.5,  # 均匀分配的权重比例
        'psw_fast_kl_threshold': 2.0,  # fast search 的 KL 阈值
        'psw_min_weight': 0.01,  # 最小权重

        'device': 'cuda',
        'savetime_interval': 7200,
        'file_name': 'connect4',
    }

    print(f"Connect4 AlphaZero Training")
    print(f"Board Size: {game.board_height} x {game.board_width}")
    print(f"Action Space: {game.action_space_size}")
    print(f"Input Planes: {game.num_planes}")
    print()

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
