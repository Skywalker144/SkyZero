"""
Connect4 Parallel Training Script
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np

# Import ParallelAlphaZero
from alphazero_parallel import ParallelAlphaZero
from connect4 import Connect4
from nets import ResNet

if __name__ == '__main__':
    # Fix for potential multiprocessing issues on Windows/CUDA
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    np.set_printoptions(precision=2, suppress=True)

    # Initialize Game
    game = Connect4(history_step=2)

    # Initialize Model (Main process)
    # We use this as the master model and for testing/validation if needed
    model = ResNet(game, num_blocks=4, num_channels=64).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    args = {
        'mode': 'train',
        'num_simulations': 700,
        'fast_simulations': 150,
        'full_search_prob': 0.25,
        'c_puct': 1.5,
        'root_temperature_init': 1.25,
        'root_temperature_final': 1.1,
        'move_temperature_init': 0.4,
        'move_temperature_final': 0.1,
        'total_dirichlet_alpha': 10.83,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 100000,
        'batch_size': 1024,
        'min_buffer_size': 10000,

        'train_steps_per_generation': 5,
        'target_ReplayRatio': 8,

        'forced_playouts': True,  # 启用强制搜索
        'forced_playout_coeff': 2.0,
        'policy_target_pruning': True,  # 启用策略目标修剪
        'noise_prune_utility_scale': 0.15,  # 噪声修剪的效用比例

        # 'Q_norm_bounds': [-1, 1],
        'Q_norm_bounds': None,

        'psw_baseline_ratio': 0.5,  # 均匀分配的权重比例
        'psw_min_weight': 0.01,  # 最小权重

        'resign_threshold': -0.99,
        'soft_resign_playout_prob': 0.3,

        'device': 'cuda',
        'savetime_interval': 7200,
        'file_name': 'connect4',
    }

    print(f"Connect4 AlphaZero Parallel Training")
    print(f"Board Size: {game.board_height} x {game.board_width}")
    print(f"Action Space: {game.action_space_size}")
    print(f"Input Planes: {game.num_planes}")
    print(f"Device: {args['device']}")
    print()

    num_workers = 20

    alphazero = ParallelAlphaZero(game, model, optimizer, args, num_workers=num_workers)

    # Try to load existing checkpoint if any
    alphazero.load_checkpoint()
    # alphazero.replay_buffer.clear()
    alphazero.learn()
