"""
Gomoku Parallel Training Script
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np

# Import ParallelAlphaZero
from alphazero_parallel import ParallelAlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    # Fix for potential multiprocessing issues on Windows/CUDA
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    np.set_printoptions(precision=2, suppress=True)

    # Initialize Game
    game = Gomoku(board_size=15, history_step=3)

    # Initialize Model (Main process)
    # We use this as the master model and for testing/validation if needed
    model = ResNet(game, num_blocks=6, num_channels=128).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)

    args = {
        'mode': 'train',
        'num_simulations': 800,
        'fast_simulations': 200,
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
        'min_buffer_size': 10000,

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
        'file_name': 'gomoku',
    }

    print(f"Gomoku AlphaZero Parallel Training")
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
