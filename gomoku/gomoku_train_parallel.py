"""
Gomoku Parallel Training Script
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np

# Import AlphaZeroParallel
from alphazero_parallel import AlphaZeroParallel
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
    game = Gomoku(board_size=15, history_step=4)

    # Initialize Model (Main process)
    # We use this as the master model and for testing/validation if needed
    model = ResNet(game, num_blocks=8, num_channels=256).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5)

    args = {
        'mode': 'train',
        'num_simulations': 800,
        'c_puct': 1.5,
        'temperature': 1.0,

        'zero_t_step': 14,

        'dirichlet_alpha': 0.05,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 100000,
        'batch_size': 1024,
        'min_buffer_size': 10000,

        'train_steps_per_generation': 5,

        # 'num_games_per_generation': 16,

        'target_ReplayRatio': 6.0,

        'playout_cap_min_ratio': 0.25,
        'playout_cap_exponent': 1.6,

        'policy_training_threshold': 0.5,

        'Q_norm_bounds': [-1, 1],

        'device': 'cuda',  # Workers will try to use this too.
        'savetime_interval': 3600,

        'file_name': 'gomoku',
    }

    print(f"Gomoku AlphaZero Parallel Training")
    print(f"Board Size: {game.board_height} x {game.board_width}")
    print(f"Action Space: {game.action_space_size}")
    print(f"Input Planes: {game.num_planes}")
    print(f"Device: {args['device']}")
    print()

    # Define model_cls and model_kwargs for workers
    model_cls = ResNet
    model_kwargs = {
        'game': game,
        'num_blocks': 8,
        'num_channels': 256
    }

    num_workers = 18

    alphazero = AlphaZeroParallel(
        game,
        model,
        optimizer,
        args,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        num_workers=num_workers
    )

    # Try to load existing checkpoint if any
    alphazero.load_checkpoint()
    alphazero.learn()
