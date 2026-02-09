"""
Connect4 Parallel Training Script
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np

# Import AlphaZeroParallel
from alphazero_parallel import AlphaZeroParallel
from connect4 import Connect4
from nets import ResNet

if __name__ == '__main__':
    # Fix for potential multiprocessing issues on Windows/CUDA
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    np.set_printoptions(precision=2, suppress=True)

    # Initialize Game
    game = Connect4(history_step=3)
    
    # Initialize Model (Main process)
    # We use this as the master model and for testing/validation if needed
    model = ResNet(game, num_blocks=2, num_channels=128).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    args = {
        'mode': 'train',
        'num_simulations': 600,
        'c_puct': 1.5,
        'temperature': 1.0,
        
        'zero_t_step': 10,
        
        'dirichlet_alpha': 1.2,
        'dirichlet_epsilon': 0.25,
        
        'buffer_size': 100000,
        'batch_size': 1024,
        'min_buffer_size': 5000,

        'train_steps_per_generation': 10,
        # 'num_games_per_generation': 20,

        'target_ReplayRatio': 6,
        
        'playout_cap_min_ratio': 0.2,
        'playout_cap_exponent': 1.5,
        
        'policy_training_threshold': 0.5,

        'Q_norm_bounds': [-1, 1],
        
        'device': 'cuda', # Workers will try to use this too. 
        'savetime_interval': 3600,

        'file_name': 'connect4', 
    }
    
    print(f"Connect4 AlphaZero Parallel Training")
    print(f"Board Size: {game.board_height} x {game.board_width}")
    print(f"Action Space: {game.action_space_size}")
    print(f"Input Planes: {game.num_planes}")
    print(f"Device: {args['device']}")
    print()

    # Define model_cls and model_kwargs for workers
    model_cls = ResNet
    model_kwargs = {
        'game': game,
        'num_blocks': 2,
        'num_channels': 128
    }

    num_workers = 20

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
