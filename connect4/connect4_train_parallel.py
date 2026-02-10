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
    game = Connect4(history_step=2)
    
    # Initialize Model (Main process)
    # We use this as the master model and for testing/validation if needed
    model = ResNet(game, num_blocks=4, num_channels=64).to('cuda')
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
        
        # Playout Cap Randomization (二选一策略)
        'fast_simulations': 100,  # 快速搜索的 simulation 数量
        'full_search_prob': 0.25,  # 全量搜索的概率

        'forced_playouts': True,
        'forced_playout_coeff': 2.0,

        'policy_target_pruning': True,

        'Q_norm_bounds': [-1, 1],

        'policy_surprise_weighting': True,  # 启用PSW
        'psw_baseline_ratio': 0.5,  # 均匀分配的权重比例
        'psw_fast_kl_threshold': 2.0,  # fast search 的 KL 阈值
        'psw_min_weight': 0.01,  # 最小权重
        'psw_stochastic': True,  # 随机采样

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
        'num_blocks': 4,
        'num_channels': 64
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
    alphazero.replay_buffer.clear()
    alphazero.learn()
