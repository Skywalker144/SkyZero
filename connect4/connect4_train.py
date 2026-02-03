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
        'c_puct': 1.5,
        'temperature': 1.0,
        
        'zero_t_step': 10,
        
        'dirichlet_alpha': 1.2,
        'dirichlet_epsilon': 0.25,
        
        'buffer_size': 100000,
        'batch_size': 512,
        'min_buffer_size': 2000,
        
        'train_steps_per_generation': 5,
        'num_games_per_generation': 20,
        
        'target_ReplayRatio': 5,
        
        'playout_cap_min_ratio': 0.3,
        'playout_cap_exponent': 1.5,
        
        'policy_training_threshold': 0.5,
        
        'device': 'cuda',
        'savetime_interval': 1800,
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
