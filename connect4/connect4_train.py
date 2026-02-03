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
    
    # 创建Connect4游戏实例
    game = Connect4(history_step=3)
    
    # 创建网络模型
    model = ResNet(game, num_blocks=2, num_channels=128).to('cuda')
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练参数
    args = {
        'mode': 'train',
        'num_simulations': 600,
        'c_puct': 1.5,
        'temperature': 1.0,
        
        'zero_t_step': 10,  # Connect4平均游戏长度较长，适当增加
        
        'dirichlet_alpha': 0.5,  # Connect4只有7个动作，alpha可以稍大
        'dirichlet_epsilon': 0.25,
        
        'buffer_size': 10000,
        'batch_size': 512,
        'min_buffer_size': 2000,
        
        'train_steps_per_generation': 10,
        'num_games_per_generation': 20,
        
        'target_ReplayRatio': 5,
        
        'playout_cap_min_ratio': 0.3,
        'playout_cap_exponent': 1.5,
        
        'policy_training_threshold': 0.5,
        
        'device': 'cuda',
        'savetime_interval': 300,  # 5分钟保存一次
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
