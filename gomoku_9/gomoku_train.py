import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=9, history_step=4)
    model = ResNet(game, num_blocks=4, num_channels=256).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-5)
    args = {
        'mode': 'train',
        'num_simulations': 600,
        'c_puct': 1.5,
        'temperature': 1.0,

        'root_temperature_start': 1.05,
        'root_temperature_end': 1.03,
        'zero_t_step': 14,

        'dirichlet_alpha': 0.05,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 100000,
        'batch_size': 1024,
        'min_buffer_size': 10000,

        'train_steps_per_generation': 5,
        'num_games_per_generation': 16,

        # Replay Ratio: (batch_size * train_steps_per_generations) / (num_games_per_generation * Avg_steps_per_game)
        # RR 通常在2到8之间
        # 太高：容易对旧数据过拟合，策略停滞，MCTS有概率策略坍缩
        # 太低：样本利用率不足，训练速度慢

        # RR = (1024 * 5) / (16 * avg_game_len) = 3.2

        'target_ReplayRatio': 6,

        'playout_cap_min_ratio': 0.2,
        'playout_cap_exponent': 1.6,

        'policy_training_threshold': 0.5,

        'device': 'cuda',
        'savetime_interval': 600,
        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
