import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=15, history_step=4)
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

        'target_ReplayRatio': 6,

        'playout_cap_min_ratio': 0.25,
        'playout_cap_exponent': 1.6,

        'policy_training_threshold': 0.5,

        'Q_norm_bounds': [-1, 1],

        'device': 'cuda',
        'savetime_interval': 3600,
        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
