import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from tictactoe import TicTacToe
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=3)
    model = ResNet(game, num_blocks=1).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    args = {
        'mode': 'train',
        'num_simulations': 200,
        'c_puct': 1.5,
        'temperature': 1.0,

        'root_temperature_start': 1.05,
        'root_temperature_end': 1.03,
        'zero_t_step': 3,

        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 3000,
        'batch_size': 128,
        'min_buffer_size': 500,

        'train_steps_per_generation': 5,
        'num_games_per_generation': 20,

        # Replay Ratio: (batch_size * train_steps_per_generations) / (num_games_per_generation * Avg_steps_per_game)
        # RR 通常在2到8之间
        # 太高：容易对旧数据过拟合，策略停滞，MCTS有概率策略坍缩
        # 太低：样本利用率不足，训练速度慢

        # RR: 256*5/20/8=8

        'target_ReplayRatio': 8,

        'playout_cap_min_ratio': 0.2,
        'playout_cap_exponent': 1.5,

        'policy_training_threshold': 0.5,

        'device': 'cuda',
        'savetime_interval': 120,
        'file_name': 'tictactoe',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
