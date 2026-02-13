import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from tictactoe import TicTacToe
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=3)
    model = ResNet(game, num_blocks=1, num_channels=64).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    args = {
        'mode': 'train',
        'num_simulations': 200,
        'fast_simulations': 50,
        'full_search_prob': 0.25,
        'c_puct': 1.5,
        'temperature': 1.0,
        'root_temperature_init': 1.25,
        'root_temperature_final': 1.1,
        'move_temperature_init': 0.8,
        'move_temperature_final': 0.2,
        'total_dirichlet_alpha': 10.83,
        'dirichlet_epsilon': 0.25,

        'buffer_size': 3000,
        'batch_size': 128,
        'min_buffer_size': 500,

        'train_steps_per_generation': 5,
        'target_ReplayRatio': 8,

        'forced_playouts': True,
        'forced_playout_coeff': 2.0,

        'policy_target_pruning': True,

        'Q_norm_bounds': [-1, 1],

        'policy_surprise_weighting': True,  # 启用PSW
        'psw_baseline_ratio': 0.5,  # 均匀分配的权重比例
        'psw_fast_kl_threshold': 2.0,  # fast search 的 KL 阈值
        'psw_min_weight': 0.01,  # 最小权重
        'psw_stochastic': True,  # 随机采样

        'device': 'cuda',
        'savetime_interval': 120,
        'file_name': 'tictactoe',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    # alphazero.load_checkpoint()
    alphazero.learn()
