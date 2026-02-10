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

        # Playout Cap Randomization (二选一策略)
        'fast_simulations': 150,  # 快速搜索的 simulation 数量
        'full_search_prob': 0.25,  # 全量搜索的概率

        'forced_playouts': True,
        'forced_playout_coeff': 2.0,

        'policy_target_pruning': True,

        # === Policy Surprise Weighting (PSW) ===
        # KataGo 的重要改进之一，通过增加"令人惊讶"的样本的采样频率来加速学习
        'policy_surprise_weighting': True,  # 是否启用 PSW
        'psw_baseline_ratio': 0.5,           # 均匀分配的权重比例 (默认 0.5)
        'psw_fast_kl_threshold': 2.0,        # fast search 被包含的 KL 散度阈值
        'psw_min_weight': 0.01,              # 最小权重
        'psw_stochastic': True,              # 是否使用随机采样

        'Q_norm_bounds': [-1, 1],

        'device': 'cuda',
        'savetime_interval': 3600,
        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
