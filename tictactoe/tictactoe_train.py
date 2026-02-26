import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import ResNet
from envs.tictactoe import TicTacToe
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
import numpy as np
import torch.optim as optim

train_args = {
    'mode': 'train',
    
    'num_workers': 16,

    'history_step': 2,
    'num_blocks': 1,
    'num_channels': 16,
    'lr': 0.001,
    'weight_decay': 3e-5,

    'full_search_num_simulations': 30,
    'fast_search_num_simulations': 5,
    'full_search_prob': 0.25,

    'c_puct': 1.5,

    'root_temperature_init': 1.25,
    'root_temperature_final': 1.1,

    'move_temperature_init': 0.8,
    'move_temperature_final': 0.4,

    'total_dirichlet_alpha': 10.83,
    'dirichlet_epsilon': 0.25,

    'batch_size': 128,
    'max_grad_norm': 1,

    'min_buffer_size': 500,
    'max_buffer_size': 10000,
    'buffer_size_k': 1,

    'train_steps_per_generation': 5,
    'target_ReplayRatio': 5,

    'fpu_reduction_max': 0,
    'root_fpu_reduction_max': 0,
    
    'savetime_interval': 12000,
    'file_name': 'tictactoe',
    'data_dir': 'data/tictactoe',
    'device': 'cuda',
    'save_on_exit': False,
}

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=train_args['history_step'])
    model = ResNet(game, num_blocks=train_args['num_blocks'], num_channels=train_args['num_channels']).to(train_args['device'])
    optimizer = optim.AdamW(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    # alphazero.load_checkpoint()
    alphazero.learn()
