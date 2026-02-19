import torch
import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet
from utils import print_board

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=15, history_step=2)
    model = ResNet(game, num_blocks=6, num_channels=128).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    args = {
        'mode': 'eval',
        'num_simulations': 600,
        'c_puct': 1.4,
        'move_temperature_init': 0.4,
        'move_temperature_final': 0.1,
        # 'Q_norm_bounds': [-1, 1],
        'min_buffer_size': 10000,
        'max_buffer_size': 20000,
        'buffer_size_k': 0.5,
        'Q_norm_bounds': None,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1,
        'device': 'cuda',
        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()

    to_play = 1
    color = 1
    state = game.get_initial_state()
    print_board(state[0])
    while not game.is_terminal(state):
        if to_play == 1:
            print(f'AlphaZero step:')
            action, info = alphazero.play(state, color)
            state = game.get_next_state(state, action, color)
            action_probs, policy, value = info['action_probs'], info['policy'], info['value']
            print(action_probs.reshape(game.board_size, game.board_size))
            print(policy)
            print(value)
        elif to_play == -1:
            print(f'AlphaZero step:')
            action, info = alphazero.play(state, color)
            state = game.get_next_state(state, action, color)
            action_probs, policy, value = info['action_probs'], info['policy'], info['value']
            print(action_probs.reshape(game.board_size, game.board_size))
            print(policy)
            print(value)
        to_play = -to_play
        color = -color
        print_board(state[0])
