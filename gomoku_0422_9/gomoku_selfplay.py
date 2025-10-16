import torch
import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet
from utils import print_board

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=9)
    model = ResNet(game, num_blocks=6, num_channels=256).to('cuda')
    model.load_state_dict(torch.load(f'gomoku_model.pt'))
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    args = {
        'mode': 'eval',
        'num_simulations': 600,
        'c_puct': 1.4,
        'temperature': 0.5,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.1,
        'device': 'cuda'
    }

    alphazero = AlphaZero(game, model, optimizer, args)

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
