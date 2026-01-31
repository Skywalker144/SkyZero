import torch
import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet
from utils import print_board

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=9, history_step=4)
    model = ResNet(game, num_blocks=4, num_channels=256, history_step=4).to('cuda')
    optimizer = None
    args = {
        'mode': 'eval',
        'num_simulations': 600,
        'c_puct': 1,
        'buffer_size': 100000,
        'file_name': 'gomoku',
        'device': 'cuda'
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()

    to_play = int(input(
        f'1 for the first move and -1 for the second move\n'
        f'The position of the piece needs to be input in coordinate form.\n'
        f'   (first input the vertical coordinate, then the horizontal coordinate).\n'
        f'Please enter:'
    ))
    color = 1
    state = game.get_initial_state()
    print_board(state)
    while not game.is_terminal(state):
        if to_play == 1:
            move = input(f"Human step:")
            i, j = map(int, move.strip().split())
            action = i * game.board_size + j
            state = game.get_next_state(state, action, color)
        elif to_play == -1:
            print(f'AlphaZero step:')
            action, info = alphazero.play(state, color)
            state = game.get_next_state(state, action, color)
            action_probs, policy, ai_winrate = info['action_probs'], info['policy'], info['ai_winrate']
            print(action_probs.reshape(game.board_size, game.board_size))
            print(policy)
            print(ai_winrate)
        to_play = -to_play
        color = -color
        print_board(state)
