import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from tictactoe import TicTacToe
from nets import ResNet
from utils import print_board

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=2)
    model = ResNet(game, num_blocks=2, num_channels=32).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    args = {
        'mode': 'eval',
        'num_simulations': 200,
        'c_puct': 1.4,
        'move_temperature_init': 0.5,
        'move_temperature_final': 0.1,
        'total_dirichlet_alpha': 10.83,
        'dirichlet_epsilon': 0,
        'buffer_size': 10000,
        'file_name': 'tictactoe',
        # 'Q_norm_bounds': [-1, 1],
        'Q_norm_bounds': None,
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
    history = []
    
    while not game.is_terminal(state):
        if to_play == 1:
            while True:
                move = input(f"Human step (enter 'undo' to take back): ")
                if move.lower() == 'undo':
                    if len(history) >= 2:
                        state = history[-2]
                        history = history[:-2]
                        print("Undoing last round...")
                        print_board(state)
                        continue
                    else:
                        print("Cannot undo further!")
                        continue
                
                try:
                    i, j = map(int, move.strip().split())
                    action = i * game.board_size + j
                    legal_actions = game.get_is_legal_actions(state)
                    if not legal_actions[action]:
                        print("Illegal move! Cell is already occupied.")
                        continue
                    break
                except ValueError:
                    print("Invalid input! Please enter coordinates (e.g., '0 1') or 'undo'.")
                except IndexError:
                    print("Coordinates out of bounds!")

            history.append(state.copy())
            state = game.get_next_state(state, action, color)
        elif to_play == -1:
            print(f'AlphaZero step:')
            action, info = alphazero.play(state, color)
            history.append(state.copy())
            state = game.get_next_state(state, action, color)
            action_probs, policy, value = info['action_probs'], info['policy'], info['value']
            print(action_probs.reshape(3, 3))
            print(policy)
            print(value)
        to_play = -to_play
        color = -color
        print_board(state)
