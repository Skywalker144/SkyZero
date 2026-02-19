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
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5)
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
    history = []
    
    while not game.is_terminal(state):
        if to_play == 1:
            while True:
                move = input(f"Human step (row col) or 'undo': ")
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
                    if not (0 <= i < game.board_size and 0 <= j < game.board_size):
                        print("Coordinates out of bounds!")
                        continue

                    action = i * game.board_size + j
                    legal_actions = game.get_is_legal_actions(state)
                    if not legal_actions[action]:
                        print("Illegal move! (Occupied or Forbidden)")
                        continue
                    break
                except ValueError:
                    print("Invalid input! Please enter 'row col' (e.g., '7 7') or 'undo'.")

            history.append(state.copy())
            state = game.get_next_state(state, action, color)
        elif to_play == -1:
            print(f'AlphaZero step:')
            action, info = alphazero.play(state, color)
            history.append(state.copy())
            state = game.get_next_state(state, action, color)
            action_probs, policy, ai_winrate = info['action_probs'], info['policy'], info['ai_winrate']
            print(action_probs.reshape(game.board_size, game.board_size))
            print(policy)
            print(ai_winrate)
        to_play = -to_play
        color = -color
        print_board(state)
