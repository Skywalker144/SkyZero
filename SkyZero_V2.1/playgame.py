import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from nets import ResNet
from utils import print_board


class GamePlayer:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.args["mode"] = "eval"

    def play(self):
        np.set_printoptions(precision=2, suppress=True)
        model = ResNet(self.game, num_blocks=self.args["num_blocks"], num_channels=self.args["num_channels"]).to(self.args["device"])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        alphazero = AlphaZero(self.game, model, optimizer, self.args)
        alphazero.load_model()
        human_side = int(input(
            f"1 for Human first move and -1 for Human second move\n"
            f"The position of the piece needs to be input in coordinate form.\n"
            f"   (first input the vertical coordinate, then the horizontal coordinate).\n"
            f"Please enter:"
        ))
        to_play = 1
        color = 1
        state = self.game.get_initial_state()
        last_action = None
        last_player = None
        history = []
        mcts_root = None
        print_board(state)
        while True:
            if self.game.is_terminal(state, last_action, last_player):
                winner = self.game.get_winner(state, last_action, last_player)
                if winner == 1:
                    print("Black wins!")
                elif winner == -1:
                    print("White wins!")
                else:
                    print("Draw!")
                
                resp = input("Game Over. 'u' to undo, 'q' to quit: ").strip().lower()
                if resp == "u":
                    if len(history) >= 2:
                        state, to_play, color, mcts_root, last_action, last_player = history.pop()
                        state, to_play, color, mcts_root, last_action, last_player = history.pop()
                        print("Undo successful.")
                        print_board(state)
                        continue
                    else:
                        print("Nothing to undo.")
                        break
                else:
                    break

            if to_play == human_side:
                while True:
                    move = input(f"Human step (row col / 'u' for undo / 'q' for quit): ").strip().lower()
                    if move == "u":
                        if len(history) >= 2:
                            state, to_play, color, mcts_root, last_action, last_player = history.pop()
                            state, to_play, color, mcts_root, last_action, last_player = history.pop()
                            print("Undo successful.")
                            print_board(state)
                            continue
                        else:
                            print("Nothing to undo.")
                            continue
                    elif move == "q":
                        print("Exiting game.")
                        return

                    try:
                        i, j = map(int, move.split())
                        action = i * self.game.board_size + j
                        if not self.game.get_is_legal_actions(state, to_play)[action]:
                            print(f"Invalid move: ({i}, {j}) is forbidden or occupied.")
                            continue
                        break
                    except (ValueError, IndexError):
                        print("Invalid input format. Please enter 'row col' (e.g., '7 7').")

                history.append((state.copy(), to_play, color, mcts_root, last_action, last_player))

                state = self.game.get_next_state(state, action, color)
                last_action = action
                last_player = to_play

                if mcts_root is not None:
                    next_root = None
                    for child in mcts_root.children:
                        if child.action_taken == action:
                            next_root = child
                            break
                    if next_root is not None:
                        next_root.parent = None
                        mcts_root = next_root
                    else:
                        mcts_root = None
            elif to_play == -human_side:
                history.append((state.copy(), to_play, color, mcts_root, last_action, last_player))
                print(f"AlphaZero step:")
                action, info, mcts_root = alphazero.play(state, color, root=mcts_root)

                state = self.game.get_next_state(state, action, color)
                last_action = action
                last_player = to_play

                if mcts_root is not None:
                    next_root = None
                    for child in mcts_root.children:
                        if child.action_taken == action:
                            next_root = child
                            break
                    if next_root is not None:
                        next_root.parent = None
                        mcts_root = next_root
                    else:
                        mcts_root = None

                print(f"MCTS Strategy:\n{info['mcts_policy']}")
                print(f"NN Strategy:\n{info['nn_policy']}")
                print(
                    f"Win  Probability: {info['nn_value_probs'][0]:.2f}\n"
                    f"Draw Probability: {info['nn_value_probs'][1]:.2f}\n"
                    f"Lose Probability: {info['nn_value_probs'][2]:.2f}"
                )
                root_value_p = info['root_value_probs']
                nn_val_p = info['nn_value_probs']
                print(
                    f"          {'Win':>6s}  {'Draw':>6s}  {'Loss':>6s}  {'W-L':>6s}\n"
                    f"  root_value:  {root_value_p[0]:6.2%}  {root_value_p[1]:6.2%}  {root_value_p[2]:6.2%}  {info['root_value']:+.2f}\n"
                    f"  nn_value:    {nn_val_p[0]:6.2%}  {nn_val_p[1]:6.2%}  {nn_val_p[2]:6.2%}  {info['nn_value']:+.2f}"
                )
                print(f"Opponent Policy:\n{info['nn_opponent_policy']}")
                print()
                print(f"Actual Search Num: {info['actual_search_num']}")
            to_play = -to_play
            color = -color
            print_board(state)
