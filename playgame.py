import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.optim as optim
import numpy as np
from alphazero_play import TreeReuseAlphaZero
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
        alphazero = TreeReuseAlphaZero(self.game, model, optimizer, self.args)
        alphazero.load_checkpoint()
        to_play = int(input(
            f"1 for the first move and -1 for the second move\n"
            f"The position of the piece needs to be input in coordinate form.\n"
            f"   (first input the vertical coordinate, then the horizontal coordinate).\n"
            f"Please enter:"
        ))
        color = 1
        state = self.game.get_initial_state()
        history = []
        mcts_root = None
        print_board(state)
        while True:
            if self.game.is_terminal(state):
                winner = self.game.get_winner(state)
                if winner == 1:
                    print("Black wins!")
                elif winner == -1:
                    print("White wins!")
                else:
                    print("Draw!")
                
                resp = input("Game Over. 'u' to undo, 'q' to quit: ").strip().lower()
                if resp == "u":
                    if len(history) >= 2:
                        state, to_play, color = history.pop() # State before AI move
                        state, to_play, color = history.pop() # State before Human move
                        mcts_root = None # Reset MCTS tree on undo
                        print("Undo successful.")
                        print_board(state)
                        continue
                    else:
                        print("Nothing to undo.")
                        break
                else:
                    break

            if to_play == 1:
                while True:
                    move = input(f"Human step (row col / 'u' for undo / 'q' for quit): ").strip().lower()
                    if move == "u":
                        if len(history) >= 2:
                            state, to_play, color = history.pop()  # Revert to state before AI move
                            state, to_play, color = history.pop()  # Revert to state before Human move
                            mcts_root = None # Reset MCTS tree on undo
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

                history.append((state.copy(), to_play, color))
                mcts_root = alphazero.apply_action(mcts_root, action)
                state = self.game.get_next_state(state, action, color)
            elif to_play == -1:
                history.append((state.copy(), to_play, color))
                print(f"AlphaZero step:")
                action, info, mcts_root = alphazero.play(state, color, root=mcts_root, additive=True)

                state = self.game.get_next_state(state, action, color)
                mcts_policy, nn_policy, value_probs, root_value, nn_value = info["mcts_policy"], info["nn_policy"], info["value_probs"], info["root_value"], info["nn_value"]
                print(f"MCTS Strategy:\n{mcts_policy}")
                print(f"NN Strategy:\n{nn_policy}")
                print(
                    f"Win  Probability: {value_probs[0]:.2f}\n"
                    f"Draw Probability: {value_probs[1]:.2f}\n"
                    f"Lose Probability: {value_probs[2]:.2f}"
                )
                print(f"root value: {root_value:.2f}")
                print(f"nn value:   {nn_value:.2f}")
                opponent_policy = info["opponent_policy"]
                ownership = info["ownership"]
                print(f"Opponent Policy:\n{opponent_policy}")
                print(f"Ownership:\n{ownership}")
                print()
                print(info["root_n"])

                # heatmaps = {
                #     "MCTS Strategy": info["mcts_policy"],
                #     "NN Strategy": info["nn_policy"],
                #     "Opponent Policy": info["opponent_policy"],
                #     "Ownership": info["ownership"]
                # }
                # plot_heatmaps(self.game.board_size, heatmaps)

            to_play = -to_play
            color = -color
            print_board(state)
