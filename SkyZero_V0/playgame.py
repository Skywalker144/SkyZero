import os
import sys
import numpy as np
import torch
from alphazero import AlphaZero, Node, temperature_transform
from nets import ResNet
from utils import print_board

class GamePlayer:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.args["mode"] = "eval"
        self.device = args.get("device", "cpu")

    def play(self):
        np.set_printoptions(precision=2, suppress=True)
        model = ResNet(self.game, num_blocks=self.args["num_blocks"], num_channels=self.args["num_channels"]).to(self.device)
        
        # We don't need an optimizer for playing, but AlphaZero init wants one
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        alphazero = AlphaZero(self.game, model, optimizer, self.args)
        if not alphazero.load_checkpoint():
            print("Warning: Model not loaded, using random weights.")
        
        human_side = int(input(
            "Enter 1 to play as Black (first), -1 to play as White (second): "
        ))
        
        state = self.game.get_initial_state()
        to_play = 1  # 1 is Black, -1 is White
        print_board(state)
        
        while not self.game.is_terminal(state):
            if to_play == human_side:
                move_str = input("Your move (row col): ").strip()
                try:
                    r, c = map(int, move_str.split())
                    action = r * self.game.board_size + c
                    if not self.game.get_is_legal_actions(state, to_play)[action]:
                        print("Illegal move!")
                        continue
                except:
                    print("Invalid input! Use 'row col'.")
                    continue
            else:
                print("AlphaZero is thinking...")
                mcts_policy = alphazero.mcts.search(state, to_play, self.args["num_simulations"])
                # In eval, we usually take the most visited move
                action = np.argmax(mcts_policy)
                r, c = action // self.game.board_size, action % self.game.board_size
                print(f"AlphaZero plays: {r} {c}")
            
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
            print_board(state)
            
        winner = self.game.get_winner(state)
        if winner == 1:
            print("Black wins!")
        elif winner == -1:
            print("White wins!")
        else:
            print("It's a draw!")
