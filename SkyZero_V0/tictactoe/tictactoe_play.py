import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tictactoe_train import train_args
from playgame import GamePlayer
from envs.tictactoe import TicTacToe


eval_args = {
    "num_blocks": train_args["num_blocks"],
    "num_channels": train_args["num_channels"],
    "num_simulations": 100,
    "c_puct": 1.5,
    "data_dir": train_args["data_dir"],
    "device": "cuda",
}

if __name__ == "__main__":
    game = TicTacToe(history_step=train_args["history_step"])
    gp = GamePlayer(game, eval_args)
    gp.play()
