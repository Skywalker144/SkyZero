import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tictactoe_train import train_args
from playgame import GamePlayer
from envs.tictactoe import TicTacToe


eval_args = {
    "mode": "eval",
    "num_blocks": train_args["num_blocks"],
    "num_channels": train_args["num_channels"],
    "lr": train_args["lr"],
    "weight_decay": train_args["weight_decay"],

    "num_simulations": 64,
    "enable_symmetry_inference_for_root": True,
    "enable_symmetry_inference_for_child": True,

    "gumbel_m": train_args["gumbel_m"],

    "fpu_reduction_max": train_args["fpu_reduction_max"],
    "root_fpu_reduction_max": train_args["root_fpu_reduction_max"],

    "file_name": "tictactoe",
    "data_dir": "data/tictactoe",
    "device": "cuda",
}

if __name__ == "__main__":
    gp = GamePlayer(TicTacToe(), eval_args)
    gp.play()
