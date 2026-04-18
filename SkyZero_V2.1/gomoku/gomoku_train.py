import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
from envs.gomoku import Gomoku
from nets import ResNet

train_args = {
    "mode": "train",

    "num_workers": 19,

    "board_size": 15,
    "num_blocks": 4,
    "num_channels": 128,
    "lr": 0.0001,
    "weight_decay": 3e-5,

    "full_search_num_simulations": 480,
    "fast_search_num_simulations": 80,
    "full_search_prob": 0.25,

    # "enable_symmetry_inference_for_root": True,
    "enable_stochastic_transform_inference_for_root": True,
    "enable_stochastic_transform_inference_for_child": True,

    "root_temperature_init": 1.25,
    "root_temperature_final": 1.1,

    "move_temperature_init": 0.8,
    "move_temperature_final": 0.2,

    "total_dirichlet_alpha": 6.75,
    "dirichlet_epsilon": 0.25,

    "batch_size": 128,
    "max_grad_norm": 1,

    "min_buffer_size": 5e5,
    "linear_threshold": 5e6,
    "alpha": 0.75,
    "max_buffer_size": 5e7,

    "train_steps_per_generation": 100,
    "target_ReplayRatio": 8,

    "fpu_reduction_max": 0.08,
    "root_fpu_reduction_max": 0,

    "savetime_interval": 7200,
    "file_name": "gomoku",
    "data_dir": "data/gomoku",
    "device": "cuda",
    "save_on_exit": True,
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=train_args["board_size"])
    game.load_openings("envs/gomoku_openings.txt", empty_board_prob=0.5)
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(train_args["device"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    # alphazero.load_checkpoint()
    alphazero.learn()
