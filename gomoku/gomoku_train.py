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

    "num_workers": 16,

    "board_size": 15,
    "history_step": 2,
    "num_blocks": 2,
    "num_channels": 128,
    "lr": 0.0001,
    "weight_decay": 3e-5,

    "full_search_num_simulations": 1000,
    "fast_search_num_simulations": 200,
    "full_search_prob": 0.25,

    "c_puct": 1.5,

    "root_temperature_init": 1.3,
    "root_temperature_final": 1.1,

    "move_temperature_init": 1,
    "move_temperature_final": 0.2,

    "total_dirichlet_alpha": 10.83,
    "dirichlet_epsilon": 0.3,

    "batch_size": 128,
    "max_grad_norm": 1,

    "min_buffer_size": 500,
    "max_buffer_size": 500000,
    "buffer_size_k": 1,

    "train_steps_per_generation": 5,
    "target_ReplayRatio": 5,

    "fpu_reduction_max": 0.2,
    "root_fpu_reduction_max": 0.0,

    "savetime_interval": 7200,
    "file_name": "gomoku",
    "data_dir": "data/gomoku",
    "device": "cuda",
    "save_on_exit": True,
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=train_args["board_size"], history_step=train_args["history_step"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(train_args["device"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
