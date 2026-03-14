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
    "history_step": 1,
    "num_blocks": 4,
    "num_channels": 128,
    "lr": 0.0001,
    "weight_decay": 3e-5,

    "num_simulations": 64,
    "batch_size": 256,

    # Gumbel settings
    "gumbel_m": 16,
    "gumbel_c_visit": 50,
    "gumbel_c_scale": 1.0,

    "enable_stochastic_transform_inference_for_child": True,
    "enable_stochastic_transform_inference_for_root": True,

    "min_buffer_size": 20480,
    "linear_threshold": 200000,
    "alpha": 0.8,
    "max_buffer_size": 1e7,

    "train_steps_per_generation": 5,
    "target_ReplayRatio": 5,

    "fpu_reduction_max": 0.08,
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
    # alphazero.load_checkpoint()
    alphazero.learn()
