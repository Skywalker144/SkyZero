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

    "num_simulations": 512,
    "batch_size": 256,

    # Gumbel settings
    "gumbel_m": 32,
    "gumbel_c_visit": 50,
    "gumbel_c_scale": 1.0,

    "enable_stochastic_transform_inference_for_child": True,
    "enable_stochastic_transform_inference_for_root": True,

    "min_buffer_size": 8e5,
    "linear_threshold": 5e6,
    "alpha": 0.75,
    "max_buffer_size": 5e7,

    "half_life": 10,

    "train_steps_per_generation": 100,
    "target_ReplayRatio": 8,

    "fpu_reduction_max": 0.08,
    "root_fpu_reduction_max": 0.0,

    "savetime_interval": 14400,
    "file_name": "gomoku",
    "data_dir": "data/gomoku",
    "device": "cuda",
    "save_on_exit": True,
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=train_args["board_size"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(train_args["device"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
