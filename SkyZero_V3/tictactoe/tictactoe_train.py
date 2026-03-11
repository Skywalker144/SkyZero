import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import ResNet
from envs.tictactoe import TicTacToe
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
import numpy as np
import torch.optim as optim

train_args = {
    "mode": "train",
    
    "num_workers": 19,

    "history_step": 1,
    "num_blocks": 2,
    "num_channels": 32,
    "lr": 0.001,
    "weight_decay": 3e-5,

    "num_simulations": 8,
    "batch_size": 256,

    # Gumbel settings
    "gumbel_m": 2,
    "gumbel_c_visit": 50,
    "gumbel_c_scale": 1.0,

    "enable_stochastic_transform_inference_for_child": True,
    "enable_stochastic_transform_inference_for_root": True,

    "min_buffer_size": 500,
    "linear_threshold": 2048,
    "alpha": 0.75,
    "max_buffer_size": 100000,

    "train_steps_per_generation": 5,
    "target_ReplayRatio": 5,

    "fpu_reduction_max": 0.1,
    "root_fpu_reduction_max": 0.0,
    
    "savetime_interval": 120,
    "file_name": "tictactoe",
    "data_dir": "data/tictactoe",
    "device": "cuda",
    "save_on_exit": True,
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=train_args["history_step"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"]).to(train_args["device"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    # alphazero.load_checkpoint()
    alphazero.learn()
