import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
from envs.gomoku import Gomoku
from nets import ResNet

train_args = {
    "num_iterations": 1000,
    
    "board_size": 15,
    "rule": "renju",  # "freestyle" | "standard" | "renju"

    "num_blocks": 6,
    "num_channels": 64,
    "lr": 0.0001,
    "weight_decay": 3e-5,

    "num_simulations": 400,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,

    "batch_size": 128,
    "train_steps_per_iteration": 100,
    "target_ReplayRatio": 4,

    "move_temperature": 1.0,
    "half_life": 20,

    "min_buffer_size": 25000,
    "max_buffer_size": 1000000,
    "window_exponent": 0.65,
    "window_expand_per_row": 0.4,
    
    "save_interval": 10,
    "print_interval": 20,
    "data_dir": "data/gomoku",
    "device": "cuda",

    "num_workers": 16,
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=train_args["board_size"], rule=train_args["rule"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
