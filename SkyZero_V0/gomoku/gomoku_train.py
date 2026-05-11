import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
from envs.gomoku import Gomoku
from nets import ResNet

train_args = {
    "num_workers": 16,
    "board_size": 15,
    "rule": "renju",  # "freestyle" | "standard" | "renju"
    "num_blocks": 8,
    "num_channels": 96,
    "lr": 0.0001,
    "weight_decay": 3e-5,

    "num_simulations": 400,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,

    "batch_size": 128,
    "num_iterations": 1000,
    "train_steps_per_iteration": 100,
    "target_ReplayRatio": 8,

    "temperature": 1.0,
    "temp_threshold": 30,

    "min_buffer_size": 25000,
    "max_buffer_size": 500000,
    "window_exponent": 0.65,
    "window_expand_per_row": 0.4,
    
    "save_interval": 10,
    "data_dir": "data/gomoku",
    "device": "cuda",
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=train_args["board_size"], rule=train_args["rule"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
