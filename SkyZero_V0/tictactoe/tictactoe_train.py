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
    "num_workers": 12,
    "history_step": 2,
    "num_blocks": 2,
    "num_channels": 32,
    "lr": 0.001,
    "weight_decay": 3e-5,

    "num_simulations": 50,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,

    "batch_size": 128,
    "num_iterations": 100,
    "num_games_per_iter": 20,
    "train_steps": 10,
    
    "temperature": 1.0,
    "temp_threshold": 6,

    "min_buffer_size": 500,
    "max_buffer_size": 20000,
    
    "save_interval": 10,
    "data_dir": "data/tictactoe",
    "device": "cuda",
}

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe(history_step=train_args["history_step"])
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
