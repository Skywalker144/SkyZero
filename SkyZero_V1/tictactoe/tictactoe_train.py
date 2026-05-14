import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import ResNet
from envs.tictactoe import TicTacToe
from alphazero import AlphaZero
from alphazero_parallel import AlphaZeroParallel
from config import load_config
import numpy as np
import torch.optim as optim

_CFG_PATH = os.path.join(os.path.dirname(__file__), "tictactoe.cfg")
train_args = load_config(_CFG_PATH)

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe()
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
