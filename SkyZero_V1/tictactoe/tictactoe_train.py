import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets import ResNet
from envs.tictactoe import TicTacToe
from skyzero import AlphaZero
from skyzero_parallel import AlphaZeroParallel
from config import load_config
import numpy as np
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.path.dirname(__file__), "tictactoe.cfg"),
        help="Path to TOML cfg (default: tictactoe/tictactoe.cfg)",
    )
    cli, _ = parser.parse_known_args()
    train_args = load_config(cli.cfg)
    print(f"[train] Loaded cfg: {cli.cfg}", flush=True)
    print(f"[train] data_dir: {train_args.get('data_dir', 'data')}", flush=True)

    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe()
    model = ResNet(game, num_blocks=train_args["num_blocks"], num_channels=train_args["num_channels"])
    optimizer = optim.AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])

    alphazero = AlphaZeroParallel(game, model, optimizer, train_args)
    alphazero.load_checkpoint()
    alphazero.learn()
