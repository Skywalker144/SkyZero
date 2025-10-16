import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from tictactoe import TicTacToe
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = TicTacToe()
    model = ResNet(game).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    args = {
        'mode': 'train',
        'num_iterations': 5,
        'num_simulations': 100,
        'c_puct': 2,
        'temperature': 1,

        'expansion_temperature': 1,
        'zero_t_step': 1,
        'memory_size': 512,
        'batch_size': 128,
        'num_epochs': 10,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,

        'device': 'cuda',
        'save_interval': 100,
        'file_name': 'tictactoe',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.learn()
