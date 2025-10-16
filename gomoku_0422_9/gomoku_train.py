import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku(board_size=9)
    model = ResNet(game, num_blocks=6, num_channels=256).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=5e-5)
    args = {
        'mode': 'train',
        'num_iterations': 100,
        'num_simulations': 800,
        'c_puct': 1.15,
        'temperature': 0.4,

        'root_temperature_start': 1.03,
        'root_temperature_end': 1.03,

        'zero_t_step': 15,

        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'memory_size': 2048,
        'batch_size': 256,
        'num_epochs': 10,
        'device': 'cuda',
        'save_interval': 20,
        #
        'file_name': 'gomoku',
    }
    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
