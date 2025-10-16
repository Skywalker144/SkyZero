import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku()
    model = ResNet(game, num_blocks=16, num_channels=256).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    args = {
        'mode': 'train',
        'num_iterations': 100,
        'num_simulations': 1000,
        'c_puct': 2.5,
        'temperature': 1,

        'root_temperature_start': 1.2,
        'root_temperature_end': 1.2,

        'expansion_temperature': 1,

        'correction_ratio': 1,

        'zero_t_step': 15,

        'dirichlet_alpha': 0.04,
        'dirichlet_epsilon': 0.25,
        'memory_size': 5120,
        'batch_size': 256,
        'num_epochs': 12,
        'device': 'cuda',
        'save_interval': 10,

        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
