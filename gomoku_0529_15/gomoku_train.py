import torch.optim as optim
import numpy as np
from alphazero import AlphaZero
from gomoku import Gomoku
from nets import ResNet

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Gomoku()
    model = ResNet(game, num_blocks=12, num_channels=256).to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=3e-5)
    args = {
        'mode': 'train',
        'num_iterations': 100,
        'num_simulations': 1600,
        'c_puct': 2,
        'temperature': 1,

        'root_temperature_start': 1.2,
        'root_temperature_end': 1.2,

        'expansion_temperature': 1.05,

        'correction_ratio': 1.01,

        'zero_t_step': 100,

        'dirichlet_alpha': 0.05,
        'dirichlet_epsilon': 0.25,
        'memory_size': 2048,
        'batch_size': 256,
        'num_epochs': 16,
        'device': 'cuda',
        'save_interval': 10,

        'file_name': 'gomoku',
    }

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    alphazero.learn()
