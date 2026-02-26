import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gomoku_train import train_args
from envs.gomoku import Gomoku
from playgame import GamePlayer


eval_args = {
    'mode': 'eval',
    'history_step': train_args['history_step'],
    'num_blocks': train_args['num_blocks'],
    'num_channels': train_args['num_channels'],
    'lr': train_args['lr'],
    'weight_decay': train_args['weight_decay'],

    'full_search_num_simulations': 600,

    'c_puct': 1.1,

    'root_temperature_init': 1,
    'root_temperature_final': 0.9,

    'move_temperature_init': 0.5,
    'move_temperature_final': 0.1,

    'fpu_reduction_max': train_args['fpu_reduction_max'],
    'root_fpu_reduction_max': train_args['root_fpu_reduction_max'],

    'file_name': 'gomoku',
    'data_dir': 'data/gomoku',
    'device': 'cuda',
}

if __name__ == '__main__':
    gp = GamePlayer(Gomoku(board_size=train_args['board_size'], history_step=train_args['history_step']), eval_args)
    gp.play()
