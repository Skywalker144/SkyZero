import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tictactoe_train import train_args
from playgame import GamePlayer
from envs.tictactoe import TicTacToe


eval_args = {
    'mode': 'eval',
    'history_step': train_args['history_step'],
    'num_blocks': train_args['num_blocks'],
    'num_channels': train_args['num_channels'],
    'lr': train_args['lr'],
    'weight_decay': train_args['weight_decay'],

    'full_search_num_simulations': 100,
    'enable_symmetry_inference_for_root': True,
    'enable_symmetry_inference_for_child': True,

    'c_puct': 1.5,

    'root_temperature_init': 1,
    'root_temperature_final': 0.9,

    'move_temperature_init': 0.5,
    'move_temperature_final': 0.1,

    'fpu_reduction_max': train_args['fpu_reduction_max'],
    'root_fpu_reduction_max': train_args['root_fpu_reduction_max'],

    'enable_forced_playout': False,

    'file_name': 'tictactoe',
    'data_dir': 'data/tictactoe',
    'device': 'cuda',
}

if __name__ == '__main__':
    gp = GamePlayer(TicTacToe(history_step=eval_args['history_step']), eval_args)
    gp.play()
