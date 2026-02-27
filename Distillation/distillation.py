import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.optim as optim
import numpy as np
from alphazero_play import TreeReuseAlphaZero
from nets import ResNet
from utils import print_board


class DataGenerater:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.args['mode'] = 'eval'
        model = ResNet(self.game, num_blocks=self.args['num_blocks'], num_channels=self.args['num_channels']).to(self.args['device'])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.alphazero = TreeReuseAlphaZero(self.game, model, optimizer, self.args)
        self.alphazero.load_checkpoint()

    def generate_train_data(self):

        to_play = 1
        state = self.game.get_initial_state()
        history = []
        mcts_root = None
        print_board(state)
        while not self.game.is_terminal(state):
            action, info, mcts_root = self.alphazero.play(state, to_play, root=mcts_root)
            mcts_root = self.alphazero.apply_action(mcts_root, action)
            state = self.game.get_next_state(state, action, to_play)
            
            nn_output = info['nn_output']
            history.append((
                self.game.encode_state(state, to_play),
                nn_output['policy_logits'],
                nn_output['opponent_policy_logits'],
                nn_output['soft_policy_logits'],
                nn_output['ownership'],
                nn_output['value_logits'],
            ))

            to_play = -to_play
