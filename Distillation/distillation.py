import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch.optim as optim
import numpy as np
from alphazero_play import TreeReuseAlphaZero
from nets import ResNet
from utils import print_board


'''
蒸馏训练：
nn_output = {
    'policy_logits': total_policy_logits[:, 0:1, :, :],
    'opponent_policy_logits': total_policy_logits[:, 1:2, :, :],
    'soft_policy_logits': soft_policy_logits,
    'ownership': ownership,
    'value_logits': value_logits,
}

训练目标：
cee:
student_nn_policy -> mcts_policy
student_opponent_policy -> next_mcts_policy
student_soft_policy -> soft_mcts_policy
student_ownership -> ownership
student_nn_value -> mcts_value and outcome
kl:
student_nn_policy -> teacher_nn_policy
student_nn_value -> teacher_nn_value
student_soft_policy -> teacher_soft_mcts_policy
student_ownership -> teacher_ownership

SelfPlay保存：
mcts_policy
next_mcts_policy
soft_mcts_policy
ownership
mcts_value and outcome
teacher_nn_policy
teacher_nn_value
teacher_soft_mcts_policy
teacher_ownership
teacher_mcts_value

'''

class DataGenerater:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.args['mode'] = 'eval'
        model = ResNet(self.game, num_blocks=self.args['num_blocks'], num_channels=self.args['num_channels']).to(self.args['device'])
        optimizer = optim.Adam(model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.alphazero = TreeReuseAlphaZero(self.game, model, optimizer, self.args)
        self.alphazero.load_checkpoint()

    def eval_selfplay(self):
        to_play = 1
        state = self.game.get_initial_state()
        memory = []
        mcts_root = None

        while not self.game.is_terminal(state):
            action, info, mcts_root = self.alphazero.play(state, to_play, root=mcts_root)
            
            if len(memory) > 0:
                memory[-1]["next_mcts_policy"] = info['mcts_policy']

            teacher_nn_output = info['nn_output']
            memory.append({
                'encoded_state': self.game.encode_state(state, to_play),
                'to_play': to_play,
                'mcts_policy': info['mcts_policy'],
                'next_mcts_policy': None,
                "root_value": info['root_value'],
                'teacher_policy_logits': teacher_nn_output['policy_logits'],
                'teacher_opponent_policy_logits': teacher_nn_output['opponent_policy_logits'],
                'teacher_soft_policy_logits': teacher_nn_output['soft_policy_logits'],
                'teacher_ownership': teacher_nn_output['ownership'],
                'teacher_value_logits': teacher_nn_output['value_logits'],
            })

            mcts_root = self.alphazero.apply_action(mcts_root, action)
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
        
        final_state = state
        winner = self.game.get_winner(state)
        return_memory = []
        for sample in memory:
            outcome = winner * sample['to_play']


