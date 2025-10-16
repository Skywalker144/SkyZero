import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import augment_data, drop_last, add_dirichlet_noise, print_board, temperature_transform, add_dirichlet_noise_sm


class Node:
    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken

        self.children = []

        self.v = 0
        self.n = 0

    def is_expanded(self):
        return len(self.children) > 0

    def get_puct(self, c):
        # PUCT Formula:
        # PUCT = v / n + c_puct * prior * sqrt(N) / (1 + n)
        if self.n == 0:
            q = 0
        else:
            q = self.v / self.n
        u = c * self.prior * (math.sqrt(self.parent.n) / (self.n + 1))
        return q + u

    def update(self, value):
        self.v += value
        self.n += 1


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args.copy()
        self.model = model.to(args['device'])
        self.model.eval()

    def select__(self, node):
        max_puct = -np.inf
        selected_child = None
        for child in node.children:
            puct = child.get_puct(self.args['c_puct'])
            if puct > max_puct:
                max_puct = puct
                selected_child = child
        return selected_child

    def select(self, node):

        child_priors = np.array([child.prior for child in node.children])
        child_visit_counts = np.array([child.n for child in node.children])
        child_values = np.array([child.v for child in node.children])

        q_values = child_values / (child_visit_counts + 1e-8)
        u_values = self.args['c_puct'] * child_priors * (math.sqrt(node.n) / (1 + child_visit_counts))

        puct_scores = q_values + u_values

        best_child_idx = np.argmax(puct_scores)
        return node.children[best_child_idx]

    def expand(self, node):
        state = node.state
        to_play = node.to_play

        policy, value = self.prediction(state, to_play)

        # policy, value = self.random_transformation_prediction(
        #     self.game.encode_state(state, to_play)
        # )

        # policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        is_legal_actions = self.game.get_is_legal_actions(state)
        policy *= is_legal_actions
        policy /= np.sum(policy)

        # print()
        # print_board(state)
        # print(self.game.get_is_legal_actions(state).reshape(self.game.board_size, self.game.board_size))
        # print(policy.reshape(self.game.board_size, self.game.board_size))

        if node.parent is None and self.args['mode'] == 'train':
            # policy = add_dirichlet_noise_origin(policy, self.args['dirichlet_alpha'], self.args['dirichlet_epsilon'])
            # policy = add_dirichlet_noise_sm(policy, self.args['dirichlet_epsilon'])
            policy = add_dirichlet_noise(policy, self.args['dirichlet_alpha'], self.args['dirichlet_epsilon'])

            step_count = np.count_nonzero(state)
            root_temperature_start = self.args['root_temperature_start']
            root_temperature_end = self.args['root_temperature_end']
            root_temperature = root_temperature_end + (root_temperature_start - root_temperature_end) * 0.5 ** (step_count / self.game.board_size)

            policy = temperature_transform(policy, root_temperature)
        elif node.parent is not None and self.args['mode'] == 'train':
            policy = temperature_transform(policy, self.args['expansion_temperature'])


        # print(f'policy after noise: {policy}')

        for action, prob in enumerate(policy):
            if prob > 0:
                child = Node(
                    state=self.game.get_next_state(state, action, to_play),
                    to_play=-to_play,
                    prior=prob,
                    parent=node,
                    action_taken=action,
                )
                node.children.append(child)
        return value.item()

    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, k, process_bar=False):

        self.args['c_puct'] = k['first_hand_args']['c_puct'] if to_play == 1 else k['second_hand_args']['c_puct']
        self.args['num_simulations'] = k['first_hand_args']['num_simulations'] if to_play == 1 else k['second_hand_args']['num_simulations']

        root = Node(state, to_play)

        if process_bar:
            for search in tqdm(range(self.args['num_simulations']), desc='MCTS: '):
                node = root

                while node.is_expanded():
                    node = self.select(node)

                if self.game.is_terminal(node.state):
                    value = self.game.get_winner(node.state)
                else:
                    value = self.expand(node) * node.to_play

                value *= node.to_play * -1
                self.backpropagate(node, value)
        else:
            for search in range(self.args['num_simulations']):
                node = root

                while node.is_expanded():
                    node = self.select(node)

                if self.game.is_terminal(node.state):
                    value = self.game.get_winner(node.state)
                else:
                    value = self.expand(node) * node.to_play

                value *= node.to_play * -1
                self.backpropagate(node, value)

        action_probs = np.zeros(self.game.board_size ** 2)
        for child in root.children:
            action_probs[child.action_taken] = child.n
        action_probs /= np.sum(action_probs)
        return action_probs

    def random_transformation_prediction(self, encoded_state):
        board_size = encoded_state.shape[1]

        transformations = [
            {
                'state_transform': lambda x: x,
                'policy_transform': lambda x: x
            },
            {
                'state_transform': lambda x: np.rot90(x, k=1, axes=(1, 2)),
                'policy_transform': lambda x: np.rot90(x.reshape(board_size, board_size), k=3).flatten()
            },
            {
                'state_transform': lambda x: np.rot90(x, k=2, axes=(1, 2)),
                'policy_transform': lambda x: np.rot90(x.reshape(board_size, board_size), k=2).flatten()
            },
            {
                'state_transform': lambda x: np.rot90(x, k=3, axes=(1, 2)),
                'policy_transform': lambda x: np.rot90(x.reshape(board_size, board_size), k=1).flatten()
            },
            {
                'state_transform': lambda x: np.fliplr(x),
                'policy_transform': lambda x: np.fliplr(x.reshape(board_size, board_size)).flatten()
            },
            {
                'state_transform': lambda x: np.flipud(x),
                'policy_transform': lambda x: np.flipud(x.reshape(board_size, board_size)).flatten()
            }
        ]

        transform = random.choice(transformations)
        state_transform = transform['state_transform']
        policy_transform = transform['policy_transform']

        # print("Encoded state shape:", encoded_state.shape)

        transformed_state = state_transform(encoded_state)

        policy, value = self.model(torch.tensor(
            transformed_state.copy(), device=self.args['device'], dtype=torch.float32
        ).unsqueeze(0))

        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        # print("Policy shape:", policy.shape)

        policy = policy_transform(policy)

        return policy, value

    def prediction(self, state, to_play):
        policy, value = self.model(torch.tensor(
            self.game.encode_state(state, to_play), device=self.args['device']
        ).unsqueeze(0))
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        return policy, value


class AlphaZero:
    def __init__(self, game, model, optimizer, args):
        self.model = model.to(args['device'])
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.losses = []

    def selfplay(self, k):
        memory = []
        to_play = 1
        state = self.game.get_initial_state()
        while not self.game.is_terminal(state):
            action_probs = self.mcts.search(state, to_play, k)

            memory.append((state, action_probs, to_play))
            if len(memory) >= self.args['zero_t_step']:
                t = 0
            else:
                t = self.args['temperature']
            action = np.random.choice(
                self.game.board_size ** 2,
                p=temperature_transform(action_probs, t)
            )
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
        final_state = state
        last_to_play = -to_play
        value = self.game.get_winner(state) * last_to_play
        return_memory = []
        for state, policy_target, to_play in memory:
            outcome = value if to_play == last_to_play else -value
            return_memory.append((
                self.game.encode_state(state, to_play),
                policy_target,
                outcome
            ))
        print_board(final_state)
        return return_memory, self.game.get_winner(final_state)

    def train(self, memory):
        losses = []
        total_loss = 0
        random.shuffle(memory)
        memory = drop_last(memory, self.args['batch_size'])
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:batch_idx + self.args['batch_size']]

            states, policy_targets, value_targets = zip(*sample)
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.args['device'])
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.args['device'])
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.args['device'])

            policy, value = self.model(states)

            policy_loss = F.cross_entropy(policy, policy_targets)
            value_loss = F.mse_loss(value, value_targets)

            # loss = policy_loss + value_loss * 1.25
            # loss = policy_loss + value_loss * 1.1
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            total_loss += loss.item()
        self.losses += losses
        return total_loss / (len(memory) / self.args['batch_size'])

    def learn(self):
        for iteration in range(1, self.args['num_iterations'] + 1):
            print(f'\nIteration: {iteration}/{self.args["num_iterations"]}')
            total_memory = []

            # Self Play
            self.model.eval()
            pbar = tqdm(total=self.args['memory_size'], desc='SelfPlay: ')
            avg_len, idx = 0, 0
            first_hand_win = 0
            second_hand_win = 0
            while len(total_memory) <= self.args['memory_size']:

                ns = self.args['num_simulations']
                c = self.args['c_puct']
                # k = {
                #     'first_hand_args': {
                #         'num_simulations': ns,
                #         'c_puct': c,
                #     },
                #     'second_hand_args': {
                #         'num_simulations': ns,
                #         'c_puct': c,
                #     },
                # }
                # if (first_hand_win > 0) or (second_hand_win > 0):
                #     if first_hand_win > second_hand_win:
                #         num_simulations = int(ns * first_hand_win / (second_hand_win + 1))
                #         c_puct = math.sqrt(num_simulations / 800) * 2.5
                #         k = {
                #             'first_hand_args': {
                #                 'num_simulations': ns,
                #                 'c_puct': c,
                #             },
                #             'second_hand_args': {
                #                 'num_simulations': num_simulations,
                #                 'c_puct': c_puct,
                #             },
                #         }
                #     elif first_hand_win < second_hand_win:
                #         num_simulations = int(ns * second_hand_win / (first_hand_win + 1))
                #         c_puct = math.sqrt(num_simulations / 800) * 2.5
                #         k = {
                #             'first_hand_args': {
                #                 'num_simulations': num_simulations,
                #                 'c_puct': c_puct,
                #             },
                #             'second_hand_args': {
                #                 'num_simulations': ns,
                #                 'c_puct': c,
                #             },
                #         }

                first_corrected_ns = int(min(
                    ns * self.args['correction_ratio'],
                    max(
                        ns,
                        ns * (second_hand_win / (first_hand_win if first_hand_win > 0 else 1))
                    )
                ))
                first_c_puct = math.sqrt(first_corrected_ns / ns) * c
                second_corrected_ns = int(min(
                    ns * self.args['correction_ratio'],
                    max(
                        ns,
                        ns * (first_hand_win / (second_hand_win if second_hand_win > 0 else 1))
                    )
                ))
                second_c_puct = math.sqrt(second_corrected_ns / ns) * c

                k = {
                    'first_hand_args': {
                        'num_simulations': first_corrected_ns,
                        'c_puct': first_c_puct,
                    },
                    'second_hand_args': {
                        'num_simulations': second_corrected_ns,
                        'c_puct': second_c_puct,
                    },
                }

                print(k)

                memory, winner = self.selfplay(k)

                if winner == 1:
                    first_hand_win += 1
                elif winner == -1:
                    second_hand_win += 1
                if len(memory) + len(total_memory) > self.args['memory_size']:
                    pbar.update(self.args['memory_size'] - len(total_memory))
                else:
                    pbar.update(len(memory))
                total_memory += memory

                avg_len += len(memory)
                idx += 1
                pbar.set_postfix_str(f'Avg Step Num: {avg_len / idx:.1f}, First Win: {first_hand_win}, Second Win: {second_hand_win}')
            pbar.close()

            total_memory = augment_data(total_memory, self.game.board_size)

            # Training
            self.model.train()
            pbar = tqdm(range(self.args['num_epochs']), desc='Training: ')
            for i in pbar:
                loss = self.train(total_memory)
                if i == 0:
                    loss_s = loss
                elif i == (self.args['num_epochs'] // 2):
                    loss_m = loss
                pbar.set_postfix_str(f'Loss: {loss:.4f}')
            pbar.close()

            torch.save(self.model.state_dict(), f"{self.args['file_name']}_model.pt")
            torch.save(self.optimizer.state_dict(), f"{self.args['file_name']}_optimizer.pt")

            if iteration % self.args['save_interval'] == 0:
                self.save_checkpoint()

            plt.yscale('log')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.plot(self.losses)
            plt.savefig(f"{self.args['file_name']}_losses.png")
            plt.close()
            print(
                f'  Loss: {loss_s:.4f} -> {loss_m:.4f} -> {loss:.4f}\n'
            )

    @torch.inference_mode()
    def play(self, state, to_play):
        self.model.eval()
        k = {
            'first_hand_args': {
                'num_simulations': self.args['num_simulations'],
                'c_puct': self.args['c_puct'],
            },
            'second_hand_args': {
                'num_simulations': self.args['num_simulations'],
                'c_puct': self.args['c_puct'],
            },
        }
        action_probs = self.mcts.search(state, to_play, k, process_bar=True)
        action = np.argmax(action_probs)
        policy, value = self.model(
            torch.tensor(self.game.encode_state(state, to_play), device=self.args['device']).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy *= self.game.get_is_legal_actions(state)
        policy /= np.sum(policy)
        policy = policy.reshape(self.game.board_size, self.game.board_size)
        value = value.item()
        action_probs = action_probs.reshape(self.game.board_size, self.game.board_size)
        info = {
            'action_probs': action_probs,
            'policy': policy,
            'value': value,
            'ai_winrate': (value + 1) / 2,
        }
        return action, info

    def save_checkpoint(self):

        checkpoint_dir = f"{self.args['file_name']}_checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, f"{self.args['file_name']}_model_{timestamp}.pt")
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, f"{self.args['file_name']}_optimizer_{timestamp}.pt")
        )

    def load_checkpoint(self):

        model_path = f"{self.args['file_name']}_model.pt"
        optimizer_path = f"{self.args['file_name']}_optimizer.pt"

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
