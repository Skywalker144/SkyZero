import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from utils import add_dirichlet_noise, print_board, temperature_transform, random_augment_batch


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

    def update(self, value):
        self.v += value
        self.n += 1


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args.copy()
        self.model = model.to(args['device'])
        self.model.eval()

    def select(self, node):

        child_priors = np.array([child.prior for child in node.children])
        child_visit_counts = np.array([child.n for child in node.children])
        child_values = np.array([child.v for child in node.children])

        q_values = -child_values / (child_visit_counts + 1e-8)
        u_values = self.args['c_puct'] * child_priors * (math.sqrt(node.n) / (1 + child_visit_counts))

        puct_scores = q_values + u_values

        best_child_idx = np.argmax(puct_scores)
        return node.children[best_child_idx]

    def expand(self, node):
        state = node.state
        to_play = node.to_play

        policy, value = self.model(torch.tensor(
            self.game.encode_state(state, to_play), device=self.args['device']
        ).unsqueeze(0))

        policy_logits = policy.squeeze(0).cpu().numpy()

        is_legal_actions = self.game.get_is_legal_actions(state)
        policy_logits = np.where(is_legal_actions, policy_logits, -np.inf)

        max_logit = np.max(policy_logits)
        policy = np.exp(policy_logits - max_logit)

        policy_sum = np.sum(policy)
        policy /= policy_sum

        if node.parent is None and self.args['mode'] == 'train':
            policy = add_dirichlet_noise(policy, self.args['dirichlet_alpha'], self.args['dirichlet_epsilon'])

            # step_count = np.count_nonzero(state)
            # root_temperature_start = self.args['root_temperature_start']
            # root_temperature_end = self.args['root_temperature_end']
            # root_temperature = root_temperature_end + (root_temperature_start - root_temperature_end) * 0.5 ** (
            #         step_count / self.game.board_size)
            #
            # policy = temperature_transform(policy, root_temperature)

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
    def search(self, state, to_play, num_simulations=None, process_bar=False):

        root = Node(state, to_play)

        if num_simulations is None:
            num_simulations = self.args['num_simulations']

        iterator = tqdm(range(num_simulations), desc='MCTS: ') if process_bar else range(num_simulations)
        for _ in iterator:
            node = root

            while node.is_expanded():
                node = self.select(node)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play
            else:
                value = self.expand(node)

            self.backpropagate(node, value)

        action_probs = np.zeros(self.game.board_size ** 2)
        for child in root.children:
            action_probs[child.action_taken] = child.n
        action_probs /= np.sum(action_probs)
        return action_probs


class AlphaZero:
    def __init__(self, game, model, optimizer, args):
        self.model = model.to(args['device'])
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.losses = []

        buffer_size = args['buffer_size']
        self.replay_buffer = ReplayBuffer(
            window_size=buffer_size,
            board_size=game.board_size,
        )

    def _get_randomized_simulations(self):

        base_simulations = self.args['num_simulations']
        min_ratio = self.args['playout_cap_min_ratio']
        exponent = self.args['playout_cap_exponent']

        random_value = np.random.random() ** exponent
        ratio = min_ratio + (1 - min_ratio) * random_value

        randomized_simulations = int(base_simulations * ratio)

        return max(1, randomized_simulations)

    def selfplay(self):
        memory = []
        to_play = 1
        state = self.game.get_initial_state()
        while not self.game.is_terminal(state):

            num_simulations = self._get_randomized_simulations()

            action_probs = self.mcts.search(state, to_play, num_simulations)

            memory.append((state, action_probs, to_play, num_simulations))
            if len(memory) >= self.args['zero_t_step']:
                t = 0.1
            else:
                t = self.args['temperature']
            action = np.random.choice(
                self.game.board_size ** 2,
                p=temperature_transform(action_probs, t)
            )
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        final_state = state
        # last_to_play = -to_play
        # value = self.game.get_winner(state) * last_to_play
        winner = self.game.get_winner(final_state)
        return_memory = []
        for state, policy_target, to_play, num_sims in memory:
            # outcome = value if to_play == last_to_play else -value
            outcome = winner * to_play
            return_memory.append((
                self.game.encode_state(state, to_play),
                policy_target,
                outcome,
                num_sims
            ))
        print_board(final_state)
        return return_memory, self.game.get_winner(final_state)

    def _train_batch(self, batch):
        batch = random_augment_batch(batch, self.game.board_size)

        states, policy_targets, value_targets, num_sims_list = zip(*batch)
        num_sims_array = np.array(num_sims_list)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.args['device'])
        policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.args['device'])
        value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.args['device'])

        policy, value = self.model(states)

        policy_training_threshold = self.args['policy_training_threshold']
        base_simulations = self.args['num_simulations']

        policy_mask = num_sims_array >= (policy_training_threshold * base_simulations)
        policy_mask_tensor = torch.tensor(policy_mask, dtype=torch.bool, device=self.args['device'])

        if policy_mask.sum() > 0:
            policy_loss = F.cross_entropy(
                policy[policy_mask_tensor],
                policy_targets[policy_mask_tensor]
            )
        else:
            policy_loss = torch.tensor(0.0, device=self.args['device'])

        value_loss = F.mse_loss(value, value_targets)

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def learn(self):
        batch_size = self.args['batch_size']
        min_buffer_size = self.args['min_buffer_size']
        train_steps_per_generation = self.args['train_steps_per_generation']
        # num_games_per_generation = self.args['num_games_per_generation']

        game_count = 0
        total_train_steps = 0
        recent_game_lens = []

        recent_winners = []
        recent_step_times = []
        recent_train_times = []
        total_start_time = time.time()

        last_save_time = time.time()
        savetime_interval = self.args['savetime_interval']

        print(f'Buffer Size: {self.replay_buffer.window_size}')
        print(f'Batch Size: {batch_size}')
        print(f'Min Buffer Size: {min_buffer_size}')
        print(f'Train Steps per Generation: {train_steps_per_generation}')
        # print(f'Games per Train: {num_games_per_generation}')
        print(f'Save Time Interval: {savetime_interval}s ({savetime_interval / 60:.1f}min)')
        print()

        num_games_per_generation = 20
        init_flag = True
        train_game_count = 0

        while True:

            self.model.eval()
            selfplay_start = time.time()
            memory, winner = self.selfplay()
            selfplay_time = time.time() - selfplay_start
            game_count += 1

            recent_winners.append(winner)
            if len(recent_winners) > 100:
                recent_winners.pop(0)

            step_count = len(memory)
            time_per_step = selfplay_time / step_count if step_count > 0 else 0
            for _ in range(step_count):
                recent_step_times.append(time_per_step)
            while len(recent_step_times) > 100:
                recent_step_times.pop(0)

            recent_game_lens.append(len(memory))
            if len(recent_game_lens) > 100:
                recent_game_lens.pop(0)
            avg_game_len = sum(recent_game_lens) / len(recent_game_lens)

            for sample in memory:
                self.replay_buffer.buffer.append(sample)

            recent_first_win = sum(1 for w in recent_winners if w == 1)
            recent_second_win = sum(1 for w in recent_winners if w == -1)
            recent_total = len(recent_winners)

            avg_step_time = sum(recent_step_times) / len(recent_step_times) if recent_step_times else 0

            current_buffer_size = len(self.replay_buffer)
            print(f'\n[Game {game_count}] Steps: {len(memory)}, Winner: {int(winner):+d}, Buffer: {current_buffer_size}, Avg Len: {avg_game_len:.1f}')
            print(f'  Win Rate (Recent {recent_total}) - First: {recent_first_win}/{recent_total} ({100 * recent_first_win / recent_total:.1f}%), '
                  f'Second: {recent_second_win}/{recent_total} ({100 * recent_second_win / recent_total:.1f}%)')
            print(f'  Selfplay Time: {selfplay_time:.2f}s ({step_count} steps), Recent {len(recent_step_times)} steps Avg: {avg_step_time * 1000:.1f}ms/step')

            if current_buffer_size < min_buffer_size:
                print(f'  [Skip Training] Buffer {current_buffer_size} < min_buffer_size {min_buffer_size}')
                continue
            elif init_flag:
                train_game_count = game_count
                init_flag = False

            current_time = time.time()
            if current_time - last_save_time >= savetime_interval:
                self.save_checkpoint()
                last_save_time = current_time

            if game_count != train_game_count:
                print(
                    f'  [Skip Training] Waiting for {train_game_count - game_count} more games')
                continue

            self.model.train()
            train_losses = []

            train_start = time.time()
            for step in range(train_steps_per_generation):
                batch = self.replay_buffer.sample(batch_size)

                loss = self._train_batch(batch)
                train_losses.append(loss)
                total_train_steps += 1

            train_time = time.time() - train_start
            recent_train_times.append(train_time)
            if len(recent_train_times) > 100:
                recent_train_times.pop(0)
            avg_train_time = sum(recent_train_times) / len(recent_train_times)
            time_per_step = train_time / train_steps_per_generation

            avg_loss = np.mean(train_losses)
            self.losses.append(avg_loss)
            print(f'  [Training] {train_steps_per_generation} steps, Avg Loss: {avg_loss:.4f}, Total Steps: {total_train_steps}')
            print(f'  Train Time: {train_time:.2f}s, Recent {len(recent_train_times)} Avg: {avg_train_time:.2f}s, Per Step: {time_per_step * 1000:.1f}ms')

            num_games_per_generation = int(self.args['batch_size'] * self.args['train_steps_per_generation'] / avg_game_len / self.args['target_ReplayRatio'])
            train_game_count = game_count + num_games_per_generation

            total_elapsed = time.time() - total_start_time
            avg_time_per_game = total_elapsed / game_count
            # print(f'  num_games_per_generation: {num_games_per_generation}')
            print(f'  Total Elapsed: {total_elapsed / 60:.1f}min, Avg/Game: {avg_time_per_game:.2f}s')

            plt.figure(figsize=(10, 6))
            plt.yscale('log')
            plt.xlabel('Games')
            plt.ylabel('Loss')
            plt.plot(self.losses)
            plt.title(f'Training Loss (Game {game_count})')
            plt.savefig(f"{self.args['file_name']}_losses.png")
            plt.close()

    @torch.inference_mode()
    def play(self, state, to_play):
        self.model.eval()
        action_probs = self.mcts.search(state, to_play, process_bar=True)
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

    def save_checkpoint(self, filepath=None):
        from datetime import datetime

        if filepath is None:
            checkpoint_dir = f"{self.args['file_name']}_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join(checkpoint_dir, f"{os.path.basename(self.args['file_name'])}_checkpoint_{timestamp}.pth")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'replay_buffer': {
                'buffer': list(self.replay_buffer.buffer),
                'window_size': self.replay_buffer.window_size,
                'board_size': self.replay_buffer.board_size,
                'games_count': self.replay_buffer.games_count,
            },
        }

        torch.save(checkpoint, filepath)

        file_size = os.path.getsize(filepath)
        size_str = f"{file_size / 1024 / 1024:.1f}MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.1f}KB"
        print(f"Checkpoint saved to {filepath} ({size_str})")

    def load_checkpoint(self, filepath=None):
        from collections import deque
        import glob

        if filepath is None:
            checkpoint_dir = f"{self.args['file_name']}_checkpoints"
            if not os.path.exists(checkpoint_dir):
                print(f"Checkpoint directory not found: {checkpoint_dir}")
                return False

            pattern = os.path.join(checkpoint_dir, "*.pth")
            checkpoint_files = glob.glob(pattern)

            if not checkpoint_files:
                print(f"No checkpoint files found in: {checkpoint_dir}")
                return False

            filepath = max(checkpoint_files, key=os.path.getmtime)
            print(f"Auto-selected latest checkpoint: {filepath}")

        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded")

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Optimizer loaded")

        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
            print(f"Losses history loaded ({len(self.losses)} data points)")

        if 'replay_buffer' in checkpoint:
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.buffer = deque(buffer_data['buffer'], maxlen=self.replay_buffer.window_size)
            self.replay_buffer.games_count = buffer_data.get('games_count', 0)
            print(f"Replay buffer loaded ({len(self.replay_buffer)} samples)")

        print(f"Checkpoint loaded from {filepath}")
        return True
