import math
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from policy_surprise_weighting import compute_policy_surprise_weights, apply_surprise_weighting_to_game
from utils import (
    temperature_transform,
    random_augment_batch,
    softmax,
    add_shaped_dirichlet_noise,
    root_temperature_transform,
)


class Node:
    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None, nn_value=0):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.nn_value = nn_value

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
        self.model = model.to(args["device"])
        self.model.eval()

    def _inference(self, state, to_play):
        nn_output = self.model(torch.tensor(
            self.game.encode_state(state, to_play), dtype=torch.float32, device=self.args["device"]
        ).unsqueeze(0))  # (1, num_planes, board_size, board_size)

        policy_logits = nn_output["policy_logits"]
        value_logits = nn_output["value_logits"]

        policy_logits = np.where(
            self.game.get_is_legal_actions(state, to_play),
            policy_logits.flatten().cpu().numpy(),
            -np.inf,
        )

        policy = softmax(policy_logits)

        value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        value = value_probs[0] - value_probs[2]  # (赢, 平, 输)
        return policy, value, value_probs

    def _inference_with_stochastic_transform(self, state, to_play):
        encoded_state = self.game.encode_state(state, to_play)  # (num_planes, board_size, board_size)
        encoded_state = torch.tensor(encoded_state, dtype=torch.float32, device=self.args["device"]).unsqueeze(0)

        transform_type = np.random.randint(0, 8)
        k = transform_type % 4
        do_flip = transform_type >= 4

        transformed_encoded_state = torch.rot90(encoded_state, k, dims=(2, 3))
        if do_flip:
            transformed_encoded_state = torch.flip(transformed_encoded_state, dims=[3])

        nn_output = self.model(transformed_encoded_state)  # (1, num_planes, board_size, board_size)

        policy_logits = nn_output["policy_logits"]
        value_logits = nn_output["value_logits"]

        if do_flip:
            policy_logits = torch.flip(policy_logits, dims=[3])
        policy_logits = torch.rot90(policy_logits, k=-k, dims=(2, 3))

        policy_logits = np.where(
            self.game.get_is_legal_actions(state, to_play),
            policy_logits.flatten().cpu().numpy(),
            -np.inf,
        )

        policy = softmax(policy_logits)

        value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        value = value_probs[0] - value_probs[2]  # (赢, 平, 输)
        return policy, value, value_probs

    def _inference_with_symmetry(self, state, to_play):
        encoded = self.game.encode_state(state, to_play)  # (num_planes, board_size, board_size)

        symmetries = []
        for do_flip in [False, True]:
            for k in range(4):
                aug = np.rot90(encoded, k, axes=(1, 2))
                if do_flip:
                    aug = np.flip(aug, axis=2)
                symmetries.append(aug)

        input_tensor = torch.tensor(np.array(symmetries), dtype=torch.float32, device=self.args["device"])
        nn_output = self.model(input_tensor)

        v_probs = F.softmax(nn_output["value_logits"], dim=1).cpu().numpy()
        v_probs = v_probs.mean(axis=0)
        value = v_probs[0] - v_probs[2]

        p_logits = nn_output["policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)
        untransformed_p = []
        for i, (do_flip, k) in enumerate([(f, r) for f in [False, True] for r in range(4)]):
            p = p_logits[i]
            if do_flip:
                p = np.flip(p, axis=1)
            p = np.rot90(p, k=-k)
            untransformed_p.append(p.flatten())

        avg_p_logits = np.mean(untransformed_p, axis=0)
        is_legal = self.game.get_is_legal_actions(state, to_play)
        avg_p_logits = np.where(is_legal, avg_p_logits, -np.inf)
        policy = softmax(avg_p_logits)

        return policy, value, v_probs

    def select(self, node, is_full_search=False):

        if (
            self.args.get("enable_forced_playout", True)
            and is_full_search
            and node.parent is None and node.n > 0
        ):
            best_forced_child = None
            best_prior = -1
            k = self.args.get("forced_playout_k", 2)

            for child in node.children:
                if child.prior > 0 and child.n > 0:
                    sqrt_node_n = math.sqrt(node.n)
                    target_visits = math.sqrt(k * child.prior) * sqrt_node_n

                    if child.n < target_visits:
                        if child.prior > best_prior:
                            best_prior = child.prior
                            best_forced_child = child
            if best_forced_child is not None:
                return best_forced_child

        total_child_weight = sum(child.n for child in node.children)
        visited_policy_mass = sum(child.prior for child in node.children if child.n > 0)

        c_puct = self.args.get("c_puct", 1.1)
        c_puct_log = self.args.get("c_puct_log", 0.45)
        c_puct_base = self.args.get("c_puct_base", 500)

        if total_child_weight > 0:
            c_puct = c_puct + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

        explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01)

        # FPU
        parent_utility = node.v / node.n if node.n > 0 else 0
        nn_utility = node.nn_value

        fpu_pow = self.args.get("fpu_pow", 1)
        avg_weight = min(1, math.pow(visited_policy_mass, fpu_pow))
        parent_utility = avg_weight * parent_utility + (1 - avg_weight) * nn_utility
        if node.parent is None:
            fpu_reduction_max = self.args.get("root_fpu_reduction_max", 0.1)
        else:
            fpu_reduction_max = self.args.get("fpu_reduction_max", 0.2)
        reduction = fpu_reduction_max * math.sqrt(visited_policy_mass)
        fpu_value = parent_utility - reduction

        best_score = -float("inf")
        best_child = None

        for child in node.children:
            if child.n == 0:
                q_value = fpu_value
            else:
                q_value = -child.v / child.n

            u_value = explore_scaling * child.prior / (1 + child.n)

            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, node):
        state = node.state
        to_play = node.to_play

        if self.args.get("enable_stochastic_transform_inference", True):
            nn_policy, nn_value, _ = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_child", False):
            nn_policy, nn_value, _ = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value, _ = self._inference(state, to_play)

        node.nn_value = nn_value

        for action, prob in enumerate(nn_policy):
            if prob > 0:
                child = Node(
                    state=self.game.get_next_state(state, action, to_play),
                    to_play=-to_play,
                    prior=prob,
                    parent=node,
                    action_taken=action,
                )
                node.children.append(child)
        return nn_value.item()

    def root_expand(self, node, enable_dirichlet):
        state = node.state
        to_play = node.to_play

        if self.args.get("enable_symmetry_inference_for_root", False):
            nn_policy, nn_value, value_probs = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value, value_probs = self._inference(state, to_play)

        node.nn_value = nn_value

        if enable_dirichlet:
            current_step = np.count_nonzero(node.state[-1])
            nn_policy = add_shaped_dirichlet_noise(
                nn_policy,
                self.args["total_dirichlet_alpha"],
                self.args["dirichlet_epsilon"],
            )
            nn_policy = root_temperature_transform(
                nn_policy, current_step, self.args, self.game.board_size
            )

        for action, prob in enumerate(nn_policy):
            if prob > 0:
                child = Node(
                    state=self.game.get_next_state(state, action, to_play),
                    to_play=-to_play,
                    prior=prob,
                    parent=node,
                    action_taken=action,
                )
                node.children.append(child)
        return nn_policy, nn_value, value_probs

    @staticmethod
    def backpropagate(node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations):

        is_full_search = num_simulations == self.args["full_search_num_simulations"]

        root = Node(state, to_play)

        nn_policy, nn_value, nn_value_probs = self.root_expand(root, enable_dirichlet=is_full_search)
        self.backpropagate(root, nn_value)

        for _ in range(num_simulations):
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        if self.args.get("enable_forced_playout", True):
            mcts_policy = np.zeros(self.game.board_size**2)

            best_child = max(root.children, key=lambda c: c.n)

            total_child_weight = root.n
            c_puct = self.args.get("c_puct", 1.1)
            c_puct_log = self.args.get("c_puct_log", 0.45)
            c_puct_base = self.args.get("c_puct_base", 500)

            if total_child_weight > 0:
                c_puct = c_puct + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

            explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01)

            q_best = -best_child.v / best_child.n if best_child.n > 0 else 0
            u_best = explore_scaling * best_child.prior * (1 + best_child.n)
            puct_best = q_best + u_best

            k = self.args.get("forced_playout_k", 2)
            sqrt_root_n = math.sqrt(root.n)

            for child in root.children:
                if child == best_child:
                    mcts_policy[child.action_taken] = puct_best
                    continue

                target_visits = math.sqrt(k * child.prior) * sqrt_root_n

                q_child = -child.v / child.n if child.n > 0 else 0
                puct_gap = puct_best - q_child
                if puct_gap <= 0:
                    max_subtract = 0
                else:
                    min_denominator = (explore_scaling * child.prior) / puct_gap
                    max_subtract = (1 + child.n) - min_denominator
                amount_to_subtract = min(target_visits, max(0, max_subtract))

                new_n = child.n - amount_to_subtract

                if new_n <= 1:
                    new_n = 0
                
                mcts_policy[child.action_taken] = max(0, new_n)

            mcts_policy_sum = np.sum(mcts_policy)
            if mcts_policy_sum > 0:
                mcts_policy /= mcts_policy_sum
            else:
                mcts_policy[best_child.action_taken] = 1
        else:
            mcts_policy = np.zeros(self.game.board_size**2)
            for child in root.children:
                mcts_policy[child.action_taken] = child.n
            mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, nn_policy, nn_value_probs, root.v / root.n

    @torch.inference_mode()
    def eval_search(self, state, to_play):
        
        num_simulations = self.args["full_search_num_simulations"]

        root = Node(state, to_play)

        _, nn_value, _ = self.root_expand(root, enable_dirichlet=False)
        self.backpropagate(root, nn_value)

        for _ in tqdm(range(num_simulations), desc="MCTS Evaluating", unit="sim"):
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search=True)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        if self.args.get("enable_forced_playout", True):
            mcts_policy = np.zeros(self.game.board_size**2)

            best_child = max(root.children, key=lambda c: c.n)

            total_child_weight = root.n
            c_puct = self.args.get("c_puct", 1.1)
            c_puct_log = self.args.get("c_puct_log", 0.45)
            c_puct_base = self.args.get("c_puct_base", 500)

            if total_child_weight > 0:
                c_puct = c_puct + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

            explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01)

            q_best = -best_child.v / best_child.n if best_child.n > 0 else 0
            u_best = explore_scaling * best_child.prior * (1 + best_child.n)
            puct_best = q_best + u_best

            k = self.args.get("forced_playout_k", 2)
            sqrt_root_n = math.sqrt(root.n)

            for child in root.children:
                if child == best_child:
                    mcts_policy[child.action_taken] = puct_best
                    continue

                target_visits = math.sqrt(k * child.prior) * sqrt_root_n

                q_child = -child.v / child.n if child.n > 0 else 0
                puct_gap = puct_best - q_child
                if puct_gap <= 0:
                    max_subtract = 0
                else:
                    min_denominator = (explore_scaling * child.prior) / puct_gap
                    max_subtract = (1 + child.n) - min_denominator
                amount_to_subtract = min(target_visits, max(0, max_subtract))

                new_n = child.n - amount_to_subtract

                if new_n <= 1:
                    new_n = 0
                
                mcts_policy[child.action_taken] = max(0, new_n)

            mcts_policy_sum = np.sum(mcts_policy)
            if mcts_policy_sum > 0:
                mcts_policy /= mcts_policy_sum
            else:
                mcts_policy[best_child.action_taken] = 1
        else:
            mcts_policy = np.zeros(self.game.board_size**2)
            for child in root.children:
                mcts_policy[child.action_taken] = child.n
            mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, root.v / root.n


class AlphaZero:
    def __init__(self, game, model, optimizer, args):
        self.model = model.to(args["device"])
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

        self.losses_dict = {
            "total_loss": [],
            "policy_loss": [],
            "soft_policy_loss": [],
            "opponent_policy_loss": [],
            "ownership_loss": [],
            "value_loss": [],
        }

        self.winrate_history = []
        self.avg_game_len_history = []
        self.game_count = 0

        self.replay_buffer = ReplayBuffer(
            min_buffer_size=args.get("min_buffer_size", 1000),
            max_buffer_size=args.get("max_buffer_size", 10000),
            buffer_size_k=args.get("buffer_size_k", 0.1),
        )

    def _get_randomized_simulations(self):
        if np.random.rand() < self.args["full_search_prob"]:
            return self.args["full_search_num_simulations"]
        else:
            return self.args["fast_search_num_simulations"]

    @torch.inference_mode()
    def selfplay(self):
        memory = []
        to_play = 1
        state = self.game.get_initial_state()

        in_soft_resign = False
        historical_root_value = []

        while not self.game.is_terminal(state):

            if in_soft_resign:
                num_simulations = self.args["fast_search_num_simulations"]
            else:
                num_simulations = self._get_randomized_simulations()

            mcts_policy, nn_policy, value_probs, root_value = self.mcts.search(state, to_play, num_simulations)

            # Soft Resign
            historical_root_value.append(root_value)
            absmin_root_value = min(abs(x) for x in historical_root_value[-self.args.get("soft_resign_step_threshold", 3):])
            if (
                not in_soft_resign
                and absmin_root_value >= self.args.get("soft_resign_threshold", 0.9)
                and np.random.rand() < self.args.get("soft_resign_prob", 0.7)
            ):
                in_soft_resign = True

            if len(memory) > 0:
                memory[-1]["next_mcts_policy"] = mcts_policy

            memory.append({
                "state": state,
                "to_play": to_play,
                "mcts_policy": mcts_policy,
                "nn_policy": nn_policy,
                "value_probs": value_probs,
                "root_value": root_value,
                "to_play": to_play,
                "is_full_search": num_simulations == self.args["full_search_num_simulations"],
                "next_mcts_policy": None,
                "sample_weight": 1 if not in_soft_resign else self.args.get("soft_resign_sample_weight", 0.1),
            })

            current_step = len(memory)
            max_t = self.args.get("move_temperature_init", 0.8)
            min_t = self.args.get("move_temperature_final", 0.2)
            t = min_t + (max_t - min_t) * (0.5 ** (current_step / self.game.board_size))

            action = np.random.choice(
                self.game.board_size**2,
                p=temperature_transform(mcts_policy, t)
            )

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        final_state = state
        winner = self.game.get_winner(final_state)
        return_memory = []
        for sample in memory:
            outcome = winner * sample["to_play"]
            opponent_policy = sample["next_mcts_policy"] if sample["next_mcts_policy"] is not None else np.zeros_like(sample["mcts_policy"])
            sample_data = {
                "final_state": final_state[-1],
                "to_play": sample["to_play"],
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "nn_policy": sample["nn_policy"],  # for psw
                "value_probs": sample["value_probs"],  # for psw
                "root_value": sample["root_value"],  # for psw
                "opponent_policy_target": opponent_policy,
                "outcome": outcome,
                "is_full_search": sample["is_full_search"],
                "sample_weight": sample["sample_weight"],
            }
            return_memory.append(sample_data)
        
        surprise_weight = compute_policy_surprise_weights(
            return_memory,
            self.game.board_size,
            policy_surprise_data_weight=self.args.get("policy_surprise_data_weight", 0.5),
            value_surprise_data_weight=self.args.get("value_surprise_data_weight", 0.1),
        )
        return_memory = apply_surprise_weighting_to_game(return_memory, surprise_weight)

        return return_memory, self.game.get_winner(final_state), len(memory), final_state

    def _train_batch(self, batch):

        batch = random_augment_batch(batch, self.game.board_size)
        batch_size = len(batch)

        final_states = torch.tensor(np.array([d["final_state"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        to_plays = torch.tensor(np.array([d["to_play"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        encoded_states = torch.tensor(np.array([d["encoded_state"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        policy_targets = torch.tensor(np.array([d["policy_target"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        opponent_policy_targets = torch.tensor(np.array([d["opponent_policy_target"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        outcomes = torch.tensor(np.array([d["outcome"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        is_full_search = torch.tensor(np.array([d["is_full_search"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        sample_weights = torch.tensor(np.array([d["sample_weight"] for d in batch]), device=self.args["device"], dtype=torch.float32)

        soft_policy_targets = torch.pow(policy_targets, 0.25)
        soft_policy_targets /= torch.sum(soft_policy_targets, dim=-1, keepdim=True)

        self.model.train()
        nn_output = self.model(encoded_states)

        policy_logits = nn_output["policy_logits"].view(batch_size, -1)
        soft_policy_logits = nn_output["soft_policy_logits"].view(batch_size, -1)
        opponent_policy_logits = nn_output["opponent_policy_logits"].view(batch_size, -1)

        def get_loss(logits, targets, weights):
            loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)
            return (loss * weights).mean()

        # Policy, Soft Policy, Opponent Policy Loss
        policy_loss = get_loss(policy_logits, policy_targets, sample_weights)
        soft_policy_loss = get_loss(soft_policy_logits, soft_policy_targets, sample_weights)
        opponent_policy_loss = get_loss(opponent_policy_logits, opponent_policy_targets, sample_weights)
        
        # Ownership Loss
        ownership_pred = nn_output["ownership"].view(batch_size, -1) * to_plays.view(batch_size, 1)
        final_states = final_states.view(batch_size, -1)
        ownership_loss = F.mse_loss(ownership_pred, final_states, reduction="none").mean(dim=-1)
        ownership_loss = (ownership_loss * sample_weights).mean()

        # Value Loss
        value_targets = (1 - outcomes).long()
        value_loss = F.cross_entropy(nn_output["value_logits"], value_targets, reduction="none")
        value_loss = (value_loss * sample_weights).mean()

        loss = (
            self.args.get("policy_loss_weight", 1) * policy_loss
            + self.args.get("soft_policy_loss_weight", 8) * soft_policy_loss
            + self.args.get("opponent_policy_loss_weight", 0.15) * opponent_policy_loss
            + self.args.get("ownership_loss_weight", 1.5) / (self.game.board_size ** 2) * ownership_loss
            + self.args.get("value_loss_weight", 1.5) * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.args.get("max_grad_norm", 1.0)
        )
        self.optimizer.step()
        loss_dict = {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "opponent_policy_loss": opponent_policy_loss.item(),
            "soft_policy_loss": soft_policy_loss.item(),
            "ownership_loss": ownership_loss.item(),
            "value_loss": value_loss.item(),
        }
        return loss_dict, is_full_search.sum().item() / len(is_full_search)

    def learn(self):
        batch_size = self.args["batch_size"]
        min_buffer_size = self.args["min_buffer_size"]
        train_steps_per_generation = self.args.get(
            "train_steps_per_generation", 5)

        # 统计信息（双端队列）
        len_statistics_queue = self.args.get("len_statistics_queue_size", 300)

        recent_game_lengths = deque(maxlen=len_statistics_queue)
        recent_sample_lengths = deque(maxlen=len_statistics_queue)
        black_win_counts = deque(maxlen=len_statistics_queue)
        white_win_counts = deque(maxlen=len_statistics_queue)

        last_save_time = time.time()
        savetime_interval = self.args.get("savetime_interval", 3600)

        print(
            f"Buffer Size: {self.replay_buffer.min_buffer_size} -> {self.replay_buffer.max_buffer_size} (k={self.replay_buffer.buffer_size_k})"
        )
        print(f"Batch Size: {batch_size}")
        print(f"Min Buffer Size: {min_buffer_size}")
        print(f"Train Steps per Generation: {train_steps_per_generation}")
        # print(f"Games per Train: {num_games_per_generation}")
        print(f"Save Time Interval: {savetime_interval}s ({savetime_interval / 60:.1f}min)")
        print()

        init_flag = True
        train_game_count = 0
        session_start_time = time.time()
        total_samples = 0

        try:
            while True:
                self.model.eval()

                memory, winner, game_len, _ = self.selfplay()
                self.replay_buffer.add_game(memory)

                self.game_count += 1
                total_samples += len(memory)
                recent_game_lengths.append(game_len)
                recent_sample_lengths.append(len(memory))

                black_win_counts.append(1 if winner == 1 else 0)
                white_win_counts.append(1 if winner == -1 else 0)

                current_buffer_size = len(self.replay_buffer)

                if self.game_count % 10 == 0:
                    avg_game_len = np.mean(recent_game_lengths)
                    # avg_sample_len = np.mean(recent_sample_lengths)

                    total_recent = len(black_win_counts)
                    b_rate = np.sum(black_win_counts) / total_recent
                    w_rate = np.sum(white_win_counts) / total_recent
                    d_rate = 1 - b_rate - w_rate

                    self.winrate_history.append(
                        (self.game_count, b_rate, w_rate, d_rate)
                    )
                    self.avg_game_len_history.append(avg_game_len)

                    elapsed_time = time.time() - session_start_time
                    sps = total_samples / elapsed_time if elapsed_time > 0 else 0

                    print(
                        f"Game: {self.game_count} | Sps: {sps:.1f} | BufferSize: {len(self.replay_buffer)} | "
                        f"WindowSize: {self.replay_buffer.get_window_size()} | "
                        f"AvgGameLen: {avg_game_len:.1f} | BWD: {b_rate:.1f} {w_rate:.1f} {d_rate:.1f}"
                    )

                    self.plot_metrics()

                if current_buffer_size < min_buffer_size:
                    print(
                        f"  [Skip Training] Buffer {current_buffer_size} < min_buffer_size {min_buffer_size}"
                    )
                    continue
                elif init_flag:
                    train_game_count = self.game_count
                    init_flag = False

                current_time = time.time()
                if current_time - last_save_time >= savetime_interval:
                    self.save_checkpoint()
                    self.plot_metrics()
                    last_save_time = current_time

                if self.game_count < train_game_count:
                    continue

                # train

                self.model.train()
                batch_loss_dict = {key: [] for key in self.losses_dict.keys()}
                full_search_ratio_list = []
                for _ in range(train_steps_per_generation):
                    batch = self.replay_buffer.sample(batch_size)
                    loss_dic, full_search_ratio = self._train_batch(batch)
                    for key in batch_loss_dict:
                        batch_loss_dict[key].append(loss_dic.get(key, 0))
                    full_search_ratio_list.append(full_search_ratio)

                for key in self.losses_dict:
                    self.losses_dict[key].append(np.mean(batch_loss_dict[key]))

                # calculate train interval by Target Replay Ratio
                avg_sample_len = np.mean(recent_sample_lengths)
                num_next = int(
                    self.args["batch_size"] * self.args["train_steps_per_generation"] / avg_sample_len / self.args["target_ReplayRatio"]
                )
                num_next = max(1, num_next)
                train_game_count = self.game_count + num_next

                print(f"  [Training] Full Search Ratio: {np.mean(full_search_ratio_list):.2f}")
                print(
                    f"  [Training] Loss: {self.losses_dict["total_loss"][-1]:.2f} | "
                    f"Policy Loss: {self.losses_dict["policy_loss"][-1]:.2f} | "
                    f"Value Loss: {self.losses_dict["value_loss"][-1]:.2f}"
                )
                print(f"  Next Train after {num_next} games")
        except KeyboardInterrupt:
            if self.args.get("save_on_exit", True):
                print("\nKeyboardInterrupt detected. Saving checkpoint...")
                self.save_checkpoint()
                print("Checkpoint saved. Exiting.")
            else:
                print("\nKeyboardInterrupt detected. Exiting without saving checkpoint.")

    @torch.inference_mode()
    def play(self, state, to_play):
        self.model.eval()
        
        # 1. MCTS 搜索
        mcts_policy, root_value = self.mcts.eval_search(state, to_play)
        action = np.argmax(mcts_policy)
        
        # 2. 准备 8 对称性输入 (向量化处理)
        encoded = self.game.encode_state(state, to_play)
        # 预计算 8 个对称状态
        symmetries = [(f, r) for f in [False, True] for r in range(4)]
        aug_list = []
        for f, r in symmetries:
            img = np.rot90(encoded, r, (1, 2))
            if f: img = np.flip(img, axis=2)
            aug_list.append(img)
        
        input_tensor = torch.from_numpy(np.stack(aug_list)).to(self.args["device"], dtype=torch.float32)
        nn_output = self.model(input_tensor)

        nn_output["ownership"] = torch.tanh(nn_output["ownership"])
        # 3. 处理 Value (标量类)
        value_probs = torch.softmax(nn_output["value_logits"], dim=1).mean(dim=0).cpu().numpy()

        # 4. 统一处理空间/策略类输出 (逆变换 + 平均)
        # 定义需要逆变换的键
        spatial_keys = ["policy_logits", "opponent_policy_logits", "ownership"]
        averaged_results = {}

        for key in spatial_keys:
            data = nn_output[key].cpu().numpy() # (8, C, H, W)
            untransformed = []
            for i, (f, r) in enumerate(symmetries):
                temp = data[i]
                # 逆变换：先翻转(如果是True)，再逆旋转
                if f: temp = np.flip(temp, axis=2)
                temp = np.rot90(temp, -r, (1, 2))
                untransformed.append(temp)
            averaged_results[key] = np.mean(untransformed, axis=0)

        # 5. 计算 Masked Softmax 的辅助函数
        def get_masked_softmax(logits, is_legal):
            logits_flat = logits.flatten()
            masked = np.where(is_legal, logits_flat, -1e9)
            return softmax(masked)

        # 6. 针对性处理各个输出
        # 当前玩家策略
        is_legal = self.game.get_is_legal_actions(state, to_play)
        policy = get_masked_softmax(averaged_results["policy_logits"], is_legal)

        # 对手策略 (基于执行当前 Action 后的状态)
        next_state = self.game.get_next_state(state, action, to_play)
        next_is_legal = self.game.get_is_legal_actions(next_state, -to_play)
        opp_policy = get_masked_softmax(averaged_results["opponent_policy_logits"], next_is_legal)

        # 7. 组装返回
        board_size = self.game.board_size
        info = {
            "mcts_policy": mcts_policy.reshape(board_size, board_size),
            "nn_policy": policy.reshape(board_size, board_size),
            "root_value": root_value,
            "value_probs": value_probs,
            "nn_value": value_probs[0] - value_probs[2],
            "opponent_policy": opp_policy.reshape(board_size, board_size),
            "ownership": averaged_results["ownership"].squeeze() # 假设 ownership 是 (1, H, W)
        }

        return action, info

    def save_checkpoint(self, filepath=None):
        from datetime import datetime

        if filepath is None:
            checkpoint_dir = self.args["data_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join(
                checkpoint_dir,
                f"{os.path.basename(self.args["file_name"])}_checkpoint_{timestamp}.pth",
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses_dict": self.losses_dict,
            "winrate_history": self.winrate_history,
            "avg_game_len_history": self.avg_game_len_history,
            "game_count": self.game_count,
            "replay_buffer": self.replay_buffer.get_state(),
        }

        torch.save(checkpoint, filepath)

        file_size = os.path.getsize(filepath)
        size_str = (
            f"{file_size / 1024 / 1024:.1f}MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.1f}KB"
        )
        print(f"Checkpoint saved to {filepath} ({size_str})")

    def load_checkpoint(self, filepath=None):
        import glob

        if filepath is None:
            checkpoint_dir = self.args["data_dir"]
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

        checkpoint = torch.load(
            filepath, map_location=self.args["device"], weights_only=False
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded")

        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Optimizer loaded")

        if "losses_dict" in checkpoint:
            loaded_losses = checkpoint["losses_dict"]
            for key in self.losses_dict:
                if key in loaded_losses:
                    self.losses_dict[key] = loaded_losses[key]
            num_points = len(self.losses_dict.get("total_loss", []))
            print(f"Losses history loaded ({num_points} data points)")

        if "winrate_history" in checkpoint:
            self.winrate_history = checkpoint["winrate_history"]
            print(
                f"Winrate history loaded ({len(self.winrate_history)} data points)")

        if "avg_game_len_history" in checkpoint:
            self.avg_game_len_history = checkpoint["avg_game_len_history"]
            print(
                f"Avg game length history loaded ({len(self.avg_game_len_history)} data points)"
            )

        if "game_count" in checkpoint:
            self.game_count = checkpoint["game_count"]
            print(f"Game count loaded ({self.game_count} games)")

        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])
            print(f"Replay buffer loaded ({len(self.replay_buffer)} samples)")

        print(f"Checkpoint loaded from {filepath}")
        self.plot_metrics()
        return True

    def plot_metrics(self):
        try:
            data_dir = self.args["data_dir"]
            os.makedirs(data_dir, exist_ok=True)

            # 1. Total Loss Image
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses_dict["total_loss"], label="Total Loss")
            plt.title(f"Total Training Loss (Game {self.game_count})")
            plt.xlabel("Training Generation")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(data_dir, f"{self.args["file_name"]}_total_loss.png"))
            plt.close()

            # 2. Individual Loss Components Image
            plt.figure(figsize=(10, 6))
            for key in self.losses_dict:
                if key == "total_loss":
                    continue
                plt.plot(self.losses_dict[key], label=key.replace("_", " ").title())
            plt.title(f"Loss Components (Game {self.game_count})")
            plt.xlabel("Training Generation")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(data_dir, f"{self.args["file_name"]}_loss_components.png"))
            plt.close()

            # 3. Win Rate Image
            if self.winrate_history:
                games, b_rates, w_rates, d_rates = zip(*self.winrate_history)
                plt.figure(figsize=(10, 6))
                plt.plot(games, b_rates, label="Black Win Rate", color="black")
                plt.plot(games, w_rates, label="White Win Rate", color="red")
                plt.plot(games, d_rates, label="Draw Rate", color="gray")
                if self.avg_game_len_history and len(self.avg_game_len_history) == len(
                    games
                ):
                    plt.plot(
                        games,
                        np.array(self.avg_game_len_history) /
                        self.game.board_size**2,
                        label="Avg Game Length Ratio",
                        color="blue",
                        linestyle="--",
                    )
                plt.title(f"Win Rates (Last {len(b_rates)} Statistics)")
                plt.xlabel("Game Count")
                plt.ylabel("Rate")
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(data_dir, f"{self.args["file_name"]}_win_rates.png"))
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")
