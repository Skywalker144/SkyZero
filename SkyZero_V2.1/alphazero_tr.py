import math
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from replaybuffer import ReplayBuffer
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
        
        self.nn_policy = None
        self.nn_value_probs = None

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

        nn_policy = softmax(policy_logits)

        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        nn_value = nn_value_probs[0] - nn_value_probs[2]  # (赢, 平, 输)
        return nn_policy, nn_value, nn_value_probs

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

        nn_policy = softmax(policy_logits)

        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()
        nn_value = nn_value_probs[0] - nn_value_probs[2]  # (赢, 平, 输)
        return nn_policy, nn_value, nn_value_probs

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

        nn_value_probs = F.softmax(nn_output["value_logits"], dim=1).cpu().numpy()
        nn_value_probs = nn_value_probs.mean(axis=0)
        nn_value = nn_value_probs[0] - nn_value_probs[2]

        p_logits = nn_output["policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)
        untransformed_p = []
        for i, (do_flip, k) in enumerate([(f, r) for f in [False, True] for r in range(4)]):
            p = p_logits[i]
            if do_flip:
                p = np.flip(p, axis=1)
            p = np.rot90(p, k=-k)
            untransformed_p.append(p.flatten())

        avg_p_logits = np.mean(untransformed_p, axis=0)
        is_legal_actions = self.game.get_is_legal_actions(state, to_play)
        avg_p_logits = np.where(is_legal_actions, avg_p_logits, -np.inf)
        nn_policy = softmax(avg_p_logits)

        return nn_policy, nn_value, nn_value_probs

    def select(self, node, is_full_search=False):
        if (
            self.args.get("enable_forced_playouts", True)
            and is_full_search
            and node.parent is None and node.n > 0
        ):
            best_forced_child = None
            best_prior = -1
            k = self.args.get("forced_playouts_k", 2)

            total_child_weight = max(0, node.n - 1)
            sqrt_node_n = math.sqrt(total_child_weight)
            for child in node.children:
                if child.prior > 0:
                    target_visits = math.sqrt(k * child.prior) * sqrt_node_n

                    if child.n < target_visits:
                        if child.prior > best_prior:
                            best_prior = child.prior
                            best_forced_child = child
            if best_forced_child is not None:
                return best_forced_child

        visited_policy_mass = sum(child.prior for child in node.children if child.n > 0)

        c_puct_init = self.args.get("c_puct", 1.1)
        c_puct_log = self.args.get("c_puct_log", 0.45)
        c_puct_base = self.args.get("c_puct_base", 500)

        total_child_weight = max(0, node.n - 1)

        c_puct = c_puct_init + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

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

        fpu_loss_prop = self.args.get("fpu_loss_prop", 0.0)
        loss_value = -1
        fpu_value = fpu_value + (loss_value - fpu_value) * fpu_loss_prop

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

        if self.args.get("enable_stochastic_transform_inference_for_child", True):
            nn_policy, nn_value, nn_value_probs = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_child", False):
            nn_policy, nn_value, nn_value_probs = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value, nn_value_probs = self._inference(state, to_play)

        node.nn_value = nn_value
        node.nn_policy = nn_policy.copy()
        node.nn_value_probs = nn_value_probs.copy() if nn_value_probs is not None else None

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
        return nn_value

    def root_expand(self, node, enable_dirichlet):
        state = node.state
        to_play = node.to_play

        if self.args.get("enable_stochastic_transform_inference_for_root", True):
            nn_policy, nn_value, nn_value_probs = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_root", False):
            nn_policy, nn_value, nn_value_probs = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value, nn_value_probs = self._inference(state, to_play)

        node.nn_value = nn_value
        node.nn_policy = nn_policy.copy()
        node.nn_value_probs = nn_value_probs.copy() if nn_value_probs is not None else None

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
        return nn_policy, nn_value, nn_value_probs

    def apply_dirichlet_to_root(self, node):
        if node.nn_policy is None:
            return node.nn_policy
            
        current_step = np.count_nonzero(node.state[-1])
        nn_policy = node.nn_policy.copy()
        
        nn_policy = add_shaped_dirichlet_noise(
            nn_policy,
            self.args["total_dirichlet_alpha"],
            self.args["dirichlet_epsilon"],
        )
        nn_policy = root_temperature_transform(
            nn_policy, current_step, self.args, self.game.board_size
        )
        
        existing_children = {child.action_taken: child for child in node.children}
        node.children = []
        
        for action, prob in enumerate(nn_policy):
            if prob > 0:
                if action in existing_children:
                    child = existing_children[action]
                    child.prior = prob
                    node.children.append(child)
                else:
                    child = Node(
                        state=self.game.get_next_state(node.state, action, node.to_play),
                        to_play=-node.to_play,
                        prior=prob,
                        parent=node,
                        action_taken=action,
                    )
                    node.children.append(child)
        return nn_policy

    @staticmethod
    def backpropagate(node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, root=None):

        is_full_search = num_simulations == self.args["full_search_num_simulations"]

        if root is None:
            root = Node(state, to_play)

        if not root.is_expanded():
            nn_policy, nn_value, nn_value_probs = self.root_expand(root, enable_dirichlet=is_full_search)
            self.backpropagate(root, nn_value)
        else:
            if is_full_search:
                nn_policy = self.apply_dirichlet_to_root(root)
            else:
                nn_policy = root.nn_policy
            nn_value_probs = root.nn_value_probs

        for _ in range(num_simulations):
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search)
                assert node is not None

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        # Policy Targe Pruning
        if self.args.get("enable_forced_playouts", True):
            mcts_policy = np.zeros(self.game.board_size**2)

            best_child = max(root.children, key=lambda c: c.n)

            c_puct = self.args.get("c_puct", 1.1)
            c_puct_log = self.args.get("c_puct_log", 0.45)
            c_puct_base = self.args.get("c_puct_base", 500)

            total_child_weight = max(0, root.n - 1)

            if root.n > 0:
                c_puct = c_puct + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

            explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01)

            q_best = -best_child.v / best_child.n if best_child.n > 0 else 0
            u_best = explore_scaling * best_child.prior / (1 + best_child.n)
            puct_best = q_best + u_best

            for child in root.children:
                if child == best_child:
                    mcts_policy[child.action_taken] = best_child.n
                    continue

                q_child = -child.v / child.n if child.n > 0 else 0
                puct_gap = puct_best - q_child
                if puct_gap <= 0:
                    max_subtract = 0
                else:
                    min_denominator = (explore_scaling * child.prior) / puct_gap
                    max_subtract = (1 + child.n) - min_denominator
                amount_to_subtract = max(0, max_subtract)

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
            mcts_policy_sum = np.sum(mcts_policy)
            if mcts_policy_sum > 0:
                mcts_policy /= mcts_policy_sum
            elif len(root.children) > 0:
                mcts_policy[root.children[0].action_taken] = 1
        return mcts_policy, root.v / root.n, nn_policy, nn_value_probs

    @torch.inference_mode()
    def eval_search(self, state, to_play, num_simulations, root=None):

        if root is None:
            root = Node(state, to_play)

        if not root.is_expanded():
            nn_policy, nn_value, nn_value_probs = self.root_expand(root, enable_dirichlet=False)
            self.backpropagate(root, nn_value)
        else:
            nn_policy = root.nn_policy
            nn_value_probs = root.nn_value_probs

        for _ in tqdm(range(num_simulations), desc="MCTS:", unit="sim"):
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search=True)
                assert node is not None

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy_sum = np.sum(mcts_policy)
        if mcts_policy_sum > 0:
            mcts_policy /= mcts_policy_sum
        elif len(root.children) > 0:
            mcts_policy[root.children[0].action_taken] = 1
        return mcts_policy, root.v / root.n, nn_policy, nn_value_probs


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
            "soft_opponent_policy_loss": [],
            "value_loss": [],
            "win_pos_loss": [],
            "remaining_steps_loss": [],
        }

        self.winrate_history = []
        self.avg_game_len_history = []
        self.game_count = 0

        len_statistics_queue = args.get("len_statistics_queue_size", 300)
        self.recent_game_lengths = deque(maxlen=len_statistics_queue)
        self.recent_sample_lengths = deque(maxlen=len_statistics_queue)
        self.black_win_counts = deque(maxlen=len_statistics_queue)
        self.white_win_counts = deque(maxlen=len_statistics_queue)

        self.replay_buffer = ReplayBuffer(
            min_buffer_size=args.get("min_buffer_size", 1000),
            linear_threshold=args.get("linear_threshold", args.get("max_buffer_size", 10000)),
            alpha=args.get("alpha", 0.75),
            max_physical_limit=args.get("max_physical_limit", args.get("max_buffer_size", 3e6)),
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

        root = Node(state, to_play)

        while not self.game.is_terminal(state):

            if in_soft_resign:
                num_simulations = self.args["fast_search_num_simulations"]
            else:
                num_simulations = self._get_randomized_simulations()

            mcts_policy, root_value, nn_policy, nn_value_probs = self.mcts.search(state, to_play, num_simulations, root=root)

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
                "nn_value_probs": nn_value_probs,
                "root_value": root_value,
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

            # Tree Advance
            next_root = None
            for child in root.children:
                if child.action_taken == action:
                    next_root = child
                    break
            
            if next_root is not None:
                next_root.parent = None
                root = next_root
            else:
                root = Node(state, to_play)

        final_state = state
        winner = self.game.get_winner(final_state)
        win_pos = self.game.get_win_pos(final_state)

        remaining_steps_weight = 1 / self.game.board_size ** 2
        
        return_memory = []
        for i, sample in enumerate(memory):

            outcome = winner * sample["to_play"]
            opponent_policy = sample["next_mcts_policy"] if sample["next_mcts_policy"] is not None else np.zeros_like(sample["mcts_policy"])
            sample_data = {
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "opponent_policy_target": opponent_policy,
                "outcome": outcome,

                "win_pos_target": win_pos,
                "remaining_steps": (len(memory) - i - 1) * remaining_steps_weight,

                "nn_policy": sample["nn_policy"],  # for psw
                "nn_value_probs": sample["nn_value_probs"],  # for psw
                "root_value": sample["root_value"],  # for psw
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

        encoded_states = torch.tensor(np.array([d["encoded_state"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        policy_targets = torch.tensor(np.array([d["policy_target"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        opponent_policy_targets = torch.tensor(np.array([d["opponent_policy_target"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        outcomes = torch.tensor(np.array([d["outcome"] for d in batch]), device=self.args["device"], dtype=torch.float32)

        win_pos_targets = torch.tensor(np.array([d["win_pos_target"] for d in batch]), device=self.args["device"], dtype=torch.float32).view(batch_size, -1)
        remaining_steps = torch.tensor(np.array([d["remaining_steps"] for d in batch]), device=self.args["device"], dtype=torch.float32)

        is_full_search = torch.tensor(np.array([d["is_full_search"] for d in batch]), device=self.args["device"], dtype=torch.float32)
        sample_weights = torch.tensor(np.array([d["sample_weight"] for d in batch]), device=self.args["device"], dtype=torch.float32)

        soft_policy_targets = torch.pow(policy_targets, 0.25)
        soft_policy_targets /= torch.sum(soft_policy_targets, dim=-1, keepdim=True) + 1e-10
        soft_opponent_policy_targets = torch.pow(opponent_policy_targets, 0.25)
        soft_opponent_policy_targets /= torch.sum(soft_opponent_policy_targets, dim=-1, keepdim=True) + 1e-10

        self.model.train()
        nn_output = self.model(encoded_states)

        policy_logits = nn_output["policy_logits"].view(batch_size, -1)
        opponent_policy_logits = nn_output["opponent_policy_logits"].view(batch_size, -1)
        soft_policy_logits = nn_output["soft_policy_logits"].view(batch_size, -1)
        soft_opponent_policy_logits = nn_output["soft_opponent_policy_logits"].view(batch_size, -1)

        win_pos_logits = nn_output["win_pos_logits"].view(batch_size, -1)
        remaining_steps_pred = nn_output["remaining_steps"].view(batch_size) 

        def get_loss(logits, targets, weights):
            loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)
            weighted_loss = (loss * weights).mean()

            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                print(f"Nan/Inf detected in loss. logits stats: min={logits.min():.2f}, max={logits.max():.2f}")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            return (loss * weights).mean()

        # Policy, Soft Policy, Opponent Policy Loss, Soft Opponent Policy Loss
        policy_loss = get_loss(policy_logits, policy_targets, sample_weights)
        opponent_policy_loss = get_loss(opponent_policy_logits, opponent_policy_targets, sample_weights)
        soft_policy_loss = get_loss(soft_policy_logits, soft_policy_targets, sample_weights)
        soft_opponent_policy_loss = get_loss(soft_opponent_policy_logits, soft_opponent_policy_targets, sample_weights)

        # Value Loss
        value_targets = (1 - outcomes).long()
        value_loss = F.cross_entropy(nn_output["value_logits"], value_targets, reduction="none")
        value_loss = (value_loss * sample_weights).mean()

        # Five Position Loss
        win_pos_loss = F.binary_cross_entropy_with_logits(win_pos_logits, win_pos_targets, reduction="none")
        win_pos_loss = (win_pos_loss.mean(dim=-1) * sample_weights).mean()

        # Remian Step Loss
        remaining_steps_loss = F.smooth_l1_loss(remaining_steps_pred, remaining_steps, reduction="none")
        remaining_steps_loss = (remaining_steps_loss * sample_weights).mean()

        loss = (
            self.args.get("policy_loss_weight", 0.93) * policy_loss
            + self.args.get("opponent_policy_loss_weight", 0.15) * opponent_policy_loss
            + self.args.get("soft_policy_loss_weight", 8) * soft_policy_loss
            + self.args.get("soft_opponent_policy_loss_weight", 0.18) * soft_opponent_policy_loss
            + self.args.get("value_loss_weight", 0.72) * value_loss
            + self.args.get("win_pos_loss_weight", 0.01) * win_pos_loss
            + self.args.get("remaining_steps_loss_weight", 0.01) * remaining_steps_loss
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
            "soft_opponent_policy_loss": soft_opponent_policy_loss.item(),
            "value_loss": value_loss.item(),
            "win_pos_loss": win_pos_loss.item(),
            "remaining_steps_loss": remaining_steps_loss.item() * 0.1,
        }
        return loss_dict, is_full_search.sum().item() / len(is_full_search)

    def learn(self):
        batch_size = self.args["batch_size"]
        min_buffer_size = self.args["min_buffer_size"]
        train_steps_per_generation = self.args.get("train_steps_per_generation", 5)

        last_save_time = time.time()
        savetime_interval = self.args.get("savetime_interval", 3600)

        print(
            "Buffer Window: "
            f"min={self.replay_buffer.min_buffer_size}, "
            f"threshold={self.replay_buffer.linear_threshold}, "
            f"alpha={self.replay_buffer.alpha}, "
            f"cap={self.replay_buffer.max_physical_limit}"
        )
        print(f"Batch Size: {batch_size}")
        print(f"Min Buffer Size: {min_buffer_size}")
        print(f"Train Steps per Generation: {train_steps_per_generation}")
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
                self.recent_game_lengths.append(game_len)
                self.recent_sample_lengths.append(len(memory))

                self.black_win_counts.append(1 if winner == 1 else 0)
                self.white_win_counts.append(1 if winner == -1 else 0)

                current_buffer_size = len(self.replay_buffer)

                if self.game_count % 10 == 0:
                    avg_game_len = np.mean(self.recent_game_lengths)
                    # avg_sample_len = np.mean(self.recent_sample_lengths)

                    total_recent = len(self.black_win_counts)
                    b_rate = np.sum(self.black_win_counts) / total_recent
                    w_rate = np.sum(self.white_win_counts) / total_recent
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
                avg_sample_len = np.mean(self.recent_sample_lengths)
                num_next = int(
                    self.args["batch_size"] * self.args["train_steps_per_generation"] / avg_sample_len / self.args["target_ReplayRatio"]
                )
                num_next = max(1, num_next)
                train_game_count = self.game_count + num_next

                print(f"  [Training] Full Search Ratio: {np.mean(full_search_ratio_list):.2f}")
                print(
                    f"  [Training] Loss: {self.losses_dict['total_loss'][-1]:.2f} | "
                    f"Policy Loss: {self.losses_dict['policy_loss'][-1]:.2f} | "
                    f"Value Loss: {self.losses_dict['value_loss'][-1]:.2f}"
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
    def play(self, state, to_play, root=None, show_progress_bar=True):
        self.model.eval()

        if root is None:
            root = Node(state, to_play)

        actual_num_simulations = self.args["full_search_num_simulations"] - root.n

        mcts_policy, root_value, _, _ = self.mcts.eval_search(state, to_play, actual_num_simulations, root)

        action = np.argmax(mcts_policy)

        if root is not None:
            for child in root.children:
                if child.action_taken == action:
                    child.parent = None
                    root = child

        # Get symmetry avg outputs
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
        
        # Value:
        nn_value_probs = F.softmax(nn_output["value_logits"], dim=1).cpu().numpy()
        nn_value_probs = nn_value_probs.mean(axis=0)
        nn_value = nn_value_probs[0] - nn_value_probs[2]
        # Remaining steps:
        remaining_steps = nn_output["remaining_steps"].view(8).cpu().numpy()
        remaining_steps = remaining_steps.mean()
        remaining_steps = remaining_steps * self.game.board_size ** 2
        # Policy, Opponent Policy, Win Position:
        policy_logits = nn_output["policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)
        opponent_policy_logits = nn_output["opponent_policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)
        win_pos_logits = nn_output["win_pos_logits"].squeeze(1).cpu().numpy()  # (8, H, W)

        untransformed_pl = []
        untransformed_opl = []
        untransformed_wpl = []
        for i, (do_flip, k) in enumerate([(f, r) for f in [False, True] for r in range(4)]):
            pl = policy_logits[i]
            opl = opponent_policy_logits[i]
            wpl = win_pos_logits[i]
            if do_flip:
                pl = np.flip(pl, axis=1)
                opl = np.flip(opl, axis=1)
                wpl = np.flip(wpl, axis=1)
            pl = np.rot90(pl, k=-k)
            opl = np.rot90(opl, k=-k)
            wpl = np.rot90(wpl, k=-k)
            untransformed_pl.append(pl.flatten())
            untransformed_opl.append(opl.flatten())
            untransformed_wpl.append(wpl.flatten())
        
        avg_pl = np.mean(untransformed_pl, axis=0)
        is_legal_actions = self.game.get_is_legal_actions(state, to_play)
        avg_pl = np.where(is_legal_actions, avg_pl, -np.inf)
        nn_policy = softmax(avg_pl)

        avg_opl = np.mean(untransformed_opl, axis=0)
        next_is_legal_actions = self.game.get_is_legal_actions(
            self.game.get_next_state(state, action, to_play),
            to_play
        )
        avg_opl = np.where(next_is_legal_actions, avg_opl, -np.inf)
        nn_opponent_policy = softmax(avg_opl)

        def stable_sigmoid(x):
            x = np.atleast_1d(x)
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )
        
        avg_wpl = np.mean(untransformed_wpl, axis=0)
        win_pos = stable_sigmoid(avg_wpl)

        info = {
            "mcts_policy": mcts_policy.reshape(self.game.board_size, self.game.board_size),
            "root_value": root_value,
            "nn_policy": nn_policy.reshape(self.game.board_size, self.game.board_size),
            "nn_opponent_policy": nn_opponent_policy.reshape(self.game.board_size, self.game.board_size),
            "nn_value": nn_value,
            "nn_value_probs": nn_value_probs,
            "win_pos": win_pos.reshape(self.game.board_size, self.game.board_size),
            "remaining_steps": remaining_steps,
            "actual_search_num": actual_num_simulations,
        }
        
        return action, info, root

    def save_model(self, filepath=None, timestamp=None):
        from datetime import datetime

        if filepath is None:
            model_dir = os.path.join(self.args["data_dir"], "models")
            os.makedirs(model_dir, exist_ok=True)

            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            filepath = os.path.join(
                model_dir,
                f"{os.path.basename(self.args['file_name'])}_model_{timestamp}.pth",
            )

        torch.save(self.model.state_dict(), filepath)

        file_size = os.path.getsize(filepath)
        size_str = (
            f"{file_size / 1024 / 1024:.1f}MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.1f}KB"
        )
        print(f"Model saved to {filepath} ({size_str})")

    def save_checkpoint(self, filepath=None):
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.save_model(timestamp=timestamp)

        if filepath is None:
            checkpoint_dir = os.path.join(self.args["data_dir"], "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            filepath = os.path.join(
                checkpoint_dir,
                f"{os.path.basename(self.args['file_name'])}_checkpoint_{timestamp}.ckpt",
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses_dict": self.losses_dict,
            "winrate_history": self.winrate_history,
            "avg_game_len_history": self.avg_game_len_history,
            "game_count": self.game_count,
            "replay_buffer": self.replay_buffer.get_state(),
            "recent_game_lengths": self.recent_game_lengths,
            "recent_sample_lengths": self.recent_sample_lengths,
            "black_win_counts": self.black_win_counts,
            "white_win_counts": self.white_win_counts,
        }

        torch.save(checkpoint, filepath)

        file_size = os.path.getsize(filepath)
        size_str = (
            f"{file_size / 1024 / 1024:.1f}MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.1f}KB"
        )
        print(f"Checkpoint saved to {filepath} ({size_str})")

    def load_model(self, filepath=None):
        import glob

        if filepath is None:
            model_dir = os.path.join(self.args["data_dir"], "models")
            if not os.path.exists(model_dir):
                print(f"Model directory not found: {model_dir}")
                return False

            pattern = os.path.join(model_dir, "*.pth")
            model_files = glob.glob(pattern)

            if not model_files:
                print(f"No model files found in: {model_dir}")
                return False

            filepath = max(model_files, key=os.path.getmtime)
            print(f"Auto-selected latest model: {filepath}")

        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False

        state_dict = torch.load(
            filepath, map_location=self.args["device"], weights_only=False
        )
        self.model.load_state_dict(state_dict)
        print("Model loaded")
        return True

    def load_checkpoint(self, filepath=None):
        import glob

        if filepath is None:
            checkpoint_dir = os.path.join(self.args["data_dir"], "checkpoints")
            if not os.path.exists(checkpoint_dir):
                print(f"Checkpoint directory not found: {checkpoint_dir}")
                return False

            pattern = os.path.join(checkpoint_dir, "*.ckpt")
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
            # Override learning rate from current args
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.get("lr", param_group['lr'])
            print(f"Optimizer loaded (LR overridden to {self.args.get('lr')})")

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
            # Override buffer parameters from current args
            self.replay_buffer.min_buffer_size = self.args.get("min_buffer_size", self.replay_buffer.min_buffer_size)
            self.replay_buffer.linear_threshold = self.args.get("linear_threshold", self.replay_buffer.linear_threshold)
            self.replay_buffer.alpha = self.args.get("alpha", self.replay_buffer.alpha)
            self.replay_buffer.max_physical_limit = int(self.args.get("max_physical_limit", self.replay_buffer.max_physical_limit))
            print(f"Replay buffer loaded ({len(self.replay_buffer)} samples, parameters overridden)")

        if "black_win_counts" in checkpoint:
            self.black_win_counts = checkpoint["black_win_counts"]
            self.white_win_counts = checkpoint["white_win_counts"]
            self.recent_game_lengths = checkpoint["recent_game_lengths"]
            self.recent_sample_lengths = checkpoint["recent_sample_lengths"]
            print("Statistics queues loaded")

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
            plt.savefig(os.path.join(data_dir, f"{self.args['file_name']}_total_loss.png"))
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
            plt.savefig(os.path.join(data_dir, f"{self.args['file_name']}_loss_components.png"))
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
                plt.savefig(os.path.join(data_dir, f"{self.args['file_name']}_win_rates.png"))
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")
