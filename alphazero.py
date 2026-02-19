import math
import os
import time
import atexit
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from utils import add_dirichlet_noise, print_board, temperature_transform, random_augment_batch, random_augment_batch_rect, random_augment_batch_connect4, \
    root_temperature_transform, add_shaped_dirichlet_noise
from policy_surprise_weighting import (
    PolicySurpriseWeighter,
    extract_policy_prior_from_root,
    compute_kl_divergence,
)


class MinMaxState:
    def __init__(self, known_bounds=None):
        if known_bounds:
            self.min_value = known_bounds[0]
            self.max_value = known_bounds[1]
        else:
            self.min_value = float('inf')
            self.max_value = -float('inf')

    def update(self, value):
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def normalize(self, value):
        if self.max_value > self.min_value:
            return (value - self.min_value) / (self.max_value - self.min_value)
        return 0.5

    def reset(self):
        self.min_value = float('inf')
        self.max_value = float('-inf')


class Node:
    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken

        self.children = []

        self.v = 0
        self.v_sq = 0
        self.n = 0

    def is_expanded(self):
        return len(self.children) > 0

    def update(self, value):
        self.v += value
        self.v_sq += value ** 2
        self.n += 1


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args.copy()
        self.model = model.to(args['device'])
        self.model.eval()
        self.minmax_state = MinMaxState(self.args['Q_norm_bounds'])

    def select(self, node):
        # === KataGo-style Forced Playouts (Root Only) ===
        # 如果是根节点，检查是否有子节点因为 policy prior 较高但访问量不足
        # KataGo 使用 rootDesiredPerChildVisitsCoeff 来确保这一点
        # 逻辑: if child.n < sqrt(coeff * prior * total_visits), force visit.
        if node.parent is None and self.args.get('forced_playouts', False):
            forced_playout_coeff = self.args.get('forced_playout_coeff', 2.0)
            if forced_playout_coeff > 0 and node.n > 0:
                for child in node.children:
                    if child.prior > 0:
                        target_visits = math.sqrt(forced_playout_coeff * child.prior * node.n)
                        if child.n < target_visits:
                            return child

        # PUCT = Q + U
        # Q = 归一化后的平均价值估计
        # U = c_puct * P * (sqrt(N) / (1 + n))
        child_priors = np.array([child.prior for child in node.children])
        child_visit_counts = np.array([child.n for child in node.children])
        child_values = np.array([child.v for child in node.children])

        # 对已访问的子节点计算 Q 值，未访问的用 FPU (First Play Urgency)
        visited = child_visit_counts > 0
        safe_counts = np.maximum(child_visit_counts, 1)
        q_values = -child_values / safe_counts

        # === KataGo-style Dynamic cPUCT with Log Scaling ===
        # cPUCT = base_cpuct + log_scale * log((visits + base) / base)
        # Default KataGo params: base_cpuct=1.0, log_scale=0.45, base=500
        base_cpuct = self.args.get('cpuct_base', 1.0)
        cpuct_log_scale = self.args.get('cpuct_log_scale', 0.45)
        cpuct_log_base = self.args.get('cpuct_log_base', 500.0)

        total_child_weight = float(node.n)
        cpuct_exploration = base_cpuct + cpuct_log_scale * math.log((total_child_weight + cpuct_log_base) / cpuct_log_base)

        # === Dynamic Variance-Scaled cPUCT ===
        parent_utility_stdev_factor = 1.0
        if self.args.get('use_dynamic_cpuct', True) and node.n > 0:
            stdev_prior = self.args.get('cpuct_utility_stdev_prior', 0.40)
            stdev_prior_weight = self.args.get('cpuct_utility_stdev_prior_weight', 2.0)
            stdev_scale = self.args.get('cpuct_utility_stdev_scale', 0.85)

            mean = node.v / node.n
            sum_sq = node.v_sq

            var_prior_term = (mean ** 2 + stdev_prior ** 2) * stdev_prior_weight
            weighted_avg_sq = (var_prior_term + sum_sq) / (stdev_prior_weight + node.n - 1.0)

            variance = max(0.0, weighted_avg_sq - mean ** 2)
            stdev = math.sqrt(variance)

            parent_utility_stdev_factor = 1.0 + stdev_scale * (stdev / stdev_prior - 1.0)

        # Final explore scaling = cpuct * sqrt(visits + tiny_offset) * stdev_factor
        explore_scaling = cpuct_exploration * math.sqrt(total_child_weight + 0.01) * parent_utility_stdev_factor

        # === FPU (First Play Urgency) Calculation ===
        # KataGo: FPU = parentUtility - reduction * sqrt(policyProbMassVisited) + lossProp * (lossValue - fpu)
        if len(node.children) > 0:
            # 计算已访问节点的策略概率总和
            policy_prob_mass_visited = np.sum(child_priors[visited])

            # 归一化已访问节点的Q值
            q_norm = np.empty_like(q_values)
            q_norm[visited] = self.minmax_state.normalize(q_values[visited])

            # 计算 FPU 值
            fpu_reduction_max = self.args.get('fpu_reduction_max', 0.2)
            fpu_loss_prop = self.args.get('fpu_loss_prop', 0.0)

            # Parent utility (from current player's perspective)
            parent_utility = -node.v / node.n if node.n > 0 else 0.0
            parent_utility_norm = self.minmax_state.normalize(parent_utility) if node.n > 0 else 0.5

            # FPU reduction based on visited policy mass
            reduction = fpu_reduction_max * math.sqrt(policy_prob_mass_visited)
            fpu_value = parent_utility_norm - reduction

            # Optional: blend towards loss value
            if fpu_loss_prop > 0:
                loss_value = 0.0  # normalized loss = 0
                fpu_value = fpu_value + (loss_value - fpu_value) * fpu_loss_prop

            q_norm[~visited] = fpu_value
            q_values = q_norm

        u_values = explore_scaling * child_priors / (1 + child_visit_counts)

        puct_scores = q_values + u_values

        best_child_idx = np.argmax(puct_scores)
        return node.children[best_child_idx]

    def expand(self, node):
        state = node.state
        to_play = node.to_play

        # 在推理时使用 policy_optimism
        policy_optimism = 0.0
        if self.args.get('mode') != 'train':
            if node.parent is None:
                # 根节点使用较小的optimism (KataGo默认 rootPolicyOptimism = min(policyOptimism, 0.2))
                policy_optimism = min(self.args.get('policy_optimism', 1.0), self.args.get('root_policy_optimism', 0.2))
            else:
                policy_optimism = self.args.get('policy_optimism', 1.0)

        policy, value = self.model(
            torch.tensor(self.game.encode_state(state, to_play), device=self.args['device']).unsqueeze(0),
            policy_optimism=policy_optimism
        )

        policy_logits = policy.squeeze(0).cpu().numpy()

        is_legal_actions = self.game.get_is_legal_actions(state)
        policy_logits = np.where(is_legal_actions, policy_logits, -np.inf)

        max_logit = np.max(policy_logits)
        policy = np.exp(policy_logits - max_logit)

        policy_sum = np.sum(policy)
        policy /= policy_sum

        if node.parent is None and self.args['mode'] == 'train':
            current_step = np.count_nonzero(node.state[-1])
            board_size = np.sqrt(self.game.board_width * self.game.board_height)
            policy = root_temperature_transform(policy, current_step, self.args, board_size)
            policy = add_shaped_dirichlet_noise(policy, self.args['total_dirichlet_alpha'], self.args['dirichlet_epsilon'])

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
            if node.parent is not None:
                q = -node.v / node.n
                self.minmax_state.update(q)
            value = -value
            node = node.parent

    def _run_simulation(self, root):
        """运行一次 MCTS 模拟（select -> expand -> backpropagate）"""
        node = root
        while node.is_expanded():
            node = self.select(node)

        if self.game.is_terminal(node.state):
            value = self.game.get_winner(node.state) * node.to_play
        else:
            value = self.expand(node)

        self.backpropagate(node, value)

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations=None, return_policy_prior=False):

        root = Node(state, to_play)
        self.minmax_state.reset()

        if num_simulations is None:
            num_simulations = self.args['num_simulations']

        for _ in range(num_simulations):
            self._run_simulation(root)

        # 提取 policy prior
        policy_prior = None
        if return_policy_prior:
            policy_prior = extract_policy_prior_from_root(root, self.game.action_space_size)

        # === Policy Target Pruning (Noise Pruning) ===
        # KataGo "Prune Noise Weight" 方法:
        # 如果一个节点的访问量(权重)远超其 Policy 比例，且 Utility 明显低于已处理节点(通常是更好节点)的平均值，
        # 则按指数衰减其权重。这比回溯 PUCT 更稳健。
        action_probs = np.zeros(self.game.action_space_size)

        if self.args.get('policy_target_pruning', False) and root.is_expanded() and len(root.children) > 1:
            # 1. 按 Policy Prior 降序排序子节点 (KataGo 逻辑: "Children are normally sorted in policy order")
            # 只有按 Policy 顺序处理，才能正确判断某个节点是否相对于"更好"的节点表现不佳
            sorted_children = sorted(root.children, key=lambda c: c.prior, reverse=True)

            utility_sum = 0.0
            weight_sum = 0.0
            policy_sum = 0.0

            pruning_scale = self.args.get('noise_prune_utility_scale', 0.15)
            # pruning_cap = 1e50 # KataGo code implies a cap but effectively infinite

            for child in sorted_children:
                if child.n > 0:
                    # Utility relative to root player (root.to_play)
                    # child.v is from child.to_play perspective (opponent)
                    utility = -child.v / child.n
                    weight = float(child.n)
                    prior = child.prior

                    new_weight = weight

                    # 只有当我们已经积累了一些权重和 policy 时才进行修剪
                    if weight_sum > 0 and policy_sum > 0:
                        avg_utility = utility_sum / weight_sum
                        utility_gap = avg_utility - utility

                        # 只有当当前节点比之前的平均水平差时才考虑修剪
                        if utility_gap > 0:
                            # 计算基于 raw policy 应该有的权重份额
                            weight_share = weight_sum * prior / policy_sum
                            # 宽松界限 (2.0倍)，允许一定的自然波动
                            lenient_share = 2.0 * weight_share

                            if weight > lenient_share:
                                excess = weight - lenient_share
                                # 根据 utility gap 指数衰减多余的权重
                                subtract = excess * (1.0 - math.exp(-utility_gap / pruning_scale))
                                new_weight = max(0.0, weight - subtract)

                    # 更新统计量 (使用修剪后的权重)
                    utility_sum += utility * new_weight
                    weight_sum += new_weight
                    policy_sum += prior

                    action_probs[child.action_taken] = new_weight
        else:
            for child in root.children:
                action_probs[child.action_taken] = child.n

        action_probs_sum = np.sum(action_probs)
        if action_probs_sum > 0:
            action_probs /= action_probs_sum

        if return_policy_prior:
            return action_probs, policy_prior, root.v / root.n
        return action_probs, root.v / root.n

    def eval_search(self, state, to_play, num_simulations=None):
        root = Node(state, to_play)
        self.minmax_state.reset()

        if num_simulations is None:
            num_simulations = self.args['num_simulations']

        for _ in tqdm(range(num_simulations), desc='MCTS Simulations', unit='sim'):
            self._run_simulation(root)

        action_probs = np.zeros(self.game.action_space_size)
        for child in root.children:
            action_probs[child.action_taken] = child.n
        action_probs_sum = np.sum(action_probs)
        if action_probs_sum > 0:
            action_probs /= action_probs_sum
        return action_probs


class AlphaZero:
    def __init__(self, game, model, optimizer, args):
        self.model = model.to(args['device'])
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.game_count = 0

        self.replay_buffer = ReplayBuffer(
            min_buffer_size=args['min_buffer_size'],
            max_buffer_size=args['max_buffer_size'],
            buffer_size_k=args['buffer_size_k']
        )

        self.halflife = np.sqrt(self.game.board_width * self.game.board_height)
        self.move_temperature_init = self.args['move_temperature_init']
        self.move_temperature_final = self.args['move_temperature_final']

        self.psw = PolicySurpriseWeighter(
            baseline_weight_ratio=args.get('psw_baseline_ratio', 0.5),
            min_weight=args.get('psw_min_weight', 0.01)
        )

        self.win_rate_history = {
            'black': [],
            'white': [],
            'draw': [],
        }

    def _get_randomized_simulations(self):
        """
        Playout Cap Randomization: KataGo风格的三种搜索模式
        返回 (num_simulations, for_train, target_weight) 元组
        
        三种模式:
        1. Full Search: 用于策略训练, target_weight=1.0
        2. Cheap Search: 不用于策略训练, target_weight=0.0 (但可能被PSW选中)
        3. Reduced Search: 部分用于训练, target_weight=0.1 (当游戏局势明显倾斜时)
        """
        base_simulations = self.args['num_simulations']
        fast_simulations = self.args['fast_simulations']
        full_search_prob = self.args.get('full_search_prob', 0.25)
        cheap_search_prob = self.args.get('cheap_search_prob', 0.75)  # KataGo默认0.75

        # 决定搜索类型
        rand_val = np.random.random()

        if rand_val < full_search_prob:
            # Full Search: 正常训练
            return base_simulations, True, 1.0
        else:
            # Fast/Cheap Search
            return fast_simulations, False, 0.0

    def selfplay(self):
        memory = []
        to_play = 1
        state = self.game.get_initial_state()

        # Soft Resignation Logic
        resign_threshold = self.args.get('resign_threshold', -0.95)
        # Probability to play out the game in soft resignation mode (vs actual resignation)
        soft_resign_playout_prob = self.args.get('soft_resign_playout_prob', 0.1)
        in_soft_resignation = False

        # For reduced search based on win rate history (KataGo style)
        reduce_visits_threshold = self.args.get('reduce_visits_threshold', 0.9)
        reduced_visits_weight = self.args.get('reduced_visits_weight', 0.1)
        historical_root_values = []

        while not self.game.is_terminal(state):

            num_simulations, for_train, target_weight = self._get_randomized_simulations()

            if in_soft_resignation:
                num_simulations = self.args.get('fast_simulations', 50)
                # Ensure we capture this data for training but with low weight
                for_train = True
                current_sample_weight = 0.01
            else:
                # KataGo: Check if we should reduce visits based on win rate history
                # If game is clearly decided, use reduced visits
                if len(historical_root_values) >= 3:
                    recent_extreme = max(abs(v) for v in historical_root_values[-3:])
                    if recent_extreme > reduce_visits_threshold:
                        # Game seems decided, reduce visits
                        proportion_through = (recent_extreme - reduce_visits_threshold) / (1.0 - reduce_visits_threshold)
                        proportion_through = min(1.0, proportion_through)
                        current_sample_weight = target_weight + proportion_through * (reduced_visits_weight - target_weight)
                    else:
                        current_sample_weight = target_weight
                else:
                    current_sample_weight = target_weight

            action_probs, policy_prior, root_value = self.mcts.search(
                state, to_play, num_simulations, return_policy_prior=True
            )

            # Track root values for reduced search decisions
            historical_root_values.append(root_value)

            # Check for resignation condition
            if not in_soft_resignation and root_value < resign_threshold:
                if np.random.random() < soft_resign_playout_prob:
                    in_soft_resignation = True
                    # Update parameters for current step?
                    # Too late for search, but applies to sample weight
                    current_sample_weight = 0.01
                else:
                    # Actual resignation
                    # We stop the game here. The winner is the opponent.
                    # Since we are 'to_play' and resigning, opponent wins.
                    # Opponent is -to_play.
                    # So winner is -to_play.
                    # We need to break loop and set winner manually.
                    # But wait, final_state logic uses game.get_winner(final_state).
                    # If we break here, final_state is current state (non-terminal).
                    # get_winner might return 0 or invalid.
                    # We should handle resignation outcome explicitly.
                    # Let's set a flag `resigned = True`.
                    break

            # 记录 root_value 用于 Short-term Value Targets
            memory.append((state, action_probs, to_play, for_train, policy_prior, current_sample_weight, root_value))

            current_step = np.count_nonzero(state[-1])
            max_t = self.move_temperature_init
            min_t = self.move_temperature_final
            t = min_t + (max_t - min_t) * (0.5 ** (current_step / self.halflife))

            action = np.random.choice(
                self.game.action_space_size,
                p=temperature_transform(action_probs, t)
            )
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        if self.game.is_terminal(state):
            final_state = state
            winner = self.game.get_winner(final_state)
        else:
            # Resigned
            # The player 'to_play' resigned. So winner is -to_play.
            # Wait, the loop breaks when 'to_play' realizes value < threshold.
            winner = -to_play
            final_state = state  # Just for logging

        # 构建带 outcome 和 STV 的数据
        raw_memory = []
        memory_len = len(memory)
        stv_steps = [6, 16, 50]  # Short-term Value 预测步数

        for i in range(memory_len):
            state, policy_target, to_play, for_train, policy_prior, sample_weight, root_value = memory[i]
            outcome = winner * to_play

            # 计算 Short-term Value Targets
            stv_targets = []
            for k in stv_steps:
                target_idx = i + k
                if target_idx < memory_len:
                    # 获取未来时刻的 root_value
                    future_root_val = memory[target_idx][6]
                    # 调整视角：如果步数差是奇数，视角相反，取反
                    if k % 2 != 0:
                        future_root_val = -future_root_val
                    stv_targets.append(future_root_val)
                else:
                    # 游戏结束，使用最终 outcome
                    stv_targets.append(outcome)

            raw_memory.append((
                self.game.encode_state(state, to_play),
                policy_target,
                outcome,
                for_train,  # 布尔值：是否用于训练 Policy
                policy_prior,  # 保留 policy prior 用于 PSW 计算
                sample_weight,  # 样本权重
                root_value,  # 当前局面的 root_value (用于 Optimistic Policy)
                stv_targets,  # Short-term Value Targets [v_6, v_16, v_50]
            ))

        # 应用 Policy Surprise Weighting
        # PSW 期望格式: (encoded_state, policy_target, outcome, for_train, policy_prior)
        weighted_memory, psw_stats = self.psw.process_game(raw_memory)

        # 移除 policy_prior，只保留训练需要的数据
        return_memory = []
        for sample in weighted_memory:
            # (encoded_state, policy_target, outcome, for_train, policy_prior, sample_weight, root_value, stv_targets)
            # drop policy_prior (idx 4)
            # keep everything else
            new_sample = list(sample)
            new_sample.pop(4)
            return_memory.append(tuple(new_sample))

        # 打印 PSW 统计信息
        if psw_stats.get('enabled', False):
            print(
                f'  [PSW] Before: {psw_stats["samples_before"]}, '
                f'After: {psw_stats["samples_after"]}, '
                f'Ratio: {psw_stats["expansion_ratio"]:.2f}, '
                f'KL_mean: {psw_stats["kl_mean"]:.4f}, '
                f'KL_max: {psw_stats["kl_max"]:.4f}'
            )

        print_board(final_state)
        # 返回: 训练数据, 胜负结果, 实际游戏步数
        return return_memory, self.game.get_winner(final_state), len(raw_memory)

    def _train_batch(self, batch):
        # 数据增强
        if self.game.action_space_size != self.game.board_height * self.game.board_width:
            # Connect4等列动作游戏：action_space_size == board_width
            batch = random_augment_batch_connect4(batch)
        elif self.game.board_height == self.game.board_width:
            # 正方形棋盘：使用8种对称变换
            batch = random_augment_batch(batch, self.game.board_size)
        else:
            # 非正方形棋盘（动作空间等于棋盘大小）：只使用水平翻转
            batch = random_augment_batch_rect(batch, self.game.board_height, self.game.board_width)

        states, policy_targets, value_targets, for_train_list, sample_weights, root_values, stv_targets = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.args['device'])
        policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.args['device'])
        value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.args['device'])
        sample_weights = torch.tensor(np.array(sample_weights), dtype=torch.float32, device=self.args['device'])
        root_values = torch.tensor(np.array(root_values).reshape(-1, 1), dtype=torch.float32, device=self.args['device'])
        stv_targets = torch.tensor(np.array(stv_targets), dtype=torch.float32, device=self.args['device'])

        outputs = self.model(states)
        # 解包 outputs (policy, value, aux_policy, optimistic_policy, short_term_value)
        policy, value, aux_policy, optimistic_policy, short_term_value = outputs

        # PSW 和 Playout Cap Randomization：
        # 只有全量搜索 或高 surprise 的快速搜索 的样本用于 policy 训练
        # 普通快速搜索的样本只用于 value 训练
        policy_mask_tensor = torch.tensor(for_train_list, dtype=torch.bool, device=self.args['device'])

        # === 1. Main Policy Loss ===
        if policy_mask_tensor.sum() > 0:
            policy_loss = F.cross_entropy(
                policy[policy_mask_tensor],
                policy_targets[policy_mask_tensor],
                reduction='none'
            )
            # 应用 sample_weight (来自 Soft Resignation 等)
            policy_loss = (policy_loss * sample_weights[policy_mask_tensor]).mean()
        else:
            policy_loss = torch.tensor(0.0, device=self.args['device'])

        # === 2. Main Value Loss ===
        value_loss = F.mse_loss(value, value_targets, reduction='none')
        value_loss = (value_loss.squeeze() * sample_weights).mean()

        # === 3. Auxiliary Policy Loss (Soft Target) ===
        aux_policy_loss = torch.tensor(0.0, device=self.args['device'])
        if aux_policy is not None and policy_mask_tensor.sum() > 0:
            # Soft Target Calculation: target^(1/T) normalized
            # T = 4 (KataGo default)
            T = 4.0
            # 只计算有效样本的 soft target
            targets = policy_targets[policy_mask_tensor]
            # avoid zero power issues
            soft_targets = torch.pow(targets + 1e-10, 1.0 / T)
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

            # Aux Loss
            aux_loss = F.cross_entropy(
                aux_policy[policy_mask_tensor],
                soft_targets,
                reduction='none'
            )
            # Weighted 8x (KataGo default)
            aux_policy_loss = (aux_loss * sample_weights[policy_mask_tensor]).mean()

        # === 4. Optimistic Policy Loss ===
        optimistic_loss = torch.tensor(0.0, device=self.args['device'])
        if optimistic_policy is not None:
            # optimistic_policy shape: [B, 2, ActionSpace] (Output 0: Long-term, Output 1: Short-term)

            # --- Long-term Optimistic Policy (Output 0) ---
            # KataGo Logic:
            # Weight = win_squared + sigmoid((score_stdevs_excess - 1.5) * 3.0)
            # win_squared discourages draws.
            # score_stdevs_excess: how much better the actual score is than expected, in units of predicted stdev.
            # Simplified here since we don't have score targets/predictions in this code base yet:
            # Use value (win rate) improvement instead.

            # value_targets is [-1, 1]. Map to [0, 1] for win probability approx
            win_prob_target = (value_targets + 1.0) / 2.0
            root_win_prob = (root_values + 1.0) / 2.0

            # 1. Win Squared (encourages winning positions)
            # KataGo: win_squared = (win_prob + 0.5 * no_result)^2. 
            # Here just win_prob^2
            win_squared = torch.square(win_prob_target)

            # 2. Unexpected Improvement (Surprise)
            # KataGo uses score stdevs. We'll use value improvement.
            # improvement = actual - expected
            improvement = value_targets - root_values
            # Heuristic: 0.1 improvement in value ~ 1.5 sigma surprise? (Just a guess without variance head)
            # Let's map improvement to a "surprise score".
            # If improvement > 0.2, we want high weight.
            # Sigmoid((improvement - 0.1) * 20) -> Center at 0.1, scale 20.
            surprise_weight = torch.sigmoid((improvement - 0.1) * 20.0)

            target_weight_long = torch.clamp(torch.max(win_squared, surprise_weight), 0.0, 1.0)

            # Filter: Don't train on lost games (value < -0.9) and only on policy training samples
            valid_opt_mask = (value_targets > -0.9).squeeze() & policy_mask_tensor

            if valid_opt_mask.sum() > 0:
                # Long-term policy loss
                opt_loss_long_vals = F.cross_entropy(
                    optimistic_policy[valid_opt_mask, 0, :],
                    policy_targets[valid_opt_mask],
                    reduction='none'
                )

                # Expand dims for combined_weights to match loss if needed, but cross_entropy returns [N]
                # sample_weights: [B, 1] or [B] -> [N]
                # target_weight_long: [B, 1] -> [N]
                combined_weights_long = sample_weights[valid_opt_mask] * target_weight_long[valid_opt_mask].squeeze()
                loss_long = (opt_loss_long_vals * combined_weights_long).mean()

                loss_short = torch.tensor(0.0, device=self.args['device'])

                # --- Short-term Optimistic Policy (Output 1) ---
                if short_term_value is not None:
                    # Use the first horizon (6 turns) for short-term surprise
                    # stv_targets: [B, 3], short_term_value: [B, 3]
                    stv_actual = stv_targets[:, 0:1]  # [B, 1]
                    stv_pred = short_term_value[:, 0:1].detach()  # [B, 1]

                    # Positive values mean better for current player
                    stv_improvement = stv_actual - stv_pred

                    # Sigmoid gate for short term
                    target_weight_short = torch.clamp(torch.sigmoid((stv_improvement - 0.1) * 20.0), 0.0, 1.0)

                    opt_loss_short_vals = F.cross_entropy(
                        optimistic_policy[valid_opt_mask, 1, :],
                        policy_targets[valid_opt_mask],
                        reduction='none'
                    )

                    combined_weights_short = sample_weights[valid_opt_mask] * target_weight_short[valid_opt_mask].squeeze()
                    loss_short = (opt_loss_short_vals * combined_weights_short).mean()

                optimistic_loss = 0.1 * loss_long + 0.2 * loss_short
            else:
                optimistic_loss = torch.tensor(0.0, device=self.args['device'])

        # === 5. Short-term Value Loss ===
        stv_loss = torch.tensor(0.0, device=self.args['device'])
        if short_term_value is not None:
            # stv_targets shape: [B, 3]
            # short_term_value shape: [B, 3]
            s_loss = F.mse_loss(short_term_value, stv_targets, reduction='none')
            # 对三个时间点平均
            s_loss = s_loss.mean(dim=1)
            stv_loss = (s_loss * sample_weights).mean()

        loss = 0.93 * policy_loss + 0.72 * value_loss + 8 * aux_policy_loss + optimistic_loss + 0.72 * stv_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item(), aux_policy_loss.item(), optimistic_loss.item(), stv_loss.item()

    def learn(self):
        atexit.register(self.save_checkpoint)

        batch_size = self.args['batch_size']
        min_buffer_size = self.args['min_buffer_size']
        train_steps_per_generation = self.args['train_steps_per_generation']
        # num_games_per_generation = self.args['num_games_per_generation']

        total_train_steps = 0
        recent_sample_counts = deque(maxlen=100)
        recent_game_steps = deque(maxlen=100)

        recent_winners = deque(maxlen=100)
        recent_train_times = deque(maxlen=100)
        total_start_time = time.time()

        # Throughput measuring (Serial)
        perf_start_time = time.time()
        perf_steps_count = 0
        recent_throughput_measurements = deque(maxlen=10)

        last_save_time = time.time()
        savetime_interval = self.args['savetime_interval']

        print(f'Buffer Size: {self.replay_buffer.max_buffer_size}')
        print(f'Batch Size: {batch_size}')
        print(f'Min Buffer Size: {min_buffer_size}')
        print(f'Train Steps per Generation: {train_steps_per_generation}')
        # print(f'Games per Train: {num_games_per_generation}')
        print(f'Save Time Interval: {savetime_interval}s ({savetime_interval / 60:.1f}min)')
        print()

        init_flag = True
        train_game_count = 0

        while True:

            self.model.eval()
            memory, winner, game_steps = self.selfplay()
            self.game_count += 1

            recent_winners.append(winner)

            sample_count = len(memory)

            # Update Throughput Stats
            perf_steps_count += sample_count

            recent_sample_counts.append(sample_count)
            recent_game_steps.append(game_steps)

            avg_sample_count = sum(recent_sample_counts) / len(recent_sample_counts)
            avg_game_steps = sum(recent_game_steps) / len(recent_game_steps)

            self.replay_buffer.add_game(memory)

            recent_first_win = sum(1 for w in recent_winners if w == 1)
            recent_second_win = sum(1 for w in recent_winners if w == -1)
            recent_draw = sum(1 for w in recent_winners if w == 0)
            recent_total = len(recent_winners)

            # Calculate Global Throughput (Sliding Window)
            current_duration = time.time() - perf_start_time
            recent_throughput_measurements.append((current_duration, perf_steps_count))

            total_window_steps = sum(s for _, s in recent_throughput_measurements)
            total_window_time = sum(t for t, _ in recent_throughput_measurements)
            global_steps_per_sec = total_window_steps / total_window_time if total_window_time > 0 else 0

            current_buffer_size = len(self.replay_buffer)
            print(
                f'\n[Game {self.game_count}] [Total Samples Added{self.replay_buffer.total_samples_added}] Winner: {int(winner):+d}, Steps: {game_steps}, Samples: {sample_count}, Buffer: {current_buffer_size}, AvgSteps: {avg_game_steps:.1f}, AvgSamples: {avg_sample_count:.1f}')
            print(f'  Speed: {global_steps_per_sec:.1f} steps/s')
            print(
                f'  Win Rate (Recent {recent_total}) - First: {recent_first_win}/{recent_total} ({100 * recent_first_win / recent_total:.1f}%), '
                f'Second: {recent_second_win}/{recent_total} ({100 * recent_second_win / recent_total:.1f}%), '
                f'Draw: {recent_draw}/{recent_total} ({100 * recent_draw / recent_total:.1f}%)'
            )
            # Reset performance counters for next game
            perf_start_time = time.time()
            perf_steps_count = 0

            if current_buffer_size < min_buffer_size:
                print(f'  [Skip Training] Buffer {current_buffer_size} < min_buffer_size {min_buffer_size}')
                continue
            elif init_flag:
                safe_avg_samples = max(1, avg_sample_count)
                num_games_per_generation = int(self.args['batch_size'] * self.args['train_steps_per_generation'] / safe_avg_samples / self.args['target_ReplayRatio'])
                train_game_count = self.game_count + num_games_per_generation
                init_flag = False
                print(f"  [Init] Skipping immediate training. Next training at game {train_game_count} (+{num_games_per_generation} games)")

            current_time = time.time()
            if current_time - last_save_time >= savetime_interval:
                self.save_checkpoint()
                last_save_time = current_time

            if self.game_count != train_game_count:
                print(
                    f'  [Skip Training] Waiting for {train_game_count - self.game_count} more games')
                continue

            self.model.train()
            train_losses = []
            train_policy_losses = []
            train_value_losses = []

            train_start = time.time()
            for step in range(train_steps_per_generation):
                batch = self.replay_buffer.sample(batch_size)

                loss, policy_loss, value_loss, _, _, _ = self._train_batch(batch)
                train_losses.append(loss)
                train_policy_losses.append(policy_loss)
                train_value_losses.append(value_loss)
                total_train_steps += 1

            train_time = time.time() - train_start
            recent_train_times.append(train_time)
            avg_train_time = sum(recent_train_times) / len(recent_train_times)
            time_per_step = train_time / train_steps_per_generation

            avg_loss = np.mean(train_losses)
            avg_policy_loss = np.mean(train_policy_losses)
            avg_value_loss = np.mean(train_value_losses)
            self.losses.append(avg_loss)
            self.policy_losses.append(avg_policy_loss)
            self.value_losses.append(avg_value_loss)
            print(f'  [Training] {train_steps_per_generation} steps, Avg Loss: {avg_loss:.4f}, Total Steps: {total_train_steps}')
            print(f'  Train Time: {train_time:.2f}s, Recent {len(recent_train_times)} Avg: {avg_train_time:.2f}s, Per Step: {time_per_step * 1000:.1f}ms')

            safe_avg_samples = max(1, avg_sample_count)
            num_games_per_generation = int(self.args['batch_size'] * self.args['train_steps_per_generation'] / safe_avg_samples / self.args['target_ReplayRatio'])
            train_game_count = self.game_count + num_games_per_generation

            total_elapsed = time.time() - total_start_time
            avg_time_per_game = total_elapsed / self.game_count
            # print(f'  num_games_per_generation: {num_games_per_generation}')
            print(f'  Total Elapsed: {total_elapsed / 60:.1f}min, Avg/Game: {avg_time_per_game:.2f}s')

            self.win_rate_history['black'].append(recent_first_win / recent_total if recent_total > 0 else 0)
            self.win_rate_history['white'].append(recent_second_win / recent_total if recent_total > 0 else 0)
            self.win_rate_history['draw'].append(recent_draw / recent_total if recent_total > 0 else 0)

            # 图1：总loss图像
            plt.figure(figsize=(10, 6))
            plt.yscale('log')
            plt.xlabel('Games')
            plt.ylabel('Loss')
            plt.plot(self.losses)
            plt.title(f'Training Loss (Game {self.game_count})')
            plt.savefig(f"{self.args['file_name']}_losses.png")
            plt.close()

            # 图2：分离的policy loss和value loss图像（上下布局）
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.set_yscale('log')
            ax1.set_xlabel('Games')
            ax1.set_ylabel('Policy Loss')
            ax1.plot(self.policy_losses, color='blue')
            ax1.set_title(f'Policy Loss (Game {self.game_count})')
            ax1.grid(True, alpha=0.3)

            ax2.set_yscale('log')
            ax2.set_xlabel('Games')
            ax2.set_ylabel('Value Loss')
            ax2.plot(self.value_losses, color='orange')
            ax2.set_title(f'Value Loss (Game {self.game_count})')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{self.args['file_name']}_losses_detail.png")
            plt.close()

            # 图3：胜率图像
            if len(self.win_rate_history['black']) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(self.win_rate_history['black']))
                ax.plot(x, self.win_rate_history['black'], color='black', label='Black (First)', linewidth=2)
                ax.plot(x, self.win_rate_history['white'], color='gray', label='White (Second)', linewidth=2)
                ax.plot(x, self.win_rate_history['draw'], color='blue', label='Draw', linewidth=2)
                ax.set_xlabel('Training Generation')
                ax.set_ylabel('Win Rate')
                ax.set_title(f'Win Rate History (Recent 100 Games, Game {self.game_count})')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.args['file_name']}_win_rate.png")
                plt.close()

    @torch.inference_mode()
    def play(self, state, to_play):
        self.model.eval()
        action_probs = self.mcts.eval_search(state, to_play)
        action = np.argmax(action_probs)

        # 获取模型输出（eval 模式下只返回 policy, value）
        outputs = self.model(
            torch.tensor(self.game.encode_state(state, to_play), device=self.args['device']).unsqueeze(0)
        )
        policy, value = outputs[0], outputs[1]

        # policy, value, aux_policy, optimistic_policy, short_term_value = outputs

        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy *= self.game.get_is_legal_actions(state)
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        value = value.item()

        # 对于动作空间等于棋盘大小的游戏，按棋盘形状reshape
        # 对于Connect4等动作空间小于棋盘大小的游戏，保持一维
        if self.game.action_space_size == self.game.board_height * self.game.board_width:
            policy = policy.reshape(self.game.board_height, self.game.board_width)
            action_probs = action_probs.reshape(self.game.board_height, self.game.board_width)

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
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'game_count': self.game_count,
            'replay_buffer': self.replay_buffer.get_state(),
            'win_rate_history': self.win_rate_history,
        }

        torch.save(checkpoint, filepath)

        file_size = os.path.getsize(filepath)
        size_str = f"{file_size / 1024 / 1024:.1f}MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.1f}KB"
        print(f"Checkpoint saved to {filepath} ({size_str})")

    def load_checkpoint(self, filepath=None):
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

        if 'policy_losses' in checkpoint:
            self.policy_losses = checkpoint['policy_losses']
            print(f"Policy losses history loaded ({len(self.policy_losses)} data points)")

        if 'value_losses' in checkpoint:
            self.value_losses = checkpoint['value_losses']
            print(f"Value losses history loaded ({len(self.value_losses)} data points)")

        if 'game_count' in checkpoint:
            self.game_count = checkpoint['game_count']
            print(f"Game count loaded ({self.game_count} games)")

        if 'replay_buffer' in checkpoint:
            self.replay_buffer.load_state(checkpoint['replay_buffer'])
            print(f"Replay buffer loaded ({len(self.replay_buffer)} samples)")

        if 'win_rate_history' in checkpoint:
            self.win_rate_history = checkpoint['win_rate_history']
            print(f"Win rate history loaded ({len(self.win_rate_history['black'])} data points)")

        print(f"Checkpoint loaded from {filepath}")
        return True
