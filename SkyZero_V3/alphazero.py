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
    random_augment_batch,
    softmax,
    temperature_transform,
)


class Node:
    _ZERO_WDL = np.zeros(3, dtype=np.float64)

    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None, nn_value_probs=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.nn_value_probs = nn_value_probs if nn_value_probs is not None else Node._ZERO_WDL.copy()  # WDL [win, draw, loss]
        
        self.nn_policy = None
        self.nn_logits = None

        self.v = np.zeros(3, dtype=np.float64)  # cumulative WDL sum [win, draw, loss]
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

        masked_logits = np.where(
            self.game.get_is_legal_actions(state, to_play),
            policy_logits.flatten().cpu().numpy(),
            -np.inf,
        )

        nn_policy = softmax(masked_logits)

        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()  # WDL [win, draw, loss]
        return nn_policy, nn_value_probs, masked_logits

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

        masked_logits = np.where(
            self.game.get_is_legal_actions(state, to_play),
            policy_logits.flatten().cpu().numpy(),
            -np.inf,
        )

        nn_policy = softmax(masked_logits)

        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()  # WDL [win, draw, loss]
        return nn_policy, nn_value_probs, masked_logits

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
        nn_value_probs = nn_value_probs.mean(axis=0)  # average WDL across 8 symmetries

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
        masked_logits = np.where(is_legal_actions, avg_p_logits, -np.inf)
        nn_policy = softmax(masked_logits)

        return nn_policy, nn_value_probs, masked_logits

    def select(self, node):

        visited_policy_mass = sum(child.prior for child in node.children if child.n > 0)

        c_puct_init = self.args.get("c_puct", 1.1)
        c_puct_log = self.args.get("c_puct_log", 0.45)
        c_puct_base = self.args.get("c_puct_base", 500)

        total_child_weight = max(0, node.n - 1)

        c_puct = c_puct_init + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)

        explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01)

        # FPU - derive scalar Q from WDL: Q = W - L, in [-1, 1]
        parent_q = node.v / node.n if node.n > 0 else np.zeros(3)
        parent_utility = parent_q[0] - parent_q[2]  # W - L, [-1, 1]
        nn_utility = node.nn_value_probs[0] - node.nn_value_probs[2]  # W - L, [-1, 1]

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
        loss_value = -1.0
        fpu_value = fpu_value + (loss_value - fpu_value) * fpu_loss_prop

        best_score = -float("inf")
        best_child = None

        for child in node.children:
            if child.n == 0:
                q_value = fpu_value
            else:
                # Child's WDL is from child's perspective; parent's Q = -(child's W - child's L)
                child_q = child.v / child.n
                q_value = child_q[2] - child_q[0]  # L - W from child, [-1, 1]

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
            nn_policy, nn_value_probs, masked_logits = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_child", False):
            nn_policy, nn_value_probs, masked_logits = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value_probs, masked_logits = self._inference(state, to_play)

        node.nn_value_probs = nn_value_probs.copy()  # WDL [win, draw, loss]
        node.nn_policy = nn_policy.copy()
        node.nn_logits = masked_logits.copy()

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
        return nn_value_probs

    def root_expand(self, node):
        state = node.state
        to_play = node.to_play

        if self.args.get("enable_stochastic_transform_inference_for_root", True):
            nn_policy, nn_value_probs, masked_logits = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_root", False):
            nn_policy, nn_value_probs, masked_logits = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value_probs, masked_logits = self._inference(state, to_play)

        node.nn_value_probs = nn_value_probs.copy()  # WDL [win, draw, loss]
        node.nn_policy = nn_policy.copy()
        node.nn_logits = masked_logits.copy()

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
        return nn_policy, nn_value_probs

    @staticmethod
    def backpropagate(node, value):
        """Backpropagate WDL value up the tree. Flips [W,D,L] -> [L,D,W] at each level."""
        while node is not None:
            node.update(value)
            value = value[[2, 1, 0]]  # flip perspective: [W,D,L] -> [L,D,W]
            node = node.parent

    def _gumbel_sequential_halving(self, root, num_simulations, is_eval=False):

        m = min(num_simulations, self.args.get("gumbel_m", 16))  # 确定初始候选动作数量m
        logits = root.nn_logits.copy()  # 直接使用网络输出的原始logits（已mask非法动作为-inf）
        is_legal = self.game.get_is_legal_actions(root.state, root.to_play)  # 合法动作掩码
        
        if is_eval and not self.args.get("gumbel_stochastic_eval", False):
            g = np.zeros_like(logits)  # 评估模式下不加噪声
        else:
            g = np.random.gumbel(size=logits.shape)  # 采样Gumbel(0)噪声
            
        scores = logits + g  # 计算Gumbel分数
        legal_scores = np.where(is_legal, scores, -np.inf)
        surviving_actions = np.argsort(legal_scores)[-m:][::-1]  # 选取分数最高的m个动作
        surviving_actions = [a for a in surviving_actions if is_legal[a]]  # 排除非法动作
        m = len(surviving_actions)  # 最终预选动作
        
        if m > 0:
            phases = int(np.ceil(np.log2(m))) if m > 1 else 1  # 计算筛选轮数，m个候选动作每次减半，总共需要log2(m)次减半
            sims_budget = num_simulations  # 总模拟次数
            
            for phase in range(phases):  # 遍历每个阶段
                if sims_budget <= 0:
                    break
                
                remaining_phases = phases - phase
                sims_this_phase = sims_budget // remaining_phases  # 动态均分剩余预算给剩余阶段
                num_actions = len(surviving_actions)
                sims_per_action = max(1, sims_this_phase // num_actions)
                
                for _ in range(sims_per_action):  # 遍历每个阶段的每次模拟
                    if sims_budget <= 0:
                        break
                    for action in surviving_actions:
                        if sims_budget <= 0:
                            break
                        # 获取action为action的子节点
                        child = next((c for c in root.children if c.action_taken == action), None)
                        if child is None:
                            continue
                            
                        node = child
                        while node.is_expanded():
                            node = self.select(node)
                            assert node is not None
                            
                        if self.game.is_terminal(node.state, node.action_taken, -node.to_play):
                            # Terminal value as WDL one-hot
                            result = self.game.get_winner(node.state, node.action_taken, -node.to_play) * node.to_play
                            if result == 1:
                                value = np.array([1.0, 0.0, 0.0])  # win
                            elif result == -1:
                                value = np.array([0.0, 0.0, 1.0])  # loss
                            else:
                                value = np.array([0.0, 1.0, 0.0])  # draw
                        else:
                            value = self.expand(node)
                            
                        self.backpropagate(node, value)
                        sims_budget -= 1
                
                # 排除掉一半的动作
                if sims_budget <= 0:
                    break
                if phase < phases - 1:
                    # 剔除环节的得分公式：
                    # Score(a) = g(a)   +       logits(a)        + (c_visit     +          max_b N(b))        * c_scale * q(a)
                    #         Gumbel噪声 NN给出的原始策略对数概率 子节点访问次数  访问次数最大的节点的访问次数
                    max_n = max([c.n for c in root.children], default=0)
                    c_visit = self.args.get("gumbel_c_visit", 50)
                    c_scale = self.args.get("gumbel_c_scale", 1.0)
                    
                    def eval_action(a):
                        c = next((child for child in root.children if child.action_taken == a), None)
                        if c and c.n > 0:
                            child_wdl = c.v / c.n
                            q = child_wdl[2] - child_wdl[0]  # parent's Q = -(child's W - child's L)
                            q = (q + 1) / 2  # normalize to [0, 1]
                        else:
                            q = 0.5  # neutral value in [0, 1]
                        return logits[a] + g[a] + (c_visit + max_n) * c_scale * q
                        
                    surviving_actions.sort(key=eval_action, reverse=True)
                    surviving_actions = surviving_actions[:max(1, len(surviving_actions) // 2)]
                    
        c_visit = self.args.get("gumbel_c_visit", 50)
        c_scale = self.args.get("gumbel_c_scale", 1.0)
        max_n = max([c.n for c in root.children]) if root.children else 0
        
        num_actions = self.game.board_size ** 2
        # WDL Q-values per action: q_wdl[a] = parent-perspective WDL = child's [L, D, W]
        q_wdl = np.zeros((num_actions, 3))
        n_values = np.zeros(num_actions)
        for c in root.children:
            if c.n > 0:
                child_wdl = c.v / c.n
                q_wdl[c.action_taken] = child_wdl[[2, 1, 0]]  # flip to parent perspective
                n_values[c.action_taken] = c.n
                
        sum_n = np.sum(n_values)
        nn_value_wdl = root.nn_value_probs  # WDL [win, draw, loss]
        visited_mask = (n_values > 0)
        if sum_n > 0:
            # Policy-weighted average of visited Q-values (WDL)
            policy_visited = root.nn_policy * visited_mask
            policy_visited_sum = np.sum(policy_visited) + 1e-12
            weighted_q_wdl = np.sum(policy_visited[:, None] * q_wdl, axis=0) / policy_visited_sum
            v_mix_wdl = (nn_value_wdl + sum_n * weighted_q_wdl) / (1 + sum_n)
        else:
            v_mix_wdl = nn_value_wdl.copy()
            
        # completed_q: use actual Q for visited actions, v_mix for unvisited
        completed_q_wdl = np.where(visited_mask[:, None], q_wdl, v_mix_wdl[None, :])
        # sigma_q needs scalar per action for adding to logits: use W - L normalized to [0, 1]
        completed_q_scalar = completed_q_wdl[:, 0] - completed_q_wdl[:, 2]  # [-1, 1]
        completed_q_scalar = (completed_q_scalar + 1) / 2  # normalize to [0, 1]
        sigma_q = (c_visit + max_n) * c_scale * completed_q_scalar
        
        improved_logits = logits + sigma_q
        improved_logits[~is_legal] = -np.inf
        
        improved_policy = softmax(improved_logits)
        
        v_mix = v_mix_wdl  # WDL vector
        
        def final_eval(a):
            return logits[a] + g[a] + sigma_q[a]
            
        max_n_surviving = max([n_values[a] for a in surviving_actions], default=0)
        most_visited_action = [a for a in surviving_actions if n_values[a] == max_n_surviving]

        gumbel_action = max(most_visited_action, key=final_eval)

        return improved_policy, gumbel_action, v_mix

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, root=None):

        if root is None:
            root = Node(state, to_play)

        if not root.is_expanded():
            nn_policy, nn_value_probs = self.root_expand(root)
            self.backpropagate(root, nn_value_probs)
        else:
            # Tree Reuse: root already expanded from previous search, use cached values
            nn_policy = root.nn_policy
            nn_value_probs = root.nn_value_probs

        mcts_policy, gumbel_action, v_mix = self._gumbel_sequential_halving(root, num_simulations, is_eval=False)
        return mcts_policy, v_mix, nn_policy, nn_value_probs, gumbel_action

    @torch.inference_mode()
    def eval_search(self, state, to_play, num_simulations, root=None):

        if root is None:
            root = Node(state, to_play)

        if not root.is_expanded():
            nn_policy, nn_value_probs = self.root_expand(root)
            self.backpropagate(root, nn_value_probs)
        else:
            # Tree Reuse: root already expanded from previous search, use cached values
            nn_policy = root.nn_policy
            nn_value_probs = root.nn_value_probs

        mcts_policy, gumbel_action, v_mix = self._gumbel_sequential_halving(root, num_simulations, is_eval=True)
        return mcts_policy, v_mix, nn_policy, nn_value_probs, gumbel_action


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
            "opponent_policy_loss": [],
            "value_loss": [],
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
            board_size=self.game.board_size,
            num_planes=self.game.num_planes,
            min_buffer_size=args.get("min_buffer_size", 1000),
            linear_threshold=args.get("linear_threshold", 10000),
            alpha=args.get("alpha", 0.75),
            max_buffer_size=args.get("max_buffer_size", 3e6),
        )

    @torch.inference_mode()
    def selfplay(self):
        memory = []
        to_play = 1
        state = self.game.get_initial_state()

        in_soft_resign = False
        historical_v_mix = []
        last_action = None
        last_player = None

        root = Node(state, to_play)  # Tree Reuse: initialize root before game loop

        while not self.game.is_terminal(state, last_action, last_player):
            
            if in_soft_resign:
                num_simulations = max(
                    self.args["num_simulations"] // 4,
                    self.args.get("min_simulations_in_soft_resign", 8)
                )
            else:
                num_simulations = self.args["num_simulations"]

            mcts_policy, v_mix, nn_policy, nn_value_probs, gumbel_action = self.mcts.search(state, to_play, num_simulations, root=root)

            # Soft Resign - derive scalar from WDL for threshold check
            v_mix_scalar = v_mix[0] - v_mix[2]  # W - L
            historical_v_mix.append(v_mix_scalar)
            absmin_v_mix = min(abs(x) for x in historical_v_mix[-self.args.get("soft_resign_step_threshold", 3):])
            if (
                not in_soft_resign
                and absmin_v_mix >= self.args.get("soft_resign_threshold", 0.9)
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
                "nn_value_probs": nn_value_probs,  # WDL vector for psw
                "v_mix": v_mix,  # WDL vector
                "next_mcts_policy": None,
                "sample_weight": 1 if not in_soft_resign else self.args.get("soft_resign_sample_weight", 0.1),
            })

            # Gumbel Zero selfplay exploration - directly use the action derived from Gumbel-Max trick
            move_count = len(memory)
            half_life = self.args.get("half_life", self.game.board_size)
            t_init = self.args.get("move_temperature_init", 1.1)
            t_final = self.args.get("move_temperature_final", 1)
            t = t_init - (move_count / half_life) * (t_init - t_final)
            if move_count < half_life:
                action = np.random.choice(
                    len(mcts_policy),
                    p=temperature_transform(mcts_policy, t)
                )
            else:
                action = gumbel_action

            last_action = action
            last_player = to_play
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
        winner = self.game.get_winner(final_state, last_action, last_player)

        return_memory = []
        for sample in memory:
            # Outcome as WDL one-hot from this player's perspective
            result = winner * sample["to_play"]
            if result == 1:
                outcome = np.array([1.0, 0.0, 0.0])  # win
            elif result == -1:
                outcome = np.array([0.0, 0.0, 1.0])  # loss
            else:
                outcome = np.array([0.0, 1.0, 0.0])  # draw

            opponent_policy = sample["next_mcts_policy"] if sample["next_mcts_policy"] is not None else np.zeros_like(sample["mcts_policy"])
            sample_data = {
                "state": sample["state"],
                "to_play": sample["to_play"],
                "policy_target": sample["mcts_policy"],
                "opponent_policy_target": opponent_policy,
                "outcome": outcome,  # WDL one-hot
                "nn_policy": sample["nn_policy"],  # for psw
                "nn_value_probs": sample["nn_value_probs"],  # WDL vector for psw
                "v_mix": sample["v_mix"],  # WDL vector for psw and value target mix
                "sample_weight": sample["sample_weight"],
            }
            return_memory.append(sample_data)
        
        # Value target construction (KataGo-style TD): recursive exponential weighted average
        # of search root value (v_mix) and game outcome, with outcome weight increasing near end.
        # value_target[last] = outcome[last]
        # value_target[i] = (1 - now_factor) * value_target[i+1]_flipped + now_factor * v_mix[i]
        naw_factor_constant = self.args.get("value_target_mix_now_factor_constant", 0.2)
        now_factor = 1.0 / (1.0 + (self.game.board_size ** 2) * naw_factor_constant)
        return_memory[-1]["value_target"] = return_memory[-1]["outcome"].copy()
        for i in range(len(return_memory) - 2, -1, -1):
            next_value_target = return_memory[i+1]["value_target"]
            next_value_target = next_value_target[[2, 1, 0]]  # flip perspective: [W,D,L] -> [L,D,W]
            return_memory[i]["value_target"] = (1.0 - now_factor) * next_value_target + now_factor * return_memory[i]["v_mix"]

        surprise_weight = compute_policy_surprise_weights(
            return_memory,
            self.game.board_size,
            policy_surprise_data_weight=self.args.get("policy_surprise_data_weight", 0.5),
            value_surprise_data_weight=self.args.get("value_surprise_data_weight", 0.1),
        )

        return_memory = apply_surprise_weighting_to_game(return_memory, surprise_weight)

        # return memory:
        #   state
        #   to_play
        #   policy_target
        #   opponent_policy_target
        #   value_target
        #   sample_weight

        return return_memory, self.game.get_winner(final_state, last_action, last_player), len(memory), final_state

    def _train_batch(self, batch):

        # Encode raw state + to_play into encoded_state for the network
        batch["encoded_state"] = self.game.encode_state_batch(batch["state"], batch["to_play"])

        batch = random_augment_batch(batch, self.game.board_size)
        batch_size = len(batch["encoded_state"])

        encoded_states = torch.as_tensor(batch["encoded_state"], device=self.args["device"], dtype=torch.float32)
        policy_targets = torch.as_tensor(batch["policy_target"], device=self.args["device"], dtype=torch.float32)
        opponent_policy_targets = torch.as_tensor(batch["opponent_policy_target"], device=self.args["device"], dtype=torch.float32)
        value_targets = torch.as_tensor(batch["value_target"], device=self.args["device"], dtype=torch.float32)

        sample_weights = torch.as_tensor(batch["sample_weight"], device=self.args["device"], dtype=torch.float32)

        self.model.train()
        nn_output = self.model(encoded_states)

        policy_logits = nn_output["policy_logits"].view(batch_size, -1)
        opponent_policy_logits = nn_output["opponent_policy_logits"].view(batch_size, -1)

        def get_loss(logits, targets, weights, mask=None):
            loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)
            if mask is not None:  # For Opponent policy loss (Final state)
                masked_loss = loss * weights * mask
                return masked_loss.sum() / (mask.sum() + 1e-8)
            else:
                return (loss * weights).mean()

        # Policy Loss
        policy_loss = get_loss(policy_logits, policy_targets, sample_weights)

        # Opponent Policy Loss
        opp_target_sum = opponent_policy_targets.sum(dim=-1)
        opp_mask = (opp_target_sum > 0.5).float()
        opponent_policy_loss = get_loss(opponent_policy_logits, opponent_policy_targets, sample_weights, opp_mask)

        # Value Loss - value_targets is already [batch_size, 3] WDL probabilities
        value_loss = -torch.sum(value_targets * F.log_softmax(nn_output["value_logits"], dim=-1), dim=-1)
        value_loss = (value_loss * sample_weights).mean()

        loss = policy_loss + 0.15 * opponent_policy_loss + value_loss

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
            "value_loss": value_loss.item(),
        }
        return loss_dict

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
            f"cap={self.replay_buffer.max_buffer_size}"
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
                        f"Game: {self.game_count} | Sps: {sps:.1f} | TotalSamples: {self.replay_buffer.total_samples_added} | "
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
                for _ in range(train_steps_per_generation):
                    batch = self.replay_buffer.sample(batch_size)
                    loss_dic = self._train_batch(batch)
                    for key in batch_loss_dict:
                        batch_loss_dict[key].append(loss_dic.get(key, 0))

                for key in self.losses_dict:
                    self.losses_dict[key].append(np.mean(batch_loss_dict[key]))

                # calculate train interval by Target Replay Ratio
                avg_sample_len = np.mean(self.recent_sample_lengths)
                num_next = int(
                    self.args["batch_size"] * self.args["train_steps_per_generation"] / avg_sample_len / self.args["target_ReplayRatio"]
                )
                num_next = max(1, num_next)
                train_game_count = self.game_count + num_next

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

        actual_num_simulations = max(1, self.args["num_simulations"] - root.n)

        mcts_policy, v_mix, _, _, gumbel_action = self.mcts.eval_search(state, to_play, actual_num_simulations, root)

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
        
        # Value: WDL probabilities averaged across symmetries
        nn_value_probs = F.softmax(nn_output["value_logits"], dim=1).cpu().numpy()
        nn_value_probs = nn_value_probs.mean(axis=0)  # WDL [win, draw, loss]
        nn_value = nn_value_probs[0] - nn_value_probs[2]  # scalar for display
        # Policy, Opponent Policy:
        policy_logits = nn_output["policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)
        opponent_policy_logits = nn_output["opponent_policy_logits"].squeeze(1).cpu().numpy()  # (8, H, W)

        untransformed_pl = []
        untransformed_opl = []
        for i, (do_flip, k) in enumerate([(f, r) for f in [False, True] for r in range(4)]):
            pl = policy_logits[i]
            opl = opponent_policy_logits[i]
            if do_flip:
                pl = np.flip(pl, axis=1)
                opl = np.flip(opl, axis=1)
            pl = np.rot90(pl, k=-k)
            opl = np.rot90(opl, k=-k)
            untransformed_pl.append(pl.flatten())
            untransformed_opl.append(opl.flatten())
        
        avg_pl = np.mean(untransformed_pl, axis=0)
        is_legal_actions = self.game.get_is_legal_actions(state, to_play)
        avg_pl = np.where(is_legal_actions, avg_pl, -np.inf)
        nn_policy = softmax(avg_pl)

        avg_opl = np.mean(untransformed_opl, axis=0)
        next_is_legal_actions = self.game.get_is_legal_actions(
            self.game.get_next_state(state, gumbel_action, to_play),
            to_play
        )
        avg_opl = np.where(next_is_legal_actions, avg_opl, -np.inf)
        nn_opponent_policy = softmax(avg_opl)

        # v_mix is now a WDL vector; derive scalar for display
        v_mix_scalar = v_mix[0] - v_mix[2]

        if gumbel_action is not None:
            action = gumbel_action
        else:
            action = np.argmax(mcts_policy)

        info = {
            "mcts_policy": mcts_policy.reshape(self.game.board_size, self.game.board_size),
            "v_mix": v_mix_scalar,
            "v_mix_probs": v_mix,  # WDL vector
            "nn_policy": nn_policy.reshape(self.game.board_size, self.game.board_size),
            "nn_opponent_policy": nn_opponent_policy.reshape(self.game.board_size, self.game.board_size),
            "nn_value": nn_value,
            "nn_value_probs": nn_value_probs,
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

        # checkpoint = torch.load(
        #     filepath, map_location=self.args["device"], weights_only=False
        # )
        checkpoint = torch.load(
            filepath, map_location="cpu", weights_only=False
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
            self.replay_buffer.max_buffer_size = int(self.args.get("max_buffer_size", self.replay_buffer.max_buffer_size))
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
