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

from alphazero import Node, MCTS, AlphaZero


class TreeReuseNode(Node):
    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None, nn_value=0, value_probs=None):
        super().__init__(state, to_play, prior, parent, action_taken, nn_value)
        self.original_prior = prior
        self.value_probs = value_probs


class TreeReuseMCTS(MCTS):
    def __init__(self, game, args, model):
        super().__init__(game, args, model)

    def expand(self, node):
        if node.is_expanded():
            return node.nn_value

        state = node.state
        to_play = node.to_play

        if self.args.get("enable_stochastic_transform_inference", True):
            nn_policy, nn_value, nn_value_probs = self._inference_with_stochastic_transform(state, to_play)
        elif self.args.get("enable_symmetry_inference_for_child", False):
            nn_policy, nn_value, nn_value_probs = self._inference_with_symmetry(state, to_play)
        else:
            nn_policy, nn_value, nn_value_probs = self._inference(state, to_play)

        node.nn_value = nn_value
        node.value_probs = nn_value_probs

        for action, prob in enumerate(nn_policy):
            if prob > 0:
                child = TreeReuseNode(
                    state=self.game.get_next_state(state, action, to_play),
                    to_play=-to_play,
                    prior=prob,
                    parent=node,
                    action_taken=action,
                )
                node.children.append(child)
        return nn_value

    def root_expand(self, node):
        if not node.is_expanded():
            state = node.state
            to_play = node.to_play

            if self.args.get("enable_symmetry_inference_for_root", False):
                nn_policy, nn_value, value_probs = self._inference_with_symmetry(state, to_play)
            else:
                nn_policy, nn_value, value_probs = self._inference(state, to_play)

            node.nn_value = nn_value
            node.value_probs = value_probs

            for action, prob in enumerate(nn_policy):
                if prob > 0:
                    child = TreeReuseNode(
                        state=self.game.get_next_state(state, action, to_play),
                        to_play=-to_play,
                        prior=prob,
                        parent=node,
                        action_taken=action,
                    )
                    node.children.append(child)
        else:
            nn_policy = np.zeros(self.game.board_size ** 2)
            for child in node.children:
                nn_policy[child.action_taken] = child.original_prior
            nn_value = node.nn_value
            value_probs = node.value_probs
        
        for child in node.children:
            child.prior = child.original_prior

        return nn_policy, nn_value, value_probs

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, root=None):

        if root is None:
            root = TreeReuseNode(state, to_play)
    
        nn_policy, nn_value, nn_value_probs = self.root_expand(root)
        self.backpropagate(root, nn_value)

        for _ in range(num_simulations):
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search=True)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, nn_policy, nn_value_probs, root.v / root.n

    @torch.inference_mode()
    def eval_search(self, state, to_play, root=None, show_progress_bar=True):
        
        num_simulations = self.args["full_search_num_simulations"]

        if root is None:
            root = TreeReuseNode(state, to_play)

        _, nn_value, _ = self.root_expand(root)
        self.backpropagate(root, nn_value)

        if show_progress_bar:
            iterator = tqdm(range(num_simulations), desc="MCTS Evaluating", unit="sim")
        else:
            iterator = range(num_simulations)

        for _ in iterator:
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search=True)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, root.v / root.n, root.n

    @torch.inference_mode()
    def additive_search(self, state, to_play, root=None, show_progress_bar=True):
        
        num_simulations = self.args["full_search_num_simulations"] - root.n

        if root is None:
            root = TreeReuseNode(state, to_play)

        _, nn_value, _ = self.root_expand(root)
        self.backpropagate(root, nn_value)

        if show_progress_bar:
            iterator = tqdm(range(num_simulations), desc="MCTS Evaluating", unit="sim")
        else:
            iterator = range(num_simulations)

        for _ in iterator:
            node = root

            while node.is_expanded():
                node = self.select(node, is_full_search=True)

            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # outcome(to_play view)
            else:
                value = self.expand(node)  # nn_value

            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy, root.v / root.n, root.n


class TreeReuseAlphaZero(AlphaZero):
    def __init__(self, game, model, optimizer, args):
        super().__init__(game, model, optimizer, args)
        self.mcts = TreeReuseMCTS(game, args, model)

    def apply_action(self, root, action):
        if root is not None:
            for child in root.children:
                if child.action_taken == action:
                    child.parent = None
                    return child
        return None

    @torch.inference_mode()
    def play(self, state, to_play, root=None, show_progress_bar=True, additive=True):
        self.model.eval()
        
        # 1. MCTS 搜索
        if root is None or not np.array_equal(root.state, state):
            root = TreeReuseNode(state, to_play)

        if additive:
            mcts_policy, root_value, root_n = self.mcts.additive_search(state, to_play, root=root, show_progress_bar=show_progress_bar)
        else:
            mcts_policy, root_value, root_n = self.mcts.eval_search(state, to_play, root=root, show_progress_bar=show_progress_bar)
        action = np.argmax(mcts_policy)

        next_root = self.apply_action(root, action)
        
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
        # nn_output = {
        #     "policy_logits": total_policy_logits[:, 0:1, :, :],
        #     "opponent_policy_logits": total_policy_logits[:, 1:2, :, :],
        #     "soft_policy_logits": soft_policy_logits,
        #     "ownership": ownership,
        #     "value_logits": value_logits,
        # }

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
            "ownership": averaged_results["ownership"].squeeze(),
            "root_n": root_n,
            "nn_output": nn_output,
        }

        return action, info, next_root
