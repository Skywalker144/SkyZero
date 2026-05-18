import math
import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from replaybuffer import ReplayBuffer
from policy_surprise_weighting import compute_policy_surprise_weights, apply_surprise_weighting
from softresign import softresign_adjust, stochastic_resample
from mcts_selectors import make_selector, compute_completed_q
from utils import (
    temperature_transform,
    random_augment_batch,
    softmax,
    add_dirichlet_noise,
)

class Node:
    def __init__(self, state, to_play, prior=0, parent=None, action_taken=None):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self.wdl = np.zeros(3, dtype=np.float32)
        self.n = 0
        # Cached NN outputs (masked logits + WDL probs). Populated when this node
        # is evaluated; required by the Gumbel non-root selection rule.
        self.nn_policy_logits = None
        self.nn_value_probs = None
        # Σ (value[2]-value[0])² over backups — feeds PuctSelector's
        # Bayesian-shrunk parent utility stdev. Sign irrelevant (squared).
        self.q_sum_sq = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def update(self, value):
        self.wdl += value
        u = float(value[2]) - float(value[0])
        self.q_sum_sq += u * u
        self.n += 1

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model.to(args["device"])
        self.model.eval()
        self.selector = make_selector(args, game)

    def _inference(self, node):
        encoded_state = self.game.encode_state(node.state, node.to_play)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.args["device"]).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value_logits = self.model(state_tensor)

        # policy_logits: (1, 4, H*W) — channel 0 = main policy for MCTS search.
        # opp / soft_main / soft_opp are training-time auxiliary heads only.
        policy_logits = policy_logits[0, 0].cpu().numpy()
        value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()

        legal_mask = self.game.get_is_legal_actions(node.state, node.to_play)
        masked_logits = np.where(legal_mask, policy_logits, -np.inf)
        policy = softmax(masked_logits)

        node.nn_policy_logits = masked_logits
        node.nn_value_probs = value_probs

        return policy, value_probs
    
    def _inference_symmetry(self, node):
        pass

    def _simulate(self, start_node):
        # One MCTS rollout from `start_node`: select down via self.selector, expand+eval
        # at the leaf, then backprop. Shared by PUCT search and Gumbel SH (the latter
        # starts each rollout at a root child rather than at the root itself).
        node = start_node
        while node.is_expanded():
            node = self.selector.select(node)

        if self.game.is_terminal(node.state):
            relative_winner = self.game.get_winner(node.state) * node.to_play
            value = np.eye(3, dtype=np.float32)[1 - relative_winner]
        else:
            policy, value = self._inference(node)
            self.expand(node, policy)

        self.backpropagate(node, value)

    def expand(self, node, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                next_state = self.game.get_next_state(node.state, action, node.to_play)
                child = Node(
                    state=next_state,
                    to_play=-node.to_play,
                    prior=prob,
                    parent=node,
                    action_taken=action
                )
                node.children.append(child)

    def backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value = value[[2, 1, 0]]
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, add_noise=True):
        root = Node(state, to_play)

        # Initial expand for root
        nn_policy, nn_value = self._inference(root)
        if add_noise:
            policy = add_dirichlet_noise(
                nn_policy,
                self.args.get("dirichlet_alpha", 10 / self.game.board_size ** 2),
                self.args.get("dirichlet_epsilon", 0.25)
            )
        else:
            policy = nn_policy
        self.expand(root, policy)
        self.backpropagate(root, nn_value)

        for _ in range(num_simulations - 1):
            self._simulate(root)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)

        root_wdl = (root.wdl / root.n).astype(np.float32) if root.n > 0 else nn_value
        info = {
            "nn_policy": nn_policy.astype(np.float32),
            "nn_value": nn_value.astype(np.float32),
            "root_wdl": root_wdl,
        }
        return mcts_policy, info
    
    @torch.inference_mode()
    def gumbel_sequential_halving(self, state, to_play, num_simulations):
        root = Node(state, to_play)

        # Root inference: keep raw masked logits and value for Gumbel scoring
        encoded_state = self.game.encode_state(state, to_play)
        state_tensor = torch.tensor(
            encoded_state, dtype=torch.float32, device=self.args["device"]
        ).unsqueeze(0)
        policy_logits, value_logits = self.model(state_tensor)
        # (1, 4, H*W) → main channel only (the rest are training-only aux heads)
        policy_logits = policy_logits[0, 0].cpu().numpy()
        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()  # WDL

        is_legal = self.game.get_is_legal_actions(state, to_play)
        masked_logits = np.where(is_legal, policy_logits, -np.inf)
        nn_policy = softmax(masked_logits)

        # Cache root NN outputs so the Gumbel selector can operate from any depth.
        root.nn_policy_logits = masked_logits
        root.nn_value_probs = nn_value_probs

        self.expand(root, nn_policy)
        self.backpropagate(root, nn_value_probs)

        # Pick m initial candidate actions via Gumbel-perturbed logits
        m = min(num_simulations, self.args.get("gumbel_m", 16))
        g = np.random.gumbel(size=masked_logits.shape)
        scores = np.where(is_legal, masked_logits + g, -np.inf)
        surviving_actions = np.argsort(scores)[-m:][::-1].tolist()  # 按照分数从大到小排列前m个动作
        surviving_actions = [a for a in surviving_actions if is_legal[a]]  # 剔除非法动作
        m = len(surviving_actions)

        c_visit = self.args.get("gumbel_c_visit", 50)
        c_scale = self.args.get("gumbel_c_scale", 1.0)

        if m > 0:
            phases = int(np.ceil(np.log2(m))) if m > 1 else 1
            sims_budget = num_simulations

            for phase in range(phases):
                if sims_budget <= 0:
                    break
                remaining_phases = phases - phase
                sims_this_phase = sims_budget // remaining_phases
                sims_per_action = max(1, sims_this_phase // len(surviving_actions))

                for _ in range(sims_per_action):
                    if sims_budget <= 0:
                        break
                    for action in surviving_actions:
                        if sims_budget <= 0:
                            break
                        child = next((c for c in root.children if c.action_taken == action), None)
                        if child is None:
                            continue
                        self._simulate(child)
                        sims_budget -= 1

                if sims_budget <= 0:
                    break
                if phase < phases - 1:
                    max_n = max((c.n for c in root.children), default=0)

                    def eval_action(a):
                        c = next((child for child in root.children if child.action_taken == a), None)
                        if c is not None and c.n > 0:
                            child_wdl = c.wdl / c.n
                            q = child_wdl[2] - child_wdl[0]  # parent's Q = -(child W - child L)
                            q = (q + 1) / 2
                        else:
                            q = 0.5
                        return masked_logits[a] + g[a] + (c_visit + max_n) * c_scale * q

                    surviving_actions.sort(key=eval_action, reverse=True)
                    surviving_actions = surviving_actions[: max(1, len(surviving_actions) // 2)]

        # Completed Q -> improved policy
        completed_q_scalar, n_values, _, v_mix_wdl = compute_completed_q(
            root, self.game.board_size ** 2
        )
        sigma_q = (c_visit + n_values.max()) * c_scale * completed_q_scalar

        improved_logits = masked_logits + sigma_q
        improved_logits[~is_legal] = -np.inf
        improved_policy = softmax(improved_logits)

        if surviving_actions:
            def final_eval(a):
                return masked_logits[a] + g[a] + sigma_q[a]

            max_n_surviving = max((n_values[a] for a in surviving_actions), default=0)
            most_visited_action = [a for a in surviving_actions if n_values[a] == max_n_surviving]
            gumbel_action = max(most_visited_action, key=final_eval)
        else:
            gumbel_action = int(np.argmax(improved_policy))

        info = {
            "nn_policy": nn_policy.astype(np.float32),
            "nn_value": nn_value_probs.astype(np.float32),
            "root_wdl": v_mix_wdl.astype(np.float32),
        }
        return improved_policy, gumbel_action, v_mix_wdl, info


class AlphaZero:
    def __init__(self, game, model, optimizer, args):
        self.game = game
        self.model = model.to(args["device"])
        self.optimizer = optimizer
        self.args = args
        self.mcts = MCTS(game, args, model)

        B = args.get("batch_size", 128)
        T = args.get("train_steps_per_iteration", 100)
        min_buffer_size = max(args.get("min_buffer_size", 250000), B * T)

        self.replay_buffer = ReplayBuffer(
            max_buffer_size=args.get("max_buffer_size", 1e7),
            min_buffer_size=min_buffer_size,
            window_exponent=args.get("window_exponent", 0.65),
            window_expand_per_row=args.get("window_expand_per_row", 0.4),
        )

        self.iteration = 1
        self.total_samples_selfplay_generated = 0
        self.total_game_count = 0

    def selfplay(self):
        memory = []
        # all_policies[i] = mcts_policy at game step i (incl. PCR fast steps);
        # memory[k]["game_step"] indexes into it. After the game, for each
        # training entry we look up all_policies[game_step+1] (the OPPONENT's
        # search at the next ply) as the opp_policy_target.
        all_policies = []
        state = self.game.get_initial_state()
        to_play = 1

        # SoftResign: 每步 push max(W, D, L) of root WDL(视角无关)。
        # 连续 lookback 步全 > threshold 时,按二次曲线减 sims 并下调样本权重。
        history_wl_max = []
        softresign_enabled = bool(self.args.get("enable_softresign", 0))

        def maybe_softresign(orig_sims):
            if not softresign_enabled:
                return orig_sims, 1.0
            return softresign_adjust(
                history_wl_max, orig_sims,
                threshold=self.args.get("softresign_threshold", 0.9),
                lookback=self.args.get("softresign_lookback", 3),
                min_sims=self.args.get("softresign_min_sims", 100),
                min_weight=self.args.get("softresign_min_weight", 0.1),
            )

        while not self.game.is_terminal(state):
            for_train = 1
            sample_weight = 1.0

            if self.args.get("algo", "puct") == "puct":
                # Playout Cap Randomization. fast 路径用固定 cheap sims、不进训练、不走 SoftResign。
                if self.args.get("enable_pcr", 0) and np.random.rand() >= self.args.get("full_search_prob", 0.25):
                    sims = self.args.get("fast_search_num_simulations", 80)
                    mcts_policy, info = self.mcts.search(state, to_play, sims, add_noise=False)
                    for_train = 0
                else:
                    sims, sample_weight = maybe_softresign(self.args["num_simulations"])
                    mcts_policy, info = self.mcts.search(state, to_play, sims)

                t = self.args.get("move_temperature", 1.0)
                if len(memory) > self.args.get("half_life", self.game.board_size):
                    t = 0.01
                action = np.random.choice(
                    len(mcts_policy),
                    p=temperature_transform(mcts_policy, t)
                )

            elif self.args.get("algo", "puct") == "gumbel":
                sims, sample_weight = maybe_softresign(self.args["num_simulations"])
                mcts_policy, action, _, info = self.mcts.gumbel_sequential_halving(state, to_play, sims)

            if softresign_enabled:
                history_wl_max.append(float(np.max(info["root_wdl"])))

            if for_train:
                entry = {
                    "state": state, "to_play": to_play,
                    "mcts_policy": mcts_policy,
                    "sample_weight": sample_weight,
                    "game_step": len(all_policies),
                }
                if self.args.get("enable_psw", 0):
                    entry["nn_policy"] = info["nn_policy"]
                    entry["nn_value_probs"] = info["nn_value"]
                memory.append(entry)

            all_policies.append(mcts_policy)
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        winner = self.game.get_winner(state)
        n_actions = self.game.board_size ** 2
        return_memory = []
        for sample in memory:
            relative_winner = winner * sample["to_play"]
            value_target = np.eye(3, dtype=np.float32)[1 - relative_winner]
            # opp_policy_target = next ply's MCTS visits (already from opp's
            # POV since to_play alternates). Missing for the final ply →
            # mask=0 zeros the loss contribution from that sample.
            next_step = sample["game_step"] + 1
            if next_step < len(all_policies):
                opp_target = all_policies[next_step].astype(np.float32)
                opp_mask = 1.0
            else:
                opp_target = np.zeros(n_actions, dtype=np.float32)
                opp_mask = 0.0
            return_memory.append({
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"].astype(np.float32),
                "opp_policy_target": opp_target,
                "opp_policy_mask": np.float32(opp_mask),
                "value_target": value_target,
            })

        if self.args.get("enable_psw", 0):
            # PSW: 把 SoftResign 的 sample_weight 作为 baseline 传进去。PSW 内部会按其计算
            # 重采样权重(policy/value surprise bonus 仍然加上)。
            game_data = []
            for sample in memory:
                relative_winner = winner * sample["to_play"]
                value_target = np.eye(3, dtype=np.float32)[1 - relative_winner]
                game_data.append({
                    "policy_target": sample["mcts_policy"],
                    "nn_policy": sample["nn_policy"],
                    "value_target": value_target,
                    "nn_value_probs": sample["nn_value_probs"],
                    "sample_weight": sample["sample_weight"],
                })
            psw_weights = compute_policy_surprise_weights(
                game_data,
                self.args.get("policy_surprise_data_weight", 0.5),
                self.args.get("value_surprise_data_weight", 0.1),
            )
            return_memory = apply_surprise_weighting(return_memory, psw_weights)
        elif softresign_enabled:
            weights = np.array([s["sample_weight"] for s in memory], dtype=np.float32)
            if (weights < 1.0).any():
                return_memory = stochastic_resample(return_memory, weights)

        game_length = np.count_nonzero(state)
        return return_memory, winner, game_length

    def _compute_needed_games_next_iter(self):
        # samples_trained = batch_size * train_steps_per_iteration
        # target new samples = samples_trained / target_ReplayRatio
        # games = target_new_samples / avg_sample_len
        if self.recent_sample_lengths:
            avg_len = np.mean(self.recent_sample_lengths)
        else:
            avg_len = self.args.get("default_avg_sample_len", 30)
        
        needed_games_next_iter = self.args["batch_size"] * self.args["train_steps_per_iteration"] / avg_len / self.args["target_ReplayRatio"]
        if not self.replay_buffer.is_ready():
            needed_games_next_iter = (self.replay_buffer.min_buffer_size - self.replay_buffer.total_samples_added) / avg_len
            needed_games_next_iter *= 1.05

        return max(1, math.ceil(needed_games_next_iter))

    def learn(self):
        needed_games_next_iter = self._compute_needed_games_next_iter()
        self.sps_history.append((time.time(), self.total_samples_selfplay_generated))
        self._learn_start_wall = time.time()

        while True:

            # Selfplay
            for game_idx in range(needed_games_next_iter):
                memory, winner, game_length = self.selfplay()
                self.replay_buffer.add_game_memory(memory)

            # Train
            if self.replay_buffer.is_ready():
                self.train()

                self.iteration += 1

            needed_games_next_iter = self._compute_needed_games_next_iter()

    def train(self):
        B = self.args["batch_size"]
        total_samples = B * self.args["train_steps_per_iteration"]
        pool = self.replay_buffer.sample(total_samples)

        for i in tqdm(range(self.args["train_steps_per_iteration"]), desc="[Train] ", ncols=100):
            batch = pool[i * B : (i + 1) * B]
            self._train_step(batch)

    def _train_step(self, batch):
        batch = random_augment_batch(batch, self.game.board_size)

        device = self.args["device"]
        states = torch.tensor(np.array([s["encoded_state"] for s in batch]), dtype=torch.float32, device=device)
        policy_targets = torch.tensor(np.array([s["policy_target"] for s in batch]), dtype=torch.float32, device=device)
        opp_policy_targets = torch.tensor(np.array([s["opp_policy_target"] for s in batch]), dtype=torch.float32, device=device)
        opp_policy_mask = torch.tensor(np.array([s["opp_policy_mask"] for s in batch]), dtype=torch.float32, device=device)
        value_targets = torch.tensor(np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=device)

        # Loss weights aligned with V5 train.py (KataGo defaults):
        #   main = 1.0, opp = 0.15, soft scale = 8.0 multiplier on top of base.
        policy_w = self.args.get("policy_loss_weight", 1.0)
        opp_w = self.args.get("opp_policy_loss_weight", 0.15)
        soft_w = self.args.get("soft_policy_loss_weight", 8.0)

        # KataGo soft target: (p + 1e-7)^0.25 normalized. Flattens visit
        # distribution so soft head learns "everything reasonable".
        soft_main_target = (policy_targets + 1e-7).clamp_min(0.0).pow(0.25)
        soft_main_target = soft_main_target / soft_main_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        soft_opp_target = (opp_policy_targets + 1e-7).clamp_min(0.0).pow(0.25)
        soft_opp_target = soft_opp_target / soft_opp_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value_logits = self.model(states)
        # (B, 4, H*W) — slice main / opp / soft_main / soft_opp.
        p_main, p_opp, p_soft, p_soft_opp = (policy_logits[:, i, :] for i in range(4))

        def soft_ce(logits, target, sample_mask=None):
            # Cross-entropy with soft target. If sample_mask is given, zero
            # out per-sample contributions (used for opp_policy on final ply).
            per_sample = -torch.sum(target * F.log_softmax(logits, dim=-1), dim=-1)
            if sample_mask is not None:
                denom = sample_mask.sum().clamp_min(1e-8)
                return (per_sample * sample_mask).sum() / denom
            return per_sample.mean()

        policy_loss = soft_ce(p_main, policy_targets)
        opp_policy_loss = soft_ce(p_opp, opp_policy_targets, opp_policy_mask)
        soft_main_loss = soft_ce(p_soft, soft_main_target)
        soft_opp_loss = soft_ce(p_soft_opp, soft_opp_target, opp_policy_mask)
        value_loss = -torch.mean(torch.sum(value_targets * F.log_softmax(value_logits, dim=1), dim=1))

        # V5 weighting: soft losses scaled by base × soft_w (so soft_main
        # carries weight policy_w * soft_w, soft_opp carries opp_w * soft_w).
        total_loss = (
            policy_w * policy_loss
            + opp_w * opp_policy_loss
            + (policy_w * soft_w) * soft_main_loss
            + (opp_w * soft_w) * soft_opp_loss
            + value_loss
        )

        total_loss.backward()
        self.optimizer.step()

