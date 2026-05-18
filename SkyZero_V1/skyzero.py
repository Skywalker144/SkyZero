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
        # Cached NN outputs (masked logits + WDL probs), populated on eval.
        # Required by the Gumbel non-root selector to compute completed Q.
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

        # (1, 4, H*W) — channel 0 = main policy for MCTS search; aux heads
        # (opp / soft_main / soft_opp) are training-time only.
        policy_logits = policy_logits[0, 0].cpu().numpy()
        value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()

        legal_mask = self.game.get_is_legal_actions(node.state, node.to_play)
        masked_logits = np.where(legal_mask, policy_logits, -np.inf)
        policy = softmax(masked_logits)

        node.nn_policy_logits = masked_logits
        node.nn_value_probs = value_probs

        return policy, value_probs

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

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations, add_noise=True):
        root = Node(state, to_play)

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
        # (1, 4, H*W) → main channel only
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
        self.cumulative_runtime = 0.0

        self.losses_dict = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": []
        }
        
        self.stats_window_size = self.args.get("stats_window_size", 300)
        self.winrate_history = deque(maxlen=self.stats_window_size)
        self.recent_sample_lengths = deque(maxlen=self.stats_window_size)
        self.recent_game_lengths = deque(maxlen=self.stats_window_size)
        self.sps_history = deque(maxlen=self.stats_window_size)

        self.stats_history = {
            "total_samples": [],
            "black_rate": [],
            "draw_rate": [],
            "white_rate": [],
            "avg_game_len": [],
        }

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
            # POV since to_play alternates). Missing on the final ply →
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
            # PSW: SoftResign 的 sample_weight 作为 baseline 传入,PSW 自己再叠 policy/value surprise bonus。
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

    def _record_iter_stats(self):
        self.stats_history["total_samples"].append(self.total_samples_selfplay_generated)
        n = len(self.winrate_history)
        if n > 0:
            self.stats_history["black_rate"].append(self.winrate_history.count(1) / n)
            self.stats_history["draw_rate"].append(self.winrate_history.count(0) / n)
            self.stats_history["white_rate"].append(self.winrate_history.count(-1) / n)
        else:
            self.stats_history["black_rate"].append(0.0)
            self.stats_history["draw_rate"].append(0.0)
            self.stats_history["white_rate"].append(0.0)
        self.stats_history["avg_game_len"].append(
            float(np.mean(self.recent_game_lengths)) if self.recent_game_lengths else 0.0
        )

    def learn(self):
        needed_games_next_iter = self._compute_needed_games_next_iter()
        self.sps_history.append((time.time(), self.total_samples_selfplay_generated))
        self._learn_start_wall = time.time()

        try:
            while True:
                if self.iteration > self.args.get("num_iterations", -1) > 0:
                    print(f"Reached max iterations ({self.args.get('num_iterations', -1)}). Stopping.")
                    break
                if self.total_samples_selfplay_generated >= self.args.get("max_total_samples", -1) > 0:
                    print(f"Reached max total samples ({self.args.get('max_total_samples', -1)}). Stopping.")
                    break
                if self.args.get("max_runtime_seconds", -1) > 0:
                    elapsed_total = self.cumulative_runtime + (time.time() - self._learn_start_wall)
                    if elapsed_total >= self.args["max_runtime_seconds"]:
                        print(f"Reached max runtime ({self.args['max_runtime_seconds']}s). Stopping.")
                        break

                print(f"\n\n ----- Iteration {self.iteration} -----")
                print(
                    f"TotalGames:{self.total_game_count} "
                    f"TotalSamples:{self.total_samples_selfplay_generated} "
                    f"Window:{self.replay_buffer.window_size()} "
                    f"BufferLen:{len(self.replay_buffer)}"
                )

                # Selfplay
                gw = len(str(needed_games_next_iter))
                for game_idx in range(needed_games_next_iter):
                    memory, winner, game_length = self.selfplay()
                    self.replay_buffer.add_game_memory(memory)

                    self.total_samples_selfplay_generated += len(memory)
                    self.total_game_count += 1
                    self.recent_sample_lengths.append(len(memory))
                    self.recent_game_lengths.append(game_length)
                    self.winrate_history.append(winner)

                    if (game_idx + 1) % self.args.get("print_interval", 20) == 0:
                        self.sps_history.append((time.time(), self.total_samples_selfplay_generated))
                        t0, s0 = self.sps_history[0]
                        t1, s1 = self.sps_history[-1]
                        sps = (s1 - s0) / max(t1 - t0, 1e-9)
                        print(
                            f"[SelfPlay] Games={game_idx+1:0{gw}d}/{needed_games_next_iter} "
                            f"Sps={sps:.1f} "
                            f"GameLen:Avg={sum(self.recent_game_lengths)/len(self.recent_game_lengths):.1f} "
                            f"Min={int(min(self.recent_game_lengths))} "
                            f"Max={int(max(self.recent_game_lengths))} "
                            f"Std={int(np.std(self.recent_game_lengths))} "
                            f"BDW={int(self.winrate_history.count(1)/len(self.winrate_history)*100)}/"
                            f"{int(self.winrate_history.count(0)/len(self.winrate_history)*100)}/"
                            f"{int(self.winrate_history.count(-1)/len(self.winrate_history)*100)}",
                            flush=True
                        )

                # Train
                if self.replay_buffer.is_ready():
                    print()
                    losses = self.train()
                    print(
                        f"[Train] TotalLoss={losses['total_loss']/self.args['train_steps_per_iteration']:.2f} "
                        f"PLoss={losses['policy_loss']/self.args['train_steps_per_iteration']:.2f} "
                        f"VLoss={losses['value_loss']/self.args['train_steps_per_iteration']:.2f}"
                    )
                    self._record_iter_stats()

                    completed_iter = self.iteration
                    self.iteration += 1
                    if completed_iter % self.args.get("save_interval", 10) == 0:
                        self.save_checkpoint(f"checkpoint_{completed_iter}.pth")

                needed_games_next_iter = self._compute_needed_games_next_iter()

                self._plot_metrics()

        except KeyboardInterrupt:
            print(f"\nStopping... saving checkpoint at iter={self.iteration}", flush=True)
            if self.iteration > 0:
                self.save_checkpoint(f"checkpoint_{self.iteration}_interrupt.pth")

    def train(self):
        B = self.args["batch_size"]
        total_samples = B * self.args["train_steps_per_iteration"]
        pool = self.replay_buffer.sample(total_samples)

        losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        for i in tqdm(range(self.args["train_steps_per_iteration"]), desc="[Train] ", ncols=100):
            batch = pool[i * B : (i + 1) * B]
            total_loss, policy_loss, value_loss = self._train_step(batch)
            losses["total_loss"] += total_loss
            losses["policy_loss"] += policy_loss
            losses["value_loss"] += value_loss

        for key in self.losses_dict:
            self.losses_dict[key].append(losses[key] / self.args["train_steps_per_iteration"])
        return losses

    def _train_step(self, batch):
        batch = random_augment_batch(batch, self.game.board_size)

        device = self.args["device"]
        states = torch.tensor(np.array([s["encoded_state"] for s in batch]), dtype=torch.float32, device=device)
        policy_targets = torch.tensor(np.array([s["policy_target"] for s in batch]), dtype=torch.float32, device=device)
        opp_policy_targets = torch.tensor(np.array([s["opp_policy_target"] for s in batch]), dtype=torch.float32, device=device)
        opp_policy_mask = torch.tensor(np.array([s["opp_policy_mask"] for s in batch]), dtype=torch.float32, device=device)
        value_targets = torch.tensor(np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=device)

        # Loss weights aligned with V5 train.py (KataGo defaults):
        #   main = 1.0, opp = 0.15, soft scale = 8.0 (multiplied on top of base).
        policy_w = self.args.get("policy_loss_weight", 1.0)
        opp_w = self.args.get("opp_policy_loss_weight", 0.15)
        soft_w = self.args.get("soft_policy_loss_weight", 8.0)

        # KataGo soft target: (p + 1e-7)^0.25 renormalized. Flattens the visit
        # distribution so the soft head learns "everything reasonable".
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

        # V5 weighting: soft losses scaled by base × soft_w.
        total_loss = (
            policy_w * policy_loss
            + opp_w * opp_policy_loss
            + (policy_w * soft_w) * soft_main_loss
            + (opp_w * soft_w) * soft_opp_loss
            + value_loss
        )

        total_loss.backward()
        self.optimizer.step()

        # Caller reports (total, policy, value). Keep that interface; the aux
        # losses are folded into total but not separately exposed for now.
        return total_loss.item(), policy_loss.item(), value_loss.item()

    def save_checkpoint(self, filename):
        ckpt_dir = os.path.join(self.args.get("data_dir", "data"), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_game_count": self.total_game_count,
            "total_samples_selfplay_generated": self.total_samples_selfplay_generated,
            "cumulative_runtime": self.cumulative_runtime + (time.time() - self._learn_start_wall),
            "losses_dict": self.losses_dict,
            "stats_history": self.stats_history,
            "winrate_history": list(self.winrate_history),
            "recent_sample_lengths": list(self.recent_sample_lengths),
            "recent_game_lengths": list(self.recent_game_lengths),
            "replay_buffer": self.replay_buffer.get_state(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename=None):
        if filename is None:
            import glob
            ckpt_dir = os.path.join(self.args.get("data_dir", "data"), "checkpoints")
            checkpoints = glob.glob(os.path.join(ckpt_dir, "*.pth"))
            if not checkpoints:
                print("No checkpoints found.")
                return False
            filename = max(checkpoints, key=os.path.getmtime)

        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=self.args["device"], weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 1)
        self.total_game_count = checkpoint.get("total_game_count", 0)
        self.total_samples_selfplay_generated = checkpoint.get("total_samples_selfplay_generated", 0)
        self.losses_dict = checkpoint.get(
            "losses_dict", {"total_loss": [], "policy_loss": [], "value_loss": []}
        )
        self.stats_history = checkpoint.get(
            "stats_history",
            {"total_samples": [], "black_rate": [], "draw_rate": [], "white_rate": [], "avg_game_len": []},
        )
        self.stats_history.setdefault("total_samples", [])
        for key, attr in (
            ("winrate_history", "winrate_history"),
            ("recent_sample_lengths", "recent_sample_lengths"),
            ("recent_game_lengths", "recent_game_lengths"),
        ):
            deq = deque(maxlen=self.stats_window_size)
            for v in checkpoint.get(key, []):
                deq.append(v)
            setattr(self, attr, deq)
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])
        self.cumulative_runtime = checkpoint.get("cumulative_runtime", 0.0)
        print(
            f"Checkpoint loaded. iter={self.iteration} "
            f"games={self.total_game_count} samples={self.total_samples_selfplay_generated}"
        )
        return True

    def _plot_metrics(self):
        try:
            logs_dir = os.path.join(self.args.get("data_dir", "data"), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            runtime_sec = int(self.cumulative_runtime + (time.time() - self._learn_start_wall))
            d, r = divmod(runtime_sec, 86400)
            h, r = divmod(r, 3600)
            m, s = divmod(r, 60)
            parts = []
            if d: parts.append(f"{d}d")
            if h: parts.append(f"{h}h")
            if m: parts.append(f"{m}m")
            parts.append(f"{s}s")
            runtime_str = " ".join(parts)


            x_samples = self.stats_history.get("total_samples", [])
            xlabel = "Total Self-Play Samples"

            total_losses = self.losses_dict.get("total_loss", [])
            n_loss = min(len(x_samples), len(total_losses))
            if n_loss > 0:
                plt.figure(figsize=(10, 6))
                plt.suptitle(f"Runtime: {runtime_str}", fontsize=10, y=0.98)
                plt.plot(x_samples[:n_loss], total_losses[:n_loss], label="Total Loss")
                plt.title(f"Total Training Loss (Games={self.total_game_count})")
                plt.xlabel(xlabel)
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both")
                plt.savefig(os.path.join(logs_dir, "total_loss.png"), dpi=300)
                plt.close()

                plt.figure(figsize=(10, 6))
                plt.suptitle(f"Runtime: {runtime_str}", fontsize=10, y=0.98)
                for key, vals in self.losses_dict.items():
                    if key == "total_loss" or not vals:
                        continue
                    n = min(len(x_samples), len(vals))
                    if n == 0:
                        continue
                    plt.plot(x_samples[:n], vals[:n], label=key.replace("_", " ").title())
                plt.title(f"Loss Components (Games={self.total_game_count})")
                plt.xlabel(xlabel)
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both")
                plt.savefig(os.path.join(logs_dir, "loss_components.png"), dpi=300)
                plt.close()

            black_rates = self.stats_history.get("black_rate", [])
            n_rate = min(len(x_samples), len(black_rates))
            if n_rate > 0:
                xs = x_samples[:n_rate]
                plt.figure(figsize=(10, 6))
                plt.suptitle(f"Runtime: {runtime_str}", fontsize=10, y=0.98)
                plt.plot(xs, self.stats_history["black_rate"][:n_rate], label="Black Win Rate", color="black")
                plt.plot(xs, self.stats_history["white_rate"][:n_rate], label="White Win Rate", color="red")
                plt.plot(xs, self.stats_history["draw_rate"][:n_rate], label="Draw Rate", color="gray")
                avg_len = self.stats_history.get("avg_game_len", [])
                if avg_len:
                    n_alen = min(n_rate, len(avg_len))
                    plt.plot(
                        x_samples[:n_alen],
                        np.array(avg_len[:n_alen]) / (self.game.board_size ** 2),
                        label="Avg Game Length Ratio",
                        color="blue",
                        linestyle="--",
                    )
                plt.title(f"Win Rates (Games={self.total_game_count})")
                plt.xlabel(xlabel)
                plt.ylabel("Rate")
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(logs_dir, "win_rates.png"), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")
