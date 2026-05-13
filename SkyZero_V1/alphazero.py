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

    def is_expanded(self):
        return len(self.children) > 0

    def update(self, value):
        self.wdl += value
        self.n += 1

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model.to(args["device"])
        self.model.eval()

    def _inference(self, node):
        encoded_state = self.game.encode_state(node.state, node.to_play)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.args["device"]).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value_logits = self.model(state_tensor)

        policy_logits = policy_logits.flatten().cpu().numpy()
        value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()

        legal_mask = self.game.get_is_legal_actions(node.state, node.to_play)
        policy_logits = np.where(legal_mask, policy_logits, -1e10)
        policy = softmax(policy_logits)

        return policy, value_probs

    def select(self, node):
        best_score = -float("inf")
        best_child = None
        c_puct = self.args.get("c_puct", 1.5)

        for child in node.children:
            q_value = -(child.wdl[0] - child.wdl[2]) / child.n if child.n > 0 else 0
            u_value = c_puct * child.prior * math.sqrt(node.n) / (1 + child.n)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

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
            node = root
            # 1. Select
            while node.is_expanded():
                node = self.select(node)

            # 2. Expand and Evaluate
            if self.game.is_terminal(node.state):
                relative_winner = self.game.get_winner(node.state) * node.to_play
                # value: wdl
                value = np.eye(3, dtype=np.float32)[1 - relative_winner]
            else:
                policy, value = self._inference(node)
                self.expand(node, policy)

            # 3. Backpropagate
            self.backpropagate(node, value)

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
        policy_logits = policy_logits.flatten().cpu().numpy()
        nn_value_probs = F.softmax(value_logits, dim=1).squeeze(0).cpu().numpy()  # WDL

        is_legal = self.game.get_is_legal_actions(state, to_play)
        masked_logits = np.where(is_legal, policy_logits, -np.inf)
        nn_policy = softmax(masked_logits)

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

                        node = child
                        while node.is_expanded():
                            node = self.select(node)

                        if self.game.is_terminal(node.state):
                            relative_winner = self.game.get_winner(node.state) * node.to_play
                            value = np.eye(3, dtype=np.float32)[1 - relative_winner]
                        else:
                            policy, value = self._inference(node)
                            self.expand(node, policy)

                        self.backpropagate(node, value)
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
        num_actions = self.game.board_size ** 2
        q_wdl = np.zeros((num_actions, 3), dtype=np.float64)
        n_values = np.zeros(num_actions, dtype=np.float64)
        for c in root.children:
            if c.n > 0:
                child_wdl = c.wdl / c.n
                q_wdl[c.action_taken] = child_wdl[[2, 1, 0]]  # flip to parent perspective
                n_values[c.action_taken] = c.n

        sum_n = np.sum(n_values)
        visited_mask = n_values > 0
        if sum_n > 0:
            policy_visited = nn_policy * visited_mask
            weighted_q_wdl = np.sum(policy_visited[:, None] * q_wdl, axis=0) / (np.sum(policy_visited) + 1e-12)
            v_mix_wdl = (nn_value_probs + sum_n * weighted_q_wdl) / (1 + sum_n)
        else:
            v_mix_wdl = nn_value_probs.copy()

        completed_q_wdl = np.where(visited_mask[:, None], q_wdl, v_mix_wdl[None, :])
        completed_q_scalar = (completed_q_wdl[:, 0] - completed_q_wdl[:, 2] + 1) / 2  # to [0, 1]
        max_n_root = max((c.n for c in root.children), default=0)
        sigma_q = (c_visit + max_n_root) * c_scale * completed_q_scalar

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
        state = self.game.get_initial_state()
        to_play = 1
        while not self.game.is_terminal(state):

            if self.args.get("algo", "puct") == "puct":

                mcts_policy, _ = self.mcts.search(state, to_play, self.args["num_simulations"])
                t = self.args.get("move_temperature", 1.0)
                if len(memory) > self.args.get("half_life", self.game.board_size):
                    t = 0.01
                
                action = np.random.choice(
                    len(mcts_policy),
                    p=temperature_transform(mcts_policy, t)
                )

            elif self.args.get("algo", "puct") == "gumbel":

                mcts_policy, action, _, _ = self.mcts.gumbel_sequential_halving(state, to_play, self.args["num_simulations"])
            
            memory.append({"state": state, "to_play": to_play, "mcts_policy": mcts_policy})
            
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
        
        winner = self.game.get_winner(state)
        return_memory = []
        for sample in memory:
            relative_winner = winner * sample["to_play"]
            value_target = np.eye(3, dtype=np.float32)[1 - relative_winner]
            return_memory.append({
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "value_target": value_target,
            })
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

        try:
            while True:
                if self.iteration > self.args.get("num_iterations", -1) > 0:
                    print(f"Reached max iterations ({self.args.get('num_iterations', -1)}). Stopping.")
                    break
                if self.total_samples_selfplay_generated >= self.args.get("max_total_samples", -1) > 0:
                    print(f"Reached max total samples ({self.args.get('max_total_samples', -1)}). Stopping.")
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

        states = torch.tensor(np.array([s["encoded_state"] for s in batch]), dtype=torch.float32, device=self.args["device"])
        policy_targets = torch.tensor(np.array([s["policy_target"] for s in batch]), dtype=torch.float32, device=self.args["device"])
        value_targets = torch.tensor(np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=self.args["device"])

        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value_logits = self.model(states)

        policy_loss = -torch.mean(torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = -torch.mean(torch.sum(value_targets * F.log_softmax(value_logits, dim=1), dim=1))
        total_loss = policy_loss + value_loss

        total_loss.backward()
        self.optimizer.step()

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
        print(
            f"Checkpoint loaded. iter={self.iteration} "
            f"games={self.total_game_count} samples={self.total_samples_selfplay_generated}"
        )
        return True

    def _plot_metrics(self):
        try:
            logs_dir = os.path.join(self.args.get("data_dir", "data"), "logs")
            os.makedirs(logs_dir, exist_ok=True)

            x_samples = self.stats_history.get("total_samples", [])
            xlabel = "Total Self-Play Samples"

            total_losses = self.losses_dict.get("total_loss", [])
            n_loss = min(len(x_samples), len(total_losses))
            if n_loss > 0:
                plt.figure(figsize=(10, 6))
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
