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
        self.args = args
        self.model = model.to(args["device"])
        self.model.eval()

    def _inference(self, node):
        encoded_state = self.game.encode_state(node.state, node.to_play)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.args["device"]).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        policy_logits = policy_logits.flatten().cpu().numpy()
        value = value.item()

        legal_mask = self.game.get_is_legal_actions(node.state, node.to_play)
        policy_logits = np.where(legal_mask, policy_logits, -1e10)
        policy = softmax(policy_logits)
        
        return policy, value

    def select(self, node):
        best_score = -float("inf")
        best_child = None
        c_puct = self.args.get("c_puct", 1.5)

        for child in node.children:
            q_value = -child.v / child.n if child.n > 0 else 0
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
            value = -value
            node = node.parent

    @torch.inference_mode()
    def search(self, state, to_play, num_simulations):
        root = Node(state, to_play)
        
        # Initial expand for root
        policy, value = self._inference(root)
        policy = add_dirichlet_noise(
            policy,
            self.args.get("dirichlet_alpha", 10 / self.game.board_size ** 2),
            self.args.get("dirichlet_epsilon", 0.25)
        )
        self.expand(root, policy)
        self.backpropagate(root, value)

        for _ in range(num_simulations - 1):
            node = root
            # 1. Select
            while node.is_expanded():
                node = self.select(node)
            
            # 2. Expand and Evaluate
            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play  # value is the relative outcome
            else:
                policy, value = self._inference(node)
                self.expand(node, policy)
            
            # 3. Backpropagate
            self.backpropagate(node, value)

        mcts_policy = np.zeros(self.game.board_size**2)
        for child in root.children:
            mcts_policy[child.action_taken] = child.n
        mcts_policy /= np.sum(mcts_policy)
        return mcts_policy

class AlphaZero:
    def __init__(self, game, model, optimizer, args):

        B = args["batch_size"]
        T = args["train_steps_per_iteration"]
        min_buffer_size = args.get("min_buffer_size", 250000)
        if min_buffer_size < B * T:
            raise ValueError(
                f"min_buffer_size must be >= batch_size * train_steps_per_iteration "
                f"({B} * {T} = {B * T}), got {min_buffer_size}"
            )

        self.game = game
        self.model = model.to(args["device"])
        self.optimizer = optimizer
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.replay_buffer = ReplayBuffer(
            max_buffer_size=args.get("max_buffer_size", 1e7),
            min_buffer_size=min_buffer_size,
            window_exponent=args.get("window_exponent", 0.65),
            window_expand_per_row=args.get("window_expand_per_row", 0.4),
        )
        self.game_count = 0
        self.losses_dict = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": []
        }
        # Recent per-game sample counts, used to back-compute the next
        # selfplay batch size from target_ReplayRatio.
        stats_window = args.get("len_statistics_queue_size", 300)
        self.recent_sample_lengths = deque(maxlen=args.get("recent_window", 100))
        self.recent_game_lengths = deque(maxlen=stats_window)
        self.black_win_counts = deque(maxlen=stats_window)
        self.white_win_counts = deque(maxlen=stats_window)
        self.winrate_history = []
        self.avg_game_len_history = []
        self._winrate_sample_every = args.get("winrate_sample_every", 10)

    def selfplay(self):
        memory = []
        state = self.game.get_initial_state()
        to_play = 1
        while not self.game.is_terminal(state):
            mcts_policy = self.mcts.search(state, to_play, self.args["num_simulations"])
            memory.append({"state": state, "to_play": to_play, "mcts_policy": mcts_policy})
            
            t = self.args.get("move_temperature", 1.0)
            if len(memory) > self.args.get("half_life", self.game.board_size):
                t = 0.01
            
            action = np.random.choice(
                len(mcts_policy),
                p=temperature_transform(mcts_policy, t)
            )
            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play
            
        winner = self.game.get_winner(state)
        game_data = []
        for sample in memory:
            game_data.append({
                "encoded_state": self.game.encode_state(sample["state"], sample["to_play"]),
                "policy_target": sample["mcts_policy"],
                "value_target": winner * sample["to_play"]
            })
        return game_data, winner

    def _train_step(self, batch):
        batch = random_augment_batch(batch, self.game.board_size)

        states = torch.tensor(np.array([s["encoded_state"] for s in batch]), dtype=torch.float32, device=self.args["device"])
        policy_targets = torch.tensor(np.array([s["policy_target"] for s in batch]), dtype=torch.float32, device=self.args["device"])
        value_targets = torch.tensor(np.array([s["value_target"] for s in batch]), dtype=torch.float32, device=self.args["device"]).unsqueeze(1)

        self.model.train()
        self.optimizer.zero_grad()
        policy_logits, value = self.model(states)

        policy_loss = -torch.mean(torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(value, value_targets)
        total_loss = policy_loss + value_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def _record_game(self, winner, game_len):
        self.recent_game_lengths.append(game_len)
        self.black_win_counts.append(1 if winner == 1 else 0)
        self.white_win_counts.append(1 if winner == -1 else 0)
        if self.game_count % self._winrate_sample_every == 0:
            total = len(self.black_win_counts)
            b = float(np.sum(self.black_win_counts)) / total
            w = float(np.sum(self.white_win_counts)) / total
            d = 1.0 - b - w
            self.winrate_history.append((self.game_count, b, w, d))
            self.avg_game_len_history.append(float(np.mean(self.recent_game_lengths)))

    def _compute_num_next_games(self):
        # samples_trained = batch_size * train_steps_per_iteration
        # target new samples = samples_trained / target_ReplayRatio
        # games = target_new_samples / avg_sample_len
        if self.recent_sample_lengths:
            avg_len = float(np.mean(self.recent_sample_lengths))
        else:
            avg_len = float(self.args.get("default_avg_sample_len", 30))
        target = self.args["batch_size"] * self.args["train_steps_per_iteration"] / avg_len / self.args["target_ReplayRatio"]
        return max(1, math.ceil(target))

    def _print_selfplay(self, it, games_done, total_games, t0, lens, winners):
        dt = max(1e-9, time.time() - t0)
        total_rows = sum(lens) if lens else 0
        sps = total_rows / dt
        gw = max(2, len(str(total_games)))
        head = f"[SelfPlay] Iter={it} Games={games_done:0{gw}d}/{total_games} Sps={sps:.1f}"
        if lens:
            arr = np.asarray(lens, dtype=np.float64)
            n = len(winners)
            b = sum(1 for w in winners if w == 1) / n * 100
            w = sum(1 for w_ in winners if w_ == -1) / n * 100
            d = 100 - b - w
            head += (f" GameLen:Avg={arr.mean():.1f} Min={int(arr.min())} "
                     f"Max={int(arr.max())} Std={int(round(arr.std()))} "
                     f"BDW={int(round(b)):02d}/{int(round(d)):02d}/{int(round(w)):02d}")
        else:
            head += " GameLen=N/A BDW=N/A"
        print(head, flush=True)

    def learn(self):
        num_iterations = self.args.get("num_iterations", 1000)
        save_interval = self.args.get("save_interval", 10)
        train_steps_per_iteration = self.args["train_steps_per_iteration"]
        unlimited = num_iterations <= 0
        total_label = "inf" if unlimited else str(num_iterations)

        num_next = self._compute_num_next_games()
        i = 0
        it = 0
        try:
            while unlimited or i < num_iterations:
                it = i + 1
                print(
                    f"\n=== Iter {it}/{total_label} | "
                    f"collect {num_next} games (RR target={self.args['target_ReplayRatio']}) ===",
                    flush=True
                )
                print(
                    f"[Stats] total_games={self.game_count} "
                    f"total_samples={self.replay_buffer.total_samples_added}",
                    flush=True
                )

                # Self-play phase
                iter_lens, iter_winners = [], []
                sp_t0 = time.time()
                report_every = max(1, num_next // 5)
                for g in range(num_next):
                    game_data, winner = self.selfplay()
                    self.replay_buffer.add_game(game_data)
                    self.recent_sample_lengths.append(len(game_data))
                    self.game_count += 1
                    self._record_game(winner, len(game_data))
                    iter_lens.append(len(game_data))
                    iter_winners.append(winner)
                    if (g + 1) % report_every == 0 or (g + 1) == num_next:
                        self._print_selfplay(
                            it, g + 1, num_next, sp_t0,
                            iter_lens, iter_winners
                        )

                # Training phase
                if self.replay_buffer.is_ready():
                    self._train_iteration(it, train_steps_per_iteration)
                else:
                    print(
                        f"[Train] iter={it} skipped (buffer not ready: "
                        f"{self.replay_buffer.total_samples_added} < "
                        f"{self.replay_buffer.min_buffer_size})", flush=True
                    )

                num_next = self._compute_num_next_games()

                self.plot_metrics()

                if it % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{it}.pth")

                i += 1
        except KeyboardInterrupt:
            print(f"\nStopping... saving checkpoint at iter={it}", flush=True)
            if it > 0:
                self.save_checkpoint(f"checkpoint_{it}_interrupt.pth")

    def _train_iteration(self, it, total_steps):
        B = self.args["batch_size"]
        pool = self.replay_buffer.sample(B * total_steps)

        losses = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        t0 = time.time()
        pbar = tqdm(range(total_steps), desc=f"Train Iter={it}")
        for i in pbar:
            batch = pool[i * B : (i + 1) * B]
            t, p, v = self._train_step(batch)
            losses["total_loss"] += t
            losses["policy_loss"] += p
            losses["value_loss"] += v
            pbar.set_postfix(loss=f"{t:.2f}", ploss=f"{p:.2f}", vloss=f"{v:.2f}")
        pbar.close()

        dt = time.time() - t0

        for key in self.losses_dict:
            self.losses_dict[key].append(losses[key] / total_steps)

        print(
            f"[Train] iter={it} steps={total_steps} "
            f"t={dt:.1f}s | total={losses['total_loss']/total_steps:.4f} "
            f"policy={losses['policy_loss']/total_steps:.4f} value={losses['value_loss']/total_steps:.4f}",
            flush=True
        )

    def save_checkpoint(self, filename):
        ckpt_dir = os.path.join(self.args.get("data_dir", "data"), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "game_count": self.game_count,
            "losses_dict": self.losses_dict,
            "winrate_history": self.winrate_history,
            "avg_game_len_history": self.avg_game_len_history,
            "recent_game_lengths": list(self.recent_game_lengths),
            "black_win_counts": list(self.black_win_counts),
            "white_win_counts": list(self.white_win_counts),
            "replay_buffer": self.replay_buffer.get_state()
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
        self.game_count = checkpoint.get("game_count", 0)
        self.losses_dict = checkpoint.get("losses_dict", {"total_loss": [], "policy_loss": [], "value_loss": []})
        self.winrate_history = checkpoint.get("winrate_history", [])
        self.avg_game_len_history = checkpoint.get("avg_game_len_history", [])
        for key, deq in (
            ("recent_game_lengths", self.recent_game_lengths),
            ("black_win_counts", self.black_win_counts),
            ("white_win_counts", self.white_win_counts),
        ):
            deq.clear()
            for v in checkpoint.get(key, []):
                deq.append(v)
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])
        print("Checkpoint loaded.")
        return True

    def plot_metrics(self):
        try:
            logs_dir = os.path.join(self.args.get("data_dir", "data"), "logs")
            os.makedirs(logs_dir, exist_ok=True)

            if self.losses_dict.get("total_loss"):
                plt.figure(figsize=(10, 6))
                plt.plot(self.losses_dict["total_loss"], label="Total Loss")
                plt.title(f"Total Training Loss (Game {self.game_count})")
                plt.xlabel("Training Iteration")
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
                    plt.plot(vals, label=key.replace("_", " ").title())
                plt.title(f"Loss Components (Game {self.game_count})")
                plt.xlabel("Training Iteration")
                plt.ylabel("Loss")
                plt.yscale("log")
                plt.legend()
                plt.grid(True, which="both")
                plt.savefig(os.path.join(logs_dir, "loss_components.png"), dpi=300)
                plt.close()

            if self.winrate_history:
                games, b_rates, w_rates, d_rates = zip(*self.winrate_history)
                plt.figure(figsize=(10, 6))
                plt.plot(games, b_rates, label="Black Win Rate", color="black")
                plt.plot(games, w_rates, label="White Win Rate", color="red")
                plt.plot(games, d_rates, label="Draw Rate", color="gray")
                if self.avg_game_len_history and len(self.avg_game_len_history) == len(games):
                    plt.plot(
                        games,
                        np.array(self.avg_game_len_history) / (self.game.board_size ** 2),
                        label="Avg Game Length Ratio",
                        color="blue",
                        linestyle="--",
                    )
                plt.title(f"Win Rates (Last {len(b_rates)} Samples)")
                plt.xlabel("Game Count")
                plt.ylabel("Rate")
                plt.ylim(0, 1)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(logs_dir, "win_rates.png"), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Plotting failed: {e}")
