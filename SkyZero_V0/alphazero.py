import math
import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
                child = Node(next_state, -node.to_play, prior=prob, parent=node, action_taken=action)
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
        policy = add_dirichlet_noise(policy, self.args.get("dirichlet_alpha", 0.3), self.args.get("dirichlet_epsilon", 0.25), is_root=True)
        self.expand(root, policy)
        self.backpropagate(root, value)

        for _ in range(num_simulations - 1):
            node = root
            # 1. Select
            while node.is_expanded():
                node = self.select(node)
            
            # 2. Expand and Evaluate
            if self.game.is_terminal(node.state):
                value = self.game.get_winner(node.state) * node.to_play
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
        self.game = game
        self.model = model.to(args["device"])
        self.optimizer = optimizer
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.replay_buffer = ReplayBuffer(
            max_buffer_size=args.get("max_buffer_size", 50000),
            min_buffer_size=args.get("min_buffer_size", 1000)
        )
        self.game_count = 0
        self.losses = {"total_loss": [], "policy_loss": [], "value_loss": []}

    def selfplay(self):
        memory = []
        state = self.game.get_initial_state()
        to_play = 1
        while not self.game.is_terminal(state):
            mcts_policy = self.mcts.search(state, to_play, self.args["num_simulations"])
            memory.append({"state": state, "to_play": to_play, "mcts_policy": mcts_policy})
            
            temp = self.args.get("temperature", 1.0)
            if len(memory) > self.args.get("temp_threshold", 10):
                temp = 0.01
            
            action = np.random.choice(len(mcts_policy), p=temperature_transform(mcts_policy, temp))
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

    def _train_step(self):
        if not self.replay_buffer.is_ready():
            return None
        
        batch = self.replay_buffer.sample(self.args["batch_size"])
        if not batch:
            return None
        
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

    def learn(self):
        for i in range(self.args.get("num_iterations", 1000)):
            print(f"Iteration {i+1}")
            for _ in tqdm(range(self.args.get("num_games_per_iter", 10)), desc="Self-play"):
                game_data, winner = self.selfplay()
                self.replay_buffer.add_game(game_data)
                self.game_count += 1
            
            t_loss, p_loss, v_loss = 0, 0, 0
            train_steps = self.args.get("train_steps", 10)
            for _ in tqdm(range(train_steps), desc="Training"):
                res = self._train_step()
                if res:
                    t_loss += res[0]
                    p_loss += res[1]
                    v_loss += res[2]
            
            if t_loss > 0:
                self.losses["total_loss"].append(t_loss / train_steps)
                self.losses["policy_loss"].append(p_loss / train_steps)
                self.losses["value_loss"].append(v_loss / train_steps)
                print(f"Loss: {self.losses['total_loss'][-1]:.4f} (P: {self.losses['policy_loss'][-1]:.4f}, V: {self.losses['value_loss'][-1]:.4f})")
            
            if (i+1) % self.args.get("save_interval", 10) == 0:
                self.save_checkpoint(f"checkpoint_{i+1}.pth")

    def save_checkpoint(self, filename):
        path = os.path.join(self.args.get("data_dir", "data"), filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "game_count": self.game_count,
            "losses": self.losses,
            "replay_buffer": self.replay_buffer.get_state()
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename=None):
        if filename is None:
            import glob
            data_dir = self.args.get("data_dir", "data")
            checkpoints = glob.glob(os.path.join(data_dir, "*.pth"))
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
        self.losses = checkpoint.get("losses", {"total_loss": [], "policy_loss": [], "value_loss": []})
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])
        print("Checkpoint loaded.")
        return True
