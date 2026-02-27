import sys
import os
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add the project root to sys.path so envs, nets and other modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gomoku import Gomoku
from nets import ResNet
from alphazero import AlphaZero
from gomoku.gomoku_train import train_args as gomoku_train_args
from utils import random_augment_batch

class OfflineDistiller:
    def __init__(self, game, model, optimizer, args):
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = args["device"]
        
        # Load dataset
        dataset_path = os.path.join(args["data_dir"], "distillation_dataset.pkl")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate mode first.")
            
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} samples from {dataset_path}")
        
    def _train_batch(self, batch):
        batch = random_augment_batch(batch, self.game.board_size)
        batch_size = len(batch)
        
        final_states = torch.tensor(np.array([d["final_state"] for d in batch]), device=self.device, dtype=torch.float32)
        to_plays = torch.tensor(np.array([d["to_play"] for d in batch]), device=self.device, dtype=torch.float32)
        encoded_states = torch.tensor(np.array([d["encoded_state"] for d in batch]), device=self.device, dtype=torch.float32)
        policy_targets = torch.tensor(np.array([d["policy_target"] for d in batch]), device=self.device, dtype=torch.float32)
        opponent_policy_targets = torch.tensor(np.array([d["opponent_policy_target"] for d in batch]), device=self.device, dtype=torch.float32)
        outcomes = torch.tensor(np.array([d["outcome"] for d in batch]), device=self.device, dtype=torch.float32)
        sample_weights = torch.tensor(np.array([d["sample_weight"] for d in batch]), device=self.device, dtype=torch.float32)
        
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
        final_states_view = final_states.view(batch_size, -1)
        ownership_loss = F.mse_loss(ownership_pred, final_states_view, reduction="none").mean(dim=-1)
        ownership_loss = (ownership_loss * sample_weights).mean()

        # Value Loss
        value_targets = (1 - outcomes).long()
        value_loss = F.cross_entropy(nn_output["value_logits"], value_targets, reduction="none")
        value_loss = (value_loss * sample_weights).mean()

        loss = (
            policy_loss +
            soft_policy_loss +
            opponent_policy_loss +
            ownership_loss +
            value_loss
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "soft_policy_loss": soft_policy_loss.item(),
            "opponent_policy_loss": opponent_policy_loss.item(),
            "ownership_loss": ownership_loss.item()
        }
        
    def train(self):
        batch_size = self.args["batch_size"]
        epochs = self.args["epochs"]
        
        for epoch in range(epochs):
            np.random.shuffle(self.data)
            epoch_losses = []
            
            pbar = tqdm(range(0, len(self.data), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
            for i in pbar:
                batch = self.data[i:i+batch_size]
                if len(batch) < batch_size:
                    continue # Skip incomplete last batch
                    
                loss_dict = self._train_batch(batch)
                epoch_losses.append(loss_dict["total_loss"])
                
                if (i // batch_size) % self.args["log_interval"] == 0:
                    pbar.set_postfix({"loss": f"{np.mean(epoch_losses[-10:]):.4f}"})
                    
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {np.mean(epoch_losses):.4f}")
            
            # Save student model checkpoint
            os.makedirs(self.args["data_dir"], exist_ok=True)
            save_path = os.path.join(self.args["data_dir"], f"student_model_epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved student model checkpoint to {save_path}")

class DataGenerater:
    def __init__(self, game, model, train_args, dist_args):
        self.game = game
        self.model = model
        self.train_args = train_args
        self.dist_args = dist_args
        
        # Initialize AlphaZero with Gomoku training parameters
        self.alphazero = AlphaZero(self.game, self.model, None, self.train_args)
        
        if self.alphazero.load_checkpoint():
            print("Successfully loaded teacher model checkpoint from", self.train_args["data_dir"])
        else:
            print("No checkpoint found in", self.train_args["data_dir"], "- using random weights for teacher model.")
        
        self.model.eval()

    def generate_and_save(self, num_samples):
        save_dir = self.dist_args["data_dir"]
        os.makedirs(save_dir, exist_ok=True)
        
        all_data = []
        pbar = tqdm(total=num_samples, desc="Generating Data")
        
        with torch.no_grad():
            while len(all_data) < num_samples:
                # selfplay() returns (return_memory, winner, memory_len, final_state)
                memory, _, length, _ = self.alphazero.selfplay()
                
                current_len = len(all_data)
                needed = num_samples - current_len
                
                # Check how much to add to not exceed num_samples
                if length > needed:
                    all_data.extend(memory[:needed])
                    pbar.update(needed)
                else:
                    all_data.extend(memory)
                    pbar.update(length)
                
        pbar.close()
        
        save_path = os.path.join(save_dir, "distillation_dataset.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"Successfully generated {num_samples} samples and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["generate", "train"], default="generate", help="Mode to run: generate data or train distillation model")
    parser.add_argument("--num_samples", type=int, default=30000, help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    
    cmd_args = parser.parse_args()
    
    # Distillation specific arguments
    dist_args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "Distillation/dataset",
        "batch_size": cmd_args.batch_size,
        "epochs": cmd_args.epochs,
        "num_workers": 0,
        "kl_loss_weight": 0.5,
        "log_interval": 10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    
    # Initialize Gomoku game
    game = Gomoku(board_size=gomoku_train_args["board_size"], history_step=gomoku_train_args["history_step"])
    
    if cmd_args.mode == "generate":
        # Instantiate teacher model based on the same config used for training
        teacher_model = ResNet(
            game, 
            num_blocks=gomoku_train_args["num_blocks"], 
            num_channels=gomoku_train_args["num_channels"]
        ).to(gomoku_train_args["device"])
        
        generator = DataGenerater(game, teacher_model, gomoku_train_args, dist_args)
        generator.generate_and_save(cmd_args.num_samples)
        
    elif cmd_args.mode == "train":
        # Use a smaller student model for distillation
        student_model = ResNet(game, num_blocks=2, num_channels=32).to(dist_args["device"])
        optimizer = optim.Adam(student_model.parameters(), lr=dist_args["lr"], weight_decay=dist_args["weight_decay"])
        distiller = OfflineDistiller(game, student_model, optimizer, dist_args)
        distiller.train()