import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from alphazero_play import TreeReuseAlphaZero
from replay_buffer import ReplayBuffer


def _random_augment_sample(sample: Dict, board_size: int) -> Dict:
    transform_type = np.random.randint(0, 8)
    k = transform_type % 4
    do_flip = transform_type >= 4

    encoded_state = sample["encoded_state"]
    policy_target = sample["policy_target"]
    soft_policy_target = sample["soft_policy_target"]
    opponent_policy_target = sample["opponent_policy_target"]
    ownership_target = sample["ownership_target"]

    aug_state = np.rot90(encoded_state, k=k, axes=(1, 2))
    p_2d = policy_target.reshape(board_size, board_size)
    sp_2d = soft_policy_target.reshape(board_size, board_size)
    opp_p_2d = opponent_policy_target.reshape(board_size, board_size)
    own_2d = ownership_target.reshape(board_size, board_size)

    aug_p_2d = np.rot90(p_2d, k=k)
    aug_sp_2d = np.rot90(sp_2d, k=k)
    aug_opp_p_2d = np.rot90(opp_p_2d, k=k)
    aug_own_2d = np.rot90(own_2d, k=k)

    if do_flip:
        aug_state = np.flip(aug_state, axis=2)
        aug_p_2d = np.flip(aug_p_2d, axis=1)
        aug_sp_2d = np.flip(aug_sp_2d, axis=1)
        aug_opp_p_2d = np.flip(aug_opp_p_2d, axis=1)
        aug_own_2d = np.flip(aug_own_2d, axis=1)

    new_sample = sample.copy()
    new_sample.update({
        "encoded_state": aug_state.copy(),
        "policy_target": aug_p_2d.flatten().copy(),
        "soft_policy_target": aug_sp_2d.flatten().copy(),
        "opponent_policy_target": aug_opp_p_2d.flatten().copy(),
        "ownership_target": aug_own_2d.copy(),
    })
    return new_sample


def _random_augment_batch(batch: List[Dict], board_size: int) -> List[Dict]:
    return [_random_augment_sample(sample, board_size) for sample in batch]


class DistillationDataGenerator:
    def __init__(self, game, teacher_model, teacher_optimizer, args: Dict):
        self.game = game
        self.args = args
        self.teacher = TreeReuseAlphaZero(game, teacher_model, teacher_optimizer, args)

    def load_teacher(self, checkpoint_path: Optional[str] = None) -> bool:
        return self.teacher.load_checkpoint(checkpoint_path)

    @torch.inference_mode()
    def collect_game(self) -> Tuple[List[Dict], Optional[int], int]:
        memory: List[Dict] = []
        to_play = 1
        state = self.game.get_initial_state()
        root = None

        while not self.game.is_terminal(state):
            action, info, root = self.teacher.play(state, to_play, root=root, show_progress_bar=False)

            memory.append({
                "encoded_state": self.game.encode_state(state, to_play),
                "policy_target": info["mcts_policy"].reshape(-1).astype(np.float32),
                "soft_policy_target": info["nn_policy"].reshape(-1).astype(np.float32),
                "opponent_policy_target": info["opponent_policy"].reshape(-1).astype(np.float32),
                "ownership_target": info["ownership"].astype(np.float32),
                "value_probs_target": info["value_probs"].astype(np.float32),
                "to_play": to_play,
                "sample_weight": 1.0,
            })

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

        winner = self.game.get_winner(state)
        return memory, winner, len(memory)

    def generate_games(self, num_games: int, replay_buffer: ReplayBuffer) -> Dict:
        recent_game_lengths = deque(maxlen=100)
        win_counts = deque(maxlen=100)
        total_samples = 0

        for _ in range(num_games):
            memory, winner, game_len = self.collect_game()
            replay_buffer.add_game(memory)
            total_samples += game_len
            recent_game_lengths.append(game_len)
            win_counts.append(1 if winner == 1 else 0)

        avg_game_len = float(np.mean(recent_game_lengths)) if recent_game_lengths else 0.0
        win_rate = float(np.mean(win_counts)) if win_counts else 0.0
        return {
            "games": num_games,
            "samples": total_samples,
            "avg_game_len": avg_game_len,
            "teacher_first_player_win_rate": win_rate,
            "buffer_size": len(replay_buffer),
        }

    def generate_samples(
        self,
        target_samples: int,
        replay_buffer: ReplayBuffer,
        max_games: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict:
        total_samples = 0
        games = 0
        start_time = time.time()

        progress = None
        if show_progress:
            progress = tqdm(total=target_samples, desc="Collect samples", unit="sample")

        try:
            while total_samples < target_samples:
                memory, _, game_len = self.collect_game()
                replay_buffer.add_game(memory)
                total_samples += game_len
                games += 1
                if progress is not None:
                    progress.update(min(game_len, target_samples - (total_samples - game_len)))
                if max_games is not None and games >= max_games:
                    break
        finally:
            if progress is not None:
                progress.close()

        elapsed = max(1e-6, time.time() - start_time)
        return {
            "games": games,
            "samples": total_samples,
            "samples_per_sec": total_samples / elapsed,
            "buffer_size": len(replay_buffer),
        }


class DistillationTrainer:
    def __init__(self, game, student_model, optimizer, args: Dict, replay_buffer: Optional[ReplayBuffer] = None):
        self.game = game
        self.args = args
        self.model = student_model.to(args["device"])
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer or ReplayBuffer(
            min_buffer_size=args.get("min_buffer_size", 1000),
            max_buffer_size=args.get("max_buffer_size", 10000),
            buffer_size_k=args.get("buffer_size_k", 0.1),
        )

        self.losses_dict = {
            "total_loss": [],
            "policy_loss": [],
            "soft_policy_loss": [],
            "opponent_policy_loss": [],
            "ownership_loss": [],
            "value_loss": [],
        }

    def _train_batch(self, batch: List[Dict]) -> Dict:
        batch = _random_augment_batch(batch, self.game.board_size)
        batch_size = len(batch)

        encoded_states = torch.tensor(
            np.array([d["encoded_state"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        policy_targets = torch.tensor(
            np.array([d["policy_target"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        soft_policy_targets = torch.tensor(
            np.array([d["soft_policy_target"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        opponent_policy_targets = torch.tensor(
            np.array([d["opponent_policy_target"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        ownership_targets = torch.tensor(
            np.array([d["ownership_target"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        value_probs_targets = torch.tensor(
            np.array([d["value_probs_target"] for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )
        sample_weights = torch.tensor(
            np.array([d.get("sample_weight", 1.0) for d in batch]),
            device=self.args["device"],
            dtype=torch.float32,
        )

        self.model.train()
        nn_output = self.model(encoded_states)

        policy_logits = nn_output["policy_logits"].view(batch_size, -1)
        soft_policy_logits = nn_output["soft_policy_logits"].view(batch_size, -1)
        opponent_policy_logits = nn_output["opponent_policy_logits"].view(batch_size, -1)

        def soft_ce_loss(logits, targets, weights):
            loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)
            return (loss * weights).mean()

        policy_loss = soft_ce_loss(policy_logits, policy_targets, sample_weights)
        soft_policy_loss = soft_ce_loss(soft_policy_logits, soft_policy_targets, sample_weights)
        opponent_policy_loss = soft_ce_loss(opponent_policy_logits, opponent_policy_targets, sample_weights)

        ownership_pred = torch.tanh(nn_output["ownership"]).view(batch_size, -1)
        ownership_targets = ownership_targets.view(batch_size, -1)
        ownership_loss = F.mse_loss(ownership_pred, ownership_targets, reduction="none").mean(dim=-1)
        ownership_loss = (ownership_loss * sample_weights).mean()

        value_loss = soft_ce_loss(nn_output["value_logits"], value_probs_targets, sample_weights)

        loss = (
            self.args.get("policy_loss_weight", 1.0) * policy_loss
            + self.args.get("soft_policy_loss_weight", 8.0) * soft_policy_loss
            + self.args.get("opponent_policy_loss_weight", 0.15) * opponent_policy_loss
            + self.args.get("ownership_loss_weight", 1.5) / (self.game.board_size ** 2) * ownership_loss
            + self.args.get("value_loss_weight", 1.5) * value_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.args.get("max_grad_norm", 1.0),
        )
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "soft_policy_loss": soft_policy_loss.item(),
            "opponent_policy_loss": opponent_policy_loss.item(),
            "ownership_loss": ownership_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_steps(self, steps: int, batch_size: int) -> Dict:
        if len(self.replay_buffer) == 0:
            return {"skipped": True, "reason": "empty_buffer"}

        batch_loss_dict = {key: [] for key in self.losses_dict.keys()}
        for _ in range(steps):
            batch = self.replay_buffer.sample(batch_size)
            if not batch:
                break
            loss_dict = self._train_batch(batch)
            for key in batch_loss_dict:
                batch_loss_dict[key].append(loss_dict[key])

        for key in self.losses_dict:
            if batch_loss_dict[key]:
                self.losses_dict[key].append(float(np.mean(batch_loss_dict[key])))

        return {key: float(np.mean(vals)) for key, vals in batch_loss_dict.items() if vals}


def save_replay_buffer(replay_buffer: ReplayBuffer, filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = replay_buffer.get_state()
    torch.save(state, filepath)
    return filepath


def load_replay_buffer(replay_buffer: ReplayBuffer, filepath: str) -> bool:
    if not os.path.exists(filepath):
        return False
    state = torch.load(filepath, weights_only=False)
    replay_buffer.load_state(state)
    return True


def save_distilled_model(model: torch.nn.Module, args: Dict, filepath: Optional[str] = None) -> str:
    from datetime import datetime

    if filepath is None:
        output_dir = args.get("distill_dir", os.path.join("distillation", "checkpoints"))
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.basename(args.get("file_name", "model"))
        filepath = os.path.join(output_dir, f"{base_name}_distilled_{timestamp}.pth")

    torch.save({"model_state_dict": model.state_dict()}, filepath)

    file_size = os.path.getsize(filepath)
    size_str = (
        f"{file_size / 1024 / 1024:.1f}MB"
        if file_size > 1024 * 1024
        else f"{file_size / 1024:.1f}KB"
    )
    print(f"Distilled model saved to {filepath} ({size_str})")
    return filepath


def run_tictactoe_distillation():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    import torch.optim as optim
    from envs.tictactoe import TicTacToe
    from nets import ResNet

    args = {
        "mode": "eval",
        "history_step": 2,
        "num_blocks": 1,
        "num_channels": 16,
        "lr": 1e-4,
        "weight_decay": 3e-5,
        "full_search_num_simulations": 100,
        "enable_symmetry_inference_for_root": True,
        "enable_symmetry_inference_for_child": True,
        "c_puct": 1.1,
        "root_temperature_init": 1,
        "root_temperature_final": 0.9,
        "move_temperature_init": 0.5,
        "move_temperature_final": 0.1,
        "enable_forced_playout": False,
        "file_name": "tictactoe",
        "data_dir": "data/tictactoe",
        "distill_dir": os.path.join("distillation", "checkpoints"),
        "device": "cuda",
        "min_buffer_size": 1000,
        "max_buffer_size": 20000,
        "buffer_size_k": 0.1,
        "target_samples": 2000,
    }

    game = TicTacToe(history_step=args["history_step"])

    teacher_model = ResNet(
        game,
        num_blocks=args["num_blocks"],
        num_channels=args["num_channels"],
    ).to(args["device"])
    teacher_optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )
    generator = DistillationDataGenerator(game, teacher_model, teacher_optimizer, args)
    generator.load_teacher()

    student_model = ResNet(game, num_blocks=1, num_channels=4).to(args["device"])
    student_optimizer = optim.AdamW(student_model.parameters(), lr=5e-4, weight_decay=1e-5)
    trainer = DistillationTrainer(game, student_model, student_optimizer, args)

    dataset_path = os.path.join("distillation", "dataset", "tictactoe_distill_buffer.pth")
    loaded = load_replay_buffer(trainer.replay_buffer, dataset_path)
    if loaded:
        print(f"Loaded distillation buffer: {dataset_path} ({len(trainer.replay_buffer)} samples)")
    else:
        print("No existing distillation buffer found, will create new data.")

    stats = generator.generate_samples(args["target_samples"], trainer.replay_buffer)
    print("Collect:", stats)

    save_replay_buffer(trainer.replay_buffer, dataset_path)
    print(f"Saved distillation buffer: {dataset_path}")

    loss = trainer.train_steps(steps=1000, batch_size=64)
    print("Train:", loss)

    save_distilled_model(student_model, args)


if __name__ == "__main__":
    run_tictactoe_distillation()
