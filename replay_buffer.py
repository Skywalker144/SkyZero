"""
Replay Buffer for AlphaZero

AlphaZero uses Replay Buffer to store self-play data from the last N games,
sampling randomly from the entire buffer during training, not just the latest game.

Main features:
1. Capacity limited by sample count (max_buffer_size), old data replaced by new
2. Support for random sampling
3. Support for saving and loading buffer state
4. Ring Buffer implementation for O(1) append and O(1) sample
"""

import random
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    """
    Replay Buffer for storing self-play generated training data.
    
    Optimized for memory and serialization:
    - Stores data in a list of dictionaries (for flexibility)
    - Uses compact data types (int8 for board states)
    - Provides efficient state for torch.save
    """

    def __init__(self, max_buffer_size: int = 500000, min_buffer_size: int = 50000, buffer_size_k: float = 1.0):
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.buffer_size_k = buffer_size_k

        self.buffer: List[dict] = []
        self.position = 0  # Current write position

        # Track statistics
        self.games_count = 0
        self.total_samples_added = 0

    def get_window_size(self) -> int:
        """Calculate dynamic buffer size limit based on samples added."""
        if self.total_samples_added < self.min_buffer_size:
            return len(self.buffer)

        L = self.max_buffer_size - self.min_buffer_size
        if L <= 0 or self.buffer_size_k <= 0:
            return len(self.buffer)

        window_size = self.max_buffer_size - L * np.exp((self.min_buffer_size - self.total_samples_added) / (L / self.buffer_size_k))
        return min(int(window_size), len(self.buffer))

    def __len__(self) -> int:
        return len(self.buffer)

    def add_game(self, game_memory: List[dict]) -> int:
        """Add a game"s data to the buffer."""
        for sample in game_memory:
            # Ensure encoded_state is int8 to save 75% memory
            if "encoded_state" in sample and sample["encoded_state"].dtype != np.int8:
                sample["encoded_state"] = sample["encoded_state"].astype(np.int8)
            if "final_state" in sample and sample["final_state"].dtype != np.int8:
                sample["final_state"] = sample["final_state"].astype(np.int8)

            if len(self.buffer) < self.max_buffer_size:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.max_buffer_size

        self.games_count += 1
        self.total_samples_added += len(game_memory)
        return len(game_memory)

    def sample(self, batch_size: int) -> List[dict]:
        current_len = len(self.buffer)
        if current_len == 0:
            return []
            
        window_size = self.get_window_size()
        # Ensure window_size is at least batch_size if we have enough samples
        window_size = min(current_len, window_size)

        # Sample from the most recent "window_size" samples
        logical_indices = random.sample(range(window_size), min(batch_size, window_size))

        # Map to physical indices in the ring buffer
        # The newest item is at (self.position - 1) % current_len
        physical_indices = [(self.position - 1 - i) % current_len for i in logical_indices]

        return [self.buffer[i] for i in physical_indices]

    def get_all(self) -> List[dict]:
        return list(self.buffer)

    def clear(self):
        self.buffer = []
        self.position = 0
        self.games_count = 0
        self.total_samples_added = 0

    def get_state(self) -> dict:
        """
        Consolidates the buffer into large numpy arrays for efficient storage.
        This avoids the overhead of pickling 100k+ dictionaries.
        """
        if not self.buffer:
            return {"buffer_empty": True}

        # Collect keys from the first sample
        keys = self.buffer[0].keys()
        consolidated_buffer = {}
        
        for key in keys:
            # Consolidate each field into a single numpy array
            # This is much more efficient for torch.save/pickle
            try:
                consolidated_buffer[key] = np.array([sample[key] for sample in self.buffer])
            except Exception as e:
                # Fallback for non-array types
                consolidated_buffer[key] = [sample[key] for sample in self.buffer]

        return {
            "consolidated_buffer": consolidated_buffer,
            "max_buffer_size": self.max_buffer_size,
            "min_buffer_size": self.min_buffer_size,
            "buffer_size_k": self.buffer_size_k,
            "position": self.position,
            "games_count": self.games_count,
            "total_samples_added": self.total_samples_added,
        }

    def load_state(self, state: dict):
        """Loads and de-consolidates the buffer."""
        if "buffer_empty" in state:
            self.clear()
            return

        cb = state["consolidated_buffer"]
        num_samples = len(next(iter(cb.values())))


        # ##############################################################
        # 新增Soft Resign导致的Sample Weight列表
        if "sample_weight" not in cb:
            cb["sample_weight"] = np.ones(num_samples, dtype=np.float32)
        # ##############################################################


        self.buffer = []
        
        # Reconstruct list of dictionaries
        # This is slow but only happens once at startup
        keys = cb.keys()
        for i in range(num_samples):
            sample = {key: cb[key][i] for key in keys}
            self.buffer.append(sample)
        
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]

        if len(self.buffer) < self.max_buffer_size:
            self.position = len(self.buffer)
        else:
            self.position = state.get("position", 0) % self.max_buffer_size

        self.total_samples_added = state.get("total_samples_added", len(self.buffer))

        saved_total = state.get("total_samples_added", len(self.buffer))
        old_max = state.get("max_buffer_size")

        if old_max and self.max_buffer_size > old_max:
            M = self.max_buffer_size
            m = self.min_buffer_size
            k = self.buffer_size_k

            if M > old_max and old_max > m:
                target_w = old_max
                T_prime = m - (M - m) / k * np.log((M - target_w) / (M - m))
                self.total_samples_added = int(min(saved_total, T_prime))
            else:
                self.total_samples_added = saved_total
        else:
            self.total_samples_added = saved_total
        
        self.games_count = state.get("games_count", 0)


# Backward compatibility alias
ParallelReplayBuffer = ReplayBuffer
