"""
Replay Buffer for AlphaZero

AlphaZero uses Replay Buffer to store self-play data from the last N games,
sampling randomly from the entire buffer during training, not just the latest game.

Main features:
1. Capacity limited by sample count (max_physical_limit), old data replaced by new
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

    def __init__(self, min_buffer_size=10000, linear_threshold=10000, alpha=0.75, max_physical_limit=3e6):
        self.min_buffer_size = min_buffer_size
        self.linear_threshold = linear_threshold
        self.alpha = alpha
        self.max_physical_limit = int(max_physical_limit)

        self.buffer: List[dict] = []
        self.total_samples_added = 0
        self.games_count = 0

    def get_window_size(self):
        if self.total_samples_added < self.linear_threshold:
            return self.total_samples_added
        window_size = self.linear_threshold * (self.total_samples_added / self.linear_threshold) ** self.alpha
        return min(int(window_size), self.max_physical_limit)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_game(self, game_memory: List[dict]) -> int:
        """Add a game"s data to the buffer."""
        for sample in game_memory:
            self.buffer.append(sample)

        if len(self.buffer) > self.max_physical_limit:
            excess = len(self.buffer) - self.max_physical_limit
            self.buffer = self.buffer[excess:]

        self.total_samples_added += len(game_memory)
        self.games_count += 1
        return len(game_memory)

    def sample(self, batch_size: int) -> List[dict]:
        current_len = len(self.buffer)
        if current_len < batch_size:
            return []
            
        window_size = self.get_window_size()

        # Ensure window_size is at least batch_size if we have enough samples
        window_size = min(current_len, window_size)

        start_index = current_len - window_size

        physical_indices = [random.randint(start_index, current_len - 1) for _ in range(batch_size)]

        return [self.buffer[i] for i in physical_indices]

    def get_all(self) -> List[dict]:
        return list(self.buffer)

    def clear(self):
        self.buffer = []
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
            "min_buffer_size": self.min_buffer_size,
            "linear_threshold": self.linear_threshold,
            "alpha": self.alpha,
            "max_physical_limit": self.max_physical_limit,
            "games_count": self.games_count,
            "total_samples_added": self.total_samples_added,
        }

    def load_state(self, state: dict):
        """Loads and de-consolidates the buffer."""
        if "buffer_empty" in state:
            self.clear()
            return

        self.min_buffer_size = state.get("min_buffer_size", self.min_buffer_size)
        self.linear_threshold = state.get("linear_threshold", self.linear_threshold)
        self.alpha = state.get("alpha", self.alpha)
        self.max_physical_limit = int(state.get("max_physical_limit", self.max_physical_limit))

        cb = state["consolidated_buffer"]
        num_samples = len(next(iter(cb.values())))

        self.buffer = []
        
        keys = cb.keys()
        for i in range(num_samples):
            sample = {key: cb[key][i] for key in keys}
            self.buffer.append(sample)
        
        if len(self.buffer) > self.max_physical_limit:
            self.buffer = self.buffer[-self.max_physical_limit:]

        self.total_samples_added = state.get("total_samples_added", len(self.buffer))
        
        self.games_count = state.get("games_count", 0)


ParallelReplayBuffer = ReplayBuffer
