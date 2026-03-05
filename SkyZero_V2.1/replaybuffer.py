import random
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
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
        self.buffer.extend(game_memory)
        if len(self.buffer) > self.max_physical_limit:
            self.buffer = self.buffer[ - self.max_physical_limit:]

        self.total_samples_added += len(game_memory)
        self.games_count += 1
        return len(game_memory)

    def sample(self, batch_size: int) -> List[dict]:
        current_len = len(self.buffer)
        if current_len < batch_size:
            return []
            
        window_size = self.get_window_size()

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

        keys = set()
        for sample in self.buffer[:100]:
            keys.update(sample.keys())
        
        consolidated_buffer = {}
        
        for key in keys:
            try:
                data_list = [sample.get(key) for sample in self.buffer]
                if any(x is None for x in data_list):
                    consolidated_buffer[key] = data_list
                else:
                    consolidated_buffer[key] = np.array(data_list)
            except Exception as e:
                consolidated_buffer[key] = [sample.get(key) for sample in self.buffer]

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
        
        keys = list(cb.keys())
        for i in range(num_samples):
            sample = {key: cb[key][i] for key in keys}
            self.buffer.append(sample)
        
        if len(self.buffer) > self.max_physical_limit:
            self.buffer = self.buffer[-self.max_physical_limit:]

        self.total_samples_added = state.get("total_samples_added", len(self.buffer))
        
        self.games_count = state.get("games_count", 0)

