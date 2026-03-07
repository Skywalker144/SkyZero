import random
from typing import List, Tuple, Dict
import numpy as np


class ReplayBuffer:
    def __init__(self, board_size: int, num_planes: int, min_buffer_size=10000, linear_threshold=10000, alpha=0.75, max_buffer_size=3e6):
        self.min_buffer_size = min_buffer_size
        self.linear_threshold = linear_threshold
        self.alpha = alpha
        self.max_buffer_size = int(max_buffer_size)

        self.board_size = board_size
        self.num_planes = num_planes

        self.ptr = 0
        self.size = 0
        
        limit = self.max_buffer_size
        action_size = board_size * board_size

        self.data = {
            "encoded_state": np.empty((limit, num_planes, board_size, board_size), dtype=np.float32),
            "policy_target": np.empty((limit, action_size), dtype=np.float32),
            "opponent_policy_target": np.empty((limit, action_size), dtype=np.float32),
            "outcome": np.empty(limit, dtype=np.float32),
            "nn_policy": np.empty((limit, action_size), dtype=np.float32),
            "nn_value_probs": np.empty((limit, 3), dtype=np.float32),
            "root_value": np.empty(limit, dtype=np.float32),
            "is_full_search": np.empty(limit, dtype=np.bool_),
            "sample_weight": np.empty(limit, dtype=np.float32),
        }

        self.total_samples_added = 0
        self.games_count = 0

    def get_window_size(self):
        if self.total_samples_added < self.linear_threshold:
            return self.total_samples_added
        window_size = self.linear_threshold * (self.total_samples_added / self.linear_threshold) ** self.alpha
        return min(int(window_size), self.max_buffer_size)

    def __len__(self) -> int:
        return self.size

    def add_game(self, game_memory: List[dict]) -> int:
        k = len(game_memory)
        if k == 0:
            return 0
        
        batch_data = {
            key: np.array([sample[key] for sample in game_memory]) 
            for key in self.data.keys()
        }
        
        limit = self.max_buffer_size
        if self.ptr + k <= limit:
            for key in self.data:
                self.data[key][self.ptr : self.ptr + k] = batch_data[key]
        else:
            part1 = limit - self.ptr
            part2 = k - part1
            for key in self.data:
                self.data[key][self.ptr : limit] = batch_data[key][:part1]
                self.data[key][0 : part2] = batch_data[key][part1:]
                
        self.ptr = (self.ptr + k) % limit
        self.size = min(limit, self.size + k)
        
        self.total_samples_added += k
        self.games_count += 1
        return k

    def sample(self, batch_size: int) -> dict:
        if self.size < batch_size:
            return {}
            
        window_size = min(self.size, self.get_window_size())

        start_index = (self.ptr - window_size) % self.max_buffer_size

        if start_index < self.ptr:
            indices = np.random.randint(start_index, self.ptr, size=batch_size)
        else:
            valid_indices = np.concatenate([
                np.arange(start_index, self.max_buffer_size),
                np.arange(0, self.ptr)
            ])
            indices = np.random.choice(valid_indices, size=batch_size, replace=True)

        return {key: self.data[key][indices] for key in self.data}

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.games_count = 0
        self.total_samples_added = 0

    def get_state(self) -> dict:
        if self.size == 0:
            return {"buffer_empty": True}

        return {
            "data": self.data,
            "ptr": self.ptr,
            "size": self.size,
            "min_buffer_size": self.min_buffer_size,
            "linear_threshold": self.linear_threshold,
            "alpha": self.alpha,
            "max_buffer_size": self.max_buffer_size,
            "games_count": self.games_count,
            "total_samples_added": self.total_samples_added,
        }

    def load_state(self, state: dict):
        """Loads the buffer."""
        if "buffer_empty" in state:
            self.clear()
            return

        self.min_buffer_size = state.get("min_buffer_size", self.min_buffer_size)
        self.linear_threshold = state.get("linear_threshold", self.linear_threshold)
        self.alpha = state.get("alpha", self.alpha)
        self.max_buffer_size = int(state.get("max_buffer_size", self.max_buffer_size))

        if "data" in state:
            self.data = state["data"]
            self.ptr = state["ptr"]
            self.size = state["size"]
        else:
            # Migration from old format
            cb = state.get("consolidated_buffer", {})
            num_samples = len(next(iter(cb.values()))) if cb else 0
            
            # Write old data to new format
            limit = self.max_buffer_size
            
            keys = list(cb.keys())
            if num_samples > limit:
                # Need to truncate from the end
                for key in self.data.keys():
                    if key in cb:
                        self.data[key][0 : limit] = cb[key][-limit:]
                self.ptr = 0
                self.size = limit
            else:
                for key in self.data.keys():
                    if key in cb:
                        self.data[key][0 : num_samples] = cb[key]
                self.ptr = num_samples % limit
                self.size = num_samples
                
        self.total_samples_added = state.get("total_samples_added", self.size)
        self.games_count = state.get("games_count", 0)
