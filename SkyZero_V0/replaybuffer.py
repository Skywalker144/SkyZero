import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_buffer_size=50000, min_buffer_size=1000):
        self.max_buffer_size = int(max_buffer_size)
        self.min_buffer_size = int(min_buffer_size)
        self.buffer = deque(maxlen=self.max_buffer_size)
        self.total_samples_added = 0

    def __len__(self):
        return len(self.buffer)

    def add_game(self, game_memory):
        self.buffer.extend(game_memory)
        self.total_samples_added += len(game_memory)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def is_ready(self):
        return len(self.buffer) >= self.min_buffer_size

    def get_state(self):
        return {
            "buffer": list(self.buffer),
            "total_samples_added": self.total_samples_added
        }

    def load_state(self, state):
        self.buffer = deque(state.get("buffer", []), maxlen=self.max_buffer_size)
        self.total_samples_added = state.get("total_samples_added", len(self.buffer))
