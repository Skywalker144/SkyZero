from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(
            self,
            max_buffer_size=50000,
            min_buffer_size=1000,
            window_exponent=0.65,
            window_expand_per_row=0.4
        ):
        self.max_buffer_size = int(max_buffer_size)
        self.min_buffer_size = int(min_buffer_size)
        self.window_exponent = float(window_exponent)
        self.window_expand_per_row = float(window_expand_per_row)
        self.buffer = deque(maxlen=self.max_buffer_size)
        self.total_samples_added = 0

    def __len__(self):
        return len(self.buffer)

    def add_game_memory(self, game_memory):
        self.buffer.extend(game_memory)
        self.total_samples_added += len(game_memory)

    def _window_size(self):
        # KataGo three-parameter power law (V6 shuffle.py:62).
        # W(N) = (N^E - M^E) / (E*M^(E-1)) * IWPR + M, anchored at (M, M)
        # with initial slope IWPR. Below M we return N — caller is expected to
        # gate on is_ready() and skip training when total < min_buffer_size.
        N = self.total_samples_added
        M = self.min_buffer_size
        if N <= M:
            return N
        E = self.window_exponent
        IWPR = self.window_expand_per_row
        scaled = (N ** E - M ** E) / (E * M ** (E - 1))
        return int(scaled * IWPR + M)

    def window_size(self):
        return self._window_size()

    def sample(self, batch_size):
        if not self.is_ready():
            return []
        # Window restricts which entries are eligible. The deque's tail is the
        # newest data; we sample uniformly without replacement from the last
        # `eligible` entries. When W > len(buffer) (deque already trimmed older
        # data via maxlen), eligible falls back to len(buffer) — the in-memory
        # equivalent of "older rows already left the window".
        W = self._window_size()
        n = len(self.buffer)
        eligible = min(n, W)
        if eligible < batch_size:
            return []
        lo = n - eligible
        idx = np.random.choice(eligible, size=batch_size, replace=False) + lo
        return [self.buffer[i] for i in idx]

    def is_ready(self):
        return self.total_samples_added >= self.min_buffer_size

    def get_state(self):
        return {
            "buffer": list(self.buffer),
            "total_samples_added": self.total_samples_added,
        }

    def load_state(self, state):
        self.buffer = deque(state.get("buffer", []), maxlen=self.max_buffer_size)
        self.total_samples_added = state.get("total_samples_added", len(self.buffer))
