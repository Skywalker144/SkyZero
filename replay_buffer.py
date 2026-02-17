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

import os
import pickle
import random
from typing import List, Tuple, Optional

import numpy as np


class ReplayBuffer:
    """
    Replay Buffer for storing self-play generated training data.
    
    Uses Ring Buffer (circular buffer) implementation:
    - O(1) append complexity
    - O(batch_size) sample complexity (no O(N) index access like deque)
    
    Args:
        max_buffer_size: Maximum number of samples to keep, old data will be removed
    """
    
    def __init__(self, max_buffer_size: int = 500000, min_buffer_size: int = 50000, buffer_size_k: float = 1.0):
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.buffer_size_k = buffer_size_k
        
        # Ring buffer: List + position pointer
        self.buffer: List[Tuple] = []
        self.position = 0  # Current write position
        
        # Track number of games added
        self.games_count = 0
        self.total_samples_added = 0
    
    def _get_window_size(self) -> int:
        """Calculate dynamic buffer size limit based on samples added beyond min_size."""
        # If we haven't reached min_buffer_size, dynamic limit is just min_buffer_size
        # (effectively no limit beyond physical count, as physical count < min_size)
        if self.total_samples_added < self.min_buffer_size:
            return self.min_buffer_size

        L = self.max_buffer_size - self.min_buffer_size
        
        # Prevent division by zero or invalid math
        if L <= 0 or self.buffer_size_k <= 0:
            return self.max_buffer_size
            
        # Formula: current = Max - (Max - Min) * exp(-x / (buffer_size_k * (Max - Min)))
        # When x=0 (just reached min_size), exp(0)=1, result = Max - L = Min
        window_size = self.max_buffer_size - L * np.exp((self.min_buffer_size - self.total_samples_added) / (L / self.buffer_size_k))
        return int(window_size)

    def __len__(self) -> int:
        """Return current number of samples in buffer"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer has reached capacity"""
        return len(self.buffer) >= self.max_buffer_size
    
    def add_game(self, game_memory: List[Tuple]) -> int:
        """
        Add a game's data to the buffer.
        
        Args:
            game_memory: List of samples, each sample is a tuple
            
        Returns:
            Number of samples added
        """
        for sample in game_memory:
            if len(self.buffer) < self.max_buffer_size:
                self.buffer.append(sample)
            else:
                # Ring buffer: overwrite oldest data
                self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.max_buffer_size
                
        self.games_count += 1
        self.total_samples_added += len(game_memory)
        return len(game_memory)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        current_len = len(self.buffer)

        window_size = self._get_window_size()
        
        # 1. Sample logical indices (0 = newest, effective_size-1 = oldest in window)
        logical_indices = random.sample(range(window_size), batch_size)
        
        # 2. Map to physical indices in the ring buffer
        # The newest item is at (self.position - 1) % current_len
        # The i-th newest is at (self.position - 1 - i) % current_len
        physical_indices = [(self.position - 1 - i) % current_len for i in logical_indices]
        
        return [self.buffer[i] for i in physical_indices]
    
    def get_all(self) -> List[Tuple]:
        """Get all data in buffer"""
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.position = 0
        self.games_count = 0
    
    def get_state(self) -> dict:
        """
        Get buffer state for serialization.
        
        Returns:
            Dictionary containing all buffer state
        """
        return {
            'buffer': self.buffer,
            'max_buffer_size': self.max_buffer_size,
            'min_buffer_size': self.min_buffer_size,
            'buffer_size_k': self.buffer_size_k,
            'position': self.position,
            'games_count': self.games_count,
            'total_samples_added': self.total_samples_added,
        }
    
    def load_state(self, state: dict):
        """
        Load buffer state from dictionary.
        
        Args:
            state: Dictionary from get_state()
        """
        self.buffer = list(state['buffer'])
        self.max_buffer_size = state['max_buffer_size']
        self.min_buffer_size = state['min_buffer_size']
        self.buffer_size_k = ['buffer_size_k']
        self.total_samples_added = ['total_samples_added']
        self.position = state['position']
        self.games_count = ['games_count']

# Backward compatibility alias
ParallelReplayBuffer = ReplayBuffer
