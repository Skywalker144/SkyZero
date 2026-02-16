"""
Replay Buffer for AlphaZero

AlphaZero uses Replay Buffer to store self-play data from the last N games,
sampling randomly from the entire buffer during training, not just the latest game.

Main features:
1. Capacity limited by sample count (window_size), old data replaced by new
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
        window_size: Maximum number of samples to keep, old data will be removed
        board_size: Board size (for potential future use in data augmentation)
    """
    
    def __init__(self, window_size: int = 500000, board_size: int = 9):
        self.window_size = window_size
        self.board_size = board_size
        
        # Ring buffer: List + position pointer
        self.buffer: List[Tuple] = []
        self.position = 0  # Current write position
        
        # Track number of games added
        self.games_count = 0
        
    def __len__(self) -> int:
        """Return current number of samples in buffer"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer has reached capacity"""
        return len(self.buffer) >= self.window_size
    
    def add_game(self, game_memory: List[Tuple]) -> int:
        """
        Add a game's data to the buffer.
        
        Args:
            game_memory: List of samples, each sample is a tuple
            
        Returns:
            Number of samples added
        """
        for sample in game_memory:
            if len(self.buffer) < self.window_size:
                self.buffer.append(sample)
            else:
                # Ring buffer: overwrite oldest data
                self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.window_size
                
        self.games_count += 1
        return len(game_memory)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Randomly sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            List of sampled tuples
            
        Raises:
            ValueError: If buffer has fewer samples than requested
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self.buffer)} samples, "
                f"but batch_size={batch_size} was requested. "
                f"Wait for more data or reduce min_buffer_size."
            )
        return random.sample(self.buffer, batch_size)
    
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
            'window_size': self.window_size,
            'board_size': self.board_size,
            'position': self.position,
            'games_count': self.games_count,
        }
    
    def load_state(self, state: dict):
        """
        Load buffer state from dictionary.
        
        Args:
            state: Dictionary from get_state()
        """
        self.buffer = list(state['buffer'])
        self.window_size = state['window_size']
        self.board_size = state['board_size']
        
        # Backwards compatibility: Handle missing 'position' field from old checkpoints
        if 'position' in state:
            self.position = state['position']
        else:
            # Infer position for old checkpoints
            if len(self.buffer) < self.window_size:
                self.position = len(self.buffer)
            else:
                self.position = 0  # Assume we start overwriting from beginning
            print(f"Warning: 'position' not found in checkpoint. Inferred position: {self.position}")
            
        self.games_count = state.get('games_count', 0)
    
    def save(self, filepath: str):
        """
        Save buffer to file.
        
        Args:
            filepath: Path to save
        """
        data = self.get_state()
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Replay buffer saved to {filepath} ({len(self.buffer)} samples)")
    
    def load(self, filepath: str) -> bool:
        """
        Load buffer from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Whether load was successful
        """
        if not os.path.exists(filepath):
            print(f"Buffer file not found: {filepath}")
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.load_state(data)
        print(f"Replay buffer loaded from {filepath} ({len(self.buffer)} samples, {self.games_count} games)")
        return True
    
    def stats(self) -> dict:
        """Return buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'games_count': self.games_count,
                'capacity': self.window_size,
                'fill_ratio': 0.0,
            }
            
        return {
            'size': len(self.buffer),
            'games_count': self.games_count,
            'capacity': self.window_size,
            'fill_ratio': len(self.buffer) / self.window_size,
        }


# Backward compatibility alias
ParallelReplayBuffer = ReplayBuffer
