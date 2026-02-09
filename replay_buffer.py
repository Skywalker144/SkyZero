"""
Replay Buffer for AlphaZero

AlphaZero使用Replay Buffer存储最近N局游戏的自我对弈数据，
训练时从整个buffer中随机采样，而不是仅使用最新一局的数据。

主要特性：
1. 按局数限制容量（window_size），旧数据会被新数据替换
2. 支持数据增强（棋盘对称变换）
3. 支持随机采样
4. 支持保存和加载buffer状态
"""

import os
import pickle
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np


class ReplayBuffer:
    """
    Replay Buffer用于存储自我对弈生成的训练数据。
    
    Args:
        window_size: 保存最近多少局游戏的数据，旧数据会被移除
        board_size: 棋盘大小，用于数据增强
    """
    
    def __init__(self, window_size: int = 500000, board_size: int = 9):
        self.window_size = window_size
        self.board_size = board_size
        
        # 使用deque存储所有样本，自动移除旧数据
        self.buffer = deque(maxlen=window_size)
        
        # 记录添加的游戏局数
        self.games_count = 0
        
    def __len__(self) -> int:
        """返回buffer中的样本数量"""
        return len(self.buffer)
    
    def add_game(self, game_memory: List[Tuple]) -> int:
        """
        添加一局游戏的数据到buffer中。
        """
        for sample in game_memory:
            self.buffer.append(sample)
            
        self.games_count += 1
        return len(game_memory)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        从buffer中随机采样一个batch。
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def get_all(self) -> List[Tuple]:
        """获取buffer中的所有数据"""
        return list(self.buffer)
    
    def clear(self):
        """清空buffer"""
        self.buffer.clear()
        self.games_count = 0

class ParallelReplayBuffer:
    """
    专为并行训练优化的Replay Buffer。
    使用List代替Deque以优化随机采样性能（避免O(N)的类型转换）。
    使用环形缓冲区（Ring Buffer）逻辑管理内存。
    """
    def __init__(self, window_size: int = 500000, board_size: int = 9):
        self.window_size = window_size
        self.board_size = board_size
        self.buffer = [] # 使用List
        self.position = 0 # 当前写入指针
        self.games_count = 0
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def add_game(self, game_memory: List[Tuple]) -> int:
        for sample in game_memory:
            if len(self.buffer) < self.window_size:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
                self.position = (self.position + 1) % self.window_size
                
        self.games_count += 1
        return len(game_memory)
        
    def sample(self, batch_size: int) -> List[Tuple]:
        # List采样是高效的，不需要转换
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
        
    def clear(self):
        self.buffer = []
        self.position = 0
        self.games_count = 0


    
    def save(self, filepath: str):
        """
        保存buffer到文件。
        
        Args:
            filepath: 保存路径
        """
        data = {
            'buffer': list(self.buffer),
            'window_size': self.window_size,
            'board_size': self.board_size,
            'games_count': self.games_count,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Replay buffer saved to {filepath} ({len(self.buffer)} samples)")
    
    def load(self, filepath: str) -> bool:
        """
        从文件加载buffer。
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(filepath):
            print(f"Buffer file not found: {filepath}")
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.buffer = deque(data['buffer'], maxlen=self.window_size)
        self.games_count = data.get('games_count', 0)
        
        print(f"Replay buffer loaded from {filepath} ({len(self.buffer)} samples, {self.games_count} games)")
        return True
    
    def stats(self) -> dict:
        """返回buffer的统计信息"""
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