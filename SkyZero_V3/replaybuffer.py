import random
from typing import List, Tuple, Dict
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, board_size: int, num_planes: int, min_buffer_size=10000, linear_threshold=10000, alpha=0.75, max_buffer_size=3e6):
        self.min_buffer_size = min_buffer_size
        self.linear_threshold = linear_threshold
        self.alpha = alpha
        self.max_buffer_size = int(max_buffer_size)

        self.board_size = board_size
        self.num_planes = num_planes
        self.action_size = board_size * board_size
        
        # Block-based storage using torch tensors for memory efficiency during save
        # Increased block size to 100,000 to reduce object overhead during saving
        self.block_size = 100000
        self.blocks = []  # List of blocks, each block is a dict of torch tensors
        self.max_blocks = (self.max_buffer_size + self.block_size - 1) // self.block_size

        self.ptr = 0
        self.size = 0
        
        self.total_samples_added = 0
        self.games_count = 0

    def _create_block(self):
        return {
            "encoded_state": torch.empty((self.block_size, self.num_planes, self.board_size, self.board_size), dtype=torch.float32),
            "policy_target": torch.empty((self.block_size, self.action_size), dtype=torch.float32),
            "opponent_policy_target": torch.empty((self.block_size, self.action_size), dtype=torch.float32),
            "outcome": torch.empty(self.block_size, dtype=torch.float32),
            "nn_policy": torch.empty((self.block_size, self.action_size), dtype=torch.float32),
            "nn_value_probs": torch.empty((self.block_size, 3), dtype=torch.float32),
            "root_value": torch.empty(self.block_size, dtype=torch.float32),
            "sample_weight": torch.empty(self.block_size, dtype=torch.float32),
        }

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
        
        # Keys to extract from game_memory
        keys = ["encoded_state", "policy_target", "opponent_policy_target", "outcome", 
                "nn_policy", "nn_value_probs", "root_value", "sample_weight"]
        
        # Prepare data in torch format (on CPU)
        batch_data = {}
        for key in keys:
            # Convert to numpy first if needed, then to torch tensor
            vals = [sample[key] for sample in game_memory]
            batch_data[key] = torch.as_tensor(np.array(vals))
        
        written = 0
        while written < k:
            b_idx = self.ptr // self.block_size
            b_offset = self.ptr % self.block_size
            
            # Ensure block exists
            while len(self.blocks) <= b_idx:
                self.blocks.append(self._create_block())
                
            can_write = min(k - written, self.block_size - b_offset)
            
            # Write to current block
            for key in keys:
                self.blocks[b_idx][key][b_offset : b_offset + can_write] = batch_data[key][written : written + can_write]
            
            written += can_write
            self.ptr = (self.ptr + can_write) % self.max_buffer_size
            self.size = min(self.max_buffer_size, self.size + can_write)
        
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
            p = np.array([self.max_buffer_size - start_index, self.ptr], dtype=np.float32)
            p /= p.sum()
            choices = np.random.choice([0, 1], size=batch_size, p=p)
            indices = np.empty(batch_size, dtype=np.int64)
            mask0 = (choices == 0)
            mask1 = (choices == 1)
            if mask0.any():
                indices[mask0] = np.random.randint(start_index, self.max_buffer_size, size=mask0.sum())
            if mask1.any():
                indices[mask1] = np.random.randint(0, self.ptr, size=mask1.sum())

        # Group indices by block for efficiency
        block_ids = indices // self.block_size
        offsets = indices % self.block_size
        
        keys = ["encoded_state", "policy_target", "opponent_policy_target", "outcome", 
                "nn_policy", "nn_value_probs", "root_value", "sample_weight"]
        
        # Determine output shapes from first block
        sample_block = self.blocks[0]
        result = {}
        for key in keys:
            shape = (batch_size,) + sample_block[key].shape[1:]
            result[key] = torch.empty(shape, dtype=sample_block[key].dtype)

        # Fill the result batch by batch from each affected block
        unique_block_ids = np.unique(block_ids)
        for b_id in unique_block_ids:
            mask = (block_ids == b_id)
            block_offsets = torch.from_numpy(offsets[mask])
            for key in keys:
                # Use index_select or simple indexing for torch tensors
                result[key][torch.from_numpy(mask)] = self.blocks[b_id][key][block_offsets]
                
        # Convert back to numpy for consistency with the rest of the code if necessary
        # But wait, AlphaZero already converts to torch, so let's keep it as torch (CPU)
        # Actually, let's convert to numpy for the random_augment_batch which might expect numpy
        return {k: v.numpy() for k, v in result.items()}

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.games_count = 0
        self.total_samples_added = 0
        self.blocks = []

    def get_state(self) -> dict:
        if self.size == 0:
            return {"buffer_empty": True}

        # Consolidate list of dicts into a dict of lists for fewer pickle objects
        keys = ["encoded_state", "policy_target", "opponent_policy_target", "outcome", 
                "nn_policy", "nn_value_probs", "root_value", "sample_weight"]
        
        consolidated = {}
        for key in keys:
            consolidated[key] = [b[key] for b in self.blocks]

        return {
            "consolidated_blocks": consolidated,
            "block_size": self.block_size, # Explicitly save block_size
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
        """Loads the buffer and handles block-size migration."""
        if "buffer_empty" in state:
            self.clear()
            return

        self.min_buffer_size = state.get("min_buffer_size", self.min_buffer_size)
        self.linear_threshold = state.get("linear_threshold", self.linear_threshold)
        self.alpha = state.get("alpha", self.alpha)
        self.max_buffer_size = int(state.get("max_buffer_size", self.max_buffer_size))
        
        # ptr and size from state
        state_ptr = state["ptr"]
        state_size = state["size"]
        self.total_samples_added = state.get("total_samples_added", state_size)
        self.games_count = state.get("games_count", 0)

        keys = ["encoded_state", "policy_target", "opponent_policy_target", "outcome", 
                "nn_policy", "nn_value_probs", "root_value", "sample_weight"]

        raw_tensors = {key: [] for key in keys}
        
        if "consolidated_blocks" in state:
            cb = state["consolidated_blocks"]
            # Detect loaded block size
            loaded_block_size = cb[keys[0]][0].shape[0]
            for key in keys:
                raw_tensors[key] = cb[key]
        elif "blocks" in state:
            blocks = state["blocks"]
            loaded_block_size = blocks[0][keys[0]].shape[0]
            for b in blocks:
                for key in keys:
                    raw_tensors[key].append(b[key])
        elif "data" in state or "consolidated_buffer" in state:
            # Old non-blocked format
            data = state.get("data", state.get("consolidated_buffer", {}))
            self.clear()
            # We can use add_game or a custom loop, but simplest is to just re-import
            # Let's mock a game_memory structure for a moment or just fill blocks
            num_samples = state_size
            self.size = 0
            self.ptr = 0
            self.blocks = []
            
            # Re-fill using current block size
            for start_idx in range(0, num_samples, self.block_size):
                end_idx = min(start_idx + self.block_size, num_samples)
                n = end_idx - start_idx
                block = self._create_block()
                for key in keys:
                    if key in data:
                        block[key][0:n] = torch.as_tensor(data[key][start_idx:end_idx])
                self.blocks.append(block)
            self.size = num_samples
            self.ptr = state_ptr % self.max_buffer_size
            return
        else:
            return

        # Migration/Re-blocking logic:
        # If the loaded block size is different from current, we re-block the data
        # to ensure the ptr/indexing logic remains consistent with self.block_size
        
        # Calculate total samples currently held in blocks
        # Note: the last block might be partially full in the old logic, 
        # but our new logic assumes blocks are always full except possibly the very last one added.
        # Actually, circular buffer means ptr is the only thing that matters.
        
        # To be safe and simple: concatenate all blocks and re-split
        # This only happens once during loading.
        full_data = {}
        for key in keys:
            # Only take the part that actually contains data if it's the old format
            # but wait, the old format had blocks of fixed size. 
            # The total samples is self.size.
            concatenated = torch.cat(raw_tensors[key], dim=0)
            # Truncate to max_buffer_size if needed
            if concatenated.shape[0] > self.max_buffer_size:
                concatenated = concatenated[:self.max_buffer_size]
            full_data[key] = concatenated

        num_samples = full_data[keys[0]].shape[0]
        self.blocks = []
        for start_idx in range(0, num_samples, self.block_size):
            end_idx = min(start_idx + self.block_size, num_samples)
            n = end_idx - start_idx
            block = self._create_block()
            for key in keys:
                block[key][0:n] = full_data[key][start_idx:end_idx]
            self.blocks.append(block)
        
        self.size = num_samples
        self.ptr = state_ptr % self.max_buffer_size
        self.max_blocks = (self.max_buffer_size + self.block_size - 1) // self.block_size
