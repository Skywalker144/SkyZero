from typing import List
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, board_size: int, num_planes: int = None, min_buffer_size=10000, linear_threshold=10000, alpha=0.75, max_buffer_size=3e6):
        self.min_buffer_size = min_buffer_size
        self.linear_threshold = linear_threshold
        self.alpha = alpha
        self.max_buffer_size = int(max_buffer_size)

        self.board_size = board_size
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
            "state": torch.empty((self.block_size, 1, self.board_size, self.board_size), dtype=torch.int8),
            "to_play": torch.empty(self.block_size, dtype=torch.int8),
            "policy_target": torch.empty((self.block_size, self.action_size), dtype=torch.float32),
            "opponent_policy_target": torch.empty((self.block_size, self.action_size), dtype=torch.float32),
            "value_target": torch.empty((self.block_size, 3), dtype=torch.float32),
            "sample_weight": torch.empty(self.block_size, dtype=torch.float32),
        }

    def get_window_size(self):
        if self.total_samples_added < self.linear_threshold:
            return self.total_samples_added
        window_size = self.linear_threshold * (self.total_samples_added / self.linear_threshold) ** self.alpha
        return min(int(window_size), self.max_buffer_size)

    def _gc(self):
        """Release blocks that are entirely outside the sampling window to free memory."""
        window_size = self.get_window_size()
        if self.size <= window_size:
            return

        # Shrink size to window_size
        self.size = window_size

        # Determine which indices are inside the window: [start_index, ptr)
        start_index = (self.ptr - window_size) % self.max_buffer_size

        # Release blocks entirely outside the window
        freed = 0
        for b_idx in range(len(self.blocks)):
            if self.blocks[b_idx] is None:
                continue
            block_start = b_idx * self.block_size
            block_end = block_start + self.block_size  # exclusive

            # Check if block has ANY overlap with the window
            if start_index < self.ptr:
                # Window is contiguous: [start_index, ptr)
                has_overlap = (block_end > start_index) and (block_start < self.ptr)
            else:
                # Window wraps: [start_index, max_buffer_size) U [0, ptr)
                has_overlap = (block_end > start_index) or (block_start < self.ptr)

            if not has_overlap:
                self.blocks[b_idx] = None  # Release memory
                freed += 1

        if freed > 0:
            print(f"[ReplayBuffer GC] Freed {freed} blocks ({freed * self.block_size:,} slots), "
                  f"window_size={window_size:,}, buffer_size={self.size:,}")

    def __len__(self) -> int:
        return self.size

    def add_game(self, game_memory: List[dict]) -> int:
        k = len(game_memory)
        if k == 0:
            return 0
        
        # Keys to extract from game_memory
        keys = ["state", "to_play", "policy_target", "opponent_policy_target", "value_target", "sample_weight"]
        
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
            
            # Ensure block exists (may have been freed by GC or not yet created)
            while len(self.blocks) <= b_idx:
                self.blocks.append(None)
            if self.blocks[b_idx] is None:
                self.blocks[b_idx] = self._create_block()
                
            can_write = min(k - written, self.block_size - b_offset)
            
            # Write to current block
            for key in keys:
                self.blocks[b_idx][key][b_offset : b_offset + can_write] = batch_data[key][written : written + can_write]
            
            written += can_write
            self.ptr = (self.ptr + can_write) % self.max_buffer_size
            self.size = min(self.max_buffer_size, self.size + can_write)
        
        self.total_samples_added += k
        self.games_count += 1
        self._gc()
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
        
        keys = ["state", "to_play", "policy_target", "opponent_policy_target", "value_target", "sample_weight"]
        
        # Determine output shapes from first non-None block
        sample_block = next(b for b in self.blocks if b is not None)
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

        # Consolidate non-None blocks into a dict of lists for fewer pickle objects
        # Record which block indices are valid (some may have been freed by GC)
        keys = ["state", "to_play", "policy_target", "opponent_policy_target", "value_target", "sample_weight"]
        
        valid_block_indices = []
        consolidated = {key: [] for key in keys}
        for b_idx, b in enumerate(self.blocks):
            if b is not None:
                valid_block_indices.append(b_idx)
                for key in keys:
                    consolidated[key].append(b[key])

        return {
            "consolidated_blocks": consolidated,
            "valid_block_indices": valid_block_indices,  # NEW: track which blocks are non-None
            "num_total_blocks": len(self.blocks),  # NEW: total block slots (including None)
            "block_size": self.block_size,
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
        """Loads the buffer state from a checkpoint."""
        if "buffer_empty" in state:
            self.clear()
            return

        self.min_buffer_size = state.get("min_buffer_size", self.min_buffer_size)
        self.linear_threshold = state.get("linear_threshold", self.linear_threshold)
        self.alpha = state.get("alpha", self.alpha)
        self.max_buffer_size = int(state.get("max_buffer_size", self.max_buffer_size))

        state_ptr = state["ptr"]
        state_size = state["size"]
        self.total_samples_added = state.get("total_samples_added", state_size)
        self.games_count = state.get("games_count", 0)

        keys = ["state", "to_play", "policy_target", "opponent_policy_target", "value_target", "sample_weight"]

        cb = state["consolidated_blocks"]
        valid_indices = state["valid_block_indices"]
        num_total_blocks = state.get("num_total_blocks", max(valid_indices) + 1 if valid_indices else 0)

        # Restore sparse block list (with None holes from GC)
        self.blocks = [None] * num_total_blocks
        for i, b_idx in enumerate(valid_indices):
            block = self._create_block()
            for key in keys:
                block[key][:] = cb[key][i]
            self.blocks[b_idx] = block

        self.size = state_size
        self.ptr = state_ptr % self.max_buffer_size
        self.max_blocks = (self.max_buffer_size + self.block_size - 1) // self.block_size

        # Run GC after loading to free blocks outside the current window
        self._gc()
