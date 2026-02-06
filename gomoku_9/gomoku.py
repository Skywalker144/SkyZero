from copy import deepcopy

import numpy as np
import tkinter as tk
from tkinter import messagebox


class Gomoku:
    def __init__(self, board_size=15, history_step=4):  # history_board * (8 - 1) and action_board
        self.board = np.zeros((history_step, board_size, board_size))
        self.board_size = board_size
        self.board_height = board_size
        self.board_width = board_size
        self.action_space_size = board_size * board_size
        self.history_step = history_step
        # self.num_planes = 2 * history_step
        self.num_planes = 2 * history_step + 1

    def get_initial_state(self):
        return np.zeros((self.history_step, self.board_size, self.board_size))

    @staticmethod
    def get_is_legal_actions(board):
        # board is history stack: (history_step, board_size, board_size)
        current_board = board[-1]
        is_legal = (current_board == 0)
        
        # Determine current player (Black=1, White=-1)
        # Assumes Black plays first (when board is empty, sum is 0, so Black's turn)
        # If sum != 0, we need to count stones.
        # Black always +1, White -1. Sum = 0 (equal turns) or 1 (Black played).
        # Wait, if Black=1, White=-1.
        # 1 stone (Black) -> Sum=1. Next is White.
        # 2 stones (B, W) -> Sum=0. Next is Black.
        # So if Sum == 0 -> Black (1). If Sum == 1 -> White (-1).
        # Actually it's safer to count non-zeros.
        
        stone_count = np.sum(current_board != 0)
        to_play = 1 if stone_count % 2 == 0 else -1
        
        # If it's White's turn (-1), no forbidden moves. Return simple check.
        if to_play == -1:
            return is_legal.flatten()
            
        # If it's Black's turn (1), mask forbidden moves.
        # Iterate over all currently legal moves (empty spots)
        legal_indices = np.argwhere(is_legal)
        
        # We need a copy of the board to simulate moves?
        # Or just pass the board and the move coordinates to a helper that doesn't modify it permanently.
        # _check_forbidden expects a 2D board (int/float).
        
        # Create a working copy of the board
        work_board = current_board.copy()
        
        rows, cols = current_board.shape
        # Result mask
        final_legal = is_legal.copy()
        
        for idx in legal_indices:
            r, c = idx[0], idx[1]
            if Gomoku._check_forbidden(work_board, r, c):
                final_legal[r, c] = False
                
        return final_legal.flatten()

    @staticmethod
    def _get_line(board, x, y, dx, dy, radius=6):
        rows, cols = board.shape
        values = []
        BOUNDARY = 2 # Treat boundary as opponent for pattern matching logic if needed
        
        # Range -radius to +radius
        for k in range(-radius, radius + 1):
            r, c = x + k*dx, y + k*dy
            if k == 0:
                center_idx = len(values)
            
            if 0 <= r < rows and 0 <= c < cols:
                values.append(board[r, c])
            else:
                values.append(BOUNDARY)
        return values, center_idx

    @staticmethod
    def _is_five_check(board, x, y):
        rows, cols = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        BLACK = 1
        
        for dx, dy in directions:
            count = 1
            r, c = x + dx, y + dy
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == BLACK:
                count += 1
                r += dx
                c += dy
            r, c = x - dx, y - dy
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == BLACK:
                count += 1
                r -= dx
                c -= dy
            if count == 5:
                return True
        return False

    @staticmethod
    def _is_overline(board, x, y):
        rows, cols = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        BLACK = 1
        
        for dx, dy in directions:
            count = 1
            r, c = x + dx, y + dy
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == BLACK:
                count += 1
                r += dx
                c += dy
            r, c = x - dx, y - dy
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == BLACK:
                count += 1
                r -= dx
                c -= dy
            if count > 5:
                return True
        return False

    @staticmethod
    def _count_fours(board, x, y):
        # Counts "Fours" created by placing Black at (x,y)
        # A Four is a threat to win (become 5).
        rows, cols = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_fours = 0
        BLACK = 1
        EMPTY = 0
        
        for dx, dy in directions:
            line, center = Gomoku._get_line(board, x, y, dx, dy, radius=9)
            threats_in_line = set()
            
            # Iterate windows of length 5
            for i in range(len(line) - 4):
                if not (i <= center < i + 5):
                    continue
                
                window = line[i : i+5]
                black_count = 0
                empty_count = 0
                empty_idx = -1
                boundary = False
                
                for k in range(5):
                    val = window[k]
                    if val == BLACK:
                        black_count += 1
                    elif val == EMPTY:
                        empty_count += 1
                        empty_idx = k
                    else: # White or Boundary
                        boundary = True
                        break
                
                if boundary:
                    continue
                    
                if black_count == 4 and empty_count == 1:
                    # Check if filling the empty spot creates Overline (invalid 5)
                    # We need to simulate filling it in the line
                    global_empty_idx = i + empty_idx
                    
                    # Quick check in the extracted line
                    temp_line = list(line)
                    temp_line[global_empty_idx] = BLACK
                    
                    # Count consecutive blacks at global_empty_idx
                    c_idx = global_empty_idx
                    l = c_idx
                    while l >= 0 and temp_line[l] == BLACK:
                        l -= 1
                    r = c_idx
                    while r < len(temp_line) and temp_line[r] == BLACK:
                        r += 1
                    
                    consecutive = (r - 1) - (l + 1) + 1
                    
                    if consecutive == 5:
                        threats_in_line.add(global_empty_idx)
            
            if len(threats_in_line) > 0:
                total_fours += 1
        return total_fours

    @staticmethod
    def _count_open_threes(board, x, y):
        rows, cols = board.shape
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_open_threes = 0
        BLACK = 1
        EMPTY = 0
        
        for dx, dy in directions:
            # Need larger radius to check for overlines when validating the "Open" status
            line, center = Gomoku._get_line(board, x, y, dx, dy, radius=9)
            found = False
            
            # Iterate windows of length 6
            # Looking for pattern that can become 011110 (Open Four)
            # Core 111 (or 1011 etc) must be able to become 1111.
            for i in range(len(line) - 5):
                if not (i <= center < i + 6):
                    continue
                
                window = line[i : i+6]
                
                # Ends must be empty
                if window[0] != EMPTY or window[5] != EMPTY:
                    continue
                
                core = window[1:5]
                b_count = 0
                e_count = 0
                empty_core_idx = -1
                
                for k in range(4):
                    val = core[k]
                    if val == BLACK:
                        b_count += 1
                    elif val == EMPTY:
                        e_count += 1
                        empty_core_idx = k
                
                if b_count == 3 and e_count == 1:
                    # We found a candidate "Open Three" pattern: 0 [B B B .] 0
                    # But it is only a TRUE Open Three if playing at the empty spot
                    # creates a LIVE Four (011110).
                    # A Live Four implies that BOTH ends (window[0] and window[5])
                    # are valid winning spots (i.e., make 5, not Overline).
                    
                    # 1. Simulate playing at the core empty spot to make it a Four
                    # empty_core_idx is relative to core start (which is window[1])
                    # Global index in 'line':
                    fill_idx = i + 1 + empty_core_idx
                    
                    # Check if this forms a Live Four
                    # Left winning spot: i
                    # Right winning spot: i + 5
                    
                    # We need to check if filling 'i' creates exactly 5 (valid win)
                    # AND filling 'i+5' creates exactly 5 (valid win).
                    # This must be done assuming 'fill_idx' is ALREADY filled with BLACK.
                    
                    temp_line = list(line)
                    temp_line[fill_idx] = BLACK
                    
                    # Check Left End (i)
                    # Place BLACK at i
                    l_check = list(temp_line)
                    l_check[i] = BLACK
                    
                    # Count consecutive at i
                    c_idx = i
                    l = c_idx
                    while l >= 0 and l_check[l] == BLACK:
                        l -= 1
                    r = c_idx
                    while r < len(l_check) and l_check[r] == BLACK:
                        r += 1
                    len_left = (r - 1) - (l + 1) + 1
                    
                    # Check Right End (i+5)
                    # Place BLACK at i+5
                    r_check = list(temp_line)
                    r_check[i+5] = BLACK
                    
                    # Count consecutive at i+5
                    c_idx = i+5
                    l = c_idx
                    while l >= 0 and r_check[l] == BLACK:
                        l -= 1
                    r = c_idx
                    while r < len(r_check) and r_check[r] == BLACK:
                        r += 1
                    len_right = (r - 1) - (l + 1) + 1
                    
                    if len_left == 5 and len_right == 5:
                        found = True
                        break
            
            if found:
                total_open_threes += 1
        return total_open_threes

    @staticmethod
    def _check_forbidden(board, x, y):
        # board is a 2D array. (x,y) is the move.
        # Returns True if forbidden.
        BLACK = 1
        EMPTY = 0
        
        # Place tentatively
        board[x, y] = BLACK
        
        # 1. Win (5) -> Valid (takes precedence over forbidden)
        if Gomoku._is_five_check(board, x, y):
            board[x, y] = EMPTY
            return False
            
        # 2. Overline -> Forbidden
        if Gomoku._is_overline(board, x, y):
            board[x, y] = EMPTY
            return True
            
        # 3. Double Four -> Forbidden
        if Gomoku._count_fours(board, x, y) >= 2:
            board[x, y] = EMPTY
            return True
            
        # 4. Double Three -> Forbidden
        if Gomoku._count_open_threes(board, x, y) >= 2:
            board[x, y] = EMPTY
            return True
            
        board[x, y] = EMPTY
        return False

    def get_next_state(self, board, action, to_play):
        board = board.copy()

        current_board = board[-1].copy()
        x = action // self.board_size
        y = action % self.board_size
        current_board[x, y] = to_play

        board[:-1] = board[1:]
        board[-1] = current_board

        return board

    @staticmethod
    def get_winner(board):
        for i in range(board[-1].shape[0]):
            for j in range(board[-1].shape[1]):
                if board[-1][i, j] != 0:
                    player = board[-1][i, j]

                    if j + 4 < board[-1].shape[1] and np.all(board[-1][i, j:j + 5] == player):
                        return player
                    if i + 4 < board[-1].shape[0] and np.all(board[-1][i:i + 5, j] == player):
                        return player
                    if i + 4 < board[-1].shape[0] and j + 4 < board[-1].shape[1] and np.all(np.diag(board[-1][i:i + 5, j:j + 5]) == player):
                        return player
                    if i + 4 < board[-1].shape[0] and j - 4 >= 0 and np.all(np.diag(np.fliplr(board[-1][i:i + 5, j - 4:j + 1])) == player):
                        return player

        if np.all(board[-1] != 0):
            return 0
        return None

    def is_terminal(self, state):
        return self.get_winner(state) is not None
    
    @staticmethod
    def encode_state(board, to_play):
        # board.shape = (history_step, board_size, board_size)
        history_len = board.shape[0]
        board_size = board.shape[1]

        encoded_state = np.zeros((history_len * 2 + 1, board_size, board_size), dtype=np.float32)
        # encoded_state = np.zeros((history_len * 2, board_size, board_size), dtype=np.float32)

        for i in range(history_len):
            encoded_state[2 * i] = (board[i] == to_play)
            encoded_state[2 * i + 1] = (board[i] == -to_play)

        encoded_state[-1] = (to_play > 0) * np.ones((board_size, board_size), dtype=np.float32)  # to_play

        return encoded_state

if __name__ == "__main__":
    game = Gomoku()
    initial_state = game.get_initial_state()
    print(f"Initial state shape: {initial_state.shape}")
    print(f"encode_state shape: {game.encode_state(initial_state, to_play=1).shape}")
