import numpy as np
from utils import print_board


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.board_size = 3
        self.num_planes = 3

    def get_initial_state(self):
        return np.zeros((1, self.board_size, self.board_size)), 1

    @staticmethod
    def get_is_legal_actions(state, to_play):
        state = state[-1].flatten()
        return state == 0

    def get_next_state(self, state, action, to_play):
        state = state.copy()

        row = action // self.board_size
        col = action % self.board_size
        
        state[0, row, col] = to_play

        return state

    @staticmethod
    def get_winner(state, last_action=None, last_player=None):
        # Check rows and columns for a winner
        for i in range(3):
            if np.all(state[-1][i, :] == 1):  # Check rows for player 1
                return 1
            if np.all(state[-1][i, :] == -1):  # Check rows for player -1
                return -1
            if np.all(state[-1][:, i] == 1):  # Check columns for player 1
                return 1
            if np.all(state[-1][:, i] == -1):  # Check columns for player -1
                return -1

        # Check diagonals for a winner
        if np.all(np.diag(state[-1]) == 1) or np.all(np.diag(np.fliplr(state[-1])) == 1):  # Player 1 diagonals
            return 1
        if np.all(np.diag(state[-1]) == -1) or np.all(np.diag(np.fliplr(state[-1])) == -1):  # Player -1 diagonals
            return -1

        # Check for a draw (no empty spaces left)
        if np.all(state[-1] != 0):
            return 0  # 0 represents a draw

        # No winner yet
        return None

    def is_terminal(self, state, last_action=None, last_player=None):
        return (np.all(state[-1] != 0)
                or self.get_winner(state, last_action, last_player) is not None)

    def encode_state(self, state, to_play):

        encoded_state = np.zeros((3, self.board_size, self.board_size), dtype=np.int8)

        encoded_state[0] = (state[0] == to_play)
        encoded_state[1] = (state[0] == -to_play)

        encoded_state[-1] = (to_play > 0) * np.ones((self.board_size, self.board_size), dtype=np.int8)  # to_play

        return encoded_state

    def encode_state_batch(self, states, to_plays):
        """Vectorized batch encoding of raw states.
        
        Args:
            states: np.ndarray, shape (B, 1, H, W), dtype int8
            to_plays: np.ndarray, shape (B,), values in {-1, 1}
        Returns:
            np.ndarray, shape (B, num_planes, H, W), dtype int8
        """
        B = states.shape[0]
        H, W = self.board_size, self.board_size
        tp = to_plays.reshape(B, 1, 1)  # (B, 1, 1) for broadcasting

        encoded = np.zeros((B, self.num_planes, H, W), dtype=np.int8)
        boards = states[:, 0]  # (B, H, W)
        encoded[:, 0] = (boards == tp)
        encoded[:, 1] = (boards == -tp)
        encoded[:, -1] = (tp > 0).astype(np.int8) * np.ones((1, H, W), dtype=np.int8)
        return encoded

    def get_win_pos(self, final_state):
        b = final_state[-1]
        pos = np.zeros((3, 3), dtype=np.int8)
        
        for i in range(3):
            if abs(np.sum(b[i, :])) == 3: pos[i, :] = 1
        for i in range(3):
            if abs(np.sum(b[:, i])) == 3: pos[:, i] = 1
        if abs(np.trace(b)) == 3:
            np.fill_diagonal(pos, 1)
        if abs(np.trace(np.fliplr(b))) == 3:
            pos[0, 2] = pos[1, 1] = pos[2, 0] = 1
            
        return pos