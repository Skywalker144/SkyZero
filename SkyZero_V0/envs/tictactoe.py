import numpy as np
from utils import print_board


class TicTacToe:
    def __init__(self):
        self.board_size = 3
        self.num_planes = 3

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    @staticmethod
    def get_is_legal_actions(state, to_play):
        return state.flatten() == 0

    def get_next_state(self, state, action, to_play):
        state = state.copy()
        x = action // self.board_size
        y = action % self.board_size
        state[x, y] = to_play
        return state

    @staticmethod
    def get_winner(state):
        for i in range(3):
            if np.all(state[i, :] == 1): return 1
            if np.all(state[i, :] == -1): return -1
            if np.all(state[:, i] == 1): return 1
            if np.all(state[:, i] == -1): return -1

        if np.all(np.diag(state) == 1) or np.all(np.diag(np.fliplr(state)) == 1):
            return 1
        if np.all(np.diag(state) == -1) or np.all(np.diag(np.fliplr(state)) == -1):
            return -1

        if np.all(state != 0):
            return 0

        return None

    def is_terminal(self, state):
        return self.get_winner(state) is not None

    @staticmethod
    def encode_state(state, to_play):
        board_size = state.shape[0]
        encoded_state = np.zeros((3, board_size, board_size), dtype=np.int8)
        encoded_state[0] = (state == to_play)
        encoded_state[1] = (state == -to_play)
        encoded_state[2] = (to_play > 0) * np.ones((board_size, board_size), dtype=np.int8)
        return encoded_state

    def get_win_pos(self, final_state):
        b = final_state
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
