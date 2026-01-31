from copy import deepcopy
import numpy as np


class TicTacToe:
    def __init__(self, history_step=3):
        self.board = np.zeros((3, 3))
        self.board_size = 3
        self.action_space_size = 9
        self.history_step = history_step

    def get_initial_state(self):
        return np.zeros((self.history_step, self.board_size, self.board_size))

    @staticmethod
    def get_is_legal_actions(state):
        state = state[-1].flatten()
        return state == 0

    def get_next_state(self, board, action, to_play):
        board = deepcopy(board)

        current_board = board[-1].copy()
        x = action // self.board_size
        y = action % self.board_size
        current_board[x, y] = to_play

        board[:-1] = board[1:]
        board[-1] = current_board

        return board

    @staticmethod
    def get_winner(state):
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

    def is_terminal(self, state):
        return (np.all(state[-1] != 0)
                or self.get_winner(state) is not None)


    @staticmethod
    def encode_state(board, to_play):
        # board.shape = (history_step, board_size, board_size)
        history_len = board.shape[0]
        board_size = board.shape[1]

        encoded_state = np.zeros((history_len * 2 + 1, board_size, board_size), dtype=np.float32)

        for i in range(history_len):
            encoded_state[2 * i] = (board[i] == to_play)
            encoded_state[2 * i + 1] = (board[i] == -to_play)

        encoded_state[-1] = (to_play > 0) * np.ones((board_size, board_size), dtype=np.float32)  # to_play

        return encoded_state

if __name__ == '__main__':
    game = TicTacToe()
    state = game.get_initial_state()
    state[0, 0] = 1
    state[1, 1] = -1
    print(game.encode_state(state, to_play=1))