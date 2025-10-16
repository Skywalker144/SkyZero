import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.board_size = 3
        self.action_space_size = 9

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size))

    @staticmethod
    def get_is_legal_actions(state):
        state = state.flatten()
        return state == 0

    def get_next_state(self, state, action, to_play):
        state = state.copy()
        x = action // self.board_size
        y = action % self.board_size
        state[x, y] = to_play
        return state

    @staticmethod
    def get_winner(state):
        # Check rows and columns for a winner
        for i in range(3):
            if np.all(state[i, :] == 1):  # Check rows for player 1
                return 1
            if np.all(state[i, :] == -1):  # Check rows for player -1
                return -1
            if np.all(state[:, i] == 1):  # Check columns for player 1
                return 1
            if np.all(state[:, i] == -1):  # Check columns for player -1
                return -1

        # Check diagonals for a winner
        if np.all(np.diag(state) == 1) or np.all(np.diag(np.fliplr(state)) == 1):  # Player 1 diagonals
            return 1
        if np.all(np.diag(state) == -1) or np.all(np.diag(np.fliplr(state)) == -1):  # Player -1 diagonals
            return -1

        # Check for a draw (no empty spaces left)
        if np.all(state != 0):
            return 0  # 0 represents a draw

        # No winner yet
        return None

    def is_terminal(self, state):
        return (np.all(state != 0)
                or self.get_winner(state) is not None)


    @staticmethod
    def encode_state(state, to_play):
        encoded_state = np.zeros((3, *state.shape), dtype=np.float32)

        encoded_state[0] = (state == 1)
        encoded_state[1] = (state == -1)
        encoded_state[2] = (to_play > 0) * 1.0

        return encoded_state

if __name__ == '__main__':
    game = TicTacToe()
    state = game.get_initial_state()
    print(game.encode_state(state, 1))
    state[0, 0] = 1