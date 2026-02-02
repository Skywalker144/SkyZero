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
        board = board[-1].copy().flatten()
        return board == 0

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
