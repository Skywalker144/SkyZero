from copy import deepcopy

import numpy as np
import tkinter as tk
from tkinter import messagebox


class Gomoku:
    def __init__(self, board_size=15):
        self.board = np.zeros((board_size, board_size))
        self.board_size = board_size
        self.action_space_size = board_size * board_size

    def get_initial_state(self):
        return np.zeros((self.board_size, self.board_size))

    @staticmethod
    def get_is_legal_actions(board):
        board = board.copy().flatten()
        return board == 0

    def get_next_state(self, board, action, to_play):
        board = deepcopy(board)
        x = action // self.board_size
        y = action % self.board_size
        board[x, y] = to_play
        return board

    @staticmethod
    def get_winner(board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != 0:
                    player = board[i, j]

                    if j + 4 < board.shape[1] and np.all(board[i, j:j + 5] == player):
                        return player
                    if i + 4 < board.shape[0] and np.all(board[i:i + 5, j] == player):
                        return player
                    if i + 4 < board.shape[0] and j + 4 < board.shape[1] and np.all(np.diag(board[i:i + 5, j:j + 5]) == player):
                        return player
                    if i + 4 < board.shape[0] and j - 4 >= 0 and np.all(np.diag(np.fliplr(board[i:i + 5, j - 4:j + 1])) == player):
                        return player

        if np.all(board != 0):
            return 0
        return None

    def is_terminal(self, state):
        return self.get_winner(state) is not None

    @staticmethod
    def encode_state(board, to_play):
        encoded_state = np.zeros((3, *board.shape), dtype=np.float32)

        encoded_state[0] = (board == 1)
        encoded_state[1] = (board == -1)
        encoded_state[2] = (to_play > 0) * 1.0

        return encoded_state
