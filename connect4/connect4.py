"""
Connect4 (四子棋) 游戏逻辑
棋盘大小: 6行 x 7列
获胜条件: 横、竖、斜连成4子
"""

import numpy as np


class Connect4:
    def __init__(self, history_step=3):
        self.board_height = 6  # 行数
        self.board_width = 7   # 列数
        self.board_size = self.board_width  # 兼容性（用于某些显示）
        self.action_space_size = self.board_width  # Connect4只能选择列（7个动作）
        self.history_step = history_step
        self.num_planes = 2 * history_step + 1

    def get_initial_state(self):
        """返回初始状态"""
        return np.zeros((self.history_step, self.board_height, self.board_width))

    def get_is_legal_actions(self, state):
        """
        返回合法动作掩码
        在Connect4中，如果某列最顶端（第0行）为空，则该列可以落子
        """
        current_board = state[-1]
        # 检查每列的顶部是否为空
        return current_board[0, :] == 0

    def get_next_state(self, state, action, to_play):
        """
        执行动作，返回新状态
        action: 选择的列号 (0-6)
        棋子会落到该列的最低空位
        """
        state = state.copy()
        current_board = state[-1].copy()
        
        # 找到该列最低的空位
        col = action
        for row in range(self.board_height - 1, -1, -1):
            if current_board[row, col] == 0:
                current_board[row, col] = to_play
                break
        
        # 更新历史记录
        state[:-1] = state[1:]
        state[-1] = current_board
        
        return state

    def _check_winner(self, board):
        """检查是否有玩家获胜"""
        # 检查横向
        for row in range(self.board_height):
            for col in range(self.board_width - 3):
                window = board[row, col:col+4]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1
        
        # 检查纵向
        for row in range(self.board_height - 3):
            for col in range(self.board_width):
                window = board[row:row+4, col]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1
        
        # 检查右下斜线
        for row in range(self.board_height - 3):
            for col in range(self.board_width - 3):
                window = [board[row+i, col+i] for i in range(4)]
                if all(w == 1 for w in window):
                    return 1
                if all(w == -1 for w in window):
                    return -1
        
        # 检查左下斜线
        for row in range(self.board_height - 3):
            for col in range(3, self.board_width):
                window = [board[row+i, col-i] for i in range(4)]
                if all(w == 1 for w in window):
                    return 1
                if all(w == -1 for w in window):
                    return -1
        
        return None

    def get_winner(self, state):
        """获取获胜者"""
        current_board = state[-1]
        winner = self._check_winner(current_board)
        if winner is not None:
            return winner
        
        # 检查是否平局（棋盘满了）
        if np.all(current_board != 0):
            return 0
        
        return None

    def is_terminal(self, state):
        """检查游戏是否结束"""
        current_board = state[-1]
        # 有获胜者或棋盘满了
        return (self._check_winner(current_board) is not None 
                or np.all(current_board != 0))

    @staticmethod
    def encode_state(state, to_play):
        """
        编码状态用于神经网络输入
        state.shape = (history_step, board_height, board_width)
        """
        history_len = state.shape[0]
        board_height = state.shape[1]
        board_width = state.shape[2]
        
        encoded_state = np.zeros((history_len * 2 + 1, board_height, board_width), dtype=np.float32)
        
        for i in range(history_len):
            encoded_state[2 * i] = (state[i] == to_play)
            encoded_state[2 * i + 1] = (state[i] == -to_play)
        
        encoded_state[-1] = (to_play > 0) * np.ones((board_height, board_width), dtype=np.float32)
        
        return encoded_state


def print_board(board):
    """打印Connect4棋盘"""
    current_board = board[-1] if board.ndim == 3 else board
    rows, cols = current_board.shape
    
    # 打印列号
    print("   ", end="")
    for col in range(cols):
        print(f"{col:2d} ", end="")
    print()
    
    # 打印棋盘
    for row in range(rows):
        print(f"{row:2d} ", end="")
        for col in range(cols):
            if current_board[row, col] == 1:
                print(" ● ", end="")  # 玩家1
            elif current_board[row, col] == -1:
                print(" ○ ", end="")  # 玩家2
            else:
                print(" · ", end="")
        print()
    
    # 打印底部分隔线
    print("   " + "---" * cols)


if __name__ == '__main__':
    game = Connect4(history_step=3)
    state = game.get_initial_state()
    print("初始状态:")
    print_board(state)
    print(f"合法动作: {game.get_is_legal_actions(state)}")
    
    # 测试落子
    state = game.get_next_state(state, 3, 1)  # 玩家1落在第3列
    state = game.get_next_state(state, 3, -1) # 玩家2落在第3列
    state = game.get_next_state(state, 4, 1)  # 玩家1落在第4列
    print("\n落子后:")
    print_board(state)
    print(f"合法动作: {game.get_is_legal_actions(state)}")
    print(f"编码状态形状: {game.encode_state(state, to_play=1).shape}")
