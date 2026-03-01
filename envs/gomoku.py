import numpy as np
from utils import print_board
import math

def get_expanded_region_square(state, k=2):
    current_board = state[-1]
    board_size = current_board.shape[0]
    
    expanded = np.zeros((board_size, board_size), dtype=bool)
    
    rows, cols = np.where(current_board != 0)
    k = math.ceil(k)
    
    for r, c in zip(rows, cols):
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    expanded[nr, nc] = True
    
    return expanded

def get_expanded_region_circle(state, k=2):
    current_board = state[-1]
    board_size = current_board.shape[0]
    
    expanded = np.zeros((board_size, board_size), dtype=bool)
    
    rows, cols = np.where(current_board != 0)
    
    k_sq = k ** 2
    
    k_int = math.ceil(k)
    
    for r, c in zip(rows, cols):
        for dr in range(-k_int, k_int + 1):
            for dc in range(-k_int, k_int + 1):
                dist_sq = dr**2 + dc**2
                
                if dist_sq <= k_sq:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        expanded[nr, nc] = True
    
    return expanded


C_EMPTY = 0
C_BLACK = 1
C_WHITE = 2
C_WALL = 3

BLACKFIVE = 24
WHITEFIVE = 25
BLACKFORBIDDEN = 26


class Rules:
    BASICRULE_FREESTYLE = 0
    BASICRULE_STANDARD = 1
    BASICRULE_RENJU = 2

    def __init__(self):
        self.basicRule = self.BASICRULE_RENJU
        self.maxMoves = 0
        self.VCNRule = 0
        self.firstPassWin = False


class ForbiddenPointFinder:
    def __init__(self, size=15):
        self.f_boardsize = size
        self.cBoard = [[C_WALL] * (size + 2) for _ in range(size + 2)]
        self.Clear()

    def Clear(self):
        for i in range(self.f_boardsize + 2):
            self.cBoard[0][i] = C_WALL
            self.cBoard[self.f_boardsize + 1][i] = C_WALL
            self.cBoard[i][0] = C_WALL
            self.cBoard[i][self.f_boardsize + 1] = C_WALL

        for i in range(1, self.f_boardsize + 1):
            for j in range(1, self.f_boardsize + 1):
                self.cBoard[i][j] = C_EMPTY

    def SetStone(self, x, y, cStone):
        self.cBoard[x + 1][y + 1] = cStone

    def AddStone(self, x, y, cStone):
        nResult = -1
        if cStone == C_BLACK:
            if self.IsFive(x, y, C_BLACK):
                nResult = BLACKFIVE
            elif self.isForbidden(x, y):
                nResult = BLACKFORBIDDEN
        elif cStone == C_WHITE:
            if self.IsFive(x, y, C_WHITE):
                nResult = WHITEFIVE

        self.SetStone(x, y, cStone)
        return nResult

    def IsFive(self, x, y, nColor, nDir=None):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False

        if nDir is None:
            self.SetStone(x, y, nColor)
            found = False
            for d in range(1, 5):
                length = self._check_line_length(x, y, nColor, d)
                if nColor == C_BLACK:
                    if length == 5:
                        found = True
                        break
                else:
                    if length >= 5:
                        found = True
                        break
            self.SetStone(x, y, C_EMPTY)
            return found

        self.SetStone(x, y, nColor)
        length = self._check_line_length(x, y, nColor, nDir)
        self.SetStone(x, y, C_EMPTY)

        if nColor == C_BLACK:
            return length == 5
        else:
            return length >= 5

    def _check_line_length(self, x, y, nColor, nDir):
        dx, dy = self._get_dir(nDir)
        length = 1

        # Positive direction
        i, j = x + dx, y + dy
        while 0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize:
            if self.cBoard[i + 1][j + 1] == nColor:
                length += 1
                i += dx
                j += dy
            else:
                break

        # Negative direction
        i, j = x - dx, y - dy
        while 0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize:
            if self.cBoard[i + 1][j + 1] == nColor:
                length += 1
                i -= dx
                j -= dy
            else:
                break
        return length

    def _get_dir(self, nDir):
        if nDir == 1: return (1, 0)  # Horizontal
        if nDir == 2: return (0, 1)  # Vertical
        if nDir == 3: return (1, 1)  # Diagonal /
        if nDir == 4: return (1, -1)  # Diagonal \
        return (0, 0)

    def IsOverline(self, x, y):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False

        self.SetStone(x, y, C_BLACK)
        bOverline = False
        for d in range(1, 5):
            length = self._check_line_length(x, y, C_BLACK, d)
            if length == 5:
                self.SetStone(x, y, C_EMPTY)
                return False
            elif length >= 6:
                bOverline = True
        self.SetStone(x, y, C_EMPTY)
        return bOverline

    def IsFour(self, x, y, nColor, nDir):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False
        if self.IsFive(x, y, nColor):
            return False
        if nColor == C_BLACK and self.IsOverline(x, y):
            return False

        self.SetStone(x, y, nColor)
        dx, dy = self._get_dir(nDir)

        found = False
        for sign in [1, -1]:
            curr_dx, curr_dy = dx * sign, dy * sign
            i, j = x + curr_dx, y + curr_dy
            while 0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize:
                c = self.cBoard[i + 1][j + 1]
                if c == nColor:
                    i += curr_dx
                    j += curr_dy
                elif c == C_EMPTY:
                    if self.IsFive(i, j, nColor, nDir):
                        found = True
                    break
                else:
                    break
            if found: break

        self.SetStone(x, y, C_EMPTY)
        return found

    def IsOpenFour(self, x, y, nColor, nDir):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return 0
        if self.IsFive(x, y, nColor):
            return 0
        if nColor == C_BLACK and self.IsOverline(x, y):
            return 0

        self.SetStone(x, y, nColor)
        dx, dy = self._get_dir(nDir)
        nLine = 1

        # Check negative direction
        i, j = x - dx, y - dy
        while True:
            if not (0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize):
                self.SetStone(x, y, C_EMPTY)
                return 0
            c = self.cBoard[i + 1][j + 1]
            if c == nColor:
                nLine += 1
                i -= dx
                j -= dy
            elif c == C_EMPTY:
                if not self.IsFive(i, j, nColor, nDir):
                    self.SetStone(x, y, C_EMPTY)
                    return 0
                break
            else:  # Wall or opponent
                self.SetStone(x, y, C_EMPTY)
                return 0

        # Check positive direction
        i, j = x + dx, y + dy
        while True:
            if not (0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize):
                break  # Hit wall
            c = self.cBoard[i + 1][j + 1]
            if c == nColor:
                nLine += 1
                i += dx
                j += dy
            elif c == C_EMPTY:
                if self.IsFive(i, j, nColor, nDir):
                    self.SetStone(x, y, C_EMPTY)
                    return 1 if nLine == 4 else 2
                break
            else:
                break

        self.SetStone(x, y, C_EMPTY)
        return 0

    def IsDoubleFour(self, x, y):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False
        if self.IsFive(x, y, C_BLACK):
            return False

        nFour = 0
        for d in range(1, 5):
            ret = self.IsOpenFour(x, y, C_BLACK, d)
            if ret == 2:
                nFour += 2
            elif self.IsFour(x, y, C_BLACK, d):
                nFour += 1

        return nFour >= 2

    def IsOpenThree(self, x, y, nColor, nDir):
        if self.IsFive(x, y, nColor):
            return False
        if nColor == C_BLACK and self.IsOverline(x, y):
            return False

        self.SetStone(x, y, nColor)
        dx, dy = self._get_dir(nDir)

        found = False
        for sign in [1, -1]:
            curr_dx, curr_dy = dx * sign, dy * sign
            i, j = x + curr_dx, y + curr_dy
            while 0 <= i < self.f_boardsize and 0 <= j < self.f_boardsize:
                c = self.cBoard[i + 1][j + 1]
                if c == nColor:
                    i += curr_dx
                    j += curr_dy
                elif c == C_EMPTY:
                    # Check if placing at (i, j) makes an open four that is not a double three/four forbidden point
                    if self.IsOpenFour(i, j, nColor, nDir) == 1:
                        if nColor == C_BLACK:
                            if not self.IsDoubleFour(i, j) and not self.IsDoubleThree(i, j):
                                found = True
                        else:
                            found = True
                    break
                else:
                    break
            if found: break

        self.SetStone(x, y, C_EMPTY)
        return found

    def IsDoubleThree(self, x, y):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False
        if self.IsFive(x, y, C_BLACK):
            return False

        nThree = 0
        for d in range(1, 5):
            if self.IsOpenThree(x, y, C_BLACK, d):
                nThree += 1

        return nThree >= 2

    def isForbidden(self, x, y):
        if self.cBoard[x + 1][y + 1] != C_EMPTY:
            return False

        nearbyBlack = 0
        # Check nearby stones to optimize
        for i in range(max(0, x - 2), min(self.f_boardsize - 1, x + 2) + 1):
            for j in range(max(0, y - 2), min(self.f_boardsize - 1, y + 2) + 1):
                if i == x and j == y: continue
                if self.cBoard[i + 1][j + 1] == C_BLACK:
                    xd, yd = abs(i - x), abs(j - y)
                    if (xd + yd) != 3:  # Exclude knight"s move points as in C++
                        nearbyBlack += 1

        if nearbyBlack < 2: return False
        return self.isForbiddenNoNearbyCheck(x, y)

    def isForbiddenNoNearbyCheck(self, x, y):
        if self.IsDoubleThree(x, y) or self.IsDoubleFour(x, y) or self.IsOverline(x, y):
            return True
        return False


class Board:
    def __init__(self, size=15):
        self.x_size = size
        self.y_size = size
        self.colors = [C_EMPTY] * (size * size)
        self.movenum = 0
        self.blackPassNum = 0
        self.whitePassNum = 0

    def isOnBoard(self, loc):
        return 0 <= loc < self.x_size * self.y_size

    def isLegal(self, loc, pla):
        if not self.isOnBoard(loc): return False
        if self.colors[loc] != C_EMPTY: return False
        return True

    def get_xy(self, loc):
        return loc % self.x_size, loc // self.x_size

    def get_loc(self, x, y):
        return y * self.x_size + x

    def isForbidden(self, loc):
        x, y = self.get_xy(loc)
        fpf = ForbiddenPointFinder(self.x_size)
        for i in range(self.x_size * self.y_size):
            cx, cy = self.get_xy(i)
            # Ensure we don"t count the stone at "loc" for the forbidden check
            if i == loc:
                fpf.SetStone(cx, cy, C_EMPTY)
            else:
                fpf.SetStone(cx, cy, self.colors[i])
        return fpf.isForbidden(x, y)


class GameLogic:
    MP_ILLEGAL = -1
    MP_NORMAL = 0
    MP_MYLIFEFOUR = 1
    MP_OPPOFOUR = 2
    MP_FIVE = 3
    MP_SUDDEN_WIN = 4
    MP_WINNING = 5
    MP_ONLY_NONLOSE_MOVES = 6

    @staticmethod
    def getOpp(pla):
        return C_WHITE if pla == C_BLACK else C_BLACK

    @staticmethod
    def connectionLengthOneDirection(board, pla, isSixWin, loc, adj):
        tmploc = loc
        conNum = 0
        isLife = False

        while True:
            tmploc += adj
            if not board.isOnBoard(tmploc): break
            if board.colors[tmploc] == pla:
                conNum += 1
            elif board.colors[tmploc] == C_EMPTY:
                isLife = True
                if not isSixWin:
                    tmploc2 = tmploc + adj
                    if board.isOnBoard(tmploc2) and board.colors[tmploc2] == pla:
                        isLife = False
                break
            else:
                break
        return conNum, isLife

    @staticmethod
    def getMovePriorityOneDirection(board, rules, pla, loc, adj):
        opp = GameLogic.getOpp(pla)
        isSixWinMe = (rules.basicRule == Rules.BASICRULE_FREESTYLE or (rules.basicRule == Rules.BASICRULE_RENJU and pla == C_WHITE))
        isSixWinOpp = (rules.basicRule == Rules.BASICRULE_FREESTYLE or (rules.basicRule == Rules.BASICRULE_RENJU and pla == C_BLACK))

        myConNum1, myLife1 = GameLogic.connectionLengthOneDirection(board, pla, isSixWinMe, loc, adj)
        myConNum2, myLife2 = GameLogic.connectionLengthOneDirection(board, pla, isSixWinMe, loc, -adj)
        myConNum = myConNum1 + myConNum2 + 1

        oppConNum1, oppLife1 = GameLogic.connectionLengthOneDirection(board, opp, isSixWinOpp, loc, adj)
        oppConNum2, oppLife2 = GameLogic.connectionLengthOneDirection(board, opp, isSixWinOpp, loc, -adj)
        oppConNum = oppConNum1 + oppConNum2 + 1

        if myConNum == 5 or (myConNum > 5 and isSixWinMe):
            return GameLogic.MP_SUDDEN_WIN
        if oppConNum == 5 or (oppConNum > 5 and isSixWinOpp):
            return GameLogic.MP_ONLY_NONLOSE_MOVES
        if myConNum == 4 and myLife1 and myLife2:
            return GameLogic.MP_WINNING
        return GameLogic.MP_NORMAL

    @staticmethod
    def getMovePriorityAssumeLegal(board, rules, pla, loc):
        adjs = [1, board.x_size, board.x_size + 1, board.x_size - 1]
        mp = GameLogic.MP_NORMAL
        for adj in adjs:
            tmpMP = GameLogic.getMovePriorityOneDirection(board, rules, pla, loc, adj)
            if tmpMP > mp:  # Higher priority is better
                mp = tmpMP
        return mp

    @staticmethod
    def checkWinnerAfterPlayed(board, rules, pla, loc):
        if loc == -1: return C_EMPTY

        if rules.basicRule == Rules.BASICRULE_RENJU and pla == C_BLACK:
            if board.isForbidden(loc):
                return GameLogic.getOpp(pla)

        isSixWinMe = (rules.basicRule == Rules.BASICRULE_FREESTYLE or (rules.basicRule == Rules.BASICRULE_RENJU and pla == C_WHITE))

        adjs = [1, board.x_size, board.x_size + 1, board.x_size - 1]
        for adj in adjs:
            conNum1, _ = GameLogic.connectionLengthOneDirection(board, pla, isSixWinMe, loc, adj)
            conNum2, _ = GameLogic.connectionLengthOneDirection(board, pla, isSixWinMe, loc, -adj)
            total = conNum1 + conNum2 + 1
            if total == 5 or (total > 5 and isSixWinMe):
                return pla
        return C_EMPTY


class Gomoku:
    def __init__(self, board_size=15, history_step=2, use_renju=True):
        self.board_size = board_size
        self.history_step = history_step
        self.num_planes = 2 * history_step + 1
        self.use_renju = use_renju
        self._fpf = ForbiddenPointFinder(board_size)

    def get_initial_state(self):
        return np.zeros((self.history_step, self.board_size, self.board_size), dtype=np.int8)

    def get_is_legal_actions(self, state, to_play):
        current_board = state[-1]
        legal_mask = (current_board.flatten() == 0)

        # Heuristic legal action limitation
        if np.sum(current_board == 1) == 0 and self.use_renju:
            self.center_loc = (self.board_size // 2) * self.board_size + (self.board_size // 2)
            legal_mask[:] = False
            legal_mask[self.center_loc] = True
        else:
            legal_mask = legal_mask & get_expanded_region_circle(state, k=3.5).flatten()

        # In Renju, only Black (1) has forbidden moves
        if self.use_renju and to_play == 1:
            self._fpf.Clear()
            # Populate FPF board (convert 1/-1 to 1/2)
            rows, cols = np.where(current_board != 0)
            for r, c in zip(rows, cols):
                val = current_board[r, c]
                stone = C_BLACK if val == 1 else C_WHITE
                self._fpf.SetStone(r, c, stone)

            # Check forbidden moves for all empty spots
            # Optimizing: only check indices that are currently legal (empty)
            indices = np.where(legal_mask)[0]
            for idx in indices:
                r, c = idx // self.board_size, idx % self.board_size
                if self._fpf.isForbidden(r, c):
                    legal_mask[idx] = False

        return legal_mask

    def get_next_state(self, state, action, to_play):
        state = state.copy()
        current_board = state[-1].copy()

        row = action // self.board_size
        col = action % self.board_size
        
        current_board[row, col] = to_play

        state[:-1] = state[1:]
        state[-1] = current_board

        return state

    def get_winner(self, state):
        current_board = state[-1]
        size = self.board_size

        # Check horizontal
        for r in range(size):
            for c in range(size - 4):
                window = current_board[r, c:c + 5]
                s = np.sum(window)
                if s == 5 or s <= -5:
                    return 1 if s > 0 else -1

        # Check vertical
        for r in range(size - 4):
            for c in range(size):
                window = current_board[r:r + 5, c]
                s = np.sum(window)
                if abs(s) == 5:
                    return 1 if s > 0 else -1

        # Check diagonal \
        for r in range(size - 4):
            for c in range(size - 4):
                s = 0
                for k in range(5):
                    s += current_board[r + k, c + k]
                if abs(s) == 5:
                    return 1 if s > 0 else -1

        # Check diagonal /
        for r in range(size - 4):
            for c in range(4, size):
                s = 0
                for k in range(5):
                    s += current_board[r + k, c - k]
                if abs(s) == 5:
                    return 1 if s > 0 else -1

        if np.all(current_board != 0):
            return 0

        return None

    def is_terminal(self, state):
        return self.get_winner(state) is not None

    @staticmethod
    def encode_state(state, to_play):
        history_len = state.shape[0]
        board_height = state.shape[1]
        board_width = state.shape[2]

        encoded_state = np.zeros((history_len * 2 + 1, board_height, board_width), dtype=np.int8)

        for i in range(history_len):
            encoded_state[2 * i] = (state[i] == to_play)
            encoded_state[2 * i + 1] = (state[i] == -to_play)

        encoded_state[-1] = (to_play > 0) * np.ones((board_height, board_width), dtype=np.int8)

        return encoded_state
