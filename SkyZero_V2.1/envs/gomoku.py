import numpy as np
import os


def is_near_occupied(board, r, c, dist, board_size):
    """Check if any stone exists within a (2*dist+1) x (2*dist+1) square centered at (r, c)."""
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if board[nr, nc] != 0:
                    return True
    return False


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
                            if not self.IsDoubleFour(i, j) and not self.IsDoubleThree(i, j) and not self.IsOverline(i, j):
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
    def __init__(self, board_size=15, use_renju=True, enable_forbidden_point_plane=True):
        self.board_size = board_size
        self.num_planes = 3 + (1 if enable_forbidden_point_plane else 0)
        self.use_renju = use_renju
        self.enable_forbidden_point_plane = enable_forbidden_point_plane
        self._fpf = ForbiddenPointFinder(board_size)
        self.openings_ = []
        self.opening_weights_ = []
        self.empty_board_prob_ = 1.0

    def load_openings(self, path, empty_board_prob=0.0):
        """Load opening book from file matching C++ load_openings logic."""
        self.empty_board_prob_ = empty_board_prob
        if not os.path.isfile(path):
            print(f"Warning: Could not open opening file: {path}. Using empty board only.")
            self.empty_board_prob_ = 1.0
            return

        with open(path, 'r') as f:
            lines = f.readlines()

        # Parse triplets: (weight_str, black_str, white_str)
        raw_list = []
        i = 0
        while i + 2 < len(lines):
            weight_str = lines[i].strip()
            black_str = lines[i + 1].strip()
            white_str = lines[i + 2].strip()
            raw_list.append((weight_str, black_str, white_str))
            i += 3

        # Calculate implicit weights
        sum_explicit = 0.0
        num_implicit = 0
        for weight_str, _, _ in raw_list:
            if weight_str == '':
                num_implicit += 1
            else:
                try:
                    sum_explicit += float(weight_str)
                except ValueError:
                    num_implicit += 1

        implicit_weight = 0.0
        if num_implicit > 0:
            implicit_weight = max(0.0, (1.0 - sum_explicit) / num_implicit)

        center = self.board_size // 2
        for weight_str, black_str, white_str in raw_list:
            board = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
            stones = 0

            # Place black stones
            if black_str:
                for coord in black_str.split():
                    if ',' in coord:
                        dx, dy = coord.split(',')
                        r = center + int(dx)
                        c = center + int(dy)
                        if 0 <= r < self.board_size and 0 <= c < self.board_size:
                            board[0, r, c] = 1
                            stones += 1

            # Place white stones
            if white_str:
                for coord in white_str.split():
                    if ',' in coord:
                        dx, dy = coord.split(',')
                        r = center + int(dx)
                        c = center + int(dy)
                        if 0 <= r < self.board_size and 0 <= c < self.board_size:
                            board[0, r, c] = -1
                            stones += 1

            to_play = 1 if stones % 2 == 0 else -1
            self.openings_.append((board, to_play))

            # Parse weight
            w = 0.0
            if weight_str == '':
                w = implicit_weight
            else:
                try:
                    w = float(weight_str)
                except ValueError:
                    w = implicit_weight
            self.opening_weights_.append(w)

        if len(self.openings_) == 0:
            self.empty_board_prob_ = 1.0

    def get_initial_state(self):
        """Return (state, to_play) tuple. May randomly select from opening book."""
        empty_state = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
        if len(self.openings_) == 0:
            return empty_state, 1

        if np.random.rand() < self.empty_board_prob_:
            return empty_state, 1

        # Weighted random selection
        weights = np.array(self.opening_weights_, dtype=np.float64)
        weights /= weights.sum()
        idx = np.random.choice(len(self.openings_), p=weights)
        board, to_play = self.openings_[idx]
        return board.copy(), to_play

    def get_is_legal_actions(self, state, to_play):
        current_board = state[-1]
        size = self.board_size
        legal_mask = np.zeros(size * size, dtype=bool)

        # Empty board: only center is legal
        if np.all(current_board == 0):
            center_loc = (size // 2) * size + (size // 2)
            legal_mask[center_loc] = True
            return legal_mask

        # Non-empty: empty positions within distance 3 of any occupied stone
        for r in range(size):
            for c in range(size):
                loc = r * size + c
                if current_board[r, c] != 0:
                    continue
                if is_near_occupied(current_board, r, c, 3, size):
                    legal_mask[loc] = True

        # Forbidden points are legal moves for Black, but playing on one
        # results in an immediate loss (checked in get_winner).

        return legal_mask

    def get_next_state(self, state, action, to_play):
        state = state.copy()

        row = action // self.board_size
        col = action % self.board_size
        
        state[0, row, col] = to_play

        return state

    def get_winner(self, state, last_action=None, last_player=None):
        current_board = state[-1]
        size = self.board_size

        # Check if Black's last move is on a forbidden point -> White wins
        if self.use_renju and last_action is not None and last_player == 1:
            row = last_action // size
            col = last_action % size
            fpf = ForbiddenPointFinder(size)
            for i in range(size * size):
                r_i = i // size
                c_i = i % size
                if i == last_action or current_board[r_i, c_i] == 0:
                    continue
                stone = C_BLACK if current_board[r_i, c_i] == 1 else C_WHITE
                fpf.SetStone(r_i, c_i, stone)
            if fpf.isForbidden(row, col):
                return -1

        # Scan all positions, check lines in 4 directions
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(size):
            for c in range(size):
                stone = current_board[r, c]
                if stone == 0:
                    continue
                for dr, dc in dirs:
                    # Skip if preceded by the same stone (avoid counting the same line twice)
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < size and 0 <= pc < size and current_board[pr, pc] == stone:
                        continue
                    # Count contiguous length
                    length = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < size and 0 <= nc < size and current_board[nr, nc] == stone:
                        length += 1
                        nr += dr
                        nc += dc
                    # Black wins with exactly 5 (overline is forbidden)
                    if stone == 1 and length == 5:
                        return 1
                    # White wins with 5 or more
                    if stone == -1 and length >= 5:
                        return -1

        if np.all(current_board != 0):
            return 0

        return None

    def is_terminal(self, state, last_action=None, last_player=None):
        return self.get_winner(state, last_action, last_player) is not None

    def encode_state(self, state, to_play):
        encoded_state = np.zeros((self.num_planes, self.board_size, self.board_size), dtype=np.int8)

        # Plane 0: current player's stones
        # Plane 1: opponent's stones
        encoded_state[0] = (state[0] == to_play)
        encoded_state[1] = (state[0] == -to_play)

        # Plane 2: forbidden points when to_play == 1 (Black)
        # Plane 3: forbidden points when to_play == -1 (White)
        # The populated plane implicitly indicates whose turn it is.
        if self.enable_forbidden_point_plane and self.use_renju:
            fpf = ForbiddenPointFinder(self.board_size)
            current_board = state[-1]
            rows, cols = np.where(current_board != 0)
            for r, c in zip(rows, cols):
                stone = C_BLACK if current_board[r, c] == 1 else C_WHITE
                fpf.SetStone(r, c, stone)

            forbidden_plane = 2 if to_play == 1 else 3
            empty_rows, empty_cols = np.where(current_board == 0)
            for r, c in zip(empty_rows, empty_cols):
                if fpf.isForbidden(r, c):
                    encoded_state[forbidden_plane, r, c] = 1

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

        # Plane 2: forbidden points when tp == 1 (Black)
        # Plane 3: forbidden points when tp == -1 (White)
        if self.enable_forbidden_point_plane and self.use_renju:
            fpf = ForbiddenPointFinder(self.board_size)
            for b in range(B):
                board = boards[b]  # (H, W)
                fpf.Clear()
                rows, cols = np.where(board != 0)
                for r, c in zip(rows, cols):
                    stone = C_BLACK if board[r, c] == 1 else C_WHITE
                    fpf.SetStone(r, c, stone)

                forbidden_plane = 2 if to_plays[b] == 1 else 3
                empty_rows, empty_cols = np.where(board == 0)
                for r, c in zip(empty_rows, empty_cols):
                    if fpf.isForbidden(r, c):
                        encoded[b, forbidden_plane, r, c] = 1

        return encoded

    def get_win_pos(self, final_state):
        current_board = final_state[-1]
        size = self.board_size
        five_pos = np.zeros((size, size), dtype=np.int8)
        
        # Check horizontal
        for r in range(size):
            for c in range(size - 4):
                window = current_board[r, c:c + 5]
                s = np.sum(window)
                if s == 5 or s == -5:
                    five_pos[r, c:c + 5] = 1
        
        # Check vertical
        for r in range(size - 4):
            for c in range(size):
                window = current_board[r:r + 5, c]
                s = np.sum(window)
                if s == 5 or s == -5:
                    five_pos[r:r + 5, c] = 1
        
        # Check diagonal \
        for r in range(size - 4):
            for c in range(size - 4):
                s = current_board[r, c] + current_board[r+1, c+1] + current_board[r+2, c+2] + current_board[r+3, c+3] + current_board[r+4, c+4]
                if s == 5 or s == -5:
                    for k in range(5):
                        five_pos[r+k, c+k] = 1
        
        # Check diagonal /
        for r in range(size - 4):
            for c in range(4, size):
                s = current_board[r, c] + current_board[r+1, c-1] + current_board[r+2, c-2] + current_board[r+3, c-3] + current_board[r+4, c-4]
                if s == 5 or s == -5:
                    for k in range(5):
                        five_pos[r+k, c-k] = 1
        
        return five_pos

if __name__ == "__main__":
    def print_board(board):
        current_board = board[-1] if board.ndim == 3 else board
        rows, cols = current_board.shape

        print("   ", end="")
        for col in range(cols):
            print(f"{col:2d} ", end="")
        print()

        for row in range(rows):
            print(f"{row:2d} ", end="")
            for col in range(cols):
                if current_board[row, col] == 1:
                    print(" × ", end="")
                elif current_board[row, col] == -1:
                    print(" ○ ", end="")
                else:
                    print(" · ", end="")
            print()
    env = Gomoku(board_size=15, use_renju=True)
    state, to_play = env.get_initial_state()
    state = env.get_next_state(state, action=112, to_play=1)
    print(state.shape)
    print(env.encode_state(state, to_play=-1))
