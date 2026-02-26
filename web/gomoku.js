const C_EMPTY = 0;
const C_BLACK = 1;
const C_WHITE = 2;
const C_WALL = 3;

class ForbiddenPointFinder {
    constructor(size = 15) {
        this.boardSize = size;
        this.cBoard = Array.from({ length: size + 2 }, () => new Int8Array(size + 2).fill(C_WALL));
        this.clear();
    }

    clear() {
        for (let i = 1; i <= this.boardSize; i++) {
            for (let j = 1; j <= this.boardSize; j++) {
                this.cBoard[i][j] = C_EMPTY;
            }
        }
    }

    setStone(x, y, cStone) {
        this.cBoard[x + 1][y + 1] = cStone;
    }

    isFive(x, y, nColor, nDir = null) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;

        if (nDir === null) {
            this.setStone(x, y, nColor);
            let found = false;
            for (let d = 1; d <= 4; d++) {
                let length = this._checkLineLength(x, y, nColor, d);
                if (nColor === C_BLACK) {
                    if (length === 5) { found = true; break; }
                } else {
                    if (length >= 5) { found = true; break; }
                }
            }
            this.setStone(x, y, C_EMPTY);
            return found;
        }

        this.setStone(x, y, nColor);
        let length = this._checkLineLength(x, y, nColor, nDir);
        this.setStone(x, y, C_EMPTY);

        return nColor === C_BLACK ? length === 5 : length >= 5;
    }

    _checkLineLength(x, y, nColor, nDir) {
        let [dx, dy] = this._getDir(nDir);
        let length = 1;

        let i = x + dx, j = y + dy;
        while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
            if (this.cBoard[i + 1][j + 1] === nColor) {
                length++; i += dx; j += dy;
            } else break;
        }

        i = x - dx; j = y - dy;
        while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
            if (this.cBoard[i + 1][j + 1] === nColor) {
                length++; i -= dx; j -= dy;
            } else break;
        }
        return length;
    }

    _getDir(nDir) {
        if (nDir === 1) return [1, 0];
        if (nDir === 2) return [0, 1];
        if (nDir === 3) return [1, 1];
        if (nDir === 4) return [1, -1];
        return [0, 0];
    }

    isOverline(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        this.setStone(x, y, C_BLACK);
        let bOverline = false;
        for (let d = 1; d <= 4; d++) {
            let length = this._checkLineLength(x, y, C_BLACK, d);
            if (length === 5) {
                this.setStone(x, y, C_EMPTY);
                return false;
            } else if (length >= 6) {
                bOverline = true;
            }
        }
        this.setStone(x, y, C_EMPTY);
        return bOverline;
    }

    isFour(x, y, nColor, nDir) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, nColor)) return false;
        if (nColor === C_BLACK && this.isOverline(x, y)) return false;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let found = false;
        for (let sign of [1, -1]) {
            let curDx = dx * sign, curDy = dy * sign;
            let i = x + curDx, j = y + curDy;
            while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
                let c = this.cBoard[i + 1][j + 1];
                if (c === nColor) { i += curDx; j += curDy; }
                else if (c === C_EMPTY) {
                    if (this.isFive(i, j, nColor, nDir)) found = true;
                    break;
                } else break;
            }
            if (found) break;
        }
        this.setStone(x, y, C_EMPTY);
        return found;
    }

    isOpenFour(x, y, nColor, nDir) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return 0;
        if (this.isFive(x, y, nColor)) return 0;
        if (nColor === C_BLACK && this.isOverline(x, y)) return 0;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let nLine = 1;

        let i = x - dx, j = y - dy;
        while (true) {
            if (!(i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize)) {
                this.setStone(x, y, C_EMPTY); return 0;
            }
            let c = this.cBoard[i + 1][j + 1];
            if (c === nColor) { nLine++; i -= dx; j -= dy; }
            else if (c === C_EMPTY) {
                if (!this.isFive(i, j, nColor, nDir)) {
                    this.setStone(x, y, C_EMPTY); return 0;
                }
                break;
            } else { this.setStone(x, y, C_EMPTY); return 0; }
        }

        i = x + dx; j = y + dy;
        while (true) {
            if (!(i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize)) break;
            let c = this.cBoard[i + 1][j + 1];
            if (c === nColor) { nLine++; i += dx; j += dy; }
            else if (c === C_EMPTY) {
                if (this.isFive(i, j, nColor, nDir)) {
                    this.setStone(x, y, C_EMPTY);
                    return nLine === 4 ? 1 : 2;
                }
                break;
            } else break;
        }
        this.setStone(x, y, C_EMPTY);
        return 0;
    }

    isDoubleFour(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, C_BLACK)) return false;
        let nFour = 0;
        for (let d = 1; d <= 4; d++) {
            let ret = this.isOpenFour(x, y, C_BLACK, d);
            if (ret === 2) nFour += 2;
            else if (this.isFour(x, y, C_BLACK, d)) nFour += 1;
        }
        return nFour >= 2;
    }

    isOpenThree(x, y, nColor, nDir) {
        if (this.isFive(x, y, nColor)) return false;
        if (nColor === C_BLACK && this.isOverline(x, y)) return false;

        this.setStone(x, y, nColor);
        let [dx, dy] = this._getDir(nDir);
        let found = false;
        for (let sign of [1, -1]) {
            let curDx = dx * sign, curDy = dy * sign;
            let i = x + curDx, j = y + curDy;
            while (i >= 0 && i < this.boardSize && j >= 0 && j < this.boardSize) {
                let c = this.cBoard[i + 1][j + 1];
                if (c === nColor) { i += curDx; j += curDy; }
                else if (c === C_EMPTY) {
                    if (this.isOpenFour(i, j, nColor, nDir) === 1) {
                        if (nColor === C_BLACK) {
                            // In Renju, an open three must be able to become a legal open four.
                            if (!this.isDoubleFour(i, j) && !this.isDoubleThree(i, j) && !this.isOverline(i, j)) found = true;
                        } else found = true;
                    }
                    break;
                } else break;
            }
            if (found) break;
        }
        this.setStone(x, y, C_EMPTY);
        return found;
    }

    isDoubleThree(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        if (this.isFive(x, y, C_BLACK)) return false;
        let nThree = 0;
        for (let d = 1; d <= 4; d++) {
            if (this.isOpenThree(x, y, C_BLACK, d)) nThree++;
        }
        return nThree >= 2;
    }

    isForbidden(x, y) {
        if (this.cBoard[x + 1][y + 1] !== C_EMPTY) return false;
        let nearbyBlack = 0;
        for (let i = Math.max(0, x - 2); i <= Math.min(this.boardSize - 1, x + 2); i++) {
            for (let j = Math.max(0, y - 2); j <= Math.min(this.boardSize - 1, y + 2); j++) {
                if (i === x && j === y) continue;
                if (this.cBoard[i + 1][j + 1] === C_BLACK) {
                    if (Math.abs(i - x) + Math.abs(j - y) !== 3) nearbyBlack++;
                }
            }
        }
        if (nearbyBlack < 2) return false;
        return this.isDoubleThree(x, y) || this.isDoubleFour(x, y) || this.isOverline(x, y);
    }
}

class Gomoku {
    constructor(boardSize = 15, historyStep = 2, useRenju = true) {
        this.boardSize = boardSize;
        this.historyStep = historyStep;
        this.useRenju = useRenju;
        this.numPlanes = 2 * historyStep + 1;
        this.fpf = new ForbiddenPointFinder(boardSize);
    }

    getInitialState() {
        return Array.from({ length: this.historyStep }, () => new Int8Array(this.boardSize * this.boardSize).fill(0));
    }

    getExpandedRegion(state, k = 2) {
        const currentBoard = state[state.length - 1];
        const expanded = new Uint8Array(this.boardSize * this.boardSize).fill(0);
        let hasStone = false;

        for (let i = 0; i < currentBoard.length; i++) {
            if (currentBoard[i] !== 0) {
                hasStone = true;
                const r = Math.floor(i / this.boardSize);
                const c = i % this.boardSize;
                for (let dr = -k; dr <= k; dr++) {
                    for (let dc = -k; dc <= k; dc++) {
                        const nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < this.boardSize && nc >= 0 && nc < this.boardSize) {
                            expanded[nr * this.boardSize + nc] = 1;
                        }
                    }
                }
            }
        }
        return { expanded, hasStone };
    }

    getLegalActions(state, toPlay, restricted = true) {
        const currentBoard = state[state.length - 1];
        const { expanded, hasStone } = this.getExpandedRegion(state, 2);
        const legalMask = new Uint8Array(this.boardSize * this.boardSize).fill(0);

        if (restricted && !hasStone && this.useRenju) {
            const center = Math.floor(this.boardSize / 2) * this.boardSize + Math.floor(this.boardSize / 2);
            legalMask[center] = 1;
            return legalMask;
        }

        this.fpf.clear();
        for (let i = 0; i < currentBoard.length; i++) {
            if (currentBoard[i] !== 0) {
                const r = Math.floor(i / this.boardSize);
                const c = i % this.boardSize;
                this.fpf.setStone(r, c, currentBoard[i] === 1 ? C_BLACK : C_WHITE);
            }
        }

        for (let i = 0; i < currentBoard.length; i++) {
            if (currentBoard[i] === 0 && (!restricted || expanded[i])) {
                if (this.useRenju && toPlay === 1) {
                    const r = Math.floor(i / this.boardSize);
                    const c = i % this.boardSize;
                    if (!this.fpf.isForbidden(r, c)) legalMask[i] = 1;
                } else {
                    legalMask[i] = 1;
                }
            }
        }
        return legalMask;
    }

    getNextState(state, action, toPlay) {
        const nextState = state.map(layer => new Int8Array(layer));
        const currentBoard = new Int8Array(nextState[nextState.length - 1]);
        currentBoard[action] = toPlay;
        nextState.shift();
        nextState.push(currentBoard);
        return nextState;
    }

    getWinner(state) {
        const board = state[state.length - 1];
        const size = this.boardSize;

        const check = (r, c, dr, dc) => {
            const color = board[r * size + c];
            if (color === 0) return 0;
            for (let k = 1; k < 5; k++) {
                if (board[(r + dr * k) * size + (c + dc * k)] !== color) return 0;
            }
            return color;
        };

        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                if (c <= size - 5 && check(r, c, 0, 1)) return check(r, c, 0, 1);
                if (r <= size - 5 && check(r, c, 1, 0)) return check(r, c, 1, 0);
                if (r <= size - 5 && c <= size - 5 && check(r, c, 1, 1)) return check(r, c, 1, 1);
                if (r <= size - 5 && c >= 4 && check(r, c, 1, -1)) return check(r, c, 1, -1);
            }
        }

        if (board.every(x => x !== 0)) return 0; // Draw
        return null;
    }

    encodeState(state, toPlay) {
        const shape = [this.numPlanes, this.boardSize, this.boardSize];
        const encoded = new Float32Array(shape[0] * shape[1] * shape[2]);
        const layerSize = this.boardSize * this.boardSize;

        for (let i = 0; i < this.historyStep; i++) {
            const layer = state[i];
            for (let j = 0; j < layerSize; j++) {
                if (layer[j] === toPlay) encoded[2 * i * layerSize + j] = 1;
                else if (layer[j] === -toPlay) encoded[(2 * i + 1) * layerSize + j] = 1;
            }
        }
        if (toPlay > 0) {
            for (let j = 0; j < layerSize; j++) encoded[(this.numPlanes - 1) * layerSize + j] = 1;
        }
        return encoded;
    }
}

if (typeof module !== 'undefined') {
    module.exports = { Gomoku, ForbiddenPointFinder };
}
