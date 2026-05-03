const C_EMPTY = 0;
const C_BLACK = 1;
const C_WHITE = 2;
const C_WALL = 3;

const MAX_BOARD_SIZE = 17;     // V5 MAX_BOARD_SIZE
const MAX_AREA = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
const NUM_SPATIAL_PLANES = 5;

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

const RULE_RENJU     = "renju";
const RULE_STANDARD  = "standard";
const RULE_FREESTYLE = "freestyle";
const VALID_RULES    = [RULE_FREESTYLE, RULE_STANDARD, RULE_RENJU];

class Gomoku {
    constructor(boardSize, rule = RULE_RENJU) {
        if (!VALID_RULES.includes(rule)) {
            throw new Error("Gomoku: unknown rule '" + rule + "' (expected freestyle|standard|renju)");
        }
        this.boardSize = boardSize;
        this.area = boardSize * boardSize;
        this.rule = rule;
        this.fpf = new ForbiddenPointFinder(boardSize);
    }

    get hasForbidden() { return this.rule !== RULE_FREESTYLE; }

    getInitialState() {
        return new Int8Array(this.area);
    }

    getNextState(state, action, toPlay) {
        const out = new Int8Array(state);
        out[action] = toPlay;
        return out;
    }

    // Returns Uint8Array(area), 1 = legal, 0 = illegal.
    // V5 semantics (cpp/envs/gomoku.h::get_is_legal_actions): any empty cell is
    // legal, regardless of rule. Forbidden points are *legal but losing* — that
    // verdict is delivered by getWinner on the just-played move.
    getLegalActions(state, _toPlay) {
        const out = new Uint8Array(this.area);
        for (let i = 0; i < this.area; i++) out[i] = (state[i] === 0) ? 1 : 0;
        return out;
    }

    /**
     * Returns +1 (black wins), -1 (white wins), 0 (draw, board full),
     * or null (ongoing). Mirrors V5 cpp/envs/gomoku.h::get_winner_v5.
     *
     * lastAction / lastPlayer describe the most recent move; required when
     * rule has forbidden semantics (renju full-forbidden, standard long-row).
     */
    getWinner(state, lastAction, lastPlayer) {
        const N = this.boardSize;

        // Step 1: forbidden-move detection on the just-played move by black.
        if (lastAction != null && lastPlayer === 1 && this.rule !== RULE_FREESTYLE) {
            const r = (lastAction / N) | 0;
            const c = lastAction % N;
            if (this.rule === RULE_RENJU) {
                // FPF.isForbidden expects (r,c) to be empty: build from board minus lastAction.
                this.fpf.clear();
                for (let i = 0; i < this.area; i++) {
                    if (i === lastAction || state[i] === 0) continue;
                    const sr = (i / N) | 0, sc = i % N;
                    this.fpf.setStone(sr, sc, state[i] === 1 ? C_BLACK : C_WHITE);
                }
                if (this.fpf.isForbidden(r, c)) return -1;
            } else {  // STANDARD: only overline (≥6) is forbidden for black
                if (this._isOverlineAt(state, r, c, 1)) return -1;
            }
        }

        // Step 2: scan for any run of length matching the per-color win condition.
        const dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                const stone = state[r * N + c];
                if (stone === 0) continue;
                for (const [dr, dc] of dirs) {
                    // Avoid double-counting: only start runs at the leftmost / topmost end.
                    const pr = r - dr, pc = c - dc;
                    if (pr >= 0 && pr < N && pc >= 0 && pc < N && state[pr * N + pc] === stone) continue;

                    let len = 1;
                    let nr = r + dr, nc = c + dc;
                    while (nr >= 0 && nr < N && nc >= 0 && nc < N && state[nr * N + nc] === stone) {
                        len++;
                        nr += dr;
                        nc += dc;
                    }
                    if (stone === 1) {
                        // Black: freestyle wins on 5+, standard/renju only on exactly 5
                        // (overline already handled in step 1 for the latter two).
                        if (this.rule === RULE_FREESTYLE) {
                            if (len >= 5) return 1;
                        } else {
                            if (len === 5) return 1;
                        }
                    } else {  // White always wins on 5+ across all rules
                        if (len >= 5) return -1;
                    }
                }
            }
        }

        // Step 3: draw if board is full.
        for (let i = 0; i < this.area; i++) if (state[i] === 0) return null;
        return 0;
    }

    // Helper: does (r, c) of `color` belong to a run of length >= 6?
    _isOverlineAt(state, r, c, color) {
        const N = this.boardSize;
        const dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
        for (const [dr, dc] of dirs) {
            let len = 1;
            for (let k = 1; ; k++) {
                const nr = r + dr * k, nc = c + dc * k;
                if (nr < 0 || nr >= N || nc < 0 || nc >= N || state[nr * N + nc] !== color) break;
                len++;
            }
            for (let k = 1; ; k++) {
                const nr = r - dr * k, nc = c - dc * k;
                if (nr < 0 || nr >= N || nc < 0 || nc >= N || state[nr * N + nc] !== color) break;
                len++;
            }
            if (len >= 6) return true;
        }
        return false;
    }

    /**
     * V5 encode: 5 planes padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE = 17×17.
     * Plane 0: on-board mask (1 inside [0, boardSize), 0 in padding)
     * Plane 1: own stones
     * Plane 2: opponent stones
     * Plane 3: forbidden points when current player is BLACK (rule != freestyle)
     * Plane 4: forbidden points when current player is WHITE
     *
     * Output is Float32Array (model expects float32 input).
     *
     * STANDARD note (matches V5 gomoku.h::encode_state_v5): we still write the
     * full FPF result into the forbidden plane. Only long-row is *enforced* by
     * the winner check, but the plane carries the same patterns and the
     * network leans on the rule one-hot (global features) to weigh them.
     */
    encodeState(state, toPlay) {
        const N = this.boardSize;
        const M = MAX_BOARD_SIZE;
        const A = MAX_AREA;
        const out = new Float32Array(NUM_SPATIAL_PLANES * A);

        // Plane 0: mask
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                out[0 * A + r * M + c] = 1;
            }
        }

        // Planes 1-2: own / opp
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                const s = state[r * N + c];
                const dst = r * M + c;
                if (s === toPlay)       out[1 * A + dst] = 1;
                else if (s === -toPlay) out[2 * A + dst] = 1;
            }
        }

        // Planes 3-4: forbidden (skip entirely under FREESTYLE)
        if (this.hasForbidden) {
            this.fpf.clear();
            for (let i = 0; i < this.area; i++) {
                if (state[i] === 0) continue;
                const r = (i / N) | 0, c = i % N;
                this.fpf.setStone(r, c, state[i] === 1 ? C_BLACK : C_WHITE);
            }
            const fbPlane = (toPlay === 1) ? 3 : 4;
            for (let r = 0; r < N; r++) {
                for (let c = 0; c < N; c++) {
                    if (state[r * N + c] !== 0) continue;
                    if (this.fpf.isForbidden(r, c)) {
                        out[fbPlane * A + r * M + c] = 1;
                    }
                }
            }
        }

        return out;
    }

    /**
     * 12-dim global features (KataGoNet.linear_global input).
     * Layout matches V5 gomoku.h::compute_global_features:
     *   [0..2] rule one-hot (freestyle, standard, renju)
     *   [3]    renju_color_sign (only fires under RENJU; black=-1, white=+1)
     *   [4]    has_forbidden (1 iff rule != freestyle)
     *   [5]    ply / board_area
     *   [6..11] VCF placeholders (zero)
     */
    computeGlobalFeatures(ply, toPlay) {
        const f = new Float32Array(12);
        f[0] = (this.rule === RULE_FREESTYLE) ? 1 : 0;
        f[1] = (this.rule === RULE_STANDARD)  ? 1 : 0;
        f[2] = (this.rule === RULE_RENJU)     ? 1 : 0;
        f[3] = (this.rule === RULE_RENJU) ? (toPlay === 1 ? -1 : +1) : 0;
        f[4] = this.hasForbidden ? 1 : 0;
        f[5] = ply / this.area;
        return f;
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        Gomoku, ForbiddenPointFinder,
        C_EMPTY, C_BLACK, C_WHITE, C_WALL,
        MAX_BOARD_SIZE, MAX_AREA, NUM_SPATIAL_PLANES,
        RULE_RENJU, RULE_STANDARD, RULE_FREESTYLE, VALID_RULES,
    };
}
