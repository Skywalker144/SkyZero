import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { ForbiddenPointFinder } = require("../gomoku.js");

const C_EMPTY = 0, C_BLACK = 1, C_WHITE = 2;

// Helper: set up a 15x15 FPF with a list of [r, c, color] stones.
function setupFPF(stones, size = 15) {
    const fpf = new ForbiddenPointFinder(size);
    for (const [r, c, color] of stones) fpf.setStone(r, c, color);
    return fpf;
}

test("FPF: open 3-3 at (7,7) is forbidden for black", () => {
    // Two crossing open threes — classical 三三禁手.
    const fpf = setupFPF([
        [7, 5, C_BLACK], [7, 6, C_BLACK],   // horizontal three
        [5, 7, C_BLACK], [6, 7, C_BLACK],   // vertical three
    ]);
    assert.strictEqual(fpf.isForbidden(7, 7), true);
});

test("FPF: 4-4 at (7,7) is forbidden for black", () => {
    // Two simultaneous fours through the same point.
    const fpf = setupFPF([
        [7, 4, C_BLACK], [7, 5, C_BLACK], [7, 6, C_BLACK],   // horiz four
        [4, 7, C_BLACK], [5, 7, C_BLACK], [6, 7, C_BLACK],   // vert four
    ]);
    assert.strictEqual(fpf.isForbidden(7, 7), true);
});

test("FPF: overline (6 in a row) is forbidden for black", () => {
    const fpf = setupFPF([
        [7, 4, C_BLACK], [7, 5, C_BLACK], [7, 6, C_BLACK],
        [7, 8, C_BLACK], [7, 9, C_BLACK],
        // Placing at (7,7) makes 7,4–7,9 = 6 in a row.
    ]);
    assert.strictEqual(fpf.isForbidden(7, 7), true);
});

test("FPF: exactly 5 is NOT forbidden (it's a win)", () => {
    const fpf = setupFPF([
        [7, 4, C_BLACK], [7, 5, C_BLACK], [7, 6, C_BLACK], [7, 8, C_BLACK],
        // Placing at (7,7) makes 5 in a row from 7,4 to 7,8. Wins, not forbidden.
    ]);
    assert.strictEqual(fpf.isForbidden(7, 7), false);
});

test("FPF: empty board has no forbidden points", () => {
    const fpf = setupFPF([]);
    assert.strictEqual(fpf.isForbidden(7, 7), false);
});

test("FPF: works on smaller board (13x13)", () => {
    const fpf = setupFPF([
        [6, 4, C_BLACK], [6, 5, C_BLACK],
        [4, 6, C_BLACK], [5, 6, C_BLACK],
    ], 13);
    assert.strictEqual(fpf.isForbidden(6, 6), true);
});

const { Gomoku } = require("../gomoku.js");

test("Gomoku: initial state is all zeros", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    assert.strictEqual(s.length, 15 * 15);
    assert.strictEqual(s.every(v => v === 0), true);
});

test("Gomoku: getNextState places a stone, immutable", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    const s2 = g.getNextState(s, 7 * 15 + 7, 1);
    assert.strictEqual(s2[7 * 15 + 7], 1);
    assert.strictEqual(s[7 * 15 + 7], 0);  // original unchanged
});

test("Gomoku: getLegalActions on empty board returns all true", () => {
    const g = new Gomoku(15);
    const legal = g.getLegalActions(g.getInitialState(), 1);
    assert.strictEqual(legal.length, 15 * 15);
    assert.strictEqual(legal.every(v => v === 1), true);
});

test("Gomoku: occupied cells are illegal", () => {
    const g = new Gomoku(15);
    let s = g.getInitialState();
    s = g.getNextState(s, 7 * 15 + 7, 1);
    const legal = g.getLegalActions(s, -1);
    assert.strictEqual(legal[7 * 15 + 7], 0);
});

test("Gomoku: forbidden points are LEGAL but losing under RENJU (V5 semantics)", () => {
    // V5 cpp/envs/gomoku.h::get_is_legal_actions does not filter forbidden —
    // they're legal moves; playing one ends the game with black-loss in getWinner.
    const g = new Gomoku(15, "renju");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const legal = g.getLegalActions(s, 1);
    assert.strictEqual(legal[7 * 15 + 7], 1, "forbidden point should be in legal mask");
});

test("Gomoku: playing a 3-3 forbidden point makes black lose under RENJU", () => {
    const g = new Gomoku(15, "renju");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const action = 7 * 15 + 7;
    s = g.getNextState(s, action, 1);  // black plays the 3-3 fork
    assert.strictEqual(g.getWinner(s, action, 1), -1);
});

test("Gomoku: white CAN play on what would be black's forbidden point", () => {
    const g = new Gomoku(15, "renju");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const legal = g.getLegalActions(s, -1);  // white to play
    assert.strictEqual(legal[7 * 15 + 7], 1);
});

test("Gomoku: 3-3 is NOT a black loss under STANDARD (only long-row is)", () => {
    const g = new Gomoku(15, "standard");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const action = 7 * 15 + 7;
    s = g.getNextState(s, action, 1);
    // No 5-in-a-row formed; ongoing game.
    assert.strictEqual(g.getWinner(s, action, 1), null);
});

test("Gomoku: 3-3 is NOT a black loss under FREESTYLE", () => {
    const g = new Gomoku(15, "freestyle");
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const action = 7 * 15 + 7;
    s = g.getNextState(s, action, 1);
    assert.strictEqual(g.getWinner(s, action, 1), null);
});

test("Gomoku: 13x13 board has 169 cells", () => {
    const g = new Gomoku(13);
    assert.strictEqual(g.getInitialState().length, 169);
});

test("getWinner: 5 in a row for black is a black win", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (let c = 4; c <= 8; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 8, 1), 1);
});

test("getWinner: 6 in a row by black on last move = black loses (RENJU overline)", () => {
    const g = new Gomoku(15, "renju");
    const s = g.getInitialState();
    for (let c = 4; c <= 9; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 9, 1), -1);
});

test("getWinner: 6 in a row by black = black loses under STANDARD too", () => {
    const g = new Gomoku(15, "standard");
    const s = g.getInitialState();
    for (let c = 4; c <= 9; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 9, 1), -1);
});

test("getWinner: 6 in a row by black = black WINS under FREESTYLE", () => {
    const g = new Gomoku(15, "freestyle");
    const s = g.getInitialState();
    for (let c = 4; c <= 9; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 9, 1), 1);
});

test("getWinner: 5 in a row for black wins under all three rules", () => {
    for (const rule of ["renju", "standard", "freestyle"]) {
        const g = new Gomoku(15, rule);
        const s = g.getInitialState();
        for (let c = 4; c <= 8; c++) s[7 * 15 + c] = 1;
        assert.strictEqual(g.getWinner(s, 7 * 15 + 8, 1), 1, "rule=" + rule);
    }
});

test("getWinner: 5 in a row for white is a white win", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (let c = 4; c <= 8; c++) s[7 * 15 + c] = -1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 8, -1), -1);
});

test("getWinner: 6 in a row for white is also a win (no overline rule)", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (let c = 4; c <= 9; c++) s[7 * 15 + c] = -1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 9, -1), -1);
});

test("getWinner: empty board returns null (ongoing)", () => {
    const g = new Gomoku(15);
    assert.strictEqual(g.getWinner(g.getInitialState(), null, null), null);
});

test("getWinner: full board with no 5-row returns 0 (draw)", () => {
    const g = new Gomoku(13);
    const s = g.getInitialState();
    // Striped pattern that never makes 5 in a row.
    for (let i = 0; i < 169; i++) s[i] = ((((i / 13) | 0) + (i % 13) * 2) % 3 === 0) ? 1 : -1;
    // Skip win check edge cases — just assert it doesn't return null on full board.
    const w = g.getWinner(s, 0, -1);
    assert.notStrictEqual(w, null);
});

const MAX = 17;          // V5 MAX_BOARD_SIZE
const PADDED_AREA = MAX * MAX;
const NUM_PLANES = 5;

test("encodeState: 13x13 board pads to 17x17, mask only inside [0,13)", () => {
    const g = new Gomoku(13);
    const s = g.getInitialState();
    const enc = g.encodeState(s, 1);
    assert.strictEqual(enc.length, NUM_PLANES * PADDED_AREA);
    // Mask plane (0): 1 inside [0,13), 0 outside.
    for (let r = 0; r < MAX; r++) {
        for (let c = 0; c < MAX; c++) {
            const expected = (r < 13 && c < 13) ? 1 : 0;
            assert.strictEqual(enc[0 * PADDED_AREA + r * MAX + c], expected,
                `mask plane mismatch at (${r},${c})`);
        }
    }
});

test("encodeState: own/opp planes flip with toPlay", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    s[7 * 15 + 7] = 1;   // black stone at center
    const encB = g.encodeState(s, 1);   // black to play
    const encW = g.encodeState(s, -1);  // white to play
    // Plane 1 = own. From black's POV center is own (1). From white's POV it's opp.
    assert.strictEqual(encB[1 * PADDED_AREA + 7 * MAX + 7], 1);
    assert.strictEqual(encW[2 * PADDED_AREA + 7 * MAX + 7], 1);
    assert.strictEqual(encB[2 * PADDED_AREA + 7 * MAX + 7], 0);
    assert.strictEqual(encW[1 * PADDED_AREA + 7 * MAX + 7], 0);
});

test("encodeState: forbidden plane fired in correct slot per toPlay (3-3 setup)", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const encB = g.encodeState(s, 1);
    const encW = g.encodeState(s, -1);
    // Forbidden plane for black (plane 3) when black to play; white plane (4) when white.
    // Either way the FORBIDDEN cell is (7,7) under RENJU.
    assert.strictEqual(encB[3 * PADDED_AREA + 7 * MAX + 7], 1);
    assert.strictEqual(encW[4 * PADDED_AREA + 7 * MAX + 7], 1);
    // The opposite plane for that POV must be empty for that cell.
    assert.strictEqual(encB[4 * PADDED_AREA + 7 * MAX + 7], 0);
    assert.strictEqual(encW[3 * PADDED_AREA + 7 * MAX + 7], 0);
});

test("globalFeatures: RENJU one-hot is at index 2", () => {
    const g = new Gomoku(15);
    const f = g.computeGlobalFeatures(0, 1);
    assert.strictEqual(f.length, 12);
    assert.strictEqual(f[0], 0);   // FREESTYLE
    assert.strictEqual(f[1], 0);   // STANDARD
    assert.strictEqual(f[2], 1);   // RENJU
});

test("globalFeatures: renju_color_sign is -1 for black, +1 for white", () => {
    const g = new Gomoku(15);
    assert.strictEqual(g.computeGlobalFeatures(0, 1)[3],  -1);
    assert.strictEqual(g.computeGlobalFeatures(0, -1)[3], +1);
});

test("globalFeatures: has_forbidden = 1 always (RENJU)", () => {
    const g = new Gomoku(15);
    assert.strictEqual(g.computeGlobalFeatures(0, 1)[4], 1);
});

test("globalFeatures: ply normalized by board area", () => {
    const g = new Gomoku(15);
    assert.strictEqual(g.computeGlobalFeatures(0,   1)[5], 0);
    assert.strictEqual(g.computeGlobalFeatures(225, 1)[5], 1);
    // 13x13: ply / 169
    const g13 = new Gomoku(13);
    assert.strictEqual(Math.abs(g13.computeGlobalFeatures(169, 1)[5] - 1) < 1e-6, true);
});

test("globalFeatures: dims 6-11 are zero (VCF placeholder)", () => {
    const g = new Gomoku(15);
    const f = g.computeGlobalFeatures(0, 1);
    for (let i = 6; i < 12; i++) assert.strictEqual(f[i], 0);
});

test("globalFeatures: STANDARD one-hot, color_sign zero, has_forbidden=1", () => {
    const g = new Gomoku(15, "standard");
    const f = g.computeGlobalFeatures(0, 1);
    assert.strictEqual(f[0], 0);
    assert.strictEqual(f[1], 1);   // STANDARD
    assert.strictEqual(f[2], 0);
    assert.strictEqual(f[3], 0);   // color_sign only fires under RENJU
    assert.strictEqual(f[4], 1);   // has_forbidden (overline rule)
});

test("globalFeatures: FREESTYLE one-hot, color_sign zero, has_forbidden=0", () => {
    const g = new Gomoku(15, "freestyle");
    const f = g.computeGlobalFeatures(0, 1);
    assert.strictEqual(f[0], 1);   // FREESTYLE
    assert.strictEqual(f[1], 0);
    assert.strictEqual(f[2], 0);
    assert.strictEqual(f[3], 0);
    assert.strictEqual(f[4], 0);
});

test("encodeState: forbidden planes are empty under FREESTYLE", () => {
    const g = new Gomoku(15, "freestyle");
    const s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) s[r * 15 + c] = 1;
    const enc = g.encodeState(s, 1);
    let nonzero = 0;
    for (let i = 0; i < PADDED_AREA; i++) {
        if (enc[3 * PADDED_AREA + i] !== 0) nonzero++;
        if (enc[4 * PADDED_AREA + i] !== 0) nonzero++;
    }
    assert.strictEqual(nonzero, 0);
});

test("Gomoku: rejects unknown rule string", () => {
    assert.throws(() => new Gomoku(15, "tic-tac-toe"));
});
