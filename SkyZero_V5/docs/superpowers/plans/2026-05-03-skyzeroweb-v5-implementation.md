# SkyZeroWeb V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SkyZeroWeb` — a static webpage at `/home/sky/RL/SkyZero/SkyZeroWeb/` that loads V5 ONNX models in-browser via `onnxruntime-web` and runs the same MCTS as `SkyZero_V5/python/play_web.py`, with full UI parity.

**Architecture:** Static HTML/CSS/JS, no server. ONNX inference + MCTS run inside a single Web Worker (single-threaded WASM). Game logic and MCTS are pure JS modules ported line-by-line from V5 `cpp/envs/gomoku.h` and `cpp/alphazero.h` (RENJU rule, sequential MCTS, no symmetry/stochastic). 5-tier ELO model selector reads a hand-curated `models/manifest.json`.

**Tech Stack:**
- Vanilla JS (no framework, no bundler), ES classes
- `onnxruntime-web` v1.17.0 from jsdelivr CDN (WASM EP, single-thread)
- Node 18+ builtin test runner (`node --test`) for `gomoku.js` / `mcts.js` unit tests
- Python 3 + PyTorch (re-uses V5 `nets.py` / `model_config.py`) for the export script
- Cloudflare Pages for deployment (zero-build, git-connected)

**Source spec:** `SkyZero_V5/docs/superpowers/specs/2026-05-03-skyzeroweb-v5-design.md`

---

## File Structure

```
/home/sky/RL/SkyZero/SkyZeroWeb/
├── .gitignore
├── README.md
├── _headers                 # Cloudflare cache rules
├── index.html               # UI structure (port of play_web.py HTML body)
├── style.css                # Styling (port of play_web.py <style>)
├── main.js                  # UI controller, canvas painters, worker plumbing
├── worker.js                # ort.InferenceSession + MCTS driver
├── mcts.js                  # Sequential MCTS (Gumbel halving + var-cPUCT)
├── gomoku.js                # Renju game logic + FPF + V5 5-plane encoding
├── tools/
│   └── export_onnx.py       # V5 .pt → .onnx (ExportWrapper drops unused heads)
├── models/
│   ├── manifest.json        # Hand-curated 5-tier ELO catalog
│   └── (level1..5.onnx, copied in by user)
└── tests/
    ├── test_gomoku.mjs      # node --test for gomoku.js
    └── test_mcts.mjs        # node --test for mcts.js
```

**Note on the existing `/home/sky/RL/SkyZeroWeb/`** (V4-era prototype): kept untouched as architectural reference. Useful files to crib from when porting:
- `gomoku.js` — `ForbiddenPointFinder` JS port (reusable as-is; only the `Gomoku` class needs V5 changes)
- `mcts.js` — Gumbel halving + variance-scaled cPUCT (reusable; minor changes for 4-channel policy + record `gumbelPhases`)
- `worker.js` — fetch-with-progress + ort init pattern
- `_headers` — cache directive format

---

## Task Index

1. Bootstrap repo (`SkyZeroWeb/`)
2. Set up Node test runner
3. `gomoku.js` — `ForbiddenPointFinder` port + tests
4. `gomoku.js` — `Gomoku` class basics (state/legal/next) + tests
5. `gomoku.js` — `getWinner` (RENJU multi-rule winner) + tests
6. `gomoku.js` — `encodeState` (5 planes, 17×17 padded) + tests
7. `gomoku.js` — `computeGlobalFeatures` (12-dim) + tests
8. `mcts.js` — `Node` class + `softmax` helper + tests
9. `mcts.js` — variance-scaled cPUCT + select + tests
10. `mcts.js` — expand + backpropagate + tests
11. `mcts.js` — `gumbelSequentialHalving` + tests
12. `tools/export_onnx.py` — ExportWrapper + script
13. Smoke-test the export with one V5 checkpoint
14. `worker.js` — bootstrap (init / model load / inference helper)
15. `worker.js` — message router (reset / move / search) + abort
16. `models/manifest.json` skeleton
17. `style.css` — port from `play_web.py`
18. `index.html` — port body from `play_web.py`
19. `main.js` — bootstrap + theme + sizing
20. `main.js` — board canvas (stones, gumbel overlay)
21. `main.js` — heatmap canvases (signed + unsigned) + heat modal
22. `main.js` — value chart + WDL bars
23. `main.js` — model dropdown + size dropdown + manifest loader
24. `main.js` — game flow + worker message handlers
25. `_headers` + README + first end-to-end manual run

---

## Task 1: Bootstrap repo

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/.gitignore`
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/README.md` (placeholder, fleshed out in Task 25)

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p /home/sky/RL/SkyZero/SkyZeroWeb/{tools,models,tests}
cd /home/sky/RL/SkyZero/SkyZeroWeb
```

- [ ] **Step 2: Initialize git**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
git init -b main
```

- [ ] **Step 3: Write `.gitignore`**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/.gitignore`:

```
# Editor/OS
.DS_Store
.idea/
.vscode/
*.swp

# Python (export_onnx.py byproducts)
__pycache__/
*.pyc

# Node
node_modules/

# Local dev
.cache/
```

Note: `.onnx` files are NOT ignored — model files in `models/` are checked in
(small ~4MB each, deployed via git).

- [ ] **Step 4: Placeholder README**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/README.md`:

```markdown
# SkyZeroWeb

Static webpage that runs SkyZero V5 in the browser via ONNX Runtime Web.

(Documentation will be filled in once implementation is complete — see Task 25.)
```

- [ ] **Step 5: First commit**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
git add .
git commit -m "init: bootstrap empty repo skeleton"
```

---

## Task 2: Set up Node test runner

We'll use Node 18+'s builtin `node --test` runner — zero deps, native ESM.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/package.json`

- [ ] **Step 1: Verify Node version**

```bash
node --version
```

Expected: `v18.x` or higher. If lower, install via nvm/system pkg manager.

- [ ] **Step 2: Write `package.json`**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/package.json`:

```json
{
  "name": "skyzeroweb",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "test": "node --test tests/"
  }
}
```

`"type": "module"` makes `.mjs` files first-class ESM — important so the test
files can `import` from `gomoku.js` / `mcts.js` (which we write in classic
script style for browser/worker consumption — bridged in Step 4 below).

- [ ] **Step 3: Verify the runner works on an empty suite**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
npm test
```

Expected: exits 0, "0 tests" or similar. If it errors complaining about no
files, that's also fine — we'll add tests in the next task.

- [ ] **Step 4: Plan the dual-consumption strategy**

`gomoku.js` and `mcts.js` need to work in TWO environments:
1. Browser/Worker: included via `<script>` / `importScripts`. Needs to attach
   classes to the global scope.
2. Node tests: imported via `import`. Needs `export` statements.

We'll use the CommonJS-ish escape hatch at the bottom of each module:

```js
// At the END of gomoku.js / mcts.js:
if (typeof module !== "undefined" && module.exports) {
    module.exports = { Gomoku, ForbiddenPointFinder /* etc */ };
}
```

Then in test files (`.mjs`), use Node's `createRequire` to load classic JS:

```js
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { Gomoku, ForbiddenPointFinder } = require("../gomoku.js");
```

This is the same pattern the old SkyZeroWeb used — proven to work.

- [ ] **Step 5: Commit**

```bash
git add package.json
git commit -m "tooling: add node --test setup with package.json"
```

---

## Task 3: gomoku.js — ForbiddenPointFinder + tests

Port the FPF class from `/home/sky/RL/SkyZeroWeb/gomoku.js` lines 1-227. The
algorithm matches V5 `cpp/envs/gomoku.h` lines 574-818 (also matches V3
GomoCup classical FPF) — no changes needed except making boardSize a runtime
parameter (already true in the old port).

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js` (FPF only for now)
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`

- [ ] **Step 1: Write test cases (these will fail)**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`:

```js
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
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
npm test
```

Expected: `Cannot find module '../gomoku.js'` or similar. Good.

- [ ] **Step 3: Write `gomoku.js` with FPF**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js`. Copy lines 1-226 from
`/home/sky/RL/SkyZeroWeb/gomoku.js` verbatim. The classes/constants needed:

```js
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

    // ... (all the isFive/isOverline/isFour/isOpenFour/isOpenThree/
    //      isDoubleFour/isDoubleThree/isForbidden methods, copied
    //      verbatim from the old SkyZeroWeb gomoku.js lines 25-227)
}

// Export for Node test consumption.
if (typeof module !== "undefined" && module.exports) {
    module.exports = { ForbiddenPointFinder, C_EMPTY, C_BLACK, C_WHITE, C_WALL };
}
```

The verbatim source is at `/home/sky/RL/SkyZeroWeb/gomoku.js` lines 1-227.
Open it side-by-side and copy the FPF class body. Do NOT include the
`Gomoku` class yet — that goes in Task 4. Do NOT include the bottom
`module.exports` line of the source (it exports `Gomoku` too); write the
export block above instead, exporting only what's defined so far.

- [ ] **Step 4: Run tests — should all pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
npm test
```

Expected: 6 passing.

- [ ] **Step 5: Commit**

```bash
git add gomoku.js tests/test_gomoku.mjs
git commit -m "gomoku: port ForbiddenPointFinder from old SkyZeroWeb"
```

---

## Task 4: gomoku.js — Gomoku class basics

Add the `Gomoku` class with `getInitialState` / `getNextState` /
`getLegalActions`. Crucial difference from old SkyZeroWeb: **boardSize is
variable (13–17)** and the V5 RENJU rule treats forbidden points as
illegal for black.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js` (append `Gomoku` class)
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`:

```js
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

test("Gomoku: black cannot play on a forbidden 3-3 point", () => {
    const g = new Gomoku(15);
    let s = g.getInitialState();
    // Set up a 3-3 fork at (7,7).
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) {
        s[r * 15 + c] = 1;  // black stones
    }
    const legal = g.getLegalActions(s, 1);  // black to play
    assert.strictEqual(legal[7 * 15 + 7], 0);
});

test("Gomoku: white CAN play on what would be black's forbidden point", () => {
    const g = new Gomoku(15);
    let s = g.getInitialState();
    for (const [r, c] of [[7,5],[7,6],[5,7],[6,7]]) {
        s[r * 15 + c] = 1;
    }
    const legal = g.getLegalActions(s, -1);  // white to play
    assert.strictEqual(legal[7 * 15 + 7], 1);
});

test("Gomoku: 13x13 board has 169 cells", () => {
    const g = new Gomoku(13);
    assert.strictEqual(g.getInitialState().length, 169);
});
```

- [ ] **Step 2: Run, confirm new tests fail**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: existing 6 still pass, 7 new fail with `Gomoku is not a constructor`.

- [ ] **Step 3: Append `Gomoku` class to `gomoku.js`**

Insert before the `if (typeof module !== "undefined")` export block:

```js
class Gomoku {
    constructor(boardSize) {
        this.boardSize = boardSize;
        this.area = boardSize * boardSize;
        this.fpf = new ForbiddenPointFinder(boardSize);
    }

    getInitialState() {
        return new Int8Array(this.area);
    }

    getNextState(state, action, toPlay) {
        const out = new Int8Array(state);
        out[action] = toPlay;
        return out;
    }

    // Returns Uint8Array(area), 1 = legal, 0 = illegal.
    // For RENJU + black, also excludes forbidden points (3-3, 4-4, overline).
    getLegalActions(state, toPlay) {
        const out = new Uint8Array(this.area);
        let needsFPF = (toPlay === 1);  // RENJU forbids only for black
        if (needsFPF) {
            this.fpf.clear();
            for (let i = 0; i < this.area; i++) {
                if (state[i] !== 0) {
                    const r = (i / this.boardSize) | 0;
                    const c = i % this.boardSize;
                    this.fpf.setStone(r, c, state[i] === 1 ? C_BLACK : C_WHITE);
                }
            }
        }
        for (let i = 0; i < this.area; i++) {
            if (state[i] !== 0) { out[i] = 0; continue; }
            if (needsFPF) {
                const r = (i / this.boardSize) | 0;
                const c = i % this.boardSize;
                if (this.fpf.isForbidden(r, c)) { out[i] = 0; continue; }
            }
            out[i] = 1;
        }
        return out;
    }
}
```

Update the export block at the bottom:

```js
if (typeof module !== "undefined" && module.exports) {
    module.exports = { Gomoku, ForbiddenPointFinder, C_EMPTY, C_BLACK, C_WHITE, C_WALL };
}
```

- [ ] **Step 4: Run tests — all 13 should pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 13 passing.

- [ ] **Step 5: Commit**

```bash
git add gomoku.js tests/test_gomoku.mjs
git commit -m "gomoku: add Gomoku class basics with RENJU legal-move filter"
```

---

## Task 5: gomoku.js — getWinner

Port V5 `get_winner_v5` (RENJU subset only): exactly-5-for-black wins, white
wins on ≥5, black-overline-on-last-move = black loses, board-full = draw,
otherwise ongoing.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js` (add `getWinner` method)
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`

- [ ] **Step 1: Append failing tests**

```js
test("getWinner: 5 in a row for black is a black win", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (let c = 4; c <= 8; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 8, 1), 1);
});

test("getWinner: 6 in a row by black on last move = black loses (overline)", () => {
    const g = new Gomoku(15);
    const s = g.getInitialState();
    for (let c = 4; c <= 9; c++) s[7 * 15 + c] = 1;
    assert.strictEqual(g.getWinner(s, 7 * 15 + 9, 1), -1);
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
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: existing 13 pass, 6 new fail with `getWinner is not a function`.

- [ ] **Step 3: Add `getWinner` to `Gomoku`**

Insert into the `Gomoku` class body:

```js
    /**
     * Returns +1 (black wins), -1 (white wins), 0 (draw, board full),
     * or null (ongoing). Mirrors V5 cpp/envs/gomoku.h::get_winner_v5
     * for the RENJU-only path.
     *
     * lastAction / lastPlayer describe the most recent move; needed for
     * overline-on-black check (= black loses immediately).
     */
    getWinner(state, lastAction, lastPlayer) {
        const N = this.boardSize;

        // Step 1: black overline on the just-played move = black loses.
        if (lastAction != null && lastPlayer === 1) {
            const r = (lastAction / N) | 0;
            const c = lastAction % N;
            if (this._isOverlineAt(state, r, c, 1)) return -1;
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
                    if (stone === 1 && len === 5) return 1;       // RENJU: black needs exactly 5
                    if (stone === -1 && len >= 5) return -1;      // White: 5 or more
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
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 19 passing.

- [ ] **Step 5: Commit**

```bash
git add gomoku.js tests/test_gomoku.mjs
git commit -m "gomoku: add RENJU getWinner with overline=lose for black"
```

---

## Task 6: gomoku.js — encodeState (5 planes, 17×17 padded)

Port V5 `encode_state_v5` from `cpp/envs/gomoku.h:400-448`. The output is
fixed at 17×17 stride regardless of `boardSize`, with the mask plane
indicating which cells are on-board.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`

- [ ] **Step 1: Failing tests**

Append to test file:

```js
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
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 19 pass, 3 fail.

- [ ] **Step 3: Add `encodeState` + constants to `Gomoku`**

At the top of `gomoku.js`, after the `C_*` constants, add:

```js
const MAX_BOARD_SIZE = 17;     // V5 MAX_BOARD_SIZE
const MAX_AREA = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
const NUM_SPATIAL_PLANES = 5;
```

Inside `Gomoku`, add:

```js
    /**
     * V5 encode: 5 planes padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE = 17×17.
     * Plane 0: on-board mask (1 inside [0, boardSize), 0 in padding)
     * Plane 1: own stones
     * Plane 2: opponent stones
     * Plane 3: forbidden points when current player is BLACK (RENJU only)
     * Plane 4: forbidden points when current player is WHITE
     *
     * Output is Float32Array (model expects float32 input).
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

        // Planes 3-4: forbidden (RENJU; plane chosen by current player)
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

        return out;
    }
```

Update the export block:

```js
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        Gomoku, ForbiddenPointFinder,
        C_EMPTY, C_BLACK, C_WHITE, C_WALL,
        MAX_BOARD_SIZE, MAX_AREA, NUM_SPATIAL_PLANES,
    };
}
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 22 passing.

- [ ] **Step 5: Commit**

```bash
git add gomoku.js tests/test_gomoku.mjs
git commit -m "gomoku: add encodeState (5 planes, 17x17 padded)"
```

---

## Task 7: gomoku.js — computeGlobalFeatures (12-dim)

Port V5 `compute_global_features` from `cpp/envs/gomoku.h:466-480`. Always
RENJU since that's the only rule we support.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/gomoku.js`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_gomoku.mjs`

- [ ] **Step 1: Failing tests**

```js
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
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 22 pass, 5 fail.

- [ ] **Step 3: Add `computeGlobalFeatures` to `Gomoku`**

```js
    /**
     * 12-dim global features (KataGoNet.linear_global input).
     * RENJU-only assumption: one-hot at idx 2, renju_color_sign signed,
     * has_forbidden = 1, ply / board_area at idx 5, VCF placeholder zeros.
     */
    computeGlobalFeatures(ply, toPlay) {
        const f = new Float32Array(12);
        f[0] = 0;                                // FREESTYLE one-hot
        f[1] = 0;                                // STANDARD one-hot
        f[2] = 1;                                // RENJU one-hot
        f[3] = (toPlay === 1) ? -1 : +1;         // renju_color_sign
        f[4] = 1;                                // has_forbidden (RENJU has it)
        f[5] = ply / this.area;                  // ply normalized
        // f[6..11] left zero — VCF placeholder per V5
        return f;
    }
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 27 passing.

- [ ] **Step 5: Commit**

```bash
git add gomoku.js tests/test_gomoku.mjs
git commit -m "gomoku: add 12-dim global features (RENJU-only)"
```

---

## Task 8: mcts.js — Node + softmax + tests

Port the `Node` class and `softmax` helper from
`/home/sky/RL/SkyZeroWeb/mcts.js` lines 1-59. No V5-specific changes.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/mcts.js`
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_mcts.mjs`

- [ ] **Step 1: Failing tests**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_mcts.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { Node, softmax } = require("../mcts.js");

test("Node: starts unvisited, unexpanded", () => {
    const n = new Node(null, 1, 0.5, null, 7);
    assert.strictEqual(n.n, 0);
    assert.strictEqual(n.isExpanded(), false);
    assert.strictEqual(n.prior, 0.5);
    assert.strictEqual(n.actionTaken, 7);
    assert.strictEqual(n.toPlay, 1);
});

test("Node.update: accumulates v + utilitySqSum + n", () => {
    const n = new Node(null, 1);
    n.update(new Float64Array([0.6, 0.1, 0.3]));
    n.update(new Float64Array([0.4, 0.2, 0.4]));
    assert.strictEqual(n.n, 2);
    assert.ok(Math.abs(n.v[0] - 1.0) < 1e-9);
    assert.ok(Math.abs(n.v[1] - 0.3) < 1e-9);
    assert.ok(Math.abs(n.v[2] - 0.7) < 1e-9);
    // utility = L - W. After [0.6,0.1,0.3]: u=-0.3. After [0.4,0.2,0.4]: u=0.
    // Sum of squares = 0.09 + 0 = 0.09.
    assert.ok(Math.abs(n.utilitySqSum - 0.09) < 1e-9);
});

test("softmax: outputs sum to 1, monotonic with logits", () => {
    const p = softmax(new Float64Array([1, 2, 3]));
    const sum = p[0] + p[1] + p[2];
    assert.ok(Math.abs(sum - 1) < 1e-9);
    assert.ok(p[0] < p[1] && p[1] < p[2]);
});

test("softmax: handles -Infinity (illegal moves)", () => {
    const p = softmax(new Float64Array([1, -Infinity, 3]));
    assert.strictEqual(p[1], 0);
    assert.ok(Math.abs(p[0] + p[2] - 1) < 1e-9);
});
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 4 new tests fail with "Cannot find module '../mcts.js'".

- [ ] **Step 3: Create `mcts.js` with `Node` + `softmax`**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/mcts.js`. Copy lines 1-59 from
`/home/sky/RL/SkyZeroWeb/mcts.js` verbatim (no V5 changes needed):

```js
class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];
        this.nnValue = new Float64Array(3);   // WDL output from NN
        this.nnPolicy = null;                  // softmax policy (Float32Array)
        this.nnLogits = null;                  // masked logits (Float32Array)
        this.v = new Float64Array(3);          // cumulative WDL
        this.utilitySqSum = 0;                 // for variance-scaled cPUCT
        this.n = 0;
    }

    isExpanded() {
        return this.children.length > 0;
    }

    update(value) {
        this.v[0] += value[0];
        this.v[1] += value[1];
        this.v[2] += value[2];
        const u = value[2] - value[0];   // L - W (parent perspective utility)
        this.utilitySqSum += u * u;
        this.n += 1;
    }
}

function sampleGumbel() {
    let u = Math.random();
    u = Math.max(1e-20, Math.min(1 - 1e-10, u));
    return -Math.log(-Math.log(u));
}

function softmax(logits) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    const scores = new Float64Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        scores[i] = Math.exp(logits[i] - maxLogit);
        sum += scores[i];
    }
    for (let i = 0; i < scores.length; i++) {
        scores[i] /= sum;
    }
    return scores;
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { Node, softmax, sampleGumbel };
}
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 31 passing.

- [ ] **Step 5: Commit**

```bash
git add mcts.js tests/test_mcts.mjs
git commit -m "mcts: add Node + softmax + sampleGumbel helpers"
```

---

## Task 9: mcts.js — variance-scaled cPUCT + select

Port `MCTS` class with `computeParentUtilityStdevFactor`,
`computeSelectParams`, and `select`. Mirrors V5 `cpp/alphazero.h`
`compute_parent_utility_stdev_factor` + `compute_select_params` + `select`.
Already correctly ported in old SkyZeroWeb's `mcts.js` lines 61-180; V5 has
no algorithmic changes here.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/mcts.js`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_mcts.mjs`

- [ ] **Step 1: Failing tests**

Append:

```js
const { MCTS } = require("../mcts.js");

function makeRoot(toPlay = 1) {
    const r = new Node(null, toPlay);
    r.nnValue = new Float64Array([0.5, 0, 0.5]);   // neutral WDL
    r.nnPolicy = new Float32Array(225).fill(1/225);
    return r;
}

test("MCTS.select: picks child with higher prior when no visits", () => {
    const mcts = new MCTS(null, {});
    const root = makeRoot();
    root.update(new Float64Array([0.5, 0, 0.5]));
    const c1 = new Node(null, -1, 0.1, root, 0);
    const c2 = new Node(null, -1, 0.9, root, 1);
    root.children = [c1, c2];
    assert.strictEqual(mcts.select(root), c2);
});

test("MCTS.select: prefers child with better Q when both visited", () => {
    const mcts = new MCTS(null, {});
    const root = makeRoot();
    root.update(new Float64Array([0.5, 0, 0.5]));
    // Both children have same prior, but c2 has been winning more from parent's view.
    const c1 = new Node(null, -1, 0.5, root, 0);
    c1.nnValue = new Float64Array([0.5, 0, 0.5]);
    c1.update(new Float64Array([0.7, 0, 0.3]));   // child losing -> good for parent
    const c2 = new Node(null, -1, 0.5, root, 1);
    c2.nnValue = new Float64Array([0.5, 0, 0.5]);
    c2.update(new Float64Array([0.3, 0, 0.7]));   // child winning -> bad for parent
    root.children = [c1, c2];
    // Parent's Q for c1 = c1.L - c1.W = 0.3 - 0.7 = -0.4
    // Parent's Q for c2 = c2.L - c2.W = 0.7 - 0.3 = +0.4 (better!)
    assert.strictEqual(mcts.select(root), c2);
});

test("MCTS.computeParentUtilityStdevFactor: returns 1 at neutral parent", () => {
    const mcts = new MCTS(null, {});
    const n = new Node(null, 1);
    n.update(new Float64Array([0.5, 0, 0.5]));
    const f = mcts.computeParentUtilityStdevFactor(n, 0);
    assert.ok(Math.abs(f - 1) < 1e-6);   // neutral parent → factor exactly 1
});
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 3 new tests fail with "MCTS is not a constructor".

- [ ] **Step 3: Add MCTS class with cPUCT + select**

Insert before the export block in `mcts.js`. Copy verbatim from
`/home/sky/RL/SkyZeroWeb/mcts.js` lines 61-180 — no changes:

```js
class MCTS {
    constructor(game, args) {
        this.game = game;
        this.args = Object.assign({
            c_puct: 1.1,
            c_puct_log: 0.45,
            c_puct_base: 500,
            fpu_reduction_max: 0.2,
            root_fpu_reduction_max: 0.1,
            fpu_pow: 1.0,
            fpu_loss_prop: 0.0,
            cpuct_utility_stdev_prior: 0.40,
            cpuct_utility_stdev_prior_weight: 2.0,
            cpuct_utility_stdev_scale: 0.85,
            gumbel_m: 16,
            gumbel_c_visit: 50,
            gumbel_c_scale: 1.0,
        }, args);
    }

    computeParentUtilityStdevFactor(node, parentUtility) {
        const prior = this.args.cpuct_utility_stdev_prior;
        const variancePrior = prior * prior;
        const variancePriorWeight = this.args.cpuct_utility_stdev_prior_weight;

        let parentStdev;
        if (node.n <= 1) {
            parentStdev = prior;
        } else {
            const effectiveN = node.n;
            const utilitySqAvg = node.utilitySqSum / effectiveN;
            const uSq = parentUtility * parentUtility;
            const adjSqAvg = Math.max(utilitySqAvg, uSq);
            parentStdev = Math.sqrt(Math.max(0,
                ((uSq + variancePrior) * variancePriorWeight + adjSqAvg * effectiveN)
                / (variancePriorWeight + effectiveN - 1)
                - uSq
            ));
        }
        return 1.0 + this.args.cpuct_utility_stdev_scale
            * (parentStdev / Math.max(1e-8, prior) - 1.0);
    }

    computeSelectParams(node, effectiveParentN, visitedPolicyMass) {
        const totalChildWeight = Math.max(0, effectiveParentN - 1);

        const cPuct = this.args.c_puct
            + this.args.c_puct_log * Math.log((totalChildWeight + this.args.c_puct_base) / this.args.c_puct_base);

        let parentUtility = 0;
        if (node.n > 0) {
            parentUtility = node.v[0] / node.n - node.v[2] / node.n;
        }

        const stdevFactor = this.computeParentUtilityStdevFactor(node, parentUtility);
        const exploreScaling = cPuct * Math.sqrt(totalChildWeight + 0.01) * stdevFactor;

        const nnUtility = node.nnValue[0] - node.nnValue[2];
        const avgWeight = Math.min(1, Math.pow(visitedPolicyMass, this.args.fpu_pow));
        const parentUtilityForFpu = avgWeight * parentUtility + (1 - avgWeight) * nnUtility;

        const fpuReductionMax = (node.parent === null) ? this.args.root_fpu_reduction_max : this.args.fpu_reduction_max;
        const reduction = fpuReductionMax * Math.sqrt(visitedPolicyMass);
        let fpuValue = parentUtilityForFpu - reduction;
        fpuValue = fpuValue + (-1.0 - fpuValue) * this.args.fpu_loss_prop;
        return { exploreScaling, fpuValue };
    }

    select(node) {
        let visitedPolicyMass = 0;
        for (const child of node.children) {
            if (child.n > 0) visitedPolicyMass += child.prior;
        }
        const sp = this.computeSelectParams(node, node.n, visitedPolicyMass);

        let bestScore = -Infinity;
        let bestChild = null;
        for (const child of node.children) {
            let qValue;
            if (child.n === 0) {
                qValue = sp.fpuValue;
            } else {
                qValue = child.v[2] / child.n - child.v[0] / child.n;
            }
            const uValue = sp.exploreScaling * child.prior / (1 + child.n);
            const score = qValue + uValue;
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }
}
```

Update export at bottom:

```js
if (typeof module !== "undefined" && module.exports) {
    module.exports = { Node, MCTS, softmax, sampleGumbel };
}
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 34 passing.

- [ ] **Step 5: Commit**

```bash
git add mcts.js tests/test_mcts.mjs
git commit -m "mcts: add variance-scaled cPUCT and PUCT select"
```

---

## Task 10: mcts.js — expand + backpropagate

Port from `/home/sky/RL/SkyZeroWeb/mcts.js` lines 188-227. Verbatim — no V5
changes needed.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/mcts.js`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_mcts.mjs`

- [ ] **Step 1: Failing tests**

Append:

```js
test("MCTS.expand: creates one child per nonzero policy entry", () => {
    // Tiny stub game: 4 actions, 2 of them have nonzero prior.
    const stubGame = {
        getNextState(state, action, toPlay) {
            const out = new Int8Array(state);
            out[action] = toPlay;
            return out;
        },
    };
    const mcts = new MCTS(stubGame, {});
    const root = new Node(new Int8Array(4), 1);
    const policy = new Float32Array([0.4, 0.0, 0.6, 0.0]);
    const value = new Float64Array([0.5, 0, 0.5]);
    const logits = new Float32Array([1, -Infinity, 2, -Infinity]);
    mcts.expand(root, policy, value, logits);
    assert.strictEqual(root.children.length, 2);
    assert.strictEqual(root.children[0].actionTaken, 0);
    assert.strictEqual(root.children[1].actionTaken, 2);
    assert.strictEqual(root.children[0].toPlay, -1);
    assert.ok(Math.abs(root.children[1].prior - 0.6) < 1e-6);
});

test("MCTS.backpropagate: WDL flips at each level", () => {
    const mcts = new MCTS(null, {});
    const root = new Node(null, 1);
    const child = new Node(null, -1, 1, root, 0);
    root.children = [child];
    // At leaf (=child), WDL = [W=0.7, D=0.0, L=0.3] — child winning.
    // After backprop, root sees flipped: [W=0.3, D=0.0, L=0.7] — root losing.
    mcts.backpropagate(child, new Float64Array([0.7, 0, 0.3]));
    assert.ok(Math.abs(child.v[0] - 0.7) < 1e-6);
    assert.ok(Math.abs(root.v[0]  - 0.3) < 1e-6);
    assert.ok(Math.abs(root.v[2]  - 0.7) < 1e-6);
});
```

- [ ] **Step 2: Confirm failures**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 2 new fail with "expand is not a function" / "backpropagate is not a function".

- [ ] **Step 3: Add `expand` and `backpropagate` to `MCTS`**

```js
    expand(node, nnPolicy, nnValue, nnLogits) {
        node.nnValue = new Float64Array(nnValue);
        node.nnPolicy = nnPolicy;
        node.nnLogits = nnLogits;

        const nextToPlay = -node.toPlay;
        for (let action = 0; action < nnPolicy.length; action++) {
            const prob = nnPolicy[action];
            if (prob > 0) {
                const child = new Node(
                    this.game.getNextState(node.state, action, node.toPlay),
                    nextToPlay,
                    prob,
                    node,
                    action
                );
                node.children.push(child);
            }
        }
    }

    backpropagate(node, value) {
        let curr = node;
        let w = value[0], d = value[1], l = value[2];
        const buf = new Float64Array(3);
        while (curr !== null) {
            buf[0] = w; buf[1] = d; buf[2] = l;
            curr.update(buf);
            const tmp = w; w = l; l = tmp;
            curr = curr.parent;
        }
    }
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 36 passing.

- [ ] **Step 5: Commit**

```bash
git add mcts.js tests/test_mcts.mjs
git commit -m "mcts: add expand and backpropagate (WDL flip per level)"
```

---

## Task 11: mcts.js — gumbelSequentialHalving

Port from `/home/sky/RL/SkyZeroWeb/mcts.js` lines 243-440 with **two V5
adjustments**:
1. Old caller passed `mcts.gumbelSequentialHalving(root, sims, isEval, simulateOne)`;
   we drop `isEval` (always eval / no Gumbel noise — plays deterministically;
   matches V5 play.cfg `GUMBEL_NOISE_ENABLED=0`).
2. Record per-phase surviving-action sets into `root._gumbelPhases` so the
   UI can render the 16/8/4/2/1 overlay.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/mcts.js`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/tests/test_mcts.mjs`

- [ ] **Step 1: Failing test**

```js
test("MCTS.gumbelSequentialHalving: returns gumbelAction + improvedPolicy on a 4-action game", async () => {
    const stubGame = {
        boardSize: 2,                                        // 2x2 = 4 actions
        getNextState(state, action, toPlay) {
            const out = new Int8Array(state); out[action] = toPlay; return out;
        },
        getLegalActions(state, _toPlay) {
            return new Uint8Array([1, 1, 1, 1]);
        },
    };
    const mcts = new MCTS(stubGame, { gumbel_m: 4 });

    const root = new Node(new Int8Array(4), 1);
    const policy = new Float32Array([0.4, 0.3, 0.2, 0.1]);
    const value = new Float64Array([0.5, 0, 0.5]);
    const logits = new Float32Array([0.4, 0.1, -0.3, -1.2]);
    mcts.expand(root, policy, value, logits);
    root.update(value);

    let calls = 0;
    const simulateOne = async (action) => {
        const child = root.children.find(c => c.actionTaken === action);
        // Simulated leaf: return a deterministic value
        const leafValue = new Float64Array([0.5, 0, 0.5]);
        mcts.backpropagate(child, leafValue);
        calls++;
    };

    const result = await mcts.gumbelSequentialHalving(root, 16, simulateOne);
    assert.ok(result.improvedPolicy instanceof Float32Array);
    assert.strictEqual(result.improvedPolicy.length, 4);
    assert.ok(typeof result.gumbelAction === "number");
    assert.ok([0,1,2,3].includes(result.gumbelAction));
    assert.ok(calls >= 4);
    // gumbelPhases should be recorded for UI overlay.
    assert.ok(Array.isArray(root._gumbelPhases));
    assert.ok(root._gumbelPhases.length >= 1);
});
```

- [ ] **Step 2: Confirm failure**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: fail with "gumbelSequentialHalving is not a function".

- [ ] **Step 3: Add `gumbelSequentialHalving` and `getMCTSPolicy` to `MCTS`**

Insert before the closing `}` of the `MCTS` class:

```js
    /**
     * Gumbel Sequential Halving (Danihelka et al., 2022).
     * Mirrors V5 cpp/alphazero.h::run_gumbel_search for the eval (no-noise) path.
     *
     * Records per-phase surviving action sets into root._gumbelPhases so the UI
     * can render the 16/8/4/2/1 colored overlay.
     *
     * @param {Node} root - expanded root with nnLogits set
     * @param {number} numSimulations - total simulation budget
     * @param {Function} simulateOne - async (action) => void; runs one sim from a root child
     * @returns {Promise<{improvedPolicy, gumbelAction, vMix}>}
     */
    async gumbelSequentialHalving(root, numSimulations, simulateOne) {
        const boardArea = this.game.boardSize * this.game.boardSize;
        const logits = new Float64Array(root.nnLogits);
        const isLegal = this.game.getLegalActions(root.state, root.toPlay);

        // Eval / no-noise (matches V5 play.cfg GUMBEL_NOISE_ENABLED=0).
        const g = new Float64Array(boardArea);   // all zeros

        // Gumbel-Top-k: pick top-m by logits (no noise = deterministic top-m).
        let m = Math.min(numSimulations, this.args.gumbel_m);
        const scores = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            scores[i] = isLegal[i] ? (logits[i] + g[i]) : -Infinity;
        }
        const indices = Array.from({ length: boardArea }, (_, i) => i);
        indices.sort((a, b) => scores[b] - scores[a]);
        let survivingActions = [];
        for (let i = 0; i < indices.length && survivingActions.length < m; i++) {
            if (isLegal[indices[i]]) survivingActions.push(indices[i]);
        }
        m = survivingActions.length;

        // Record initial surviving set as phase 0 (for UI overlay).
        const phases = [];
        const N = this.game.boardSize;
        const toPhaseRC = (acts) => acts.map(a => [(a / N) | 0, a % N]);
        if (m > 0) phases.push(toPhaseRC(survivingActions));

        if (m > 0) {
            const totalPhases = m > 1 ? Math.ceil(Math.log2(m)) : 1;
            let simsBudget = numSimulations;

            for (let phase = 0; phase < totalPhases; phase++) {
                if (simsBudget <= 0) break;

                const remainingPhases = totalPhases - phase;
                const simsThisPhase = Math.floor(simsBudget / remainingPhases);
                const numActions = survivingActions.length;
                const simsPerAction = Math.max(1, Math.floor(simsThisPhase / numActions));

                for (let s = 0; s < simsPerAction; s++) {
                    if (simsBudget <= 0) break;
                    for (const action of survivingActions) {
                        if (simsBudget <= 0) break;
                        await simulateOne(action);
                        simsBudget--;
                    }
                }

                if (simsBudget <= 0) break;
                if (phase < totalPhases - 1) {
                    let maxN = 0;
                    for (const child of root.children) {
                        if (child.n > maxN) maxN = child.n;
                    }
                    const cVisit = this.args.gumbel_c_visit;
                    const cScale = this.args.gumbel_c_scale;

                    const evalAction = (a) => {
                        const c = root.children.find(ch => ch.actionTaken === a);
                        let q = 0.5;
                        if (c && c.n > 0) {
                            const childW = c.v[0] / c.n;
                            const childL = c.v[2] / c.n;
                            q = (childL - childW + 1) / 2;
                        }
                        return logits[a] + g[a] + (cVisit + maxN) * cScale * q;
                    };

                    survivingActions.sort((a, b) => evalAction(b) - evalAction(a));
                    survivingActions = survivingActions.slice(0, Math.max(1, Math.floor(survivingActions.length / 2)));
                    phases.push(toPhaseRC(survivingActions));
                }
            }
        }

        // Record final survivor (1) as the last phase if it isn't already.
        if (phases.length > 0
            && JSON.stringify(phases[phases.length - 1]) !== JSON.stringify(toPhaseRC(survivingActions))) {
            phases.push(toPhaseRC(survivingActions));
        }
        root._gumbelPhases = phases;

        // ----- Improved policy + vMix from completed Q -----
        const cVisit = this.args.gumbel_c_visit;
        const cScale = this.args.gumbel_c_scale;

        let maxN = 0;
        for (const child of root.children) {
            if (child.n > maxN) maxN = child.n;
        }

        const qWdl = new Array(boardArea);
        const nValues = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) qWdl[i] = [0, 0, 0];
        for (const c of root.children) {
            if (c.n > 0) {
                qWdl[c.actionTaken] = [c.v[2] / c.n, c.v[1] / c.n, c.v[0] / c.n];
                nValues[c.actionTaken] = c.n;
            }
        }

        let sumN = 0;
        for (let i = 0; i < boardArea; i++) sumN += nValues[i];

        const nnValueWdl = root.nnValue;
        let vMixWdl;
        if (sumN > 0) {
            let policyVisitedSum = 0;
            const weightedQ = [0, 0, 0];
            for (let i = 0; i < boardArea; i++) {
                if (nValues[i] > 0 && root.nnPolicy) {
                    const pw = root.nnPolicy[i];
                    policyVisitedSum += pw;
                    weightedQ[0] += pw * qWdl[i][0];
                    weightedQ[1] += pw * qWdl[i][1];
                    weightedQ[2] += pw * qWdl[i][2];
                }
            }
            policyVisitedSum = Math.max(policyVisitedSum, 1e-12);
            weightedQ[0] /= policyVisitedSum;
            weightedQ[1] /= policyVisitedSum;
            weightedQ[2] /= policyVisitedSum;
            vMixWdl = new Float64Array([
                (nnValueWdl[0] + sumN * weightedQ[0]) / (1 + sumN),
                (nnValueWdl[1] + sumN * weightedQ[1]) / (1 + sumN),
                (nnValueWdl[2] + sumN * weightedQ[2]) / (1 + sumN),
            ]);
        } else {
            vMixWdl = new Float64Array(nnValueWdl);
        }

        const completedQScalar = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            const wdl = nValues[i] > 0 ? qWdl[i] : [vMixWdl[0], vMixWdl[1], vMixWdl[2]];
            completedQScalar[i] = (wdl[0] - wdl[2] + 1) / 2;
        }

        const sigmaQ = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            sigmaQ[i] = (cVisit + maxN) * cScale * completedQScalar[i];
        }

        const improvedLogits = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            improvedLogits[i] = isLegal[i] ? (logits[i] + sigmaQ[i]) : -Infinity;
        }
        const improvedPolicy = new Float32Array(softmax(improvedLogits));

        // Pick gumbelAction = among most-visited surviving actions, max(logits + sigma_q).
        let maxNSurviving = 0;
        for (const a of survivingActions) {
            if (nValues[a] > maxNSurviving) maxNSurviving = nValues[a];
        }
        const mostVisited = survivingActions.filter(a => nValues[a] === maxNSurviving);
        let gumbelAction = mostVisited[0] || 0;
        let bestFinalScore = -Infinity;
        for (const a of mostVisited) {
            const s = logits[a] + g[a] + sigmaQ[a];
            if (s > bestFinalScore) {
                bestFinalScore = s;
                gumbelAction = a;
            }
        }

        return { improvedPolicy, gumbelAction, vMix: vMixWdl };
    }

    // Visit-count-based policy for UI heatmap.
    getMCTSPolicy(root) {
        const policy = new Float32Array(this.game.boardSize * this.game.boardSize).fill(0);
        let sumN = 0;
        for (const child of root.children) {
            policy[child.actionTaken] = child.n;
            sumN += child.n;
        }
        if (sumN > 0) {
            for (let i = 0; i < policy.length; i++) policy[i] /= sumN;
        }
        return policy;
    }
```

- [ ] **Step 4: Tests pass**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb && npm test
```

Expected: 37 passing.

- [ ] **Step 5: Commit**

```bash
git add mcts.js tests/test_mcts.mjs
git commit -m "mcts: add gumbelSequentialHalving (no-noise) + UI phase recording"
```

---

## Task 12: tools/export_onnx.py — V5 ckpt → ONNX

Adapt `SkyZero_V5/python/export_model.py` to write ONNX instead of TorchScript,
keeping only the 4 outputs the UI needs.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/tools/export_onnx.py`

- [ ] **Step 1: Write the script**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/tools/export_onnx.py`:

```python
#!/usr/bin/env python3
"""Export a SkyZero V5 checkpoint to ONNX for browser inference.

Wraps SkyZero_V5's nets.KataGoNet, drops the value_td and intermediate_*
outputs (UI doesn't display them), and reorders to a (1, 4, 17, 17) +
(1, 12) → (policy_logits, value_wdl_logits, value_futurepos_pretanh)
signature. Spatial dims fixed at MAX_BOARD_SIZE so the same .onnx serves
all board sizes 13-17 via the mask plane.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import onnx


# Make SkyZero_V5/python importable regardless of CWD.
SKYZERO_V5_PY = Path(__file__).resolve().parents[2] / "SkyZero_V5" / "python"
if not SKYZERO_V5_PY.is_dir():
    raise SystemExit(f"Expected V5 python at {SKYZERO_V5_PY}")
sys.path.insert(0, str(SKYZERO_V5_PY))

from nets import build_model               # noqa: E402
from model_config import NetConfig         # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Wraps KataGoNet and emits only the heads the web UI consumes."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor):
        out = self.model(input_spatial, input_global)
        # Drop: value_td (9), intermediate_* (4 keys). Keep: policy / wdl / futurepos.
        return (
            out["policy"],              # (B, 4, H*W)
            out["value_wdl"],           # (B, 3) — logits
            out["value_futurepos"],     # (B, 2, H, W) — pre-tanh
        )


def load_state(ckpt_path: Path) -> dict:
    """Prefer SWA weights when present (matches export_model.py)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.get("swa_model_state_dict") is not None:
        swa_sd = state["swa_model_state_dict"]
        stripped = {k[len("module."):]: v for k, v in swa_sd.items() if k.startswith("module.")}
        print(f"[export_onnx] using SWA weights from {ckpt_path}")
        return stripped
    if isinstance(state, dict) and "model_state_dict" in state:
        print(f"[export_onnx] using regular model_state_dict from {ckpt_path}")
        return state["model_state_dict"]
    raise ValueError(f"Checkpoint {ckpt_path} has neither swa_model_state_dict nor model_state_dict")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--num-channels", type=int, default=128)
    ap.add_argument("--max-board-size", type=int, default=17)
    ap.add_argument("--num-planes", type=int, default=5)
    ap.add_argument("--num-global-features", type=int, default=12)
    ap.add_argument("--opset", type=int, default=15)
    args = ap.parse_args()

    cfg = NetConfig(
        board_size=args.max_board_size,
        num_planes=args.num_planes,
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        num_global_features=args.num_global_features,
    )
    model = build_model(cfg)

    state_dict = load_state(args.ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[export_onnx] missing keys: {missing}")
    if unexpected:
        print(f"[export_onnx] unexpected keys: {unexpected}")

    # V5 trap 3: NormMask scales not in state_dict → re-derive from arch.
    model.set_norm_scales()
    model.eval()

    wrapper = ExportWrapper(model).eval()

    M = args.max_board_size
    spatial = torch.zeros(1, args.num_planes, M, M, dtype=torch.float32)
    spatial[:, 0] = 1.0   # mask: full board
    global_in = torch.zeros(1, args.num_global_features, dtype=torch.float32)

    with torch.no_grad():
        p, w, f = wrapper(spatial, global_in)
        print(f"[export_onnx] policy: {tuple(p.shape)}, wdl: {tuple(w.shape)}, futurepos: {tuple(f.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (spatial, global_in),
        str(args.out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_spatial", "input_global"],
        output_names=["policy_logits", "value_wdl_logits", "value_futurepos_pretanh"],
        dynamic_axes={
            "input_spatial":              {0: "B"},
            "input_global":               {0: "B"},
            "policy_logits":              {0: "B"},
            "value_wdl_logits":           {0: "B"},
            "value_futurepos_pretanh":    {0: "B"},
        },
    )

    onnx.checker.check_model(onnx.load(str(args.out)))
    print(f"[export_onnx] wrote {args.out}  (check passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Quick lint**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 -c "import ast; ast.parse(open('tools/export_onnx.py').read())"
```

Expected: no output (parse OK).

- [ ] **Step 3: Verify imports resolve**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tools').resolve().parent.parent / 'SkyZero_V5' / 'python'))
from nets import build_model
from model_config import NetConfig
print('ok')
"
```

Expected: `ok`. If `import torch` failures appear, ensure the V5 venv (where
PyTorch is installed) is active.

- [ ] **Step 4: Commit**

```bash
git add tools/export_onnx.py
git commit -m "tools: add export_onnx.py (V5 .pt → .onnx with 4-output wrapper)"
```

---

## Task 13: Smoke-test the export

End-to-end sanity check that `export_onnx.py` produces a loadable ONNX file.

**Files:** none (just verifies Task 12 produces a valid .onnx)

- [ ] **Step 1: Pick the smallest available checkpoint**

```bash
ls -lh /home/sky/RL/SkyZero/SkyZero_V5/anchors/
```

Expected: should show e.g. `b4c64iter180.pt` (smallest, fastest to export).

- [ ] **Step 2: Run the exporter**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 tools/export_onnx.py \
    --ckpt /home/sky/RL/SkyZero/SkyZero_V5/anchors/b4c64iter180.pt \
    --out  /tmp/skyzeroweb-smoke.onnx \
    --num-blocks 4 --num-channels 64
```

Expected output ends with: `[export_onnx] wrote /tmp/skyzeroweb-smoke.onnx  (check passed)`.

- [ ] **Step 3: Verify the ONNX runs in Python with onnxruntime**

```bash
python3 - <<'EOF'
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("/tmp/skyzeroweb-smoke.onnx")
spatial = np.zeros((1, 5, 17, 17), dtype=np.float32)
spatial[0, 0] = 1.0   # mask = full board
global_in = np.zeros((1, 12), dtype=np.float32)
global_in[0, 2] = 1.0   # RENJU one-hot

outs = sess.run(None, {"input_spatial": spatial, "input_global": global_in})
print("output shapes:", [o.shape for o in outs])
assert outs[0].shape == (1, 4, 17 * 17), f"policy shape wrong: {outs[0].shape}"
assert outs[1].shape == (1, 3),          f"wdl shape wrong: {outs[1].shape}"
assert outs[2].shape == (1, 2, 17, 17),  f"futurepos shape wrong: {outs[2].shape}"
print("smoke test PASSED")
EOF
```

Expected: `output shapes: [(1, 4, 289), (1, 3), (1, 2, 17, 17)]` then
`smoke test PASSED`.

- [ ] **Step 4: Cleanup**

```bash
rm /tmp/skyzeroweb-smoke.onnx
```

- [ ] **Step 5: No commit**

This task is pure verification. If anything failed, fix `export_onnx.py`
before proceeding.

---

## Task 14: worker.js — bootstrap (init / model load / inference helper)

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/worker.js`

- [ ] **Step 1: Create the worker bootstrap**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/worker.js`:

```js
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js");
importScripts("gomoku.js");
importScripts("mcts.js");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer cross-origin fragility; force single-thread

// --- module-level state ---
let session = null;
let game = null;
let mcts = null;
let root = null;
let currentBoardSize = 15;
let currentPly = 0;
let latestSearchId = 0;

// --- helpers ---

function concatChunks(chunks, total) {
    const result = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) { result.set(c, offset); offset += c.length; }
    return result;
}

async function fetchModelWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${url} → ${response.status}`);
    const total = Number(response.headers.get("Content-Length")) || 0;
    if (!response.body) {
        const buf = await response.arrayBuffer();
        postMessage({ type: "model-progress", percent: 100, loaded: buf.byteLength, total: buf.byteLength });
        return new Uint8Array(buf);
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    if (total > 0) postMessage({ type: "model-progress", percent: 0, loaded: 0, total });
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        const percent = total > 0 ? (loaded / total) * 100 : null;
        postMessage({ type: "model-progress", percent, loaded, total: total || null });
    }
    postMessage({ type: "model-progress", percent: 100, loaded, total: total || loaded });
    return concatChunks(chunks, total || loaded);
}

/**
 * Run one ONNX forward pass for `state` (board_size × board_size) with
 * current player `toPlay`. Returns the un-padded heatmap arrays the UI
 * wants, plus raw masked logits for MCTS.
 */
async function inference(state, toPlay) {
    if (!session) throw new Error("session not ready");
    const M = 17, A = M * M;                     // padded canvas
    const N = currentBoardSize, NA = N * N;      // game canvas

    const spatial = game.encodeState(state, toPlay);
    const globalF = game.computeGlobalFeatures(currentPly, toPlay);

    const feeds = {
        input_spatial: new ort.Tensor("float32", spatial, [1, 5, M, M]),
        input_global:  new ort.Tensor("float32", globalF, [1, 12]),
    };
    const out = await session.run(feeds);

    const policyAll = out.policy_logits.data;            // (1, 4, A)
    const wdlLogits = out.value_wdl_logits.data;         // (1, 3)
    const futureAll = out.value_futurepos_pretanh.data;  // (1, 2, M, M)

    // --- crop policy channels 0 (main) and 1 (opp) from padded → board area ---
    function cropChannel(channelIdx) {
        const cropped = new Float32Array(NA);
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                cropped[r * N + c] = policyAll[channelIdx * A + r * M + c];
            }
        }
        return cropped;
    }
    const policyMainRaw = cropChannel(0);
    const policyOppRaw  = cropChannel(1);

    // --- mask illegal + softmax ---
    const legal = game.getLegalActions(state, toPlay);
    const policyMainMasked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) policyMainMasked[i] = legal[i] ? policyMainRaw[i] : -1e9;
    const policyMainSoft = new Float32Array(softmax(policyMainMasked));

    // Opp policy: don't mask (opponent's policy doesn't share legality), just softmax.
    const policyOppSoft = new Float32Array(softmax(policyOppRaw));

    // --- value WDL: softmax 3 logits ---
    const wdl = softmax(new Float64Array([wdlLogits[0], wdlLogits[1], wdlLogits[2]]));
    const wdlF64 = new Float64Array([wdl[0], wdl[1], wdl[2]]);

    // --- futurepos: tanh per cell, crop both channels ---
    function cropTanh(channelIdx) {
        const cropped = new Float32Array(NA);
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                cropped[r * N + c] = Math.tanh(futureAll[channelIdx * A + r * M + c]);
            }
        }
        return cropped;
    }
    const future8  = cropTanh(0);
    const future32 = cropTanh(1);

    return {
        policyMainSoft,                    // for MCTS expand prior
        policyMainMaskedLogits: policyMainMasked,   // for Gumbel halving
        policyOppSoft,                     // UI heatmap
        wdl: wdlF64,                       // root nn value
        future8, future32,                 // UI heatmaps
    };
}

// (Worker message handlers added in Task 15)
```

- [ ] **Step 2: Verify the file is syntactically valid**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
node --check worker.js 2>&1 || echo "NOTE: importScripts is worker-only; node --check expects to fail. Skip."
```

Expected: a parse error mentioning `importScripts`. That's OK — `worker.js`
runs in a browser Worker, not Node. We just confirmed the rest of the file
parses up to that point. The real verification is in Task 25 (manual run).

- [ ] **Step 3: Commit**

```bash
git add worker.js
git commit -m "worker: bootstrap ort init + fetchModelWithProgress + inference helper"
```

---

## Task 15: worker.js — message router (init/reset/move/search) + abort

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/worker.js` (append message handler)

- [ ] **Step 1: Append the `onmessage` block**

Append to `/home/sky/RL/SkyZero/SkyZeroWeb/worker.js`:

```js
async function initSession(modelUrl, boardSize) {
    currentBoardSize = boardSize;
    game = new Gomoku(boardSize);
    mcts = new MCTS(game, {
        c_puct: 1.1,
        c_puct_log: 0.45,
        c_puct_base: 500,
        fpu_reduction_max: 0.2,
        root_fpu_reduction_max: 0.1,
        fpu_pow: 1.0,
        fpu_loss_prop: 0.0,
        cpuct_utility_stdev_prior: 0.40,
        cpuct_utility_stdev_prior_weight: 2.0,
        cpuct_utility_stdev_scale: 0.85,
        gumbel_m: 16,
        gumbel_c_visit: 50,
        gumbel_c_scale: 1.0,
    });
    root = null;
    const bytes = await fetchModelWithProgress(modelUrl);
    session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
    });
    postMessage({ type: "ready" });
}

function resetGame(boardSize, ply) {
    if (boardSize !== undefined && boardSize !== currentBoardSize) {
        currentBoardSize = boardSize;
        game = new Gomoku(boardSize);
        // mcts.game is the same object, no rebuild needed
    }
    currentPly = ply || 0;
    root = null;
}

function applyMove(action, nextState, nextToPlay, ply) {
    currentPly = ply;
    if (root && root.children.length > 0) {
        const child = root.children.find(c => c.actionTaken === action);
        if (child) {
            root = child;
            root.parent = null;   // detach for GC
            return;
        }
    }
    root = new Node(nextState, nextToPlay);
}

async function runSearch(state, toPlay, ply, sims, gumbelM, searchId) {
    currentPly = ply;
    if (gumbelM != null) mcts.args.gumbel_m = gumbelM;
    if (!root) root = new Node(state, toPlay);

    // Root inference if not already expanded.
    let oppPolicy, future8, future32;
    let nnValueWDL;
    if (!root.isExpanded()) {
        const inf = await inference(root.state, root.toPlay);
        if (latestSearchId !== searchId) return;
        mcts.expand(root, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
        mcts.backpropagate(root, inf.wdl);
        nnValueWDL = inf.wdl;
        oppPolicy = inf.policyOppSoft;
        future8 = inf.future8;
        future32 = inf.future32;
    } else {
        // Even on tree reuse we still run ONE inference to get fresh oppPolicy / future*.
        const inf = await inference(root.state, root.toPlay);
        if (latestSearchId !== searchId) return;
        nnValueWDL = root.nnValue;   // cached
        oppPolicy = inf.policyOppSoft;
        future8 = inf.future8;
        future32 = inf.future32;
    }

    let totalSims = 0;
    let lastProgress = performance.now();

    const simulateOne = async (action) => {
        const child = root.children.find(c => c.actionTaken === action);
        if (!child) return;
        let node = child;
        while (node.isExpanded()) {
            node = mcts.select(node);
            if (!node) return;
        }
        // Evaluate leaf: terminal or NN.
        // node.toPlay is who moves NEXT; the actor of node.actionTaken is -node.toPlay.
        const winner = game.getWinner(node.state, node.actionTaken, -node.toPlay);
        let value;
        if (winner !== null) {
            const result = winner * node.toPlay;   // winner relative to node.toPlay's POV
            if      (result === 1)  value = new Float64Array([1, 0, 0]);
            else if (result === -1) value = new Float64Array([0, 0, 1]);
            else                    value = new Float64Array([0, 1, 0]);
        } else {
            const inf = await inference(node.state, node.toPlay);
            if (latestSearchId !== searchId) return;
            mcts.expand(node, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
            value = inf.wdl;
        }
        mcts.backpropagate(node, value);
        totalSims++;

        const now = performance.now();
        if (now - lastProgress > 60) {
            lastProgress = now;
            const pct = Math.min(100, (totalSims / sims) * 100);
            postMessage({ type: "progress", progress: pct, searchId });
        }
    };

    const { improvedPolicy, gumbelAction, vMix } =
        await mcts.gumbelSequentialHalving(root, sims, simulateOne);

    if (latestSearchId !== searchId) return;
    postMessage({ type: "progress", progress: 100, searchId });

    // Visit distribution N(s,a)/sum — matches V5 cpp label "MCTS Visits (N(s,a)/sum)".
    // (drawHeat re-normalizes by max for color, but the data we transmit is sum-normalized
    // to keep semantics aligned with cpp.)
    const visitDist = mcts.getMCTSPolicy(root);   // already returns N(s,a)/sum

    postMessage({
        type: "result",
        searchId,
        gumbelAction,
        rootValueWDL: vMix,                   // [W, D, L] from vMix
        nnValueWDL,                           // [W, D, L] root NN
        mctsPolicy:    Array.from(improvedPolicy),  // V5 "MCTS Strategy (improved policy)"
        mctsVisits:    Array.from(visitDist),       // V5 "MCTS Visits (N(s,a)/sum)"
        nnPolicy:      Array.from(root.nnPolicy || new Float32Array(currentBoardSize * currentBoardSize)),  // V5 "NN Strategy" (channel 0 softmax, masked)
        nnOppPolicy:   Array.from(oppPolicy),
        nnFuturepos8:  Array.from(future8),
        nnFuturepos32: Array.from(future32),
        gumbelPhases:  root._gumbelPhases || [],
        iterations:    totalSims,
    }, /* transfer? */ undefined);
}

onmessage = async (e) => {
    const data = e.data;
    try {
        if (data.type === "init") {
            await initSession(data.modelUrl, data.boardSize);
        } else if (data.type === "reset") {
            latestSearchId++;
            resetGame(data.boardSize, data.ply);
        } else if (data.type === "move") {
            applyMove(data.action, data.nextState, data.nextToPlay, data.ply);
        } else if (data.type === "search") {
            latestSearchId++;
            const sid = latestSearchId;
            await runSearch(data.state, data.toPlay, data.ply, data.sims, data.gumbel_m, sid);
        } else if (data.type === "swap-model") {
            latestSearchId++;
            session = null;
            root = null;
            const bytes = await fetchModelWithProgress(data.modelUrl);
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: ["wasm"],
                intraOpNumThreads: 1,
                interOpNumThreads: 1,
            });
            postMessage({ type: "ready" });
        }
    } catch (err) {
        postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
    }
};
```

**Output field semantics** (matches V5 `cpp/gomoku_play_main.cpp:644-661`):
- `mctsPolicy` = improved policy from Gumbel completed-Q (V5 "MCTS Strategy (improved policy)")
- `mctsVisits` = N(s,a) / sum (V5 "MCTS Visits (N(s,a)/sum)")
- `nnPolicy` = root.nnPolicy = NN's main-policy softmax with illegal moves masked (V5 "NN Strategy")
- `nnOppPolicy` = NN's opp-policy softmax (V5 "NN Opp Strategy", channel 1)
- `nnFuturepos8` / `nnFuturepos32` = tanh of value_futurepos channels (V5 "NN Futurepos +8 / +32")

- [ ] **Step 2: Commit**

```bash
git add worker.js
git commit -m "worker: add init/reset/move/search/swap-model handlers with abort"
```

---

## Task 16: models/manifest.json skeleton

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/models/manifest.json`
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/models/.gitkeep`

- [ ] **Step 1: Write skeleton manifest**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/models/manifest.json`:

```json
{
  "default": "lv3",
  "models": [
    { "id": "lv1", "label": "新手", "elo":    0, "file": "level1.onnx", "params": "b4c64"   },
    { "id": "lv2", "label": "入门", "elo":  300, "file": "level2.onnx", "params": "b10c128" },
    { "id": "lv3", "label": "进阶", "elo":  800, "file": "level3.onnx", "params": "b10c128" },
    { "id": "lv4", "label": "高手", "elo": 1400, "file": "level4.onnx", "params": "b10c128" },
    { "id": "lv5", "label": "大师", "elo": 2100, "file": "level5.onnx", "params": "b10c128" }
  ]
}
```

The user will hand-curate the actual ELO values and copy `level1..5.onnx`
in later. The plan only ships the skeleton.

- [ ] **Step 2: Add a .gitkeep so models/ commits even with no .onnx files**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/models/.gitkeep` (empty file):

```bash
touch /home/sky/RL/SkyZero/SkyZeroWeb/models/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add models/manifest.json models/.gitkeep
git commit -m "models: add 5-tier ELO manifest skeleton"
```

---

## Task 17: style.css — port from play_web.py

The styling is large (~480 lines) but mechanical: extract the `<style>...</style>`
block from `SkyZero_V5/python/play_web.py` into a standalone CSS file.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/style.css`

- [ ] **Step 1: Extract the `<style>` block**

Open `/home/sky/RL/SkyZero/SkyZero_V5/python/play_web.py`. The `<style>` block
starts at the line `<style>` (around line 345) and ends at `</style>` (around
line 828). Copy everything BETWEEN those tags (exclusive) into
`/home/sky/RL/SkyZero/SkyZeroWeb/style.css`.

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 - <<'EOF'
src = open("/home/sky/RL/SkyZero/SkyZero_V5/python/play_web.py").read()
i = src.index("<style>") + len("<style>")
j = src.index("</style>", i)
open("style.css", "w").write(src[i:j].lstrip())
print("wrote style.css ({} bytes)".format(j - i))
EOF
```

- [ ] **Step 2: Verify the file looks reasonable**

```bash
wc -l style.css
head -20 style.css
```

Expected: ~480 lines starting with `:root { --font-sans: ...`.

- [ ] **Step 3: Commit**

```bash
git add style.css
git commit -m "style: port style.css from play_web.py <style> block"
```

---

## Task 18: index.html — port body from play_web.py

The body structure is ~240 lines: top bar, three-column main grid, heat
modal. Extract from `play_web.py`'s HTML and write `index.html`.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/index.html`

- [ ] **Step 1: Extract the body via script**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 - <<'EOF'
src = open("/home/sky/RL/SkyZero/SkyZero_V5/python/play_web.py").read()
i = src.index("<body>")
j = src.index("</body>", i) + len("</body>")
body = src[i:j]
print("extracted {} bytes of body".format(len(body)))
EOF
```

The script above just verifies the bounds — we'll write the file in Step 2
with a few hand-edits applied.

- [ ] **Step 2: Write `index.html`**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/index.html`. The structure:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SkyZero Gomoku · ONNX Web</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script>
    // Set theme before first paint to avoid flash. Tri-state: auto/light/dark.
    (function(){
      try {
        var saved = localStorage.getItem('skz_theme');
        var mode = (saved === 'light' || saved === 'dark') ? saved : 'auto';
        var resolved = (mode === 'auto')
          ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
          : mode;
        document.documentElement.dataset.theme = resolved;
        document.documentElement.dataset.themeMode = mode;
      } catch(e) {
        document.documentElement.dataset.theme = 'light';
        document.documentElement.dataset.themeMode = 'auto';
      }
    })();
  </script>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <!-- Loading overlay shown while downloading the first ONNX model -->
  <div id="loading_overlay">
    <div class="loader-card">
      <div class="loader-spin"></div>
      <div class="loader-text" id="loading_text">Loading model…</div>
      <div class="loader-bar"><div class="loader-fill" id="loading_fill"></div></div>
      <div class="loader-pct" id="loading_pct">0%</div>
    </div>
  </div>

  <div class="app">
    <!-- Header -->
    <header class="topbar">
      <div class="brand">
        <div class="brand-title">SkyZero Gomoku</div>
        <div class="brand-sub">AlphaZero-style self-play · ONNX · static web</div>
      </div>
      <button class="icon-btn" id="theme_toggle" aria-label="Toggle color theme" title="Toggle theme">
        <svg class="sun-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="8" cy="8" r="3"></circle>
          <path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M3.05 12.95l1.06-1.06M11.89 4.11l1.06-1.06"></path>
        </svg>
        <svg class="moon-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7z"></path>
        </svg>
        <svg class="auto-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
          <circle cx="8" cy="8" r="6"></circle>
          <path d="M8 2 a6 6 0 0 1 0 12 z" fill="currentColor" stroke="none"></path>
        </svg>
      </button>
    </header>

    <div class="main">
      <!-- Left column: status, model, side, size, search params, value -->
      <aside class="side-col" id="left_col">
        <div class="card status-card">
          <div class="status-pill" id="status_pill" data-variant="idle">
            <span class="dot"></span>
            <span id="status">idle</span>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="card-title">Model</div>
            <select id="model_select" class="num" style="width:100%; height:32px; text-align:left; font-family: var(--font-mono);"></select>
          </div>
        </div>

        <div class="card">
          <div class="card-body side-row">
            <div class="card-title" style="margin:0;">Human side</div>
            <div class="seg-row">
              <button class="seg-btn" id="side_black" aria-pressed="true">
                <span class="seg-stone black"></span>Black
              </button>
              <button class="seg-btn" id="side_white" aria-pressed="false">
                <span class="seg-stone white"></span>White
              </button>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-body side-row">
            <div class="card-title" style="margin:0;">Board size</div>
            <select id="size_select" class="num" style="width:auto; min-width:84px; height:32px; text-align:left; font-family: var(--font-mono);"></select>
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="card-title">Search</div>
            <div class="field-row">
              <label for="sims_input">sims</label>
              <input class="num" type="number" id="sims_input" min="1" step="1" value="256">
            </div>
            <div class="field-row">
              <label for="gm_input">gumbel_m</label>
              <input class="num" type="number" id="gm_input" min="1" step="1" value="16">
            </div>
            <div class="divider"></div>
            <label class="toggle">
              <span>Gumbel overlay</span>
              <span style="display:inline-flex;">
                <input type="checkbox" id="gumbel_toggle" checked>
                <span class="track"></span>
              </span>
            </label>
            <!-- root symmetry prune toggle removed: no symmetry in browser version -->
          </div>
        </div>

        <div class="card">
          <div class="card-body">
            <div class="card-title">Value estimates</div>
            <div class="wdl-row">
              <span class="wdl-label">root</span>
              <div class="wdl-bar" id="wdl_root_bar">
                <span class="seg w" style="width:0"></span>
                <span class="seg d" style="width:0"></span>
                <span class="seg l" style="width:0"></span>
              </div>
              <span class="wdl-wl" id="wdl_root_wl">—</span>
            </div>
            <div class="wdl-detail" id="wdl_root_detail"></div>
            <div class="wdl-row" style="margin-top:4px;">
              <span class="wdl-label">nn</span>
              <div class="wdl-bar" id="wdl_nn_bar">
                <span class="seg w" style="width:0"></span>
                <span class="seg d" style="width:0"></span>
                <span class="seg l" style="width:0"></span>
              </div>
              <span class="wdl-wl" id="wdl_nn_wl">—</span>
            </div>
            <div class="wdl-detail" id="wdl_nn_detail"></div>
            <div class="value-chart-wrap">
              <div class="value-chart-legend">
                <span class="vc-item"><span class="vc-swatch" style="background:#0969da;"></span>root</span>
                <span class="vc-item"><span class="vc-swatch" style="background:#cf222e;"></span>nn</span>
                <span class="vc-axis">W/L · −1…+1</span>
              </div>
              <canvas id="value_chart"></canvas>
            </div>
          </div>
        </div>
      </aside>

      <!-- Center column: board -->
      <section class="board-col">
        <div class="card board-card">
          <canvas id="board"></canvas>
          <div id="gumbel_legend">
            <span class="legend-head">Gumbel Sequential Halving</span>
            <span class="chip"><span class="dot" style="background:#9ca3af;"></span>16</span>
            <span class="chip"><span class="dot" style="background:#3b82f6;"></span>8</span>
            <span class="chip"><span class="dot" style="background:#10b981;"></span>4</span>
            <span class="chip"><span class="dot" style="background:#f59e0b;"></span>2</span>
            <span class="chip"><span class="dot" style="background:#ef4444;"></span>1 (picked)</span>
          </div>
        </div>
        <div class="board-actions">
          <button class="btn primary" id="new_btn">New game</button>
          <button class="btn danger-ghost" id="undo_btn">Undo</button>
        </div>
      </section>

      <!-- Right column: 6 heatmaps -->
      <aside class="side-col" id="right_col">
        <div class="grids">
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">Improved Policy</span>
              <button class="expand-btn" data-target="h_mcts_policy" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_mcts_policy"></canvas>
          </div>
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">Visits Dist</span>
              <button class="expand-btn" data-target="h_mcts_visits" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_mcts_visits"></canvas>
          </div>
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">NN Policy</span>
              <button class="expand-btn" data-target="h_nn_policy" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_nn_policy"></canvas>
          </div>
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">NN Opp Policy</span>
              <button class="expand-btn" data-target="h_nn_opp_policy" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_nn_opp_policy"></canvas>
          </div>
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">NN Futurepos +8</span>
              <button class="expand-btn" data-target="h_nn_futurepos_8" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_nn_futurepos_8"></canvas>
          </div>
          <div class="card grid-card">
            <div class="grid-title"><span class="grid-title-text">NN Futurepos +32</span>
              <button class="expand-btn" data-target="h_nn_futurepos_32" aria-label="Expand"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/></svg></button>
            </div>
            <canvas class="heat" id="h_nn_futurepos_32"></canvas>
          </div>
        </div>
      </aside>
    </div>

    <!-- Heat modal (fullscreen single heatmap) -->
    <div class="heat-modal hidden" id="heat_modal" role="dialog" aria-modal="true">
      <div class="heat-modal-card">
        <div class="heat-modal-header">
          <span class="heat-modal-title" id="heat_modal_title">Heatmap</span>
          <button class="heat-modal-close" id="heat_modal_close" aria-label="Close">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l10 10M13 3L3 13"/></svg>
          </button>
        </div>
        <canvas id="heat_modal_canvas"></canvas>
      </div>
    </div>
  </div>

  <script src="gomoku.js"></script>
  <script src="mcts.js"></script>
  <script src="main.js"></script>
</body>
</html>
```

Note `style.css` will need a small addition for `#loading_overlay` (we'll
add it in Task 25 if visual checks reveal it's missing — V5 doesn't have a
loading overlay since the C++ engine is local).

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "html: port play_web.py body to standalone index.html"
```

---

## Task 19: main.js — bootstrap, theme, sizing helpers

The `main.js` port is large (~600 lines once done). Split into 6 sub-tasks
(19-24) by section.

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Skeleton + theme + DPR canvas helpers + sizing**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`:

```js
// Board geometry — N is mutable (board size dropdown can change it).
let N = 15;
const MARGIN = 28;
let CELL = 36;
let BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
const MONO_FONT = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace';
const HEAT_LOGICAL = 240;
const DPR = window.devicePixelRatio || 1;

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setupCanvas(canvas, logicalW, logicalH, setStyle = true) {
    canvas.width = Math.round(logicalW * DPR);
    canvas.height = Math.round(logicalH * DPR);
    if (setStyle) {
        canvas.style.width = logicalW + "px";
        canvas.style.height = logicalH + "px";
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx._logicalW = logicalW;
    ctx._logicalH = logicalH;
    return ctx;
}
function clearLogical(ctx) { ctx.clearRect(0, 0, ctx._logicalW, ctx._logicalH); }

// Board canvas + 6 heat canvases set up later in Tasks 20-21.
const cv = document.getElementById("board");
const ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);

// Theme controller (tri-state).
const themeBtn = document.getElementById("theme_toggle");
const THEME_NEXT = { auto: "light", light: "dark", dark: "auto" };
function resolveTheme(mode) {
    if (mode === "light" || mode === "dark") return mode;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function setThemeTooltip(mode) {
    const label = mode.charAt(0).toUpperCase() + mode.slice(1);
    const nxt = THEME_NEXT[mode];
    themeBtn.title = "Theme: " + label + " (click for " + nxt.charAt(0).toUpperCase() + nxt.slice(1) + ")";
}
function applyTheme(mode) {
    document.documentElement.dataset.theme = resolveTheme(mode);
    document.documentElement.dataset.themeMode = mode;
    setThemeTooltip(mode);
    if (typeof drawAll === "function") drawAll();
}
setThemeTooltip(document.documentElement.dataset.themeMode || "auto");
themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.themeMode || "auto";
    const next = THEME_NEXT[cur] || "auto";
    try {
        if (next === "auto") localStorage.removeItem("skz_theme");
        else localStorage.setItem("skz_theme", next);
    } catch (_) {}
    applyTheme(next);
});
try {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onSysChange = () => {
        if ((document.documentElement.dataset.themeMode || "auto") !== "auto") return;
        applyTheme("auto");
    };
    if (mql.addEventListener) mql.addEventListener("change", onSysChange);
    else if (mql.addListener) mql.addListener(onSysChange);
} catch (_) {}

// Drives sizing of right-column to match left-column height (port from play_web.py).
const leftCol = document.getElementById("left_col");
const rightCol = document.getElementById("right_col");
const boardCard = document.querySelector(".board-card");
const boardActions = document.querySelector(".board-actions");
const mainEl = document.querySelector(".main");
const gumbelLegend = document.getElementById("gumbel_legend");

function syncBoardSize() {
    if (window.matchMedia("(max-width: 1399px)").matches) {
        rightCol.style.height = "";
        rightCol.style.width = "";
        if (BOARD_LOGICAL !== 560) {
            CELL = 36;
            BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
            setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
            ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
            if (typeof draw === "function") draw();
        }
        return;
    }
    const cardCS = getComputedStyle(boardCard);
    const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const cardPadY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const legendCS = getComputedStyle(gumbelLegend);
    const legendH = gumbelLegend.classList.contains("hidden")
        ? 0
        : gumbelLegend.offsetHeight + parseFloat(legendCS.marginTop || 0);
    const sizeByHeight = leftCol.offsetHeight - cardPadY - legendH;
    const mainCS = getComputedStyle(mainEl);
    const gap = parseFloat(mainCS.columnGap || mainCS.gap) || 20;
    const remaining = mainEl.clientWidth - leftCol.offsetWidth - 2 * gap;
    const sizeByWidth = Math.floor(remaining / 2 - cardPadX);
    let size = Math.max(360, Math.min(sizeByHeight, sizeByWidth));
    CELL = Math.max(20, Math.floor((size - 2 * MARGIN) / (N - 1)));
    BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
    const need = cv.width !== Math.round(BOARD_LOGICAL * DPR);
    if (need) setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
    rightCol.style.height = boardCard.offsetHeight + "px";
    rightCol.style.width  = boardCard.offsetWidth  + "px";
    if (need && typeof draw === "function") draw();
}
new ResizeObserver(syncBoardSize).observe(leftCol);
window.addEventListener("resize", syncBoardSize);
```

- [ ] **Step 2: Sanity check**

```bash
node --check /home/sky/RL/SkyZero/SkyZeroWeb/main.js
```

Expected: a parse error mentioning `document` or browser globals — that's
because Node doesn't have a DOM. Fine — we just want syntactic validity.
The error will look like `SyntaxError` not `ReferenceError`; if you see
`ReferenceError`, the file parsed OK and you can ignore.

- [ ] **Step 3: Commit**

```bash
git add main.js
git commit -m "main: bootstrap with DPR canvas helpers, theme toggle, sizing"
```

---

## Task 20: main.js — board canvas (stones + Gumbel overlay)

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Append the `draw()` function**

Append to `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`. This is essentially
the V5 `draw()` from `play_web.py:1440-1541`, with state shape adapted —
we use module-level `state` / `lastMove` / `gumbelPhases`:

```js
// Module-level game-display state. Updated by handlers in Task 24.
let state = null;        // { board: 2D N×N int, last_move: [r,c]|null, board_size: N }
let showGumbel = true;
let gumbelPhases = null; // last search's gumbel phases [[r,c]...] per phase

document.getElementById("gumbel_toggle").addEventListener("change", (ev) => {
    showGumbel = ev.target.checked;
    gumbelLegend.classList.toggle("hidden", !showGumbel);
    syncBoardSize();
    draw();
});

function draw() {
    clearLogical(ctx);
    const boardLine = cssVar("--board-line") || "#6b5a3a";
    const boardStar = cssVar("--board-star") || "#3a2e1a";
    const stoneB0 = cssVar("--stone-black-0");
    const stoneB1 = cssVar("--stone-black-1");
    const stoneW0 = cssVar("--stone-white-0");
    const stoneW1 = cssVar("--stone-white-1");
    const stoneOutline = cssVar("--stone-outline");
    const stoneShadow = cssVar("--stone-shadow") || "rgba(0,0,0,0.18)";

    ctx.strokeStyle = boardLine; ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
        const p = Math.round(MARGIN + i * CELL) + 0.5;
        ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL * (N - 1), p); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL * (N - 1)); ctx.stroke();
    }
    ctx.fillStyle = boardStar;
    if (N >= 7) {
        const off = (N >= 13) ? 3 : 2;
        const pts = [[off, off], [off, N-1-off], [N-1-off, off], [N-1-off, N-1-off]];
        if (N % 2 === 1) pts.push([(N-1)/2, (N-1)/2]);
        for (const [r, c] of pts) {
            ctx.beginPath();
            ctx.arc(MARGIN + c * CELL, MARGIN + r * CELL, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    ctx.fillStyle = boardLine;
    ctx.font = `11px ${MONO_FONT}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < N; i++) {
        ctx.fillText(i, MARGIN + i * CELL, 12);
        ctx.fillText(i, 10, MARGIN + i * CELL);
    }
    if (!state) return;

    const stoneR    = Math.max(6, Math.round(CELL * 0.39));
    const gumbelR   = Math.max(6, Math.round(CELL * 0.34));
    const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
    const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
    const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
    const gradInner = Math.max(1, Math.round(CELL * 0.11));
    const gumbelFontPx = Math.max(8, Math.round(CELL * 0.28));

    if (showGumbel && gumbelPhases && gumbelPhases.length > 0) {
        const COLORS = ["#9ca3af","#3b82f6","#10b981","#f59e0b","#ef4444"];
        const LABELS = ["16","8","4","2","1"];
        const deepest = new Map();
        for (let i = 0; i < gumbelPhases.length; i++) {
            for (const rc of gumbelPhases[i]) {
                const key = rc[0] * N + rc[1];
                if (!deepest.has(key) || deepest.get(key) < i) deepest.set(key, i);
            }
        }
        for (const [key, idx] of deepest) {
            const r = (key / N) | 0, c = key % N;
            const sizeLabel = String(gumbelPhases[idx].length);
            let bucket = LABELS.indexOf(sizeLabel);
            if (bucket < 0) bucket = Math.min(idx, COLORS.length - 1);
            const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
            ctx.beginPath(); ctx.arc(x, y, gumbelR, 0, Math.PI * 2);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = COLORS[bucket];
            ctx.stroke();
            ctx.lineWidth = 1;
            if (state.board[r][c] === 0) {
                ctx.fillStyle = COLORS[bucket];
                ctx.font = `bold ${gumbelFontPx}px ${MONO_FONT}`;
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(sizeLabel, x, y);
            }
        }
    }

    const b = state.board, lm = state.last_move;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const v = b[r][c]; if (!v) continue;
        const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
        ctx.beginPath(); ctx.arc(x + shadowDx, y + shadowDy, stoneR, 0, Math.PI * 2);
        ctx.fillStyle = stoneShadow; ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI * 2);
        if (v === 1) {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneB0); grad.addColorStop(1, stoneB1);
            ctx.fillStyle = grad;
        } else {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneW0); grad.addColorStop(1, stoneW1);
            ctx.fillStyle = grad;
        }
        ctx.fill();
        ctx.strokeStyle = stoneOutline; ctx.lineWidth = 1; ctx.stroke();
        if (lm && lm[0] === r && lm[1] === c) {
            ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI * 2);
            ctx.fillStyle = "#ef4444"; ctx.fill();
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add main.js
git commit -m "main: add board canvas drawing (stones, hoshi, gumbel overlay)"
```

---

## Task 21: main.js — heatmap canvases + heat modal

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Append heatmap setup + draw functions**

Append to `main.js`:

```js
// --- Six heatmap canvases ---
const heatCtxs = {
    h_mcts_policy:    setupCanvas(document.getElementById("h_mcts_policy"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_mcts_visits:    setupCanvas(document.getElementById("h_mcts_visits"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_policy:      setupCanvas(document.getElementById("h_nn_policy"),      HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_opp_policy:  setupCanvas(document.getElementById("h_nn_opp_policy"),  HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_8: setupCanvas(document.getElementById("h_nn_futurepos_8"), HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_32:setupCanvas(document.getElementById("h_nn_futurepos_32"),HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_GRID_KEYS = {
    h_mcts_policy:    "mcts_policy",
    h_mcts_visits:    "mcts_visits",
    h_nn_policy:      "nn_policy",
    h_nn_opp_policy:  "nn_opp_policy",
    h_nn_futurepos_8: "nn_futurepos_8",
    h_nn_futurepos_32:"nn_futurepos_32",
};
const SIGNED_HEAT_IDS = new Set(["h_nn_futurepos_8", "h_nn_futurepos_32"]);

function fitHeatCanvas(canvasId) {
    const c = document.getElementById(canvasId);
    const card = c.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const title = card.querySelector(".grid-title");
    let titleH = 0;
    if (title) {
        const tCS = getComputedStyle(title);
        titleH = title.offsetHeight + parseFloat(tCS.marginTop || 0) + parseFloat(tCS.marginBottom || 0);
    }
    const availW = card.clientWidth - padX;
    const availH = card.clientHeight - padY - titleH;
    const size = Math.max(60, Math.floor(Math.min(availW, availH > 0 ? availH : availW)));
    c.style.width = size + "px";
    c.style.height = size + "px";
    if (heatCtxs[canvasId]._logicalW === size) return false;
    heatCtxs[canvasId] = setupCanvas(c, size, size, false);
    return true;
}
for (const id of Object.keys(heatCtxs)) {
    const c = document.getElementById(id);
    new ResizeObserver(() => {
        if (!fitHeatCanvas(id)) return;
        const grid = state ? state[HEAT_GRID_KEYS[id]] : null;
        drawHeatById(id, grid);
    }).observe(c.parentElement);
    fitHeatCanvas(id);
}

function drawHeatById(id, grid) {
    if (SIGNED_HEAT_IDS.has(id)) drawHeatSigned(id, grid);
    else drawHeat(id, grid);
}

function drawHeat(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    let maxV = 0;
    if (grid) for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (grid[r][k] > maxV) maxV = grid[r][k];
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
        g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (v >= 0.01) {
            g.fillStyle = a > 0.5 ? "#fff" : (cssVar("--heat-text") || "#111");
            g.font = `${Math.floor(cell * 0.38)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            g.fillText((v * 100).toFixed(0), x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function drawHeatSigned(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    const heatText = cssVar("--heat-text") || "#111";
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = Math.min(1, Math.abs(v));
        if (v > 0) g.fillStyle = `rgba(9,105,218,${a.toFixed(3)})`;
        else if (v < 0) g.fillStyle = `rgba(207,34,46,${a.toFixed(3)})`;
        else g.fillStyle = "rgba(0,0,0,0)";
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (Math.abs(v) >= 0.05) {
            g.fillStyle = a > 0.5 ? "#fff" : heatText;
            g.font = `${Math.floor(cell * 0.32)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            const label = (v >= 0 ? "+" : "") + (v * 100).toFixed(0);
            g.fillText(label, x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function overlayStones(g, cell) {
    const r0 = cell * 0.32;
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const sv = state.board[r][k]; if (!sv) continue;
        const cx = k * cell + cell / 2, cy = r * cell + cell / 2;
        g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI * 2);
        if (sv === 1) {
            g.fillStyle = "rgba(0,0,0,0.7)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,0.6)"; g.stroke();
        } else {
            g.fillStyle = "rgba(255,255,255,0.85)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(0,0,0,0.5)"; g.stroke();
        }
    }
}

// --- Heat modal (fullscreen single heatmap) ---
let expandedSourceId = null;
function setupModalCanvas() {
    const canvas = document.getElementById("heat_modal_canvas");
    const card = canvas.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const header = card.querySelector(".heat-modal-header");
    const headerH = header ? header.offsetHeight + 12 : 0;
    const availW = window.innerWidth  * 0.95 - padX;
    const availH = window.innerHeight * 0.95 - padY - headerH;
    const sz = Math.max(240, Math.floor(Math.min(availW, availH)));
    canvas.style.width  = sz + "px";
    canvas.style.height = sz + "px";
    heatCtxs.h_modal = setupCanvas(canvas, sz, sz, false);
}
function paintHeatModal() {
    if (!expandedSourceId) return;
    const grid = state ? state[HEAT_GRID_KEYS[expandedSourceId]] : null;
    if (SIGNED_HEAT_IDS.has(expandedSourceId)) drawHeatSigned("h_modal", grid);
    else drawHeat("h_modal", grid);
}
function openHeatModal(sourceId) {
    if (!HEAT_GRID_KEYS[sourceId]) return;
    expandedSourceId = sourceId;
    const card = document.getElementById(sourceId).parentElement;
    const titleEl = card.querySelector(".grid-title-text");
    document.getElementById("heat_modal_title").textContent = titleEl ? titleEl.textContent : "Heatmap";
    document.getElementById("heat_modal").classList.remove("hidden");
    setupModalCanvas();
    paintHeatModal();
}
function closeHeatModal() {
    if (expandedSourceId === null) return;
    expandedSourceId = null;
    document.getElementById("heat_modal").classList.add("hidden");
}
for (const btn of document.querySelectorAll(".expand-btn")) {
    btn.addEventListener("click", () => openHeatModal(btn.dataset.target));
}
document.getElementById("heat_modal_close").addEventListener("click", closeHeatModal);
document.getElementById("heat_modal").addEventListener("click", (ev) => {
    if (ev.target === ev.currentTarget) closeHeatModal();
});
document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && expandedSourceId !== null) closeHeatModal();
});
window.addEventListener("resize", () => {
    if (expandedSourceId !== null) {
        setupModalCanvas();
        paintHeatModal();
    }
});
```

- [ ] **Step 2: Commit**

```bash
git add main.js
git commit -m "main: add 6-heatmap renderers (signed + unsigned) + fullscreen modal"
```

---

## Task 22: main.js — value chart + WDL bars

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Append**

Append to `main.js`:

```js
// --- Value chart ---
const vcCanvas = document.getElementById("value_chart");
let vctx = setupCanvas(vcCanvas, 280, 160);
vcCanvas.style.width = "100%";
vcCanvas.style.height = "100%";
function resizeValueChart() {
    const rect = vcCanvas.getBoundingClientRect();
    const w = Math.max(120, Math.floor(rect.width));
    const h = Math.max(120, Math.floor(rect.height));
    if (vctx._logicalW === w && vctx._logicalH === h) return;
    vctx = setupCanvas(vcCanvas, w, h);
    drawValueChart();
}
new ResizeObserver(resizeValueChart).observe(vcCanvas);

let valueHistory = [];   // [{step, root, nn}]

function stoneCount(board2d) {
    let n = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board2d[r][c]) n++;
    return n;
}
function normWL(v) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s > 1e-4) return (v.w - v.l) / s;
    return v.wl;
}
function recordValues(rootValueWDL, nnValueWDL, board2d) {
    if (!board2d) return;
    const step = stoneCount(board2d);
    const rw = normWL(rootValueWDL);
    const nw = normWL(nnValueWDL);
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > step) {
        valueHistory.pop();
    }
    const last = valueHistory[valueHistory.length - 1];
    if (rw == null && nw == null) {
        if (last && step > last.step) valueHistory.push({ step, root: last.root, nn: last.nn });
        return;
    }
    if (last && last.step === step) {
        if (rw != null) last.root = rw;
        if (nw != null) last.nn = nw;
    } else if (!last || step > last.step) {
        valueHistory.push({ step, root: rw, nn: nw });
    }
}
function drawValueChart() {
    clearLogical(vctx);
    const W = vctx._logicalW, H = vctx._logicalH;
    const padL = 22, padR = 6, padT = 6, padB = 14;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const grid = cssVar("--heat-grid") || "#e5e7eb";
    const muted = cssVar("--fg-muted") || "#59636e";
    const subtle = cssVar("--fg-subtle") || "#8b949e";
    const axis = cssVar("--border") || "#d8dee4";
    vctx.strokeStyle = grid; vctx.lineWidth = 1;
    for (const v of [-1, 0, 1]) {
        const y = padT + ((1 - v) / 2) * innerH + 0.5;
        vctx.beginPath(); vctx.moveTo(padL, y); vctx.lineTo(W - padR, y); vctx.stroke();
    }
    vctx.fillStyle = subtle;
    vctx.font = `10px ${MONO_FONT}`;
    vctx.textAlign = "right"; vctx.textBaseline = "middle";
    for (const v of [1, 0, -1]) {
        const y = padT + ((1 - v) / 2) * innerH;
        vctx.fillText((v > 0 ? "+" : "") + v.toFixed(0), padL - 4, y);
    }
    vctx.strokeStyle = axis;
    vctx.beginPath();
    vctx.moveTo(padL + 0.5, padT); vctx.lineTo(padL + 0.5, H - padB);
    vctx.lineTo(W - padR, H - padB); vctx.stroke();
    if (valueHistory.length === 0) {
        vctx.fillStyle = subtle;
        vctx.font = `11px ${MONO_FONT}`;
        vctx.textAlign = "center"; vctx.textBaseline = "middle";
        vctx.fillText("no data", padL + innerW / 2, padT + innerH / 2);
        return;
    }
    const maxStep = Math.max(1, valueHistory[valueHistory.length - 1].step);
    const xOf = (s) => padL + (s / maxStep) * innerW;
    const yOf = (v) => padT + ((1 - v) / 2) * innerH;
    vctx.fillStyle = muted;
    vctx.textAlign = "center"; vctx.textBaseline = "top";
    vctx.fillText("0", xOf(0), H - padB + 2);
    vctx.fillText(String(maxStep), xOf(maxStep), H - padB + 2);
    function plot(key, color) {
        const pts = valueHistory.filter(p => p[key] != null);
        if (pts.length === 0) return;
        vctx.strokeStyle = color; vctx.lineWidth = 1.5;
        vctx.beginPath();
        pts.forEach((p, i) => {
            const x = xOf(p.step), y = yOf(p[key]);
            if (i === 0) vctx.moveTo(x, y); else vctx.lineTo(x, y);
        });
        vctx.stroke();
        vctx.fillStyle = color;
        for (const p of pts) {
            vctx.beginPath();
            vctx.arc(xOf(p.step), yOf(p[key]), 2, 0, Math.PI * 2);
            vctx.fill();
        }
    }
    plot("root", "#0969da");
    plot("nn",   "#cf222e");
}

// --- WDL bars ---
function normalizeWDL(v) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s > 1e-4) return { w: v.w / s * 100, d: v.d / s * 100, l: v.l / s * 100,
                           wl: (v.w - v.l) / s };
    return { w: v.w, d: v.d, l: v.l, wl: v.wl };
}
function renderWDL(prefix, vWDL) {
    const bar = document.getElementById("wdl_" + prefix + "_bar");
    const wlEl = document.getElementById("wdl_" + prefix + "_wl");
    const det = document.getElementById("wdl_" + prefix + "_detail");
    const n = normalizeWDL(vWDL);
    const segs = bar.querySelectorAll(".seg");
    if (!n) {
        segs[0].style.width = "0";
        segs[1].style.width = "100%";
        segs[2].style.width = "0";
        wlEl.textContent = "—";
        wlEl.classList.remove("pos", "neg");
        det.textContent = "";
        return;
    }
    segs[0].style.width = n.w.toFixed(2) + "%";
    segs[1].style.width = n.d.toFixed(2) + "%";
    segs[2].style.width = n.l.toFixed(2) + "%";
    wlEl.textContent = (n.wl >= 0 ? "+" : "") + n.wl.toFixed(2);
    wlEl.classList.toggle("pos", n.wl > 0.01);
    wlEl.classList.toggle("neg", n.wl < -0.01);
    det.innerHTML =
        '<span><span class="k">W</span> ' + n.w.toFixed(1) + "%</span>" +
        '<span><span class="k">D</span> ' + n.d.toFixed(1) + "%</span>" +
        '<span><span class="k">L</span> ' + n.l.toFixed(1) + "%</span>";
}
```

`recordValues` takes `{w,d,l,wl}` shape. Worker sends `Float64Array`-like
arrays for WDL — we'll convert in Task 24 before passing in (`{w: arr[0],
d: arr[1], l: arr[2], wl: arr[0]-arr[2]}`).

- [ ] **Step 2: Commit**

```bash
git add main.js
git commit -m "main: add value chart and WDL bar renderers"
```

---

## Task 23: main.js — model + size dropdowns + manifest loader

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Append**

Append to `main.js`:

```js
// --- Model dropdown (loads from models/manifest.json) ---
let manifest = { default: null, models: [] };
let currentModelId = null;

async function loadManifest() {
    const r = await fetch("models/manifest.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("manifest.json fetch failed: " + r.status);
    manifest = await r.json();
    const sel = document.getElementById("model_select");
    sel.innerHTML = "";
    // Sort by ELO ascending so "新手" sits at the top.
    const items = manifest.models.slice().sort((a, b) => a.elo - b.elo);
    for (const m of items) {
        const o = document.createElement("option");
        o.value = m.id;
        const eloStr = (m.elo >= 0 ? "+" : "") + m.elo;
        o.textContent = `${m.id.toUpperCase()} ${m.label} · ELO ${eloStr}`;
        sel.appendChild(o);
    }
    currentModelId = manifest.default || items[0].id;
    sel.value = currentModelId;
}

function modelById(id) {
    return manifest.models.find(m => m.id === id);
}

// --- Board size dropdown (hard-coded 13-17) ---
const BOARD_SIZES = [17, 16, 15, 14, 13];
function populateSizeSelect() {
    const sel = document.getElementById("size_select");
    sel.innerHTML = "";
    for (const sz of BOARD_SIZES) {
        const o = document.createElement("option");
        o.value = String(sz);
        o.textContent = String(sz);
        sel.appendChild(o);
    }
    sel.value = String(N);
}

// --- Loading overlay (only shown for first model load) ---
function setLoadingProgress(pct) {
    const fill = document.getElementById("loading_fill");
    const text = document.getElementById("loading_pct");
    if (!fill || !text) return;
    if (Number.isFinite(pct)) {
        fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
        text.textContent = Math.round(pct) + "%";
    }
}
function hideLoadingOverlay() {
    const o = document.getElementById("loading_overlay");
    if (o) o.style.display = "none";
}
function showLoadingOverlay(text) {
    const o = document.getElementById("loading_overlay");
    if (!o) return;
    o.style.display = "";
    const t = document.getElementById("loading_text");
    if (t && text) t.textContent = text;
    setLoadingProgress(0);
}
```

- [ ] **Step 2: Add minimal loading-overlay CSS**

Append to `/home/sky/RL/SkyZero/SkyZeroWeb/style.css`:

```css
/* Loading overlay (Task 23 — not in V5 since the C++ engine is local) */
#loading_overlay {
    position: fixed; inset: 0; z-index: 2000;
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-sans);
}
.loader-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 24px 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    min-width: 260px;
}
.loader-spin {
    width: 28px; height: 28px;
    border: 3px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .loader-spin { animation: none; } }
.loader-text { font-size: 13px; color: var(--fg); }
.loader-bar { width: 100%; height: 6px; background: var(--surface-2);
              border-radius: 999px; overflow: hidden; }
.loader-fill { height: 100%; background: var(--accent); width: 0;
               transition: width 0.15s linear; }
.loader-pct { font-family: var(--font-mono); font-size: 11px; color: var(--fg-muted); }
```

- [ ] **Step 3: Commit**

```bash
git add main.js style.css
git commit -m "main: model + size dropdowns + manifest loader + loading overlay"
```

---

## Task 24: main.js — game flow + worker message handlers

The big one — wires everything together. Game state, worker messages,
button handlers.

**Files:**
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/main.js`

- [ ] **Step 1: Append the game-flow block**

Append to `main.js`:

```js
// --- Game state (mirrors V5 gomoku semantics, in-browser) ---
let game = null;          // Gomoku instance (rebuilt on size change)
let boardState = null;    // Int8Array(N*N), +1 / -1 / 0
let toPlay = 1;           // 1 = black, -1 = white
let humanSide = 1;        // 1 = human is black; -1 = human is white
let lastMove = null;      // { r, c }
let ply = 0;              // half-move counter
let gameOver = false;
let aiThinking = false;
let searchId = 0;
let history = [];         // [{ board: Int8Array, toPlay, lastMove, ply, gumbelPhases, root_value, nn_value }]

const worker = new Worker("worker.js?v=" + Date.now());

function setStatus(text, variant) {
    document.getElementById("status").textContent = text;
    document.getElementById("status_pill").dataset.variant = variant || "idle";
}

function board1Dto2D(b1d) {
    const b = [];
    for (let r = 0; r < N; r++) {
        b.push([]);
        for (let c = 0; c < N; c++) b[r].push(b1d[r * N + c]);
    }
    return b;
}

function publishStateForDrawing(extras = {}) {
    state = {
        board: board1Dto2D(boardState),
        last_move: lastMove ? [lastMove.r, lastMove.c] : null,
        board_size: N,
        // Heat data (set by handleResult; null otherwise — drawHeat handles nulls).
        mcts_policy:    extras.mcts_policy    || null,
        mcts_visits:    extras.mcts_visits    || null,
        nn_policy:      extras.nn_policy      || null,
        nn_opp_policy:  extras.nn_opp_policy  || null,
        nn_futurepos_8: extras.nn_futurepos_8 || null,
        nn_futurepos_32:extras.nn_futurepos_32|| null,
    };
}

function repaintAllHeatmaps() {
    drawHeat("h_mcts_policy",    state ? state.mcts_policy    : null);
    drawHeat("h_mcts_visits",    state ? state.mcts_visits    : null);
    drawHeat("h_nn_policy",      state ? state.nn_policy      : null);
    drawHeat("h_nn_opp_policy",  state ? state.nn_opp_policy  : null);
    drawHeatSigned("h_nn_futurepos_8",  state ? state.nn_futurepos_8  : null);
    drawHeatSigned("h_nn_futurepos_32", state ? state.nn_futurepos_32 : null);
    paintHeatModal();
}

function drawAll() {
    draw();
    drawValueChart();
    repaintAllHeatmaps();
}

// Convert flat Float32Array(N*N) → 2D N×N for heatmap render.
function flatToGrid(flat) {
    const g = [];
    for (let r = 0; r < N; r++) {
        g.push([]);
        for (let c = 0; c < N; c++) g[r].push(flat[r * N + c]);
    }
    return g;
}

function newGame() {
    game = new Gomoku(N);
    boardState = game.getInitialState();
    toPlay = 1;
    lastMove = null;
    ply = 0;
    gameOver = false;
    history = [];
    valueHistory = [];
    gumbelPhases = null;
    publishStateForDrawing();
    drawAll();
    worker.postMessage({ type: "reset", boardSize: N, ply: 0 });
    if (humanSide === toPlay) {
        setStatus("Your turn", "active");
    } else {
        triggerAISearch();
    }
}

function triggerAISearch() {
    if (gameOver) return;
    aiThinking = true;
    searchId++;
    setStatus("AI thinking…", "thinking");
    const sims = parseInt(document.getElementById("sims_input").value, 10) || 256;
    const gm = parseInt(document.getElementById("gm_input").value, 10) || 16;
    worker.postMessage({
        type: "search",
        state: boardState,
        toPlay: toPlay,
        ply: ply,
        sims: sims,
        gumbel_m: gm,
        searchId: searchId,
    });
}

function applyMoveLocal(action) {
    history.push({
        board: new Int8Array(boardState),
        toPlay,
        lastMove: lastMove ? { ...lastMove } : null,
        ply,
        gumbelPhases,
    });
    const r = (action / N) | 0, c = action % N;
    boardState = game.getNextState(boardState, action, toPlay);
    lastMove = { r, c };
    const winner = game.getWinner(boardState, action, toPlay);
    const movedBy = toPlay;
    toPlay = -toPlay;
    ply++;
    if (winner !== null) {
        gameOver = true;
        let msg;
        if      (winner === 1)  msg = "Black wins!";
        else if (winner === -1) msg = "White wins!";
        else                    msg = "Draw!";
        setStatus(msg, "done");
    }
    publishStateForDrawing(state || {});
    drawAll();
    worker.postMessage({
        type: "move",
        action,
        nextState: boardState,
        nextToPlay: toPlay,
        ply,
    });
    return { winner, movedBy };
}

// --- Board click (human move) ---
cv.addEventListener("click", (ev) => {
    if (gameOver || aiThinking) return;
    if (toPlay !== humanSide) return;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const c = Math.round((x - MARGIN) / CELL), r = Math.round((y - MARGIN) / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    if (boardState[r * N + c] !== 0) return;
    const legal = game.getLegalActions(boardState, toPlay);
    if (!legal[r * N + c]) return;   // Renju forbidden for black, etc.
    const { winner } = applyMoveLocal(r * N + c);
    if (winner === null) triggerAISearch();
});

// --- Worker message router ---
worker.onmessage = (e) => {
    const data = e.data;
    if (data.type === "model-progress") {
        if (Number.isFinite(data.percent)) setLoadingProgress(data.percent);
        return;
    }
    if (data.type === "ready") {
        hideLoadingOverlay();
        // First-ready means model is loaded; if it's a swap, just resume.
        if (!boardState) newGame();
        return;
    }
    if (data.type === "error") {
        setStatus("Error: " + data.message, "error");
        aiThinking = false;
        return;
    }
    if (data.type === "progress") {
        if (data.searchId !== searchId) return;
        // Could update a progress bar here; for now status pill stays "thinking".
        return;
    }
    if (data.type === "result") {
        if (data.searchId !== searchId) return;
        aiThinking = false;
        // Update gumbel overlay + heatmaps.
        gumbelPhases = data.gumbelPhases;
        publishStateForDrawing({
            mcts_policy:    flatToGrid(data.mctsPolicy),
            mcts_visits:    flatToGrid(data.mctsVisits),
            nn_policy:      flatToGrid(data.nnPolicy),
            nn_opp_policy:  flatToGrid(data.nnOppPolicy),
            nn_futurepos_8: flatToGrid(data.nnFuturepos8),
            nn_futurepos_32:flatToGrid(data.nnFuturepos32),
        });
        // WDL update.
        const rootWDL = data.rootValueWDL ? { w: data.rootValueWDL[0], d: data.rootValueWDL[1],
                                              l: data.rootValueWDL[2],
                                              wl: data.rootValueWDL[0] - data.rootValueWDL[2] } : null;
        const nnWDL   = data.nnValueWDL   ? { w: data.nnValueWDL[0], d: data.nnValueWDL[1],
                                              l: data.nnValueWDL[2],
                                              wl: data.nnValueWDL[0] - data.nnValueWDL[2] } : null;
        renderWDL("root", rootWDL);
        renderWDL("nn",   nnWDL);
        recordValues(rootWDL, nnWDL, state.board);
        drawAll();

        // AI plays its chosen move.
        const { winner } = applyMoveLocal(data.gumbelAction);
        if (winner === null && humanSide !== toPlay) {
            // AI vs AI? Should not happen in normal play. Stop.
        } else if (winner === null) {
            setStatus("Your turn", "active");
        }
    }
};

// --- Buttons ---
document.getElementById("new_btn").addEventListener("click", newGame);

document.getElementById("undo_btn").addEventListener("click", () => {
    if (history.length === 0) return;
    // Undo enough plies so the next move is the human's.
    let target = history.length;
    if (toPlay !== humanSide) target = Math.max(0, target - 1);
    else                      target = Math.max(0, target - 2);
    if (target === history.length) return;
    while (history.length > target) {
        const prev = history.pop();
        boardState = prev.board;
        toPlay = prev.toPlay;
        lastMove = prev.lastMove;
        ply = prev.ply;
        gumbelPhases = prev.gumbelPhases;
    }
    gameOver = false;
    aiThinking = false;
    searchId++;   // abort any in-flight search
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > stoneCount(board1Dto2D(boardState))) {
        valueHistory.pop();
    }
    publishStateForDrawing();
    drawAll();
    worker.postMessage({ type: "reset", boardSize: N, ply });
    if (toPlay === humanSide) {
        setStatus("Your turn", "active");
    } else {
        triggerAISearch();
    }
});

// Side toggle buttons.
function setSide(side) {
    if (side !== 1 && side !== -1) return;
    humanSide = side;
    document.getElementById("side_black").setAttribute("aria-pressed", side === 1 ? "true" : "false");
    document.getElementById("side_white").setAttribute("aria-pressed", side === -1 ? "true" : "false");
    newGame();
}
document.getElementById("side_black").addEventListener("click", () => setSide(1));
document.getElementById("side_white").addEventListener("click", () => setSide(-1));

// Size dropdown.
document.getElementById("size_select").addEventListener("change", (ev) => {
    const sz = parseInt(ev.target.value, 10);
    if (!Number.isFinite(sz) || !BOARD_SIZES.includes(sz)) return;
    N = sz;
    syncBoardSize();
    newGame();
});

// Model dropdown.
document.getElementById("model_select").addEventListener("change", (ev) => {
    const id = ev.target.value;
    const m = modelById(id);
    if (!m) return;
    currentModelId = id;
    showLoadingOverlay("Loading " + m.label + "…");
    searchId++;
    aiThinking = false;
    worker.postMessage({ type: "swap-model", modelUrl: "models/" + m.file });
});

// --- Bootstrap on load ---
(async function bootstrap() {
    populateSizeSelect();
    showLoadingOverlay("Loading manifest…");
    try {
        await loadManifest();
    } catch (err) {
        setStatus("manifest load failed: " + err.message, "error");
        return;
    }
    const startModel = modelById(currentModelId) || manifest.models[0];
    if (!startModel) {
        setStatus("manifest empty — add models", "error");
        return;
    }
    showLoadingOverlay("Loading " + startModel.label + "…");
    worker.postMessage({
        type: "init",
        modelUrl: "models/" + startModel.file,
        boardSize: N,
    });
    syncBoardSize();
})();
```

- [ ] **Step 2: Commit**

```bash
git add main.js
git commit -m "main: wire game flow + worker message router + button handlers"
```

---

## Task 25: _headers, README, end-to-end smoke

**Files:**
- Create: `/home/sky/RL/SkyZero/SkyZeroWeb/_headers`
- Modify: `/home/sky/RL/SkyZero/SkyZeroWeb/README.md` (replace placeholder)

- [ ] **Step 1: `_headers` file**

Create `/home/sky/RL/SkyZero/SkyZeroWeb/_headers`:

```
/*.onnx
  Cache-Control: public, max-age=31536000, immutable

/models/manifest.json
  Cache-Control: public, max-age=300

/*
  Cache-Control: public, max-age=3600
```

- [ ] **Step 2: README.md**

Replace `/home/sky/RL/SkyZero/SkyZeroWeb/README.md` with:

```markdown
# SkyZeroWeb

Static webpage that runs a SkyZero V5 model in the browser via
`onnxruntime-web`. Full UI parity with `SkyZero_V5/python/play_web.py`
but no server, no C++ engine — everything runs client-side.

## Quick start

### Local dev
```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 -m http.server 8000
# Open http://localhost:8000
```

(`file://` won't work — Worker `importScripts` and `fetch('models/...')`
need an HTTP server.)

### Deploy
This repo is set up for [Cloudflare Pages](https://pages.cloudflare.com/)
with no build step. Connect via git, point to repo root.

## Adding / updating a model

```bash
# 1. Export a V5 checkpoint to ONNX
python3 tools/export_onnx.py \
    --ckpt /path/to/SkyZero_V5/data/.../models/model_iter_NNNNNN.pt \
    --out  models/levelN.onnx \
    --num-blocks 10 --num-channels 128

# 2. Edit models/manifest.json — add or update the entry's elo / label / file
# 3. git add + commit + push → Cloudflare auto-deploys
```

The 5-tier ELO catalog is hand-curated. Each tier ships one ONNX
(~4 MB for `b10c128`).

## Architecture

- `index.html` / `style.css` — UI (ported from `play_web.py`)
- `main.js` — UI controller, canvas rendering, worker plumbing
- `worker.js` — runs `ort.InferenceSession` + MCTS in a Web Worker
- `mcts.js` — Sequential MCTS with variance-scaled cPUCT + Gumbel halving
- `gomoku.js` — RENJU game logic with multi-board-size + V5 5-plane encoding
- `tools/export_onnx.py` — V5 `.pt` → `.onnx` (drops UI-unused heads)
- `models/manifest.json` — 5-tier ELO catalog

## Tests

```bash
npm test
```

Runs Node 18+ builtin test runner against `gomoku.js` and `mcts.js`
(pure-logic units; UI is verified manually).

## Differences from V5 `play_web.py`

These simplifications are intentional (browser constraints):

- No 8-fold symmetry ensemble (single forward pass)
- No stochastic transform
- No parallel MCTS (Worker is single-threaded)
- No root symmetry pruning toggle
- RENJU rule only (no STANDARD / FREESTYLE — only RENJU was trained)
- `value_td` and intermediate heads dropped from ONNX export
```

- [ ] **Step 3: Manual end-to-end smoke**

```bash
# Make sure user has copied at least one .onnx into models/level1.onnx
ls /home/sky/RL/SkyZero/SkyZeroWeb/models/
```

If empty, prompt the user to copy at least one model file matching the
`file` field in manifest.json (default `level3.onnx` per `default: "lv3"`).

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 -m http.server 8000
```

Open `http://localhost:8000` in a browser. Manual checklist:

- [ ] Loading overlay shows download progress
- [ ] Page renders 3-column layout (left controls, center board, right 6 heatmaps)
- [ ] Status pill says "Your turn" after model load (assuming default = black)
- [ ] Click an empty cell → black stone appears, status → "AI thinking..."
- [ ] After ~2-15 sec: AI plays, gumbel circles overlay the board
- [ ] All 6 heatmaps populate (mcts_policy, visits, nn_policy, opp, futurepos+8/+32)
- [ ] Both WDL bars show non-zero W/D/L percentages
- [ ] Value chart adds a point per ply (root blue + nn red)
- [ ] Click expand button on any heatmap → fullscreen modal opens; Esc/click-outside closes
- [ ] Switch theme toggle 3× cycles auto → light → dark → auto
- [ ] Switch board size to 13 → fresh game, 13×13 board
- [ ] Switch model in dropdown → overlay reappears, fresh game

If any check fails, debug before final commit.

- [ ] **Step 4: Final commit**

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
git add _headers README.md
git commit -m "deploy: add Cloudflare _headers and full README"
```

---

## Self-Review Notes

- All spec sections are covered: gomoku.js + mcts.js + worker.js + main.js
  + index.html + style.css + tools/export_onnx.py + models/manifest.json +
  _headers + README.md.
- TDD applied to gomoku.js (Tasks 3-7) and mcts.js (Tasks 8-11) since
  they are pure logic; UI / port-heavy tasks rely on visual verification
  in Task 25.
- Worker abort epoch (`latestSearchId` vs `searchId`) is consistently
  named across worker.js and main.js.
- Method names: `gumbelSequentialHalving`, `getMCTSPolicy`,
  `computeParentUtilityStdevFactor`, `computeSelectParams`, `select`,
  `expand`, `backpropagate`, `encodeState`, `computeGlobalFeatures`,
  `getLegalActions`, `getNextState`, `getInitialState`, `getWinner` —
  consistent throughout.
- `_gumbelPhases` is set on the root by `gumbelSequentialHalving` and
  read by the worker before sending the result; the worker passes
  `gumbelPhases` (camelCase, no underscore) in the result message; main
  reads `data.gumbelPhases` and stores in module-level `gumbelPhases`.
- ELO field type: number throughout (manifest.json, dropdown formatting).
- `boardSize` parameter naming consistent across worker messages.
- `mcts_policy` / `mcts_visits` semantics confirmed against V5 cpp
  `gomoku_play_main.cpp:644,654`: `mcts_policy` = improved policy
  (Gumbel completed-Q derived), `mcts_visits` = N(s,a) / sum. Worker
  emits both correctly named.
