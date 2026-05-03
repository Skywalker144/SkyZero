import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { Node, MCTS, softmax } = require("../mcts.js");

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
