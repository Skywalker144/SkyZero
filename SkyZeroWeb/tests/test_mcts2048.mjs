// Cross-checks mcts2048.js (browser afterstate Stochastic Gumbel AlphaZero MCTS)
// against a fixture generated from SkyZero_2048/python/mcts.py. Both sides run
// the SAME deterministic synthetic evaluator (gumbel_noise OFF), so the whole
// search must agree: a real porting bug shows up as a gross mismatch, not a
// float ulp. Regenerate the fixture with tests/gen_mcts2048_fixture.py if
// python/mcts.py changes.
import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const AI = require("../ai2048.js");
const MCTS = require("../mcts2048.js");
const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(here, "mcts2048_fixture.json"), "utf8"));

const CFG = { num_simulations: 64, gamma: 0.999, c_puct: 1.25,
              gumbel_c_visit: 50.0, gumbel_c_scale: 1.0, gumbel_noise: false };

// Same synthetic evaluator as gen_mcts2048_fixture.py:
//   value = sum 2**exp ; logits[a] = applyMove(state,a).reward * 0.01.
function runNet(flat, B) {
  const P = AI.NUM_PLANES * AI.AREA;
  const logits = new Float32Array(B * 4);
  const values = new Float32Array(B);
  for (let s = 0; s < B; s++) {
    // Decode the one-hot planes back to an exponent board.
    const st = new Array(AI.AREA).fill(0);
    let sum = 0;
    for (let loc = 0; loc < AI.AREA; loc++) {
      for (let p = 0; p < AI.NUM_PLANES; p++) {
        if (flat[s * P + p * AI.AREA + loc] !== 0) { st[loc] = p; if (p > 0) sum += (1 << p); }
      }
    }
    values[s] = sum;
    for (let a = 0; a < 4; a++) logits[s * 4 + a] = AI.applyMove(st, a).reward * 0.01;
  }
  return Promise.resolve({ logits, values });
}

test("mcts2048 matches python/mcts.py on best action (deterministic, no noise)", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    assert.strictEqual(res.action, rec.best_action,
      `best_action mismatch for ${rec.state}: js=${res.action} py=${rec.best_action}`);
  }
});

test("mcts2048 root value + improved policy match python to tolerance", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    const relV = Math.abs(res.value - rec.root_value) / Math.max(1, Math.abs(rec.root_value));
    assert.ok(relV < 1e-4,
      `root_value mismatch for ${rec.state}: js=${res.value} py=${rec.root_value}`);
    for (let a = 0; a < 4; a++) {
      assert.ok(Math.abs(res.qs[a] - rec.improved_policy[a]) < 1e-4,
        `improved_policy[${a}] mismatch for ${rec.state}: js=${res.qs[a]} py=${rec.improved_policy[a]}`);
    }
  }
});

test("mcts2048 visit counts match python (sum = sims, per-action equal)", async () => {
  for (const rec of cases) {
    const res = await MCTS.chooseMoveMCTS(rec.state, runNet, CFG);
    const sum = res.visits.reduce((x, y) => x + y, 0);
    assert.strictEqual(sum, CFG.num_simulations, `visit sum != sims for ${rec.state}`);
    assert.deepStrictEqual(res.visits, rec.visits,
      `visit counts mismatch for ${rec.state}: js=${res.visits} py=${rec.visits}`);
  }
});
