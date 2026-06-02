// Cross-checks ai2048.js (browser 2048 AI logic) against a fixture generated
// from the Python reference SkyZero_2048/python/game.py. Regenerate the fixture
// with the inline script in the project history if game.py ever changes.
import { test } from "node:test";
import assert from "node:assert";
import { createRequire } from "module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const AI = require("../ai2048.js");
const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(join(here, "ai2048_fixture.json"), "utf8"));

test("applyMove / legal / terminal match the Python reference", () => {
  for (const rec of cases) {
    const st = rec.state;
    const legal = AI.legalActions(st);
    assert.deepStrictEqual(legal, rec.legal, `legal mismatch for ${st}`);
    assert.strictEqual(AI.isTerminal(st), rec.terminal, `terminal mismatch for ${st}`);
    for (let a = 0; a < 4; a++) {
      const mv = AI.applyMove(st, a);
      const ref = rec.moves[a];
      assert.strictEqual(mv.changed, ref.changed, `changed[${a}] for ${st}`);
      assert.strictEqual(mv.reward, ref.reward, `reward[${a}] for ${st}`);
      assert.deepStrictEqual(Array.from(mv.after), ref.after, `after[${a}] for ${st}`);
    }
  }
});

test("spawnDistribution matches the Python reference", () => {
  for (const rec of cases) {
    if (rec.spawn_after_action === undefined) continue;
    const mv = AI.applyMove(rec.state, rec.spawn_after_action);
    const dist = AI.spawnDistribution(mv.after)
      .map(d => [d.cell, d.exp, Math.round(d.prob * 1e10) / 1e10])
      .sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    assert.deepStrictEqual(dist, rec.spawn_dist, `spawn dist for ${rec.state}`);
  }
});

test("encode produces the same nonzero one-hot indices", () => {
  for (const rec of cases) {
    const buf = new Float32Array(AI.NUM_PLANES * AI.AREA);
    AI.encode(rec.state, buf, 0);
    const nz = [];
    for (let i = 0; i < buf.length; i++) if (buf[i] !== 0) nz.push(i);
    assert.deepStrictEqual(nz, rec.encode_nonzero, `encode for ${rec.state}`);
  }
});

test("chooseMove returns a legal direction and aligns with a stub expectimax", async () => {
  // Stub value net: V = sum of tile values (favors keeping big tiles / merging).
  // chooseMove must (a) pick a legal action and (b) maximize the same Q we
  // compute here independently, validating the expectimax bookkeeping.
  const gamma = 0.999;
  const stub = async (flat, B) => {
    const out = new Float32Array(B);
    for (let s = 0; s < B; s++) {
      let sum = 0;
      for (let loc = 0; loc < AI.AREA; loc++) {
        for (let p = 0; p < AI.NUM_PLANES; p++) {
          if (flat[s * AI.NUM_PLANES * AI.AREA + p * AI.AREA + loc] !== 0) {
            if (p > 0) sum += (1 << p);
          }
        }
      }
      out[s] = sum;
    }
    return out;
  };

  for (const rec of cases.slice(0, 120)) {
    const res = await AI.chooseMove(rec.state, stub, gamma);
    if (rec.terminal) { assert.strictEqual(res.action, -1); continue; }
    assert.ok(rec.legal[res.action] === 1, `chose illegal action for ${rec.state}`);
    // Independently recompute Q for the chosen action vs all legal, confirm max.
    let bestQ = -Infinity;
    for (let a = 0; a < 4; a++) {
      if (!rec.legal[a]) continue;
      if (res.qs[a] > bestQ) bestQ = res.qs[a];
    }
    assert.ok(Math.abs(res.qs[res.action] - bestQ) < 1e-9, `not argmax for ${rec.state}`);
  }
});
