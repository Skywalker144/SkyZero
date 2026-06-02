// Browser-side 2048 AI: afterstate logic + a 1-ply expectimax over the value
// network. Faithful JS port of SkyZero_2048/python/game.py (slide / spawn /
// encode) and the chance-node value backup from python/mcts.py, but instead of
// a full Gumbel MCTS this does a single expectimax expansion:
//
//     Q(a) = reward(a) + gamma * E_spawn[ V(decision_state) ]
//
// where V is the network's scalar value head (raw 2048 points; the ONNX export
// already folds value_scale in). The move with the highest Q is played. This is
// strong for 2048 (it uses the value net the way the search does at depth 1)
// while needing only one batched forward and no tree.
//
// State convention mirrors game.py: a length-16 array of EXPONENTS, row-major
// (loc = r*4 + c); 0 = empty, e>0 = tile 2**e. Directions: 0=up 1=right 2=down
// 3=left, matching both game.py and 2048.html's `lineCoord` slide convention.
//
// Loads in a Web Worker (attaches to `self`) and in Node (module.exports) so
// tests/ can exercise the slide logic against the Python reference.
(function (root) {
  'use strict';

  var SIZE = 4;
  var AREA = 16;
  var NUM_ACTIONS = 4;
  var NUM_PLANES = 16;
  var PROB_2 = 0.9;   // spawns exponent 1 (tile 2)
  var PROB_4 = 0.1;   // spawns exponent 2 (tile 4)
  var DIR_NAMES = ['up', 'right', 'down', 'left'];

  // The 4 board locs of `line` for `action`, ordered from the moving edge
  // inward (slot 0 = edge). Mirrors game.py:_line_indices.
  function lineIndices(action, line) {
    var idx = new Array(SIZE);
    for (var i = 0; i < SIZE; i++) {
      var r, c;
      if (action === 0) { r = i; c = line; }            // up
      else if (action === 1) { r = line; c = SIZE - 1 - i; } // right
      else if (action === 2) { r = SIZE - 1 - i; c = line; } // down
      else { r = line; c = i; }                         // left
      idx[i] = r * SIZE + c;
    }
    return idx;
  }

  // Compress + single-merge a 4-exponent line toward index 0.
  // Returns { out:[4], reward, changed }. Mirrors game.py:_slide_line.
  function slideLine(vals) {
    var packed = [];
    for (var k = 0; k < SIZE; k++) if (vals[k] !== 0) packed.push(vals[k]);
    var out = [];
    var reward = 0;
    var i = 0;
    while (i < packed.length) {
      if (i + 1 < packed.length && packed[i] === packed[i + 1]) {
        var merged = packed[i] + 1;
        out.push(merged);
        reward += (1 << merged);   // merged tile value = 2**merged
        i += 2;
      } else {
        out.push(packed[i]);
        i += 1;
      }
    }
    while (out.length < SIZE) out.push(0);
    var changed = false;
    for (var j = 0; j < SIZE; j++) if (out[j] !== vals[j]) { changed = true; break; }
    return { out: out, reward: reward, changed: changed };
  }

  // Apply a slide to an exponent board. Returns { after, reward, changed }.
  // `after` is a new array; on no-change it is a copy of `state`.
  function applyMove(state, action) {
    var after = state.slice();
    var reward = 0;
    var changed = false;
    for (var line = 0; line < SIZE; line++) {
      var locs = lineIndices(action, line);
      var vals = [state[locs[0]], state[locs[1]], state[locs[2]], state[locs[3]]];
      var res = slideLine(vals);
      reward += res.reward;
      if (res.changed) changed = true;
      for (var j = 0; j < SIZE; j++) after[locs[j]] = res.out[j];
    }
    if (!changed) return { after: state.slice(), reward: 0, changed: false };
    return { after: after, reward: reward, changed: true };
  }

  function legalActions(state) {
    var legal = [0, 0, 0, 0];
    for (var a = 0; a < NUM_ACTIONS; a++) legal[a] = applyMove(state, a).changed ? 1 : 0;
    return legal;
  }

  function isTerminal(state) {
    for (var a = 0; a < NUM_ACTIONS; a++) if (applyMove(state, a).changed) return false;
    return true;
  }

  // Enumerate the known spawn distribution of an afterstate.
  // Returns [{ cell, exp, prob }]. Mirrors game.py:spawn_distribution.
  function spawnDistribution(afterstate) {
    var empties = [];
    for (var i = 0; i < AREA; i++) if (afterstate[i] === 0) empties.push(i);
    if (empties.length === 0) return [];
    var per = 1.0 / empties.length;
    var out = [];
    for (var e = 0; e < empties.length; e++) {
      out.push({ cell: empties[e], exp: 1, prob: per * PROB_2 });
      out.push({ cell: empties[e], exp: 2, prob: per * PROB_4 });
    }
    return out;
  }

  // One-hot encode an exponent board into NUM_PLANES*AREA floats, plane-major:
  // enc[plane*AREA + loc]. Matches game.py:encode_state flattened to (16,4,4).
  function encode(state, outBuf, offset) {
    offset = offset || 0;
    for (var loc = 0; loc < AREA; loc++) {
      var p = state[loc];
      if (p < 0) p = 0; else if (p > NUM_PLANES - 1) p = NUM_PLANES - 1;
      outBuf[offset + p * AREA + loc] = 1.0;
    }
  }

  // Convert 2048.html's value grid[r][c] (actual tile values, 0 = empty) into a
  // flat length-16 exponent array.
  function gridValuesToExps(grid) {
    var exps = new Array(AREA).fill(0);
    for (var r = 0; r < SIZE; r++) {
      for (var c = 0; c < SIZE; c++) {
        var v = grid[r][c];
        exps[r * SIZE + c] = v > 0 ? Math.round(Math.log2(v)) : 0;
      }
    }
    return exps;
  }

  // Pick a move by 1-ply expectimax.
  //   state    : length-16 exponent array
  //   runBatch : async (Float32Array[B*NUM_PLANES*AREA], B) -> Float32Array[B]
  //              of value-head outputs in raw 2048 points
  //   gamma    : discount (default 0.999, matching Config.gamma)
  // Returns { action, dir, qs:[4], value } ; action -1 (dir null) if terminal.
  async function chooseMove(state, runBatch, gamma) {
    if (gamma == null) gamma = 0.999;
    var legal = legalActions(state);
    var actions = [];
    for (var a = 0; a < NUM_ACTIONS; a++) if (legal[a]) actions.push(a);
    if (actions.length === 0) return { action: -1, dir: null, qs: [0, 0, 0, 0], value: 0 };

    // Build the batch of non-terminal spawn decision-states, remembering which
    // action and probability each contributes to.
    var rewards = {};                 // action -> immediate reward
    var contrib = [];                 // {action, prob} per batched state
    var states = [];                  // exponent arrays needing a net eval
    var qConst = [0, 0, 0, 0];        // accumulated reward + gamma*prob*0 part

    for (var ai = 0; ai < actions.length; ai++) {
      var act = actions[ai];
      var mv = applyMove(state, act);
      rewards[act] = mv.reward;
      qConst[act] = mv.reward;        // terminal spawns contribute V=0
      var dist = spawnDistribution(mv.after);
      for (var d = 0; d < dist.length; d++) {
        var sp = dist[d];
        var ns = mv.after.slice();
        ns[sp.cell] = sp.exp;
        if (isTerminal(ns)) continue; // V(terminal) = 0
        contrib.push({ action: act, prob: sp.prob });
        states.push(ns);
      }
    }

    var values = null;
    if (states.length > 0) {
      var B = states.length;
      var flat = new Float32Array(B * NUM_PLANES * AREA);
      for (var s = 0; s < B; s++) encode(states[s], flat, s * NUM_PLANES * AREA);
      values = await runBatch(flat, B);
    }

    var qs = qConst.slice();
    if (values) {
      for (var i = 0; i < contrib.length; i++) {
        qs[contrib[i].action] += gamma * contrib[i].prob * values[i];
      }
    }

    var best = actions[0];
    for (var j = 1; j < actions.length; j++) if (qs[actions[j]] > qs[best]) best = actions[j];
    return { action: best, dir: DIR_NAMES[best], qs: qs, value: qs[best] };
  }

  var api = {
    SIZE: SIZE, AREA: AREA, NUM_ACTIONS: NUM_ACTIONS, NUM_PLANES: NUM_PLANES,
    DIR_NAMES: DIR_NAMES,
    lineIndices: lineIndices, slideLine: slideLine, applyMove: applyMove,
    legalActions: legalActions, isTerminal: isTerminal,
    spawnDistribution: spawnDistribution, encode: encode,
    gridValuesToExps: gridValuesToExps, chooseMove: chooseMove,
  };

  root.AI2048 = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : this);
