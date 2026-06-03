// Browser-side afterstate Stochastic *Gumbel* AlphaZero MCTS for 2048.
//
// Faithful JS port of SkyZero_2048/python/mcts.py (the GameSearch class). The
// tree alternates DECISION nodes (player picks 1 of 4 directions) and CHANCE
// nodes (env spawns a tile from a KNOWN distribution). Single-agent => scalar
// value = expected discounted future score; backup is a plain discounted
// accumulation G = r + gamma*G_child with NO perspective flip. PUCT compares Q
// after MuZero-style min-max normalization (returns reach tens of thousands).
// The root runs Gumbel-Top-k + sequential halving (Danihelka et al. 2022).
//
// Unlike the Python loop (which batches many independent games per sim step),
// the browser plays one game, so each simulation evaluates a single leaf via an
// async `runNet(flat, B) -> {logits:Float32Array[B*4], values:Float32Array[B]}`
// (raw 2048 points; the ONNX export folds value_scale + h_inv into the graph).
//
// Mirrors mcts.js (the Gomoku web MCTS): attaches to `self` in a Worker and to
// module.exports in Node so tests/ can cross-check against python/mcts.py.
(function (root) {
  'use strict';

  var AI = root.AI2048 ||
    (typeof require !== 'undefined' ? require('./ai2048.js') : null);
  if (!AI) throw new Error('mcts2048.js requires ai2048.js (AI2048) to be loaded first');

  var NEG = -1e30;

  // Defaults mirror SkyZero_2048 model_config.Config + configs/vtransform/run.cfg.
  // gumbel_noise OFF => deterministic strongest play (the demo always plays best).
  var DEFAULTS = {
    gamma: 0.999,
    num_simulations: 64,
    c_puct: 1.25,
    gumbel_c_visit: 50.0,
    gumbel_c_scale: 1.0,
    gumbel_noise: false,
  };

  function softmax(x) {
    var m = -Infinity, i;
    for (i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    var e = new Array(x.length), s = 0;
    for (i = 0; i < x.length; i++) { e[i] = Math.exp(x[i] - m); s += e[i]; }
    if (s > 0) { for (i = 0; i < x.length; i++) e[i] /= s; }
    else { for (i = 0; i < x.length; i++) e[i] = 1.0 / x.length; }
    return e;
  }

  function sampleGumbel() {
    var u = Math.random();
    u = Math.max(1e-20, Math.min(1 - 1e-10, u));
    return -Math.log(-Math.log(u));
  }

  // MuZero-style running min/max for Q normalization (mirrors mcts.py:_MinMax).
  function MinMax() { this.lo = Infinity; this.hi = -Infinity; }
  MinMax.prototype.update = function (v) {
    if (v < this.lo) this.lo = v;
    if (v > this.hi) this.hi = v;
  };
  MinMax.prototype.norm = function (v) {
    if (this.hi > this.lo) return (v - this.lo) / (this.hi - this.lo);
    return 0.5;
  };

  function Decision(state) {
    this.state = state;
    this.terminal = false;
    this.expanded = false;
    this.prior = [0, 0, 0, 0];
    this.children = [null, null, null, null];  // Chance | null per action
    this.nnValue = 0.0;
    this.n = 0;
    this.w = 0.0;
    this.logits = [NEG, NEG, NEG, NEG];
  }
  Decision.prototype.value = function () {
    return this.n > 0 ? this.w / this.n : this.nnValue;
  };

  function Chance(afterstate, reward) {
    this.afterstate = afterstate;
    this.reward = reward;
    // each edge: { prob, cell, exp, child:Decision|null }
    this.edges = [];
    this.n = 0;
    this.w = 0.0;
  }
  Chance.prototype.q = function () {
    return this.n > 0 ? this.w / this.n : 0.0;
  };

  // One game's search tree + Gumbel root scheduler. Leaf evaluation is deferred
  // to the async driver via selectLeaf()/applyEval() (mirrors mcts.py).
  function GameSearch(state, cfg) {
    this.cfg = cfg;
    this.root = new Decision(state);
    this.root.terminal = AI.isTerminal(state);
    this.stats = new MinMax();
    this._pendingPath = null;
    this._g = [0, 0, 0, 0];
    this._rootLogits = [NEG, NEG, NEG, NEG];
    this._active = [];
    this._phase = 0;
    this._phaseBudgets = [];
    this._inPhase = 0;
    this._rr = 0;
    this._ready = false;
  }

  // ---- root expansion ----
  GameSearch.prototype.rootLeaf = function () {
    return this.root.terminal ? null : this.root.state;
  };

  GameSearch.prototype.applyRootEval = function (logits, value) {
    if (this.root.terminal) return;
    this._expand(this.root, logits, value);
    this._backupNode(this.root, value);
    this._setupGumbel();
  };

  GameSearch.prototype._setupGumbel = function () {
    var legal = AI.legalActions(this.root.state);
    var a;
    for (a = 0; a < 4; a++) {
      this._rootLogits[a] = legal[a] > 0 ? this.root.logits[a] : NEG;
    }
    this._g = [0, 0, 0, 0];
    if (this.cfg.gumbel_noise) {
      for (a = 0; a < 4; a++) this._g[a] = sampleGumbel();
    }
    var scores = [0, 0, 0, 0];
    for (a = 0; a < 4; a++) scores[a] = this._rootLogits[a] + this._g[a];
    var cands = [];
    for (a = 0; a < 4; a++) if (legal[a] > 0) cands.push(a);
    cands.sort(function (x, y) { return scores[y] - scores[x]; });
    this._active = cands;
    var m = cands.length;
    var phases = m > 1 ? Math.max(1, Math.ceil(Math.log2(m))) : 1;
    var n = this.cfg.num_simulations;
    var base = Math.floor(n / phases);
    var rem = n - base * phases;
    this._phaseBudgets = [];
    for (var i = 0; i < phases; i++) this._phaseBudgets.push(base + (i < rem ? 1 : 0));
    this._phase = 0;
    this._inPhase = 0;
    this._rr = 0;
    this._ready = true;
  };

  // ---- one simulation: pick root action, descend, return leaf to eval ----
  GameSearch.prototype.selectLeaf = function () {
    if (this.root.terminal || !this._ready || this._active.length === 0) return null;
    var a = this._nextRootAction();
    var path = [this.root];
    var rewards = [];
    var chance = this.root.children[a];
    path.push(chance);
    rewards.push(chance.reward);
    var node = this._descendChance(chance);
    while (true) {
      path.push(node);
      if (node.terminal) {
        this._backupPath(path, rewards, 0.0);
        return null;
      }
      if (!node.expanded) {
        this._pendingPath = { path: path, rewards: rewards };
        return node.state;
      }
      var a2 = this._selectAction(node);
      var ch = node.children[a2];
      path.push(ch);
      rewards.push(ch.reward);
      node = this._descendChance(ch);
    }
  };

  GameSearch.prototype.applyEval = function (logits, value) {
    var path = this._pendingPath.path;
    var rewards = this._pendingPath.rewards;
    var leaf = path[path.length - 1];
    this._expand(leaf, logits, value);
    this._backupPath(path, rewards, value);
    this._pendingPath = null;
  };

  // ---- Gumbel sequential halving over root candidates ----
  GameSearch.prototype._nextRootAction = function () {
    var m = this._active.length;
    var a = this._active[this._rr % m];
    this._rr += 1;
    this._inPhase += 1;
    if (this._phase < this._phaseBudgets.length - 1 &&
        this._inPhase >= this._phaseBudgets[this._phase]) {
      var self = this;
      this._active.sort(function (x, y) { return self._rootScore(y) - self._rootScore(x); });
      var keep = Math.max(1, Math.floor((m + 1) / 2));
      this._active = this._active.slice(0, keep);
      this._phase += 1;
      this._inPhase = 0;
      this._rr = 0;
    }
    return a;
  };

  GameSearch.prototype._rootScore = function (a) {
    var ch = this.root.children[a];
    var q = (ch && ch.n > 0) ? ch.q() : this.root.value();
    return this._rootLogits[a] + this._g[a] + this._sigma(q);
  };

  GameSearch.prototype._sigma = function (q) {
    var maxN = 0;
    for (var a = 0; a < 4; a++) {
      var c = this.root.children[a];
      if (c && c.n > maxN) maxN = c.n;
    }
    return (this.cfg.gumbel_c_visit + maxN) * this.cfg.gumbel_c_scale * this.stats.norm(q);
  };

  // ---- PUCT at non-root decision nodes ----
  GameSearch.prototype._selectAction = function (node) {
    var sqrtN = Math.sqrt(Math.max(1, node.n));
    var best = -Infinity, bestA = -1;
    for (var a = 0; a < 4; a++) {
      var ch = node.children[a];
      if (ch === null) continue;
      var q = ch.n > 0 ? ch.q() : (ch.reward + this.cfg.gamma * node.value());
      var score = this.stats.norm(q) +
        this.cfg.c_puct * node.prior[a] * sqrtN / (1 + ch.n);
      if (score > best) { best = score; bestA = a; }
    }
    return bestA;
  };

  // ---- chance node: deterministic "most under-represented" descent ----
  GameSearch.prototype._descendChance = function (ch) {
    var total = ch.n;
    var bestI = -1, bestDef = -Infinity;
    for (var i = 0; i < ch.edges.length; i++) {
      var e = ch.edges[i];
      var cn = e.child !== null ? e.child.n : 0;
      var frac = total > 0 ? cn / total : 0.0;
      var deficit = e.prob - frac;
      if (deficit > bestDef) { bestDef = deficit; bestI = i; }
    }
    var edge = ch.edges[bestI];
    if (edge.child === null) {
      var ns = edge_apply(ch.afterstate, edge);
      var d = new Decision(ns);
      d.terminal = AI.isTerminal(ns);
      edge.child = d;
    }
    return edge.child;
  };

  function edge_apply(afterstate, edge) {
    var ns = afterstate.slice();
    ns[edge.cell] = edge.exp;
    return ns;
  }

  // ---- expand / backup ----
  GameSearch.prototype._expand = function (node, logits, value) {
    node.expanded = true;
    node.nnValue = value;
    var legal = AI.legalActions(node.state);
    var masked = [NEG, NEG, NEG, NEG];
    var a;
    for (a = 0; a < 4; a++) masked[a] = legal[a] > 0 ? logits[a] : NEG;
    node.prior = softmax(masked);
    node.logits = masked;
    for (a = 0; a < 4; a++) {
      if (legal[a] > 0) {
        var mv = AI.applyMove(node.state, a);
        var ch = new Chance(mv.after, mv.reward);
        var dist = AI.spawnDistribution(mv.after);
        for (var d = 0; d < dist.length; d++) {
          ch.edges.push({ prob: dist[d].prob, cell: dist[d].cell, exp: dist[d].exp, child: null });
        }
        node.children[a] = ch;
      }
    }
  };

  GameSearch.prototype._backupNode = function (node, value) {
    node.n += 1;
    node.w += value;
    this.stats.update(value);
  };

  // path = [dec0, ch0, dec1, ch1, ..., decK]; rewards align with chance nodes.
  GameSearch.prototype._backupPath = function (path, rewards, leafValue) {
    var g = leafValue;
    var leaf = path[path.length - 1];
    leaf.n += 1;
    leaf.w += g;
    this.stats.update(g);
    var ci = rewards.length - 1;
    var i = path.length - 2;
    while (i >= 0) {
      var node = path[i];
      if (node instanceof Chance) {
        g = rewards[ci] + this.cfg.gamma * g;
        ci -= 1;
      }
      node.n += 1;
      node.w += g;
      this.stats.update(g);
      i -= 1;
    }
  };

  // ---- results ----
  GameSearch.prototype.improvedPolicy = function () {
    var legal = AI.legalActions(this.root.state);
    var vmix = this._vMix();
    var comp = [NEG, NEG, NEG, NEG];
    for (var a = 0; a < 4; a++) {
      if (legal[a] > 0) {
        var ch = this.root.children[a];
        var q = (ch && ch.n > 0) ? ch.q() : vmix;
        comp[a] = this._rootLogits[a] + this._sigma(q);
      }
    }
    return softmax(comp);
  };

  GameSearch.prototype._vMix = function () {
    var sumN = 0, a, c;
    for (a = 0; a < 4; a++) { c = this.root.children[a]; if (c) sumN += c.n; }
    if (sumN === 0) return this.root.nnValue;
    var wq = 0.0, psum = 1e-12;
    for (a = 0; a < 4; a++) {
      c = this.root.children[a];
      if (c && c.n > 0) {
        var p = this.root.prior[a];
        wq += p * c.q();
        psum += p;
      }
    }
    wq /= psum;
    return (this.root.nnValue + sumN * wq) / (1 + sumN);
  };

  GameSearch.prototype.nnPolicy = function () {
    return this.root.prior.slice();
  };

  GameSearch.prototype.visitCounts = function () {
    var vc = [0, 0, 0, 0];
    for (var a = 0; a < 4; a++) { var c = this.root.children[a]; vc[a] = c ? c.n : 0; }
    return vc;
  };

  GameSearch.prototype.bestAction = function () {
    if (this.root.terminal) return -1;
    if (this._active.length > 0) {
      var self = this;
      var best = this._active[0], bestS = this._rootScore(best);
      for (var k = 1; k < this._active.length; k++) {
        var s = self._rootScore(this._active[k]);
        if (s > bestS) { bestS = s; best = this._active[k]; }
      }
      return best;
    }
    var vc = this.visitCounts();
    var sum = vc[0] + vc[1] + vc[2] + vc[3];
    if (sum === 0) return -1;
    var ba = 0;
    for (var a = 1; a < 4; a++) if (vc[a] > vc[ba]) ba = a;
    return ba;
  };

  GameSearch.prototype.rootValue = function () {
    return this.root.value();
  };

  // High-level driver: run a full search from `state` and return the move.
  //   state  : length-16 exponent array (game.py convention)
  //   runNet : async (Float32Array[B*256], B) -> {logits:Float32Array[B*4],
  //            values:Float32Array[B]} ; value in RAW 2048 points
  //   userCfg: optional overrides of DEFAULTS
  // Returns { action, dir, qs (improved policy[4]), visits[4], value (root value) }.
  async function chooseMoveMCTS(state, runNet, userCfg) {
    var cfg = Object.assign({}, DEFAULTS, userCfg || {});
    var gs = new GameSearch(state, cfg);
    if (gs.root.terminal) {
      return { action: -1, dir: null, qs: [0, 0, 0, 0], visits: [0, 0, 0, 0], value: 0 };
    }
    var r = await evalStates([gs.rootLeaf()], runNet);
    gs.applyRootEval(r.logits[0], r.values[0]);
    for (var i = 0; i < cfg.num_simulations; i++) {
      var leaf = gs.selectLeaf();
      if (leaf === null) continue;
      var e = await evalStates([leaf], runNet);
      gs.applyEval(e.logits[0], e.values[0]);
    }
    var a = gs.bestAction();
    return {
      action: a,
      dir: a >= 0 ? AI.DIR_NAMES[a] : null,
      qs: gs.improvedPolicy(),
      visits: gs.visitCounts(),
      value: gs.rootValue(),
    };
  }

  // Encode a list of states and run the net, splitting the flat outputs into
  // per-state [4]-logit arrays and scalar values.
  async function evalStates(states, runNet) {
    var P = AI.NUM_PLANES * AI.AREA;
    var B = states.length;
    var flat = new Float32Array(B * P);
    for (var s = 0; s < B; s++) AI.encode(states[s], flat, s * P);
    var out = await runNet(flat, B);
    var logits = [], values = [];
    for (var i = 0; i < B; i++) {
      logits.push([out.logits[i * 4], out.logits[i * 4 + 1],
                   out.logits[i * 4 + 2], out.logits[i * 4 + 3]]);
      values.push(out.values[i]);
    }
    return { logits: logits, values: values };
  }

  var api = {
    DEFAULTS: DEFAULTS,
    MinMax: MinMax, Decision: Decision, Chance: Chance, GameSearch: GameSearch,
    softmax: softmax, chooseMoveMCTS: chooseMoveMCTS, evalStates: evalStates,
  };

  root.MCTS2048 = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : this);
