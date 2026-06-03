"""Afterstate Stochastic *Gumbel* AlphaZero MCTS for 2048 (Python).

Tree alternates DECISION nodes (player picks 1 of 4 directions) and CHANCE
nodes (env spawns a tile). Single-agent => scalar value = expected discounted
future score; backup is a plain discounted accumulation G = r + gamma*G_child
with NO perspective flip. PUCT compares Q after MuZero-style min-max
normalization (returns reach tens of thousands).

Root uses Gumbel-Top-k + sequential halving (Danihelka et al. 2022); the policy
training target is the completed-Q "improved policy".

`batch_search` runs many independent games in lockstep — every simulation step
collects one pending leaf per game and evaluates them all in a single batched
NN forward, which is what makes Python self-play fast enough.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

import game as G
from model_config import Config

# A NN evaluator: takes a list of states, returns (policy_logits[B,4],
# value[B] in RAW points). Built by net_evaluator() below.
EvalFn = Callable[[list[np.ndarray]], tuple[np.ndarray, np.ndarray]]


class _MinMax:
    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, v: float) -> None:
        if v < self.lo: self.lo = v
        if v > self.hi: self.hi = v

    def norm(self, v: float) -> float:
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return 0.5


class _Decision:
    __slots__ = ("state", "terminal", "expanded", "prior", "children",
                 "nn_value", "n", "w", "logits")

    def __init__(self, state: np.ndarray) -> None:
        self.state = state
        self.terminal = False
        self.expanded = False
        self.prior = np.zeros(4, dtype=np.float64)
        self.children: list[_Chance | None] = [None, None, None, None]
        self.nn_value = 0.0
        self.n = 0
        self.w = 0.0
        self.logits = np.full(4, -1e30, dtype=np.float64)

    def value(self) -> float:
        return self.w / self.n if self.n > 0 else self.nn_value


class _Chance:
    __slots__ = ("afterstate", "reward", "edges", "n", "w")

    def __init__(self, afterstate: np.ndarray, reward: int) -> None:
        self.afterstate = afterstate
        self.reward = reward
        # each edge: [prob, cell, exp, child_decision_or_None]
        self.edges: list[list] = []
        self.n = 0
        self.w = 0.0

    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class GameSearch:
    """One game's search tree + Gumbel root scheduler. Leaf evaluation is
    deferred to the batched driver via select_leaf()/apply_eval()."""

    def __init__(self, state: np.ndarray, cfg: Config, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.root = _Decision(state)
        self.root.terminal = G.is_terminal(state)
        self.stats = _MinMax()
        self._pending_path: list | None = None
        # Gumbel scheduler state (set up after root expansion).
        self._g = np.zeros(4, dtype=np.float64)
        self._root_logits = np.full(4, -1e30, dtype=np.float64)
        self._active: list[int] = []     # surviving candidate actions
        self._phase = 0
        self._phase_budgets: list[int] = []
        self._in_phase = 0
        self._rr = 0
        self._ready = False

    # ---- root expansion (driver supplies the eval) ----
    def root_leaf(self) -> np.ndarray | None:
        return None if self.root.terminal else self.root.state

    def apply_root_eval(self, logits: np.ndarray, value: float) -> None:
        if self.root.terminal:
            return
        self._expand(self.root, logits, value)
        self._backup_node(self.root, value)
        self._setup_gumbel()

    def _setup_gumbel(self) -> None:
        legal = G.legal_actions(self.root.state)
        self._root_logits = np.where(legal > 0, self.root.logits, -1e30)
        if self.cfg.gumbel_noise:
            self._g = self.rng.gumbel(0.0, 1.0, size=4)
        else:
            self._g = np.zeros(4, dtype=np.float64)
        scores = self._root_logits + self._g
        cands = [a for a in range(4) if legal[a] > 0]
        cands.sort(key=lambda a: scores[a], reverse=True)
        self._active = cands
        m = len(cands)
        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        n = self.cfg.num_simulations
        base = n // phases
        rem = n - base * phases
        self._phase_budgets = [base + (1 if i < rem else 0) for i in range(phases)]
        self._phase = 0
        self._in_phase = 0
        self._rr = 0
        self._ready = True

    # ---- one simulation: pick root action, descend, return leaf to eval ----
    def select_leaf(self):
        """Returns a state to evaluate, or None if this sim needed no NN
        (terminal leaf already backed up, or root terminal)."""
        if self.root.terminal or not self._ready or not self._active:
            return None
        a = self._next_root_action()
        path = [self.root]
        rewards: list[int] = []
        chance = self.root.children[a]
        path.append(chance)
        rewards.append(chance.reward)
        node = self._descend_chance(chance)
        # descend further while inside the tree
        while True:
            path.append(node)
            if node.terminal:
                self._backup_path(path, rewards, 0.0)
                return None
            if not node.expanded:
                self._pending_path = (path, rewards)
                return node.state
            a2 = self._select_action(node)
            ch = node.children[a2]
            path.append(ch)
            rewards.append(ch.reward)
            node = self._descend_chance(ch)

    def apply_eval(self, logits: np.ndarray, value: float) -> None:
        path, rewards = self._pending_path
        leaf = path[-1]
        self._expand(leaf, logits, value)
        self._backup_path(path, rewards, value)
        self._pending_path = None

    # ---- Gumbel sequential halving over root candidates ----
    def _next_root_action(self) -> int:
        m = len(self._active)
        a = self._active[self._rr % m]
        self._rr += 1
        self._in_phase += 1
        if (self._phase < len(self._phase_budgets) - 1
                and self._in_phase >= self._phase_budgets[self._phase]):
            # prune: keep top ceil(m/2) by gumbel score + completed-Q
            self._active.sort(key=self._root_score, reverse=True)
            keep = max(1, (m + 1) // 2)
            self._active = self._active[:keep]
            self._phase += 1
            self._in_phase = 0
            self._rr = 0
        return a

    def _root_score(self, a: int) -> float:
        ch = self.root.children[a]
        q = ch.q() if (ch and ch.n > 0) else self.root.value()
        return self._root_logits[a] + self._g[a] + self._sigma(q)

    def _sigma(self, q: float) -> float:
        max_n = max((c.n for c in self.root.children if c is not None), default=0)
        return (self.cfg.gumbel_c_visit + max_n) * self.cfg.gumbel_c_scale * self.stats.norm(q)

    # ---- PUCT at non-root decision nodes ----
    def _select_action(self, node: _Decision) -> int:
        sqrt_n = math.sqrt(max(1, node.n))
        best, best_a = -math.inf, -1
        for a in range(4):
            ch = node.children[a]
            if ch is None:
                continue
            if ch.n > 0:
                q = ch.q()
            else:
                q = ch.reward + self.cfg.gamma * node.value()
            score = self.stats.norm(q) + self.cfg.c_puct * node.prior[a] * sqrt_n / (1 + ch.n)
            if score > best:
                best, best_a = score, a
        return best_a

    # ---- chance node: deterministic "most under-represented" descent ----
    def _descend_chance(self, ch: _Chance) -> _Decision:
        total = ch.n
        best_i, best_def = -1, -math.inf
        for i, e in enumerate(ch.edges):
            child = e[3]
            cn = child.n if child is not None else 0
            frac = (cn / total) if total > 0 else 0.0
            deficit = e[0] - frac
            if deficit > best_def:
                best_def, best_i = deficit, i
        e = ch.edges[best_i]
        if e[3] is None:
            ns = ch.afterstate.copy()
            ns[e[1]] = e[2]
            d = _Decision(ns)
            d.terminal = G.is_terminal(ns)
            e[3] = d
        return e[3]

    # ---- expand / backup ----
    def _expand(self, node: _Decision, logits: np.ndarray, value: float) -> None:
        node.expanded = True
        node.nn_value = value
        legal = G.legal_actions(node.state)
        masked = np.where(legal > 0, logits, -1e30)
        node.prior = _softmax(masked)
        node.logits = masked  # used by root Gumbel setup
        for a in range(4):
            if legal[a] > 0:
                after, reward, _ = G.apply_move(node.state, a)
                ch = _Chance(after, reward)
                ch.edges = [[p, cell, exp, None] for (cell, exp, p) in G.spawn_distribution(after)]
                node.children[a] = ch

    def _backup_node(self, node: _Decision, value: float) -> None:
        node.n += 1
        node.w += value
        self.stats.update(value)

    def _backup_path(self, path: list, rewards: list[int], leaf_value: float) -> None:
        # path = [dec0, ch0, dec1, ch1, ..., decK]; rewards align with chance nodes.
        g = leaf_value
        leaf = path[-1]
        leaf.n += 1
        leaf.w += g
        self.stats.update(g)
        # walk back over (chance, decision) pairs
        ci = len(rewards) - 1
        i = len(path) - 2  # first chance node from the end
        while i >= 0:
            node = path[i]
            if isinstance(node, _Chance):
                g = rewards[ci] + self.cfg.gamma * g
                ci -= 1
            node.n += 1
            node.w += g
            self.stats.update(g)
            i -= 1

    # ---- results ----
    def improved_policy(self) -> np.ndarray:
        legal = G.legal_actions(self.root.state)
        vmix = self._v_mix()
        comp = np.full(4, -1e30, dtype=np.float64)
        for a in range(4):
            if legal[a] > 0:
                ch = self.root.children[a]
                q = ch.q() if (ch and ch.n > 0) else vmix
                comp[a] = self._root_logits[a] + self._sigma(q)
        return _softmax(comp)

    def _v_mix(self) -> float:
        sum_n = sum(c.n for c in self.root.children if c is not None)
        if sum_n == 0:
            return self.root.nn_value
        wq, psum = 0.0, 1e-12
        for a in range(4):
            ch = self.root.children[a]
            if ch is not None and ch.n > 0:
                p = self.root.prior[a]
                wq += p * ch.q()
                psum += p
        wq /= psum
        return (self.root.nn_value + sum_n * wq) / (1 + sum_n)

    def nn_policy(self) -> np.ndarray:
        """Raw network prior over the 4 directions (softmax of legal-masked
        logits) — i.e. what the policy head says before search."""
        return self.root.prior.copy()

    def visit_counts(self) -> np.ndarray:
        return np.array([c.n if c is not None else 0 for c in self.root.children], dtype=np.float64)

    def best_action(self) -> int:
        # Gumbel selection among surviving candidates (or most-visited).
        if self.root.terminal:
            return -1
        if self._active:
            return max(self._active, key=self._root_score)
        vc = self.visit_counts()
        return int(vc.argmax()) if vc.sum() > 0 else -1

    def root_value(self) -> float:
        return self.root.value()


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / len(x)


# ---------------------------------------------------------------------------
# Batched driver: run many independent games in lockstep, batching all leaf
# evaluations of each simulation step into one NN forward.
# ---------------------------------------------------------------------------
def batch_search(searches: list[GameSearch], eval_fn: EvalFn, cfg: Config) -> None:
    """Run num_simulations on every search in `searches`. After this returns,
    each search exposes improved_policy()/best_action()/root_value()."""
    # Root expansion, batched.
    root_states, root_idx = [], []
    for i, s in enumerate(searches):
        st = s.root_leaf()
        if st is not None:
            root_states.append(st)
            root_idx.append(i)
    if root_states:
        logits, values = eval_fn(root_states)
        for k, i in enumerate(root_idx):
            searches[i].apply_root_eval(logits[k], float(values[k]))

    for _ in range(cfg.num_simulations):
        leaf_states, leaf_idx = [], []
        for i, s in enumerate(searches):
            st = s.select_leaf()
            if st is not None:
                leaf_states.append(st)
                leaf_idx.append(i)
        if not leaf_states:
            continue
        logits, values = eval_fn(leaf_states)
        for k, i in enumerate(leaf_idx):
            searches[i].apply_eval(logits[k], float(values[k]))


def net_evaluator(net, cfg: Config, device: str | None = None) -> EvalFn:
    """Wrap a Net2048 into an EvalFn returning (logits[B,4], value[B] in RAW
    points). value head outputs scaled units => multiply by value_scale (and,
    when value_transform is on, invert the MuZero h() to get back to points)."""
    import torch
    import value_transform

    dev = device or cfg.device

    def _eval(states: list[np.ndarray]):
        enc = G.encode_batch(states)
        with torch.no_grad():
            x = torch.from_numpy(enc).to(dev)
            logits, value = net(x)
        v = value.float().cpu().numpy() * cfg.value_scale
        if cfg.value_transform:
            v = value_transform.from_h_np(v).astype(np.float32)
        return logits.float().cpu().numpy(), v

    return _eval


def stub_evaluator(cfg: Config) -> EvalFn:
    """Uniform policy, zero value — for testing the search without a net."""
    def _eval(states: list[np.ndarray]):
        b = len(states)
        return np.zeros((b, 4), dtype=np.float32), np.zeros(b, dtype=np.float32)
    return _eval

