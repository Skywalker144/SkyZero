"""Replay buffers + n-step machinery, shared by the value-based agents (Rainbow /
QR-DQN) and SAC.

* :class:`NStepCollector` — one per parallel env, turns 1-step transitions into
  n-step ones ``(s_t, a_t, R_t^{(m)}, s_{t+m}, terminal, gamma^m)``. Storing
  ``gamma^m`` per transition keeps the short windows flushed at an episode
  boundary correct (a time-limit truncation still bootstraps; only a real death
  sets ``terminal``). Works for both discrete (int action) and continuous
  (vector action) agents.
* :class:`PrioritizedReplay` — proportional PER over a :class:`SumTree`
  (Rainbow / QR-DQN).
* :class:`UniformReplay` — plain ring buffer with the same transition layout
  (SAC); ``sample`` returns unit IS-weights so the training code is uniform.

The buffers keep ``obs`` and ``next_obs`` as separate dense arrays — doubles obs
memory (tens of MB here) but lets the collector own episode boundaries so the
buffer never reasons about resets.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# n-step collector (one per env)
# --------------------------------------------------------------------------- #
class NStepCollector:
    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.buf = []          # dicts: s, a, r, s_next, terminal

    def _make(self, k):
        """n-step transition anchored at buf[0], summing up to k steps."""
        R, disc, last, terminal = 0.0, 1.0, None, False
        for j in range(k):
            tr = self.buf[j]
            R += disc * tr["r"]
            disc *= self.gamma
            last = tr
            if tr["terminal"]:
                terminal = True
                break
        return {"obs": self.buf[0]["s"], "action": self.buf[0]["a"], "ret": R,
                "next_obs": last["s_next"], "terminal": terminal, "disc": disc}

    def push(self, s, a, r, s_next, terminal):
        self.buf.append({"s": s, "a": a, "r": r, "s_next": s_next, "terminal": terminal})
        out = []
        if terminal:
            return out                     # caller flushes the whole window
        if len(self.buf) >= self.n:
            out.append(self._make(self.n))
            self.buf.pop(0)
        return out

    def flush(self):
        out = [self._make(len(self.buf) - i) for i in range(len(self.buf))]
        self.buf.clear()
        return out


# --------------------------------------------------------------------------- #
# storage backends
# --------------------------------------------------------------------------- #
class _Storage:
    """Dense transition arrays shared by both buffers. ``action`` may be a scalar
    (discrete) or a vector (continuous)."""

    def __init__(self, capacity, obs_shape, act_shape):
        self.capacity = int(capacity)
        obs_shape = tuple(obs_shape)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        if act_shape is None:                      # discrete
            self.actions = np.zeros(self.capacity, dtype=np.int64)
        else:
            self.actions = np.zeros((self.capacity, *tuple(act_shape)), dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.disc = np.zeros(self.capacity, dtype=np.float32)
        self.terminal = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def _write(self, i, tr):
        self.obs[i] = tr["obs"]
        self.actions[i] = tr["action"]
        self.returns[i] = tr["ret"]
        self.next_obs[i] = tr["next_obs"]
        self.terminal[i] = tr["terminal"]
        self.disc[i] = tr["disc"]

    def _gather(self, idxs):
        return {
            "obs": self.obs[idxs], "actions": self.actions[idxs],
            "returns": self.returns[idxs], "next_obs": self.next_obs[idxs],
            "terminal": self.terminal[idxs], "disc": self.disc[idxs],
        }


class UniformReplay(_Storage):
    def __init__(self, capacity, obs_shape, act_shape=None):
        super().__init__(capacity, obs_shape, act_shape)

    def add(self, tr):
        self._write(self.pos, tr)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = self._gather(idxs)
        batch["weights"] = np.ones(batch_size, dtype=np.float32)
        batch["idxs"] = idxs
        return batch

    def update_priorities(self, idxs, priorities):
        pass


class SumTree:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total(self):
        return float(self.tree[0])

    def update(self, data_idx, priority):
        idx = data_idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1)

    def leaf_value(self, data_idx):
        return float(self.tree[data_idx + self.capacity - 1])


class PrioritizedReplay(_Storage):
    def __init__(self, capacity, obs_shape, act_shape=None, alpha=0.5, eps=1e-6):
        super().__init__(capacity, obs_shape, act_shape)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)
        self.max_prio = 1.0

    def add(self, tr):
        i = self.pos
        self._write(i, tr)
        self.tree.update(i, self.max_prio ** self.alpha)   # seen at least once
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        idxs = np.empty(batch_size, dtype=np.int64)
        prios = np.empty(batch_size, dtype=np.float64)
        total = self.tree.total()
        seg = total / batch_size
        for k in range(batch_size):
            s = np.random.uniform(seg * k, seg * (k + 1))
            di = self.tree.get(s)
            idxs[k] = di
            prios[k] = self.tree.leaf_value(di)
        probs = prios / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        batch = self._gather(idxs)
        batch["weights"] = weights.astype(np.float32)
        batch["idxs"] = idxs
        return batch

    def update_priorities(self, idxs, priorities):
        priorities = np.abs(priorities) + self.eps
        self.max_prio = max(self.max_prio, float(priorities.max()))
        pa = priorities ** self.alpha
        for di, p in zip(idxs, pa):
            self.tree.update(int(di), float(p))
