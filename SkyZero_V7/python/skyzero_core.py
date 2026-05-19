"""SkyZero V7 algorithm reference — pure-Python, single-thread, no I/O.

A single-file distillation of what the V7 C++ selfplay + Python training stack
actually computes. All threading, batching, file I/O, logging, checkpointing,
AMP / SWA / Lookahead / LR warmup, and network-architecture details are
stripped; only the algorithm remains.

Reference style matches `../SkyZero_V1/skyzero_core.py`.

Layers covered:
  * Gumbel-MCTS with KataGo-aligned select rule
      - PUCT + FPU
      - Variance-scaled cPUCT (Bayesian-shrunk parent-utility stdev)
      - v_mix root value + sigma_q → improved policy
      - Tree reuse across plies
  * KataGomo-style balanced opening (random-then-NN-balance moves)
  * KataGomo reduceVisits SoftResign (smooth quadratic on signed-extreme v_mix)
  * Policy-Surprise Weighting (KL-divergence based, with short-horizon TD bootstrap)
  * Per-game target construction
      - Pure-outcome WDL value target
      - TD(λ) value targets at 3 horizons (long / mid / short)
      - Futurepos targets at +8 and +32 plies
  * Multi-head training loss
      - main / opp / soft_main / soft_opp policy cross-entropy
      - WDL value + TD value + futurepos MSE

NOT in this file (intentionally):
  * Network architecture (see python/nets.py)
  * Stochastic D4 inference transform & root_symmetry_pruning (call site detail)
  * Policy-init opening tweak (minor variation on balanced opening)
  * Multi-headed parallel MCTS (thread/virtual-loss machinery is in C++)
  * Replay buffer internals & shuffle/data pipeline
  * Optimizer scaffolding (AMP / SWA / Lookahead / LR warmup)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# External interfaces — duck-typed; supply your own implementations.
# ---------------------------------------------------------------------------

class Game(Protocol):
    """Two-player zero-sum board game with WDL outcomes.

    board_size : int
    encode_state(state, to_play)         -> (C, H, W) float array  (NN input)
    get_is_legal_actions(state, to_play) -> bool[area]             (legal mask)
    get_initial_state()                  -> state
    get_next_state(state, action, player) -> next_state
    is_terminal(state, last_action=None, last_player=None) -> bool
    get_winner(state, last_action=None, last_player=None)  -> int  (+1/-1/0)
    """


class Model(Protocol):
    """A multi-head value/policy network.

    Forward returns a dict with keys:
      "policy"           : (B, 4, area)   — main / opp / soft_main / soft_opp logits
      "value_wdl"        : (B, 3)         — WDL logits (current player POV)
      "value_td"         : (B, 9)         — 3 horizons × WDL logits
      "value_futurepos"  : (B, 2, H, W)   — pre-tanh +8 / +32 ply occupancy
    """


class ReplayBuffer(Protocol):
    def add_game_memory(self, samples): ...
    def sample(self, n) -> list: ...
    def is_ready(self) -> bool: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    m = np.max(x[finite])
    e = np.exp(x - m)
    e[~finite] = 0.0
    s = e.sum()
    return e / s if s > 0 else np.zeros_like(x)


def flip_wdl(v: np.ndarray) -> np.ndarray:
    """Swap W and L so the value is from the opponent's POV."""
    return np.array([v[2], v[1], v[0]], dtype=v.dtype)


def wdl_utility(v) -> float:
    """W - L ∈ [-1, 1] from current player POV."""
    return float(v[0]) - float(v[2])


def temperature_transform(probs: np.ndarray, temp: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if temp <= 1e-10:
        m = (probs == probs.max())
        return m.astype(np.float64) / m.sum()
    if abs(temp - 1.0) < 1e-10:
        return probs
    probs = np.maximum(probs, 1e-10)
    logits = np.log(probs) / temp
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def random_d4(arrs: list[np.ndarray], board_size: int):
    """Per-sample D4 augmentation. Spatial arrays are rotated/flipped in-place
    in the per-sample axis; everything else is passed through.
    The state is (C, H, W); policies are flat (H*W,)."""
    out = []
    k = np.random.randint(0, 4)
    flip = bool(np.random.rand() < 0.5)
    for a in arrs:
        if a.ndim == 3:                            # (C, H, W)
            x = np.rot90(a, k, axes=(1, 2))
            if flip: x = np.flip(x, axis=2)
        elif a.ndim == 1:                          # (H*W,) policy
            x = np.rot90(a.reshape(board_size, board_size), k)
            if flip: x = np.flip(x, axis=1)
            x = x.flatten()
        else:                                      # invariants (value, td, ...)
            x = a
        out.append(np.ascontiguousarray(x))
    return out


# ---------------------------------------------------------------------------
# Config — algorithm-relevant fields only.
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # MCTS budget
    num_simulations: int = 64
    gumbel_m: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0

    # PUCT / FPU
    c_puct: float = 1.1
    c_puct_log: float = 0.45
    c_puct_base: float = 500.0
    fpu_pow: float = 1.0
    fpu_reduction_max: float = 0.08
    fpu_loss_prop: float = 0.0

    # Variance-scaled cPUCT (KataGo searchexplorehelpers.cpp:280-297).
    # Multiplies cPUCT by (1 + scale * (stdev/prior - 1)); high-variance
    # subtrees get more exploration. scale=0 disables.
    cpuct_utility_stdev_prior: float = 0.40
    cpuct_utility_stdev_prior_weight: float = 2.0
    cpuct_utility_stdev_scale: float = 0.85

    # Tree reuse across plies (root ← child after action).
    enable_tree_reuse: bool = True

    # Balanced opening (KataGomo random_opening.cpp).
    balance_opening_prob: float = 0.8
    balanced_opening_max_tries: int = 20
    balanced_opening_avg_dist_factor: float = 0.8
    balanced_opening_reject_prob: float = 0.995
    balanced_opening_reject_prob_fallback: float = 0.8
    balanced_opening_value_exponent: float = 4.0

    # SoftResign (KataGomo reduceVisits-aligned).
    soft_resign_threshold: float = 0.9
    soft_resign_step_threshold: int = 3
    soft_resign_sample_weight: float = 0.1
    reduced_visits_fraction: float = 0.25
    reduced_visits_min_floor: int = 16

    # Policy-Surprise Weighting
    enable_psw: bool = True
    policy_surprise_data_weight: float = 0.5
    value_surprise_data_weight: float = 0.1

    # Training
    batch_size: int = 256
    train_steps_per_iteration: int = 100

    # Loss weights (KataGo defaults).
    policy_loss_weight: float = 1.0
    opp_policy_loss_weight: float = 0.15
    soft_policy_loss_weight: float = 8.0
    value_loss_weight: float = 0.72
    td_value_loss_weight: float = 0.72
    futurepos_loss_weight: float = 0.25

    # TD(λ) constants. KataGomo trainingwrite.cpp: nf = 1/(1 + area*c) for
    # c ∈ {0.176 long, 0.056 mid, 0.016 short}. Per-horizon recurrence is a
    # geometric mixture of v_mix and the next-step (perspective-flipped) target.
    td_constants: tuple = (0.176, 0.056, 0.016)
    futurepos_offsets: tuple = (8, 32)

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("state", "to_play", "prior", "parent", "action_taken",
                 "children", "nn_policy", "nn_logits", "nn_value_probs",
                 "v", "n", "q_sum_sq")

    def __init__(self, state, to_play, prior=0.0, parent=None, action_taken=-1):
        self.state = state
        self.to_play = to_play
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken
        self.children: list[Node] = []

        # Filled in at expansion time.
        self.nn_policy: np.ndarray | None = None    # softmax over legal actions
        self.nn_logits: np.ndarray | None = None    # raw logits, -inf for illegal
        self.nn_value_probs = np.array([0., 1., 0.], dtype=np.float32)  # WDL

        # Backup statistics.
        self.v = np.zeros(3, dtype=np.float32)     # Σ WDL backups
        self.n = 0                                  # visit count
        # Σ (value[2] - value[0])² over backups — feeds the variance-scaled
        # cPUCT formula. Sign irrelevant (squared).
        self.q_sum_sq = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def update(self, value: np.ndarray) -> None:
        self.v += value
        u = float(value[2]) - float(value[0])
        self.q_sum_sq += u * u
        self.n += 1


# ---------------------------------------------------------------------------
# PUCT + FPU helpers (KataGo searchexplorehelpers.cpp).
# ---------------------------------------------------------------------------

def compute_select_params(node: Node, visited_policy_mass: float, cfg: Config):
    """Returns (explore_scaling, fpu_value) used by PUCT child scoring.

    explore_scaling   = cPUCT(N) * sqrt(N - 1) * variance_scale
    fpu_value         = blended parent utility for unvisited children, reduced
                        by fpu_reduction_max * sqrt(visited_policy_mass)
    """
    total_child_weight = max(0, node.n - 1)
    c_puct = cfg.c_puct + cfg.c_puct_log * math.log(
        (total_child_weight + cfg.c_puct_base) / cfg.c_puct_base
    )

    parent_q = node.v / node.n if node.n > 0 else np.zeros(3, dtype=np.float32)
    parent_utility = wdl_utility(parent_q)

    # --- Variance-scaled cPUCT (KataGo searchexplorehelpers.cpp:280-297) ---
    # Bayesian-shrinks the per-visit utility variance toward a fixed prior, so
    # noisy/sharp subtrees get more exploration without amplifying single-visit
    # noise. utility_sq_avg = q_sum_sq / n; shrunk variance combines empirical
    # second moment with `prior_weight` virtual observations of stdev `prior`.
    stdev_factor = 1.0
    if cfg.cpuct_utility_stdev_scale != 0.0:
        weight_sum = float(node.n)
        if node.n <= 0 or weight_sum <= 1.0:
            stdev = cfg.cpuct_utility_stdev_prior
        else:
            util_sq_avg = node.q_sum_sq / weight_sum
            util_sq = parent_utility * parent_utility
            # Numerical guard: observed 2nd moment must be ≥ mean² for variance ≥ 0.
            if util_sq_avg < util_sq:
                util_sq_avg = util_sq
            prior_var = cfg.cpuct_utility_stdev_prior ** 2
            pw = cfg.cpuct_utility_stdev_prior_weight
            num = (util_sq + prior_var) * pw + util_sq_avg * weight_sum
            den = pw + weight_sum - 1.0
            shrunk_var = max(0.0, num / den - util_sq)
            stdev = math.sqrt(shrunk_var)
        stdev_factor = 1.0 + cfg.cpuct_utility_stdev_scale * (
            stdev / cfg.cpuct_utility_stdev_prior - 1.0
        )

    explore_scaling = c_puct * math.sqrt(total_child_weight + 0.01) * stdev_factor

    # --- FPU (KataGo first-play urgency) ---
    nn_utility = wdl_utility(node.nn_value_probs)
    # Blend between empirical Q (when many children visited) and NN value
    # (when few visited). Power lets the blend be sub-linear in mass.
    blend_w = min(1.0, visited_policy_mass ** cfg.fpu_pow)
    blended = blend_w * parent_utility + (1.0 - blend_w) * nn_utility
    reduction = cfg.fpu_reduction_max * math.sqrt(visited_policy_mass)
    fpu = blended - reduction
    # Optional pull toward loss for very-low-prior children.
    fpu = fpu + ((-1.0) - fpu) * cfg.fpu_loss_prop
    return explore_scaling, fpu


# ---------------------------------------------------------------------------
# MCTS — Gumbel sequential halving at root, PUCT below it.
# ---------------------------------------------------------------------------

class MCTS:
    def __init__(self, game: Game, cfg: Config, model: Model):
        self.game = game
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.model.eval()

    @torch.inference_mode()
    def _inference(self, state, to_play):
        """Returns (legal-masked policy, WDL value probs, raw masked logits)."""
        x = self.game.encode_state(state, to_play)
        t = torch.tensor(x, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        out = self.model(t)
        # Slim head returns (1, 4, area); main head is channel 0. The other three
        # heads (opp / soft_main / soft_opp) are training-only and unused at search.
        policy_logits = out["policy"][0, 0].cpu().numpy()
        value = F.softmax(out["value_wdl"], dim=1).squeeze(0).cpu().numpy()

        legal = self.game.get_is_legal_actions(state, to_play)
        masked = np.where(legal, policy_logits, -np.inf)
        return softmax(masked), value.astype(np.float32), masked

    def expand(self, node: Node, policy: np.ndarray, value: np.ndarray, logits: np.ndarray):
        node.nn_policy = policy
        node.nn_value_probs = value
        node.nn_logits = logits
        for a, p in enumerate(policy):
            if p <= 0.0:
                continue
            child = Node(
                state=self.game.get_next_state(node.state, a, node.to_play),
                to_play=-node.to_play,
                prior=float(p),
                parent=node,
                action_taken=a,
            )
            node.children.append(child)

    @staticmethod
    def backpropagate(node: Node, value: np.ndarray):
        while node is not None:
            node.update(value)
            value = flip_wdl(value)
            node = node.parent

    def select_child(self, node: Node) -> Node:
        visited_mass = sum(c.prior for c in node.children if c.n > 0)
        explore_scaling, fpu = compute_select_params(node, visited_mass, self.cfg)

        best_score = -math.inf
        best_child = None
        for c in node.children:
            if c.n > 0:
                # Q is the child's average value from the *parent's* POV,
                # i.e. (child_L - child_W) — opposite of the child's own WDL.
                q = (c.v[2] - c.v[0]) / c.n
            else:
                q = fpu
            u = explore_scaling * c.prior / (1.0 + c.n)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = c
        return best_child

    def _simulate(self, start_node: Node):
        node = start_node
        while node.is_expanded():
            node = self.select_child(node)
        # Leaf: either terminal (use the game outcome) or evaluate the NN.
        if self.game.is_terminal(node.state):
            # Winner from node.to_play's POV. Typically -1 since the previous
            # move (by the opponent) just ended the game in their favor.
            r = self.game.get_winner(node.state) * node.to_play
            value = np.array([1., 0., 0.], dtype=np.float32) if r == 1 else \
                    np.array([0., 0., 1.], dtype=np.float32) if r == -1 else \
                    np.array([0., 1., 0.], dtype=np.float32)
        else:
            policy, value, logits = self._inference(node.state, node.to_play)
            self.expand(node, policy, value, logits)
        self.backpropagate(node, value)

    @torch.inference_mode()
    def gumbel_sequential_halving(self, root: Node, num_simulations: int):
        """Gumbel-MCTS at the root with Sequential Halving over m initial actions.

        Returns dict with `improved_policy`, `gumbel_action`, `v_mix`, `visit_counts`.

        On entry, `root` must be expanded (root.nn_logits / nn_policy / nn_value_probs
        populated). When tree-reuse is on, the caller passes the child of a
        previous root that already satisfies this; otherwise root_expand() handles
        the first NN call.
        """
        cfg = self.cfg
        area = len(root.nn_logits)
        logits = root.nn_logits.copy()
        is_legal = np.isfinite(logits)

        # --- Initial m candidates: top-m by (logit + Gumbel noise) ---
        g = np.random.gumbel(size=area)
        scores = np.where(is_legal, logits + g, -np.inf)
        m = min(num_simulations, cfg.gumbel_m)
        surviving = [a for a in np.argsort(scores)[::-1] if is_legal[a]][:m]
        m = len(surviving)
        if m == 0:
            improved = softmax(logits)
            return {"improved_policy": improved, "gumbel_action": int(np.argmax(improved)),
                    "v_mix": root.nn_value_probs.copy(),
                    "visit_counts": np.zeros(area, dtype=np.float32)}

        # --- Sequential Halving: each phase, give every surviving action an
        #     equal share of the remaining sim budget, then halve. ---
        c_visit, c_scale = cfg.gumbel_c_visit, cfg.gumbel_c_scale
        phases = max(1, int(math.ceil(math.log2(m))))
        sims_budget = num_simulations
        for phase in range(phases):
            if sims_budget <= 0 or not surviving:
                break
            sims_this_phase = sims_budget // (phases - phase)
            sims_per_action = max(1, sims_this_phase // len(surviving))
            for _ in range(sims_per_action):
                for a in surviving:
                    if sims_budget <= 0:
                        break
                    child = next((c for c in root.children if c.action_taken == a), None)
                    if child is None:
                        continue
                    self._simulate(child)
                    sims_budget -= 1
                if sims_budget <= 0:
                    break

            # Halve surviving actions by (logit + g + (c_visit+max_n) * c_scale * q).
            if phase < phases - 1 and len(surviving) > 1:
                max_n = max((c.n for c in root.children), default=0)

                def eval_action(a):
                    c = next((c for c in root.children if c.action_taken == a), None)
                    if c is not None and c.n > 0:
                        # Q from parent's POV, mapped to [0, 1].
                        q = ((c.v[2] - c.v[0]) / c.n + 1.0) * 0.5
                    else:
                        q = 0.5
                    return logits[a] + g[a] + (c_visit + max_n) * c_scale * q

                surviving.sort(key=eval_action, reverse=True)
                surviving = surviving[: max(1, len(surviving) // 2)]

        # --- v_mix: prior-weighted average of visited children Q's, blended
        #     with the root's own NN value via the (1 + sum_n) denominator
        #     (KataGo searchexplorehelpers.cpp v_mix formula). ---
        n_values = np.zeros(area, dtype=np.float32)
        q_wdl = np.zeros((area, 3), dtype=np.float32)
        for c in root.children:
            if c.n > 0:
                # Stored from child's POV; flip to parent's POV by reversing W/L.
                q_wdl[c.action_taken] = (c.v[[2, 1, 0]] / c.n).astype(np.float32)
                n_values[c.action_taken] = c.n
        sum_n = float(n_values.sum())
        v_mix = root.nn_value_probs.copy()
        if sum_n > 0:
            policy_sum = 1e-12
            weighted_q = np.zeros(3, dtype=np.float64)
            for a in range(area):
                if n_values[a] > 0:
                    p = root.nn_policy[a]
                    weighted_q += p * q_wdl[a]
                    policy_sum += p
            weighted_q /= policy_sum
            v_mix = ((root.nn_value_probs + sum_n * weighted_q) / (1.0 + sum_n)).astype(np.float32)

        # --- Improved policy: softmax(logits + sigma_q), where unvisited
        #     children use v_mix as their Q surrogate. sigma_q is scaled by
        #     (c_visit + max_n) * c_scale so that the modification grows with
        #     search depth. ---
        max_n = n_values.max() if sum_n > 0 else 0
        sigma_q = np.zeros(area, dtype=np.float32)
        for a in range(area):
            q = q_wdl[a] if n_values[a] > 0 else v_mix
            s = (q[0] - q[2] + 1.0) * 0.5
            sigma_q[a] = (c_visit + max_n) * c_scale * s

        improved_logits = np.where(is_legal, logits + sigma_q, -np.inf)
        improved_policy = softmax(improved_logits)

        # --- Final Gumbel action: among most-visited survivors, pick the one
        #     with max (logit + g + sigma_q). ---
        if surviving:
            max_visits = max(n_values[a] for a in surviving)
            top_visited = [a for a in surviving if n_values[a] == max_visits]
            gumbel_action = max(top_visited, key=lambda a: logits[a] + g[a] + sigma_q[a])
        else:
            gumbel_action = int(np.argmax(improved_policy))

        return {
            "improved_policy": improved_policy,
            "gumbel_action": int(gumbel_action),
            "v_mix": v_mix,
            "visit_counts": n_values,
        }

    def search(self, root: Node, num_simulations: int):
        """Public entry: expand root if needed, then run Gumbel SH."""
        if not root.is_expanded():
            policy, value, logits = self._inference(root.state, root.to_play)
            self.expand(root, policy, value, logits)
            self.backpropagate(root, value)
        out = self.gumbel_sequential_halving(root, num_simulations)
        out["nn_policy"] = root.nn_policy
        out["nn_value_probs"] = root.nn_value_probs
        return out


# ---------------------------------------------------------------------------
# Balanced Opening (KataGomo random_opening.cpp).
#
# Each game with probability `balance_opening_prob` runs this. We sample a few
# random "nearby" stones then a final NN-balance move chosen so the resulting
# position's |value| (from opponent's POV) is small. Multiple try-once retries
# are used if the NN judges the position too one-sided.
# ---------------------------------------------------------------------------

class BalancedOpening:
    # Distribution over the # of pre-balance random moves (KataGomo NOVC weights).
    MOVE_NUM_WEIGHTS = np.array(
        [10, 30, 50, 80, 60, 40, 20, 10, 5, 1, 0, 0], dtype=np.float64
    )

    def __init__(self, game: Game, infer_fn, cfg: Config):
        self.game = game
        self.infer = infer_fn        # callable: state, to_play -> (policy, wdl, logits)
        self.cfg = cfg

    def _board_value(self, state, to_play):
        """W - L from `to_play`'s POV; close to 0 means balanced."""
        _, value, _ = self.infer(state, to_play)
        return float(value[0]) - float(value[2])

    def _random_nearby_move(self, state, avg_dist):
        """Center-biased on empty boards; distance-weighted to existing stones
        otherwise. Returns a flat action index."""
        N = self.game.board_size
        if np.all(state == 0):
            xd = np.clip(np.random.randn(), -1.5, 1.5) / 3.0
            yd = np.clip(np.random.randn(), -1.5, 1.5) / 3.0
            x = int(round(xd * N + 0.5 * (N - 1)))
            y = int(round(yd * N + 0.5 * (N - 1)))
            x = np.clip(x, 0, N - 1); y = np.clip(y, 0, N - 1)
            return y * N + x

        prob = np.zeros(N * N, dtype=np.float64)
        nonzero = np.argwhere(state.reshape(N, N) != 0)
        if len(nonzero) == 0:
            return -1
        # 1/d² distance weighting from nearest stone, with a mild center bonus.
        for r in range(N):
            for c in range(N):
                if state.reshape(N, N)[r, c] != 0:
                    continue
                d2_acc = 0.0
                for (rr, cc) in nonzero:
                    d2 = (r - rr) ** 2 + (c - cc) ** 2 + avg_dist ** 2
                    d2_acc += d2 ** -2.0
                prob[r * N + c] = d2_acc
        s = prob.sum()
        return int(np.random.choice(N * N, p=prob / s)) if s > 0 else -1

    def _balance_move(self, state, to_play, reject_prob):
        """Returns a legal action that minimizes |value| from opponent POV,
        or -1 if the position should be rejected (too one-sided)."""
        legal = self.game.get_is_legal_actions(state, to_play)
        # Reject if either side's NN value is too negative (one-sided position).
        for sign in (1, -1):
            v = self._board_value(state, sign * to_play)
            if v < 0.0:
                rej = 1.0 - math.exp(-3.0 * v * v)
                if np.random.rand() < rej and np.random.rand() < reject_prob:
                    return -1

        area = self.game.board_size ** 2
        prob = np.zeros(area, dtype=np.float64)
        for a in range(area):
            if not legal[a]:
                continue
            next_state = self.game.get_next_state(state, a, to_play)
            if self.game.is_terminal(next_state):
                continue
            v_opp = self._board_value(next_state, -to_play)
            prob[a] = max(0.0, 1.0 - v_opp * v_opp) ** self.cfg.balanced_opening_value_exponent
        max_p = prob.max()
        if np.random.rand() < (1.0 - max_p) and np.random.rand() < reject_prob:
            return -1
        s = prob.sum()
        return int(np.random.choice(area, p=prob / s)) if s > 0 else -1

    def _try_once(self, reject_prob):
        weights = self.MOVE_NUM_WEIGHTS / self.MOVE_NUM_WEIGHTS.sum()
        random_move_num = int(np.random.choice(len(weights), p=weights))
        avg_dist = np.random.exponential(1.0) * self.cfg.balanced_opening_avg_dist_factor

        state = self.game.get_initial_state()
        to_play = 1
        for _ in range(random_move_num):
            a = self._random_nearby_move(state, avg_dist)
            if a < 0 or not self.game.get_is_legal_actions(state, to_play)[a]:
                # Fall back to any legal move.
                legal = self.game.get_is_legal_actions(state, to_play)
                if not legal.any():
                    return None
                a = int(np.argmax(legal))
            state = self.game.get_next_state(state, a, to_play)
            if self.game.is_terminal(state):
                return None
            to_play = -to_play

        a = self._balance_move(state, to_play, reject_prob)
        if a < 0:
            return None
        state = self.game.get_next_state(state, a, to_play)
        if self.game.is_terminal(state):
            return None
        return state, -to_play

    def generate(self):
        reject = self.cfg.balanced_opening_reject_prob
        tries = 0
        while True:
            out = self._try_once(reject)
            if out is not None:
                return out
            tries += 1
            if tries > self.cfg.balanced_opening_max_tries:
                tries = 0
                reject = self.cfg.balanced_opening_reject_prob_fallback


# ---------------------------------------------------------------------------
# SoftResign — KataGomo reduceVisits-aligned smooth quadratic interpolation.
# Compresses sims when a player has been "winning" v_mix-fixed-frame for K
# consecutive plies, and downweights the resulting training sample.
# ---------------------------------------------------------------------------

def soft_resign(history_v_mix_fixed: list[float], num_simulations: int, cfg: Config):
    """Returns (effective_num_simulations, sample_weight).

    history is the per-ply (v_mix[0] - v_mix[2]) * to_play, i.e. WDL-utility
    in a FIXED reference frame so a signed extreme over the lookback window
    detects "same player consistently winning K turns".
    """
    if len(history_v_mix_fixed) < cfg.soft_resign_step_threshold:
        return num_simulations, 1.0
    window = history_v_mix_fixed[-cfg.soft_resign_step_threshold:]
    signed_extreme = max(max(window), -min(window))
    amount = signed_extreme - cfg.soft_resign_threshold
    if amount <= 0.0:
        return num_simulations, 1.0
    prop = amount / (1.0 - cfg.soft_resign_threshold)
    p2 = prop * prop
    eff_min = max(
        cfg.reduced_visits_min_floor,
        int(round(num_simulations * cfg.reduced_visits_fraction)),
    )
    sims = int(round(num_simulations + p2 * (eff_min - num_simulations)))
    sims = max(sims, eff_min)
    weight = 1.0 + p2 * (cfg.soft_resign_sample_weight - 1.0)
    return sims, float(weight)


# ---------------------------------------------------------------------------
# Policy-Surprise Weighting (KataGomo).
#
# After a game finishes, each sample's resampling weight is composed of a
# baseline component (proportional to the existing sample_weight) and two
# surprise components: policy-surprise (KL of MCTS policy vs NN prior) and
# value-surprise (KL of short-horizon TD bootstrap vs NN value).
# ---------------------------------------------------------------------------

def kl_divergence(target, prior, eps=1e-10):
    t = np.clip(np.asarray(target, dtype=np.float64), eps, 1.0)
    p = np.clip(np.asarray(prior, dtype=np.float64), eps, 1.0)
    if t.sum() <= eps or p.sum() <= eps:
        return 0.0
    t = t / t.sum()
    p = p / p.sum()
    mask = np.asarray(target) > 0
    return float(max(0.0, np.sum(t[mask] * (np.log(t[mask]) - np.log(p[mask])))))


def compute_psw_weights(samples: list[dict], cfg: Config) -> np.ndarray:
    n = len(samples)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    p_surprise = np.zeros(n, dtype=np.float64)
    v_surprise = np.zeros(n, dtype=np.float64)
    base_w = np.array([s["sample_weight"] for s in samples], dtype=np.float64)

    for i, s in enumerate(samples):
        p_surprise[i] = kl_divergence(s["policy_target"], s["nn_policy"])
        # KataGomo aligns value surprise to the SHORT-horizon TD bootstrap,
        # not the pure outcome — short_td is a much smoother / earlier signal.
        short_td = s["td_value_target"][6:9]
        v_surprise[i] = min(1.0, kl_divergence(short_td, s["nn_value_probs"]))

    sum_w = base_w.sum()
    if sum_w <= 1e-8:
        return np.zeros(n, dtype=np.float32)

    avg_p = float(np.dot(p_surprise, base_w) / sum_w)
    avg_v = float(np.dot(v_surprise, base_w) / sum_w)

    # Damp the value-surprise contribution when it's overall small.
    v_weight = cfg.value_surprise_data_weight
    if avg_v < 0.01:
        v_weight *= avg_v / 0.01
    baseline_ratio = max(0.0, 1.0 - cfg.policy_surprise_data_weight - v_weight)
    p_threshold = avg_p * 1.5

    p_prob = base_w * p_surprise + (1.0 - base_w) * np.maximum(0.0, p_surprise - p_threshold)
    v_prob = base_w * v_surprise
    p_sum = max(p_prob.sum(), 1e-10)
    v_sum = max(v_prob.sum(), 1e-10)

    final = (baseline_ratio * base_w
             + cfg.policy_surprise_data_weight * p_prob * sum_w / p_sum
             + v_weight * v_prob * sum_w / v_sum)
    return final.astype(np.float32)


def stochastic_resample(samples: list[dict], weights: np.ndarray) -> list[dict]:
    """Replicate sample i ⌊wᵢ⌋ times and one extra with probability frac(wᵢ).

    Used for both PSW and pure SoftResign. Output samples carry the *original*
    sample_weight (the SoftResign-level baseline), not the PSW-derived weight.
    """
    out = []
    for s, w in zip(samples, weights):
        w = float(w)
        if w <= 0.0:
            continue
        whole = int(math.floor(w))
        out.extend(dict(s) for _ in range(whole))
        if np.random.rand() < (w - whole):
            out.append(dict(s))
    return out


# ---------------------------------------------------------------------------
# Per-game target construction (KataGomo trainingwrite.cpp).
# ---------------------------------------------------------------------------

def build_value_target(outcome: np.ndarray, n_samples: int) -> list[np.ndarray]:
    """Pure-outcome WDL target, propagated backward with per-step perspective
    flip. Equivalent to KataGomo's fillValueTDTargets with nowFactor=0."""
    targets = [None] * n_samples
    cur = outcome.copy()
    targets[-1] = cur.copy()
    for i in range(n_samples - 2, -1, -1):
        cur = flip_wdl(cur)
        targets[i] = cur.copy()
    return targets


def build_td_targets(v_mix_per_step: list[np.ndarray], outcome: np.ndarray,
                     board_area: int, td_constants: tuple) -> np.ndarray:
    """3-horizon TD(λ) value targets, shape (N, 9) flattened as
    [long_W, long_D, long_L, mid..., short...] per sample.

    Per-horizon recurrence (matches KataGomo fillValueTDTargets exactly):
        T(virtual N) = outcome                                    (anchor)
        T(i)         = nf * v_mix[i] + (1 - nf) * flip(T(i+1))
    where nf = 1 / (1 + boardArea * c_h).
    """
    N = len(v_mix_per_step)
    out = np.zeros((N, 9), dtype=np.float32)
    for h, c in enumerate(td_constants):
        nf = 1.0 / (1.0 + board_area * c)
        rd = 1.0 - nf
        nx = outcome.astype(np.float64).copy()
        for i in range(N - 1, -1, -1):
            v = v_mix_per_step[i].astype(np.float64)
            cur = nf * v + rd * nx
            out[i, 3 * h : 3 * h + 3] = cur.astype(np.float32)
            # Perspective flips between plies → swap W and L.
            nx = np.array([cur[2], cur[1], cur[0]], dtype=np.float64)
    return out


def build_futurepos_targets(states_per_step: list, to_play_per_step: list[int],
                            offsets: tuple, terminal_state) -> list[np.ndarray]:
    """Per-sample (2, H, W) int8 occupancy of the board +OFFSET plies ahead,
    clamped to game end. Values: +1 own, -1 opponent, 0 empty."""
    N = len(states_per_step)
    out = []
    for i in range(N):
        pla = to_play_per_step[i]
        fp = []
        for off in offsets:
            j = min(N, i + off)
            board = states_per_step[j] if j < N else terminal_state
            # Cast to current player's POV without rotating the board layout.
            arr = np.where(board == pla, 1, np.where(board == -pla, -1, 0)).astype(np.int8)
            fp.append(arr)
        out.append(np.stack(fp, axis=0))
    return out


# ---------------------------------------------------------------------------
# SkyZero — selfplay + training loop.
# ---------------------------------------------------------------------------

class SkyZero:
    def __init__(self, game: Game, model: Model, optimizer, cfg: Config,
                 replay_buffer: ReplayBuffer):
        self.game = game
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.mcts = MCTS(game, cfg, model)
        self.iteration = 1

    # -----------------------------------------------------------------------
    # Selfplay — one game.
    # -----------------------------------------------------------------------
    def selfplay(self) -> tuple[list[dict], int, int]:
        cfg = self.cfg
        infer = lambda s, p: self.mcts._inference(s, p)

        # --- Opening: with prob `balance_opening_prob`, generate a balanced
        #     non-empty start; otherwise start from the empty board. ---
        if np.random.rand() < cfg.balance_opening_prob:
            state, to_play = BalancedOpening(self.game, infer, cfg).generate()
        else:
            state, to_play = self.game.get_initial_state(), 1

        # SoftResign history is kept in a FIXED reference frame:
        #   (v_mix[0] - v_mix[2]) * to_play
        # so a signed extreme over the lookback window detects "the same
        # player has been winning K consecutive moves".
        history_v_mix_fixed: list[float] = []
        memory: list[dict] = []
        all_mcts_policies: list[np.ndarray] = []  # for opp_policy_target lookup

        root = Node(state, to_play)

        while not self.game.is_terminal(state):
            # --- Decide num_simulations & sample_weight from PRIOR history.
            #     Decision uses [0..N-1], matches KataGomo's pre-search alteration. ---
            num_sims, sample_weight = soft_resign(history_v_mix_fixed, cfg.num_simulations, cfg)

            sr = self.mcts.search(root, num_sims)

            # Record fixed-frame v_mix for the NEXT move's SoftResign decision.
            v_mix_pov = sr["v_mix"][0] - sr["v_mix"][2]
            history_v_mix_fixed.append(v_mix_pov * to_play)

            memory.append({
                "state": state, "to_play": to_play,
                "policy_target": sr["improved_policy"].astype(np.float32),
                "nn_policy": sr["nn_policy"].astype(np.float32),
                "nn_value_probs": sr["nn_value_probs"].astype(np.float32),
                "v_mix": sr["v_mix"].astype(np.float32),
                "sample_weight": float(sample_weight),
            })
            all_mcts_policies.append(sr["improved_policy"])

            # Root action comes directly from Gumbel SH — no temperature
            # sampling. SH already injects exploration via root Gumbel noise.
            action = sr["gumbel_action"]

            state = self.game.get_next_state(state, action, to_play)
            to_play = -to_play

            # --- Tree reuse: navigate to the child for `action` rather than
            #     rebuilding the tree. Gumbel state is search-local so nothing
            #     on the node needs reset. ---
            next_root = None
            if cfg.enable_tree_reuse:
                for c in root.children:
                    if c.action_taken == action:
                        next_root = c
                        c.parent = None
                        break
            root = next_root if next_root is not None else Node(state, to_play)

        # --- Build per-sample targets after game ends. ---
        return self._build_samples(memory, all_mcts_policies, state), \
               self.game.get_winner(state), \
               int(np.count_nonzero(state))

    # -----------------------------------------------------------------------
    # Post-game target construction.
    # -----------------------------------------------------------------------
    def _build_samples(self, memory: list[dict], all_policies: list[np.ndarray],
                       terminal_state) -> list[dict]:
        if not memory:
            return []
        cfg = self.cfg
        winner = self.game.get_winner(terminal_state)
        N = len(memory)
        area = self.game.board_size ** 2

        # Per-step outcome in WDL form from each step's perspective.
        last_outcome = self._winner_to_wdl(winner, memory[-1]["to_play"])
        value_targets = build_value_target(last_outcome, N)

        td_targets = build_td_targets(
            [m["v_mix"] for m in memory], last_outcome,
            self.game.board_size ** 2, cfg.td_constants,
        )

        states = [m["state"] for m in memory]
        to_plays = [m["to_play"] for m in memory]
        fp_targets = build_futurepos_targets(states, to_plays,
                                             cfg.futurepos_offsets, terminal_state)

        samples = []
        for i, m in enumerate(memory):
            # opp_policy_target = next ply's MCTS visits (POV alternates with to_play).
            # Last ply has no next move → has_opp=False and the loss masks it out.
            if i + 1 < N:
                opp_target = all_policies[i + 1].astype(np.float32)
                has_opp = True
            else:
                opp_target = np.zeros(area, dtype=np.float32)
                has_opp = False

            samples.append({
                "encoded_state": self.game.encode_state(m["state"], m["to_play"]),
                "policy_target": m["policy_target"],
                "opp_policy_target": opp_target,
                "opp_policy_mask": np.float32(1.0 if has_opp else 0.0),
                "value_target": value_targets[i].astype(np.float32),
                "td_value_target": td_targets[i],
                "futurepos_target": fp_targets[i],
                "sample_weight": np.float32(m["sample_weight"]),
                # Kept for PSW computation; not consumed by the loss.
                "nn_policy": m["nn_policy"],
                "nn_value_probs": m["nn_value_probs"],
            })

        # --- PSW: compute per-sample weights, stochastically replicate.
        #     Replicated samples keep the SoftResign baseline weight. ---
        if cfg.enable_psw:
            psw_weights = compute_psw_weights(samples, cfg)
            samples = stochastic_resample(samples, psw_weights)
        else:
            # Without PSW, still honor SoftResign by stochastic replication on
            # the baseline weight (so reduced samples ≤ 1.0 are subsampled).
            weights = np.array([s["sample_weight"] for s in samples], dtype=np.float32)
            if (weights < 1.0).any():
                samples = stochastic_resample(samples, weights)

        return samples

    @staticmethod
    def _winner_to_wdl(winner: int, to_play: int) -> np.ndarray:
        r = winner * to_play
        if r == 1:  return np.array([1., 0., 0.], dtype=np.float32)
        if r == -1: return np.array([0., 0., 1.], dtype=np.float32)
        return np.array([0., 1., 0.], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Training — multi-head soft-CE loss + D4 augmentation.
    # -----------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        total = cfg.batch_size * cfg.train_steps_per_iteration
        pool = self.replay_buffer.sample(total)
        for i in range(cfg.train_steps_per_iteration):
            batch = pool[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            self._train_step(batch)

    def _train_step(self, batch: list[dict]):
        cfg = self.cfg
        device = cfg.device

        # --- D4 augmentation transforms state + policy targets + futurepos.
        #     Value/TD/value_wdl are D4-invariant. ---
        for j, s in enumerate(batch):
            s["encoded_state"], s["policy_target"], s["opp_policy_target"], s["futurepos_target"] = \
                random_d4([s["encoded_state"], s["policy_target"],
                           s["opp_policy_target"], s["futurepos_target"]],
                          self.game.board_size)

        def stack(key, dtype=torch.float32):
            return torch.tensor(
                np.array([s[key] for s in batch]), dtype=dtype, device=device
            )

        states          = stack("encoded_state").float()
        policy_target   = stack("policy_target")
        opp_target      = stack("opp_policy_target")
        opp_mask        = stack("opp_policy_mask")
        value_target    = stack("value_target")
        td_target       = stack("td_value_target").view(-1, 3, 3)
        fp_target       = stack("futurepos_target").float()
        sample_weight   = stack("sample_weight")
        opp_weight      = sample_weight * opp_mask

        # --- KataGo soft target: (p + ε)^0.25 normalized, masked to legal cells. ---
        B = states.shape[0]
        on_board_flat = states[:, 0:1, :, :].reshape(B, -1)        # mask plane
        soft_main = self._soft_target(policy_target, on_board_flat)
        soft_opp  = self._soft_target(opp_target, on_board_flat)

        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(states)
        policy_all = out["policy"]
        p_main = policy_all[:, 0, :].reshape(B, -1).float()
        p_opp  = policy_all[:, 1, :].reshape(B, -1).float()
        p_soft = policy_all[:, 2, :].reshape(B, -1).float()
        p_sopp = policy_all[:, 3, :].reshape(B, -1).float()

        policy_loss     = self._weighted_soft_ce(p_main, policy_target, sample_weight)
        opp_policy_loss = self._weighted_soft_ce(p_opp,  opp_target,    opp_weight)
        soft_main_loss  = self._weighted_soft_ce(p_soft, soft_main,     sample_weight)
        soft_opp_loss   = self._weighted_soft_ce(p_sopp, soft_opp,      opp_weight)
        value_loss      = self._weighted_soft_ce(out["value_wdl"].float(), value_target, sample_weight)
        td_value_loss   = self._weighted_td_ce(out["value_td"].view(B, 3, 3).float(),
                                               td_target, sample_weight)
        futurepos_loss  = self._weighted_fp_mse(out["value_futurepos"].float(), fp_target,
                                                states[:, 0:1, :, :], sample_weight)

        # KataGomo soft-policy weighting: main soft = policy_w * soft_w,
        # opp soft = opp_w * soft_w.
        soft_main_w = cfg.policy_loss_weight * cfg.soft_policy_loss_weight
        soft_opp_w  = cfg.opp_policy_loss_weight * cfg.soft_policy_loss_weight
        total_loss = (
            cfg.policy_loss_weight     * policy_loss
            + cfg.opp_policy_loss_weight * opp_policy_loss
            + soft_main_w                * soft_main_loss
            + soft_opp_w                 * soft_opp_loss
            + cfg.value_loss_weight      * value_loss
            + cfg.td_value_loss_weight   * td_value_loss
            + cfg.futurepos_loss_weight  * futurepos_loss
        )

        total_loss.backward()
        self.optimizer.step()

    # ---- Loss building blocks ----
    @staticmethod
    def _soft_target(p: torch.Tensor, on_board: torch.Tensor) -> torch.Tensor:
        soft = (p + 1e-7).clamp_min(0.0).pow(0.25) * on_board
        return soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _weighted_soft_ce(logits, target, weight):
        per_sample = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        denom = weight.sum().clamp_min(1e-8)
        return (per_sample * weight).sum() / denom

    @staticmethod
    def _weighted_td_ce(pred_logits, target_probs, weight):
        # CE - H(target) per horizon, summed over horizons, sample-weighted.
        log_p = F.log_softmax(pred_logits, dim=-1)             # (B, 3, 3)
        ce  = -(target_probs * log_p).sum(dim=-1)              # (B, 3)
        H_t = -(target_probs * (target_probs + 1e-30).log()).sum(dim=-1)
        per_sample = (ce - H_t).sum(dim=-1)                    # (B,)
        denom = weight.sum().clamp_min(1e-8)
        return (per_sample * weight).sum() / denom

    @staticmethod
    def _weighted_fp_mse(pred_pretanh, target, mask, weight):
        # tanh maps to KataGomo's {-1, 0, +1} target range. Per-channel weight
        # 1.0/0.25 (the +32 horizon contributes less).
        pred = torch.tanh(pred_pretanh)
        err = (pred - target).pow(2) * mask
        ch_w = err.new_tensor([1.0, 0.25]).view(1, 2, 1, 1)
        err = err * ch_w
        mask_sum_hw = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_sample = err.sum(dim=(1, 2, 3)) / mask_sum_hw.sqrt()
        denom = weight.sum().clamp_min(1e-8)
        return (per_sample * weight).sum() / denom

    # -----------------------------------------------------------------------
    # Outer loop.
    # -----------------------------------------------------------------------
    def learn(self, target_replay_ratio: float = 1.0,
              default_avg_sample_len: float = 30.0):
        avg_len = default_avg_sample_len
        while True:
            # Generate enough new games to keep the replay-ratio close to target.
            # target_new = batch_size * train_steps / target_replay_ratio
            # needed_games = target_new / avg_sample_len
            target_new = (self.cfg.batch_size * self.cfg.train_steps_per_iteration
                          / target_replay_ratio)
            needed_games = max(1, math.ceil(target_new / avg_len))

            recent_lens = []
            for _ in range(needed_games):
                samples, _winner, game_len = self.selfplay()
                self.replay_buffer.add_game_memory(samples)
                if game_len > 0:
                    recent_lens.append(game_len)

            if recent_lens:
                avg_len = float(np.mean(recent_lens))

            if self.replay_buffer.is_ready():
                self.train()
                self.iteration += 1
