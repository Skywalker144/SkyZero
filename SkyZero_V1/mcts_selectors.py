import math
import numpy as np

from utils import softmax


def compute_completed_q(node, num_actions, eps=1e-12):
    """Gumbel MuZero 的 completed Q 值。

    用 node 上缓存的 NN 输出 (nn_policy_logits, nn_value_probs) 加上已访问子节点
    回传的 Q,补出所有动作的 Q 估计;未访问动作回退到 v_mix(已访问 Q 的策略加权
    与 NN value 的混合)。父节点视角。

    Returns
    -------
    completed_q : (A,) float, 标量 Q ∈ [0, 1]
    n_values    : (A,) float, 每个动作的子节点访问数(未展开则为 0)
    sum_n       : float, 子节点总访问数
    v_mix_wdl   : (3,) float, 用于未访问动作的混合 WDL 估值
    """
    logits = node.nn_policy_logits
    nn_value = node.nn_value_probs
    nn_policy = softmax(logits)

    q_wdl = np.zeros((num_actions, 3), dtype=np.float64)
    n_values = np.zeros(num_actions, dtype=np.float64)
    for c in node.children:
        if c.n > 0:
            q_wdl[c.action_taken] = (c.wdl / c.n)[[2, 1, 0]]  # flip child→parent
            n_values[c.action_taken] = c.n

    sum_n = n_values.sum()
    visited = n_values > 0
    if sum_n > 0:
        pv = nn_policy * visited
        weighted_q = (pv[:, None] * q_wdl).sum(axis=0) / (pv.sum() + eps)
        v_mix_wdl = (nn_value + sum_n * weighted_q) / (1 + sum_n)
    else:
        v_mix_wdl = nn_value.copy()

    completed_q_wdl = np.where(visited[:, None], q_wdl, v_mix_wdl[None, :])
    completed_q = (completed_q_wdl[:, 0] - completed_q_wdl[:, 2] + 1) / 2
    return completed_q, n_values, sum_n, v_mix_wdl


class NonRootSelector:
    """非根节点的子节点选择规则。子类实现 select(node)。"""

    def __init__(self, args, game):
        self.args = args
        self.game = game

    def select(self, node):
        raise NotImplementedError


class PuctSelector(NonRootSelector):
    """KataGo-style PUCT: FPU + variance-scaled cpuct.
    Port of SkyZero_V5/cpp_ab_puct/alphazero.h:202-257 (compute_select_params)
    + alphazero_parallel.h:343-372 (select).
    """

    def select(self, node):
        # Defaults aligned with KataGo SearchParams::basicDecentParams()
        # (searchparams.cpp:328-360). See also forTestsV2() for the
        # near-identical training-time profile.
        c_puct = self.args.get("c_puct", 1.0)
        c_puct_log = self.args.get("c_puct_log", 0.45)
        c_puct_base = self.args.get("c_puct_base", 500.0)
        fpu_reduction_max = self.args.get("fpu_reduction_max", 0.2)
        fpu_pow = self.args.get("fpu_pow", 1.0)
        fpu_loss_prop = self.args.get("fpu_loss_prop", 0.0)
        stdev_prior = self.args.get("cpuct_utility_stdev_prior", 0.40)
        stdev_prior_weight = self.args.get("cpuct_utility_stdev_prior_weight", 2.0)
        stdev_scale = self.args.get("cpuct_utility_stdev_scale", 0.85)

        parent_n = node.n
        parent_utility = (node.wdl[0] - node.wdl[2]) / parent_n if parent_n > 0 else 0.0

        # Variance-scaled cpuct: Bayesian-shrink observed utility stdev toward
        # stdev_prior, then scale cpuct by 1 + stdev_scale * (stdev/prior - 1).
        # High-variance subtrees get more exploration.
        stdev_factor = 1.0
        if stdev_scale != 0.0:
            if parent_n <= 1:
                parent_utility_stdev = stdev_prior
            else:
                utility_sq_avg = node.q_sum_sq / parent_n
                utility_sq = parent_utility * parent_utility
                # observed 2nd moment ≥ mean² for variance to be non-negative.
                if utility_sq_avg < utility_sq:
                    utility_sq_avg = utility_sq
                variance_prior = stdev_prior * stdev_prior
                numerator = (utility_sq + variance_prior) * stdev_prior_weight + utility_sq_avg * parent_n
                denominator = stdev_prior_weight + parent_n - 1.0
                shrunk_variance = max(0.0, numerator / denominator - utility_sq)
                parent_utility_stdev = math.sqrt(shrunk_variance)
            stdev_factor = 1.0 + stdev_scale * (parent_utility_stdev / stdev_prior - 1.0)

        # KataGo's "totalChildWeight" = Σ child weights at the parent. With
        # weight=1 per visit, that's parent_n - 1 (parent's own NN eval ate
        # one). +0.01 offset (TOTALCHILDWEIGHT_PUCT_OFFSET) keeps u nonzero
        # when no children visited yet, so prior order still breaks ties.
        total_child_weight = max(0, parent_n - 1)

        # Log-augmented cpuct (searchexplorehelpers.cpp:9-12): exploration
        # grows slowly with N — negligible at small N (~+0.03 at N=32),
        # meaningful for long searches (~+1.37 at N=10k).
        c_puct_effective = c_puct + c_puct_log * math.log((total_child_weight + c_puct_base) / c_puct_base)
        explore_scaling = c_puct_effective * math.sqrt(total_child_weight + 0.01) * stdev_factor

        # FPU: q for unvisited children = mix(parent_q, nn_value) - reduction.
        # As more siblings get visited, unvisited ones look progressively worse.
        visited_policy_mass = sum(c.prior for c in node.children if c.n > 0)
        nn_v = node.nn_value_probs  # cached at expand time; never None here
        nn_utility = float(nn_v[0]) - float(nn_v[2])
        avg_weight = min(1.0, visited_policy_mass ** fpu_pow)
        parent_utility_for_fpu = avg_weight * parent_utility + (1.0 - avg_weight) * nn_utility
        reduction = fpu_reduction_max * math.sqrt(visited_policy_mass)
        fpu_value = parent_utility_for_fpu - reduction
        fpu_value += (-1.0 - fpu_value) * fpu_loss_prop

        best, best_score = None, -math.inf
        for child in node.children:
            if child.n > 0:
                q = -(child.wdl[0] - child.wdl[2]) / child.n
            else:
                q = fpu_value
            u = explore_scaling * child.prior / (1 + child.n)
            s = q + u
            if s > best_score:
                best_score, best = s, child
        return best


class GumbelSelector(NonRootSelector):
    """Gumbel MuZero 的确定性比例规则:argmax(pi_improved - N/(1+sumN))。"""

    def select(self, node):
        num_actions = self.game.board_size ** 2
        completed_q, n_values, sum_n, _ = compute_completed_q(node, num_actions)

        c_visit = self.args.get("gumbel_c_visit", 50)
        c_scale = self.args.get("gumbel_c_scale", 1.0)
        sigma_q = (c_visit + n_values.max()) * c_scale * completed_q

        improved_policy = softmax(node.nn_policy_logits + sigma_q)
        score = improved_policy - n_values / (1 + sum_n)
        return max(node.children, key=lambda c: score[c.action_taken])


def make_selector(args, game):
    name = args.get("non_root_selector", "puct")
    if name == "puct":
        return PuctSelector(args, game)
    if name == "gumbel":
        return GumbelSelector(args, game)
    raise ValueError(f"unknown non_root_selector: {name!r}")
