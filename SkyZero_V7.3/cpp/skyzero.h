#ifndef SKYZERO_ALPHAZERO_H
#define SKYZERO_ALPHAZERO_H

// Config, MCTSNode, and shared helpers for MCTS.
// Ported from CSkyZero_V3/alphazero.h with:
//   * Subtree Value Bias (SVB) removed.
//   * Dynamic variance-scaled cPUCT removed (stdev_factor == 1.0).
//   * Single-threaded `MCTS` class removed (selfplay uses ParallelMCTS only).
//   * Torch save/load `AlphaZero` class removed (Python handles training & checkpoints).
//   * Training / replay-buffer / playout-cap configuration not included here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace skyzero {

// ---------------------------------------------------------------------------
// Config — only the fields the C++ selfplay path needs.
// Python side owns everything training-related.
// ---------------------------------------------------------------------------
struct SkyZeroConfig {
    int board_size = 15;

    // Gumbel MCTS
    int num_simulations = 64;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_noise_enabled = true;

    // Non-root (in-tree) action selection. The root always uses Gumbel
    // sequential halving; this only switches how children are picked at
    // non-root nodes during descent:
    //   kPuct   — KataGo-style PUCT + FPU + variance-scaled cpuct (default).
    //   kGumbel — the Gumbel paper's deterministic "Full Gumbel" rule
    //             (Danihelka et al. 2022, eq. 14): argmax_a [ π'(a) −
    //             N(a)/(1+Σ_b N(b)) ], where π' is the completed-Q improved
    //             policy at that node (same construction as the Gumbel root
    //             target). See gumbel_deterministic_select() below.
    enum class NonRootSearchAlgo { kPuct, kGumbel };
    NonRootSearchAlgo non_root_search_algo = NonRootSearchAlgo::kPuct;
    static NonRootSearchAlgo parse_non_root_search_algo(const std::string& s) {
        return s == "gumbel" ? NonRootSearchAlgo::kGumbel : NonRootSearchAlgo::kPuct;
    }

    // PUCT / FPU
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.08f;
    float fpu_loss_prop = 0.0f;

    // Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297, v1.9+).
    // Multiplies cpuct by `1 + scale * (stdev/prior - 1)` where stdev is
    // Bayesian-shrunk per-visit utility stdev at the parent. High-variance
    // subtrees get more exploration. Applied at non-root only (Gumbel root
    // uses sequential halving, not PUCT). scale=0 disables.
    // KataGo SearchParams() ctor: 0.25 / 1.0 / 0.0; forBot()/forTestsV2(): 0.40 / 2.0 / 0.85 (used here).
    float cpuct_utility_stdev_prior = 0.40f;
    float cpuct_utility_stdev_prior_weight = 2.0f;
    float cpuct_utility_stdev_scale = 0.85f;

    // Stochastic transform / symmetry at inference time
    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;
    bool enable_symmetry_inference_for_root = false;
    bool enable_symmetry_inference_for_child = false;

    // KataGo-style root symmetry pruning. At root, symmetry-equivalent legal
    // moves are masked to -inf logit so only the orbit's canonical
    // representative (smallest loc) is searched. Leave OFF for training even
    // when the dataloader does D4 augmentation: pruning still prevents the
    // NN from receiving visit-count signal at 7/8 of equivalent positions.
    // Recommended ON for elo / human play.
    bool root_symmetry_pruning = false;

    // Surprise weighting
    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;

    // Balanced opening (KataGomo-style). Each game samples r ~ U(0,1):
    //   r <  balance_opening_prob → NN-scored random opening;
    //   r >= balance_opening_prob → empty-board start.
    float balance_opening_prob = 0.8f;
    int balanced_opening_max_tries = 20;
    float balanced_opening_avg_dist_factor = 0.8f;
    float balanced_opening_reject_prob = 0.995f;
    float balanced_opening_reject_prob_fallback = 0.8f;
    // Power applied to (1 - v^2) when sampling the final balance move. KataGomo
    // uses 4 for selfplay (looser, more diverse) and 10 for match-mode Elo
    // (sharper concentration on |v|≈0). See randomopening.cpp:152.
    float balanced_opening_value_exponent = 4.0f;

    // Policy-initialization (KataGomo initGamesWithPolicy). After balanced
    // opening, play ~Exp(1)*policy_init_avg_move_num extra moves sampled from
    // the NN policy^(1/temperature) to push the game off the balance plateau.
    // Set avg_move_num <= 0 to disable.
    float policy_init_avg_move_num = 0.0f;
    float policy_init_temperature = 1.0f;

    // Soft resign (KataGomo reduceVisits-aligned: smooth quadratic interpolation,
    // non-sticky, signed-extreme over fixed-frame v_mix; proportional floor
    // adapted for Gumbel-MCTS warmup-stage NUM_SIMULATIONS).
    // eff_min = max(reduced_visits_min_floor, round(num_simulations * reduced_visits_fraction))
    float soft_resign_threshold = 0.9f;        // reduceVisitsThreshold
    int soft_resign_step_threshold = 3;        // reduceVisitsThresholdLookback
    float soft_resign_sample_weight = 0.1f;    // reducedVisitsWeight
    float reduced_visits_fraction = 0.25f;     // 碾压时压到 num_simulations * fraction
    int reduced_visits_min_floor = 16;         // 绝对下限 (= GUMBEL_M)

    // Sprint 2 #3: keep the subtree under the played action as the new
    // root instead of rebuilding from scratch each ply. Gumbel state is
    // search-local (recomputed in gumbel_sequential_halving), so no
    // per-action noise reset is needed.
    bool enable_tree_reuse = true;

    torch::Device device = torch::kCPU;
};

// ---------------------------------------------------------------------------
// Selfplay parallelism config consumed by ParallelMCTS / SelfplayEngine.
// ---------------------------------------------------------------------------
struct SelfplayParallelConfig {
    int num_workers = 32;
    int num_inference_servers = 2;
    int inference_batch_size = 128;
    int inference_batch_wait_us = 100;
    int leaf_batch_size = 8;
    int max_result_queue_size = 0;  // 0 = auto (2 * num_workers); <0 = unbounded
};

// ---------------------------------------------------------------------------
// MCTS Node
// ---------------------------------------------------------------------------
struct MCTSNode {
    std::vector<int8_t> state;
    int to_play = 1;
    float prior = 0.0f;
    MCTSNode* parent = nullptr;
    int action_taken = -1;

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<float> nn_policy;          // softmax probabilities (legal-masked)
    std::vector<float> nn_logits;          // raw logits (legal-masked, -inf for illegal)
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};  // WDL

    std::array<float, 3> v{0.0f, 0.0f, 0.0f};
    int n = 0;
    float q_sum_sq = 0.0f;   // Σ u_i² where u_i = value[2]-value[0] per backup; feeds variance-scaled cpuct in compute_select_params.
    int vloss = 0;

    MCTSNode() = default;
    MCTSNode(std::vector<int8_t> s, int p,
             float pr = 0.0f, MCTSNode* par = nullptr, int act = -1)
        : state(std::move(s)), to_play(p), prior(pr), parent(par), action_taken(act) {}

    bool is_expanded() const { return !children.empty(); }

    void update(const std::array<float, 3>& value) {
        v[0] += value[0];
        v[1] += value[1];
        v[2] += value[2];
        const float u = value[2] - value[0];
        q_sum_sq += u * u;
        n += 1;
    }
};

// ---------------------------------------------------------------------------
// Search output
// ---------------------------------------------------------------------------
struct MCTSSearchOutput {
    std::vector<float> mcts_policy;                         // improved policy (Gumbel)
    std::array<float, 3> v_mix{0.0f, 0.0f, 0.0f};          // WDL v_mix (search root value)
    std::vector<float> nn_policy;                           // raw NN policy
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};  // raw NN value
    int gumbel_action = -1;                                 // selected action by Gumbel (selfplay/eval/play)
    std::vector<float> visit_counts;                        // raw root-child visit counts N(s,a)
    std::vector<std::vector<int>> gumbel_phases;            // surviving actions at each halving phase (16,8,4,2,1)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline std::array<float, 3> flip_wdl(const std::array<float, 3>& in) {
    return {in[2], in[1], in[0]};
}

inline float wdl_utility(const std::array<float, 3>& v) {
    return v[0] - v[2];
}

// ---------------------------------------------------------------------------
// PUCT + FPU helpers.
// Variance-scaled cPUCT is active when cfg.cpuct_utility_stdev_scale != 0
// (KataGo searchexplorehelpers.cpp:280-297); set scale=0 to fix factor at 1.0.
// ---------------------------------------------------------------------------

struct SelectParams {
    float explore_scaling;
    float fpu_value;
};

// effective_parent_n = node.n (single-thread) or node.n + node.vloss (parallel).
inline SelectParams compute_select_params(
    const MCTSNode& node,
    int effective_parent_n,
    float visited_policy_mass,
    const SkyZeroConfig& cfg
) {
    const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));

    const float c_puct = cfg.c_puct + cfg.c_puct_log
        * std::log((total_child_weight + cfg.c_puct_base) / cfg.c_puct_base);

    std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
    if (node.n > 0) {
        parent_q = {node.v[0] / node.n, node.v[1] / node.n, node.v[2] / node.n};
    }
    const float parent_utility = wdl_utility(parent_q);

    // Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297). Sign
    // of u (W-L vs L-W) doesn't matter — variance is invariant under sign
    // flip. node.q_sum_sq accumulates (value[2]-value[0])² per backup, so
    // utility_sq_avg = q_sum_sq / n is the empirical second moment.
    float parent_utility_stdev_factor = 1.0f;
    if (cfg.cpuct_utility_stdev_scale != 0.0f) {
        const float weight_sum = static_cast<float>(node.n);
        float parent_utility_stdev;
        if (node.n <= 0 || weight_sum <= 1.0f) {
            parent_utility_stdev = cfg.cpuct_utility_stdev_prior;
        } else {
            float utility_sq_avg = node.q_sum_sq / weight_sum;
            const float utility_sq = parent_utility * parent_utility;
            // numerical guard (KataGo line 286-287): observed second moment
            // must be ≥ mean² for variance to be non-negative.
            if (utility_sq_avg < utility_sq) utility_sq_avg = utility_sq;
            const float variance_prior = cfg.cpuct_utility_stdev_prior * cfg.cpuct_utility_stdev_prior;
            const float prior_weight = cfg.cpuct_utility_stdev_prior_weight;
            const float numerator = (utility_sq + variance_prior) * prior_weight + utility_sq_avg * weight_sum;
            const float denominator = prior_weight + weight_sum - 1.0f;
            const float shrunk_variance = std::max(0.0f, numerator / denominator - utility_sq);
            parent_utility_stdev = std::sqrt(shrunk_variance);
        }
        parent_utility_stdev_factor = 1.0f + cfg.cpuct_utility_stdev_scale
            * (parent_utility_stdev / cfg.cpuct_utility_stdev_prior - 1.0f);
    }

    const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f) * parent_utility_stdev_factor;

    const float nn_utility = wdl_utility(node.nn_value_probs);
    const float avg_weight = std::min(1.0f, static_cast<float>(std::pow(visited_policy_mass, cfg.fpu_pow)));
    const float parent_utility_for_fpu = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;

    const float reduction = cfg.fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * cfg.fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

// ---------------------------------------------------------------------------
// Gumbel deterministic non-root action selection (Danihelka et al. 2022,
// "Policy improvement by planning with Gumbel", eq. 14 — the "Full Gumbel"
// in-tree rule that replaces PUCT). Selects
//     argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ]
// where π'(a) = softmax_a( logit(a) + σ(completedQ(a)) ) is the completed-Q
// improved policy at this node, the same construction as the Gumbel root
// target:
//   * visited children (N>0) use their empirical node-perspective utility;
//   * unvisited children use v_mix, the prior-weighted blend of the node's NN
//     value and the visited children's utilities.
//   σ(q) = (c_visit + max_b N(b)) · c_scale · (q+1)/2,  q ∈ [−1,1].
// Subtracting N(a)/(1+Σ_b N(b)) makes the visit counts track π' over the
// search (the action just visited is penalised on the next descent), so this
// also provides path diversification when N includes virtual loss.
//
// `stats` holds one entry per child, gathered by the caller under whatever
// locking its backend needs; N(a) is the effective count (n + vloss).
// `node_nn_utility` = W−L of the node's own NN value (node-to-play view).
// Returns the index into `stats` of the chosen child, or -1 if empty.
// ---------------------------------------------------------------------------
struct GumbelChildStat {
    float prior;      // P(a) = node.nn_policy[action]
    int eff_n;        // N(a) = child.n + child.vloss
    float utility;    // node-perspective (W−L)/N for visited children; unused if eff_n==0
    float logit;      // node.nn_logits[action]
};

inline int gumbel_deterministic_select(
    const std::vector<GumbelChildStat>& stats,
    float node_nn_utility,
    float c_visit,
    float c_scale
) {
    const int k = static_cast<int>(stats.size());
    if (k <= 1) return k - 1;  // 0 children → -1; 1 child → 0

    int sum_n = 0;
    int max_n = 0;
    float wq_num = 0.0f;   // Σ_{N>0} π(a)·u(a)
    float wq_den = 0.0f;   // Σ_{N>0} π(a)
    for (const auto& s : stats) {
        sum_n += s.eff_n;
        if (s.eff_n > max_n) max_n = s.eff_n;
        if (s.eff_n > 0) {
            wq_num += s.prior * s.utility;
            wq_den += s.prior;
        }
    }

    // v_mix (node-perspective utility) completes the unvisited actions.
    float v_mix = node_nn_utility;
    if (sum_n > 0 && wq_den > 0.0f) {
        const float wq = wq_num / wq_den;
        v_mix = (node_nn_utility + static_cast<float>(sum_n) * wq)
                / (1.0f + static_cast<float>(sum_n));
    }

    const float sigma_coeff = (c_visit + static_cast<float>(max_n)) * c_scale;

    // improved logits = logit + σ(completedQ). softmax over children is the
    // same as over legal actions (children exist only for legal moves), so
    // π'(a) for a child equals its entry in the full improved policy.
    std::vector<float> improved(static_cast<size_t>(k));
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        const float u = (stats[i].eff_n > 0) ? stats[i].utility : v_mix;
        const float q_norm = (u + 1.0f) * 0.5f;
        improved[i] = stats[i].logit + sigma_coeff * q_norm;
        if (improved[i] > max_logit) max_logit = improved[i];
    }
    float exp_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        improved[i] = std::exp(improved[i] - max_logit);
        exp_sum += improved[i];
    }
    const float inv_sum = (exp_sum > 1e-20f) ? (1.0f / exp_sum) : 0.0f;

    // eq. 14: argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ].
    const float inv_total = 1.0f / (1.0f + static_cast<float>(sum_n));
    int best_i = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        const float pi_prime = (inv_sum > 0.0f)
            ? improved[i] * inv_sum
            : 1.0f / static_cast<float>(k);
        const float score = pi_prime - static_cast<float>(stats[i].eff_n) * inv_total;
        if (score > best_score) {
            best_score = score;
            best_i = i;
        }
    }
    return best_i;
}

}  // namespace skyzero

#endif
