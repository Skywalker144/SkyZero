#ifndef SKYZERO_ALPHAZERO_H
#define SKYZERO_ALPHAZERO_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/nn/functional.h>
#include <torch/torch.h>

#include "policy_surprise_weighting.h"
#include "subtree_value_bias.h"
#include "utils.h"

namespace skyzero {

// ---------------------------------------------------------------------------
// Config — aligned to Python V3 args dict
// ---------------------------------------------------------------------------
struct AlphaZeroConfig {
    int board_size = 15;

    // Gumbel MCTS
    int num_simulations = 512;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;

    // Playout Cap Randomization (KataGo-style training efficiency)
    int cheap_simulations = 64;
    int cheap_gumbel_m = 8;
    float full_search_prob = 0.25f;
    float cheap_sample_weight = 0.0f;  // KataGomo-aligned: cheap rows drop entirely (cheapSearchTargetWeight=0)

    // Exploration temperature
    int half_life = -1;                  // Python: args.get("half_life", game.board_size)
    float move_temperature_init = 1.1f;
    float move_temperature_final = 1.0f;

    // PUCT / FPU (KataGomo-aligned; see searchexplorehelpers.cpp:242-298)
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_reduction_max = 0.2f;
    float root_fpu_reduction_max = 0.1f;
    float fpu_loss_prop = 0.0f;
    float root_fpu_loss_prop = 0.0f;
    // parent_utility blended with NN utility by visited policy mass: avg_w = min(1, visited^pow)
    // When true (KataGo selfplay default), FPU base = avg_w * parent_utility + (1-avg_w) * nn_utility.
    // When false and fpu_parent_weight > 0: fixed-ratio NN blend.
    // When false and fpu_parent_weight == 0: FPU base = parent_utility (KataGo raw default).
    bool fpu_parent_weight_by_visited_policy = true;
    float fpu_parent_weight_by_visited_policy_pow = 1.0f;
    float fpu_parent_weight = 0.0f;

    // Virtual-loss multiplier: each in-flight thread pulls child utility toward loss with this weight.
    // KataGo default 3.0; see searchexplorehelpers.cpp:135-143.
    float num_virtual_losses_per_thread = 3.0f;

    // Dynamic Variance-Scaled cPUCT
    float cpuct_utility_stdev_prior = 0.40f;
    float cpuct_utility_stdev_prior_weight = 2.0f;
    float cpuct_utility_stdev_scale = 0.85f;

    // Uncertainty-Weighted MCTS Backup (KataGo-style; requires value_error head)
    // Weight formula: w = coeff / (u^exp + coeff/max_weight)
    bool enable_uncertainty_weighting = false;
    float uncertainty_coeff = 0.25f;
    float uncertainty_exponent = 1.0f;
    float uncertainty_max_weight = 8.0f;

    // Subtree Value Bias
    bool enable_subtree_value_bias = false;
    float subtree_value_bias_factor = 0.35f;
    float subtree_value_bias_weight_exponent = 0.85f;
    float subtree_value_bias_free_prop = 0.8f;
    int subtree_value_bias_table_shards = 4096;
    int subtree_value_bias_pattern_radius = 2;

    // Stochastic transform
    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;
    bool enable_symmetry_inference_for_root = false;
    bool enable_symmetry_inference_for_child = false;

    // Surprise weighting / value target
    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;
    float value_target_mix_now_factor_constant = 0.2f;

    // Soft resign
    float soft_resign_threshold = 0.9f;
    int soft_resign_step_threshold = 3;
    float soft_resign_prob = 0.7f;
    float soft_resign_sample_weight = 0.1f;
    int min_simulations_in_soft_resign = 8;

    // Fork side positions (KataGo-style data diversity)
    float fork_side_position_prob = 0.04f;
    int max_fork_queue_size = 1000;
    int fork_skip_first_n_moves = 3;  // don't fork during opening

    // Selfplay output
    int max_games_total = 4000;
    int max_rows_per_file = 25000;
    std::string model_dir = "data/models";
    std::string output_dir = "data/selfplay";

    torch::Device device = torch::kCPU;
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
    std::vector<float> nn_policy;          // softmax probabilities (legal‑masked)
    std::vector<float> nn_logits;          // raw logits (legal‑masked, -inf for illegal)
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};  // WDL

    std::array<float, 3> v{0.0f, 0.0f, 0.0f};
    float utility_sq_sum = 0.0f;  // cumulative squared utility (parent perspective)
    int n = 0;                     // raw visit count (for vloss / soft-resign)
    float weighted_n = 0.0f;       // backup weight sum (uncertainty-weighted)
    int vloss = 0;

    // Cached NN-predicted value error for this node (post-softplus).
    // Used by uncertainty-weighted backup; 0.0 means uninitialized.
    float nn_value_error = 0.0f;

    // Subtree Value Bias bookkeeping
    std::shared_ptr<SubtreeValueBiasEntry> svb_entry;
    float last_svb_delta_sum = 0.0f;
    float last_svb_weight = 0.0f;

    bool is_expanded() const { return !children.empty(); }

    void update(const std::array<float, 3>& value, float weight = 1.0f) {
        v[0] += weight * value[0];
        v[1] += weight * value[1];
        v[2] += weight * value[2];
        // parent-perspective utility = child_loss - child_win
        const float u = value[2] - value[0];
        utility_sq_sum += weight * u * u;
        weighted_n += weight;
        n += 1;
    }
};

// ---------------------------------------------------------------------------
// Search output
// ---------------------------------------------------------------------------
struct MCTSSearchOutput {
    std::vector<float> mcts_policy;                         // improved policy (Gumbel)
    std::array<float, 3> v_mix{0.0f, 0.0f, 0.0f};         // WDL v_mix (search root value)
    std::vector<float> nn_policy;                           // raw NN policy
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f}; // raw NN value
    int gumbel_action = -1;                                 // selected action by Gumbel
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
// Shared MCTS helpers: dynamic variance-scaled cPUCT + FPU
// ---------------------------------------------------------------------------

struct SelectParams {
    float explore_scaling;
    float fpu_value;
};

// Compute the stdev-based scaling factor for cPUCT at a given node.
inline float compute_parent_utility_stdev_factor(
    const MCTSNode& node,
    float parent_utility,
    const AlphaZeroConfig& cfg
) {
    const float variance_prior = cfg.cpuct_utility_stdev_prior * cfg.cpuct_utility_stdev_prior;
    const float variance_prior_weight = cfg.cpuct_utility_stdev_prior_weight;

    float parent_stdev;
    if (node.weighted_n <= 1.0f) {
        parent_stdev = cfg.cpuct_utility_stdev_prior;
    } else {
        const float effective_n = node.weighted_n;
        const float utility_sq_avg = node.utility_sq_sum / effective_n;
        const float u_sq = parent_utility * parent_utility;
        const float adj_sq_avg = std::max(utility_sq_avg, u_sq);
        parent_stdev = std::sqrt(std::max(0.0f,
            ((u_sq + variance_prior) * variance_prior_weight + adj_sq_avg * effective_n)
            / (variance_prior_weight + effective_n - 1.0f)
            - u_sq
        ));
    }

    return 1.0f + cfg.cpuct_utility_stdev_scale
        * (parent_stdev / std::max(1e-8f, cfg.cpuct_utility_stdev_prior) - 1.0f);
}

// Compute explore_scaling and fpu_value for a node's children.
// effective_parent_weight = node.weighted_n + node.vloss (parallel) or node.weighted_n (single).
inline SelectParams compute_select_params(
    const MCTSNode& node,
    float effective_parent_weight,
    float visited_policy_mass,
    const AlphaZeroConfig& cfg
) {
    const float total_child_weight = std::max(0.0f, effective_parent_weight - 1.0f);

    const float c_puct = cfg.c_puct + cfg.c_puct_log
        * std::log((total_child_weight + cfg.c_puct_base) / cfg.c_puct_base);

    std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
    if (node.weighted_n > 0.0f) {
        parent_q = {node.v[0] / node.weighted_n, node.v[1] / node.weighted_n, node.v[2] / node.weighted_n};
    }
    float parent_utility = wdl_utility(parent_q);

    const float stdev_factor = compute_parent_utility_stdev_factor(node, parent_utility, cfg);
    const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f) * stdev_factor;

    float parent_utility_for_fpu = parent_utility;
    if (cfg.fpu_parent_weight_by_visited_policy) {
        const float nn_utility = wdl_utility(node.nn_value_probs);
        const float avg_weight = std::min(1.0f, static_cast<float>(
            std::pow(visited_policy_mass, cfg.fpu_parent_weight_by_visited_policy_pow)));
        parent_utility_for_fpu = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;
    } else if (cfg.fpu_parent_weight > 0.0f) {
        const float nn_utility = wdl_utility(node.nn_value_probs);
        parent_utility_for_fpu = cfg.fpu_parent_weight * nn_utility
                               + (1.0f - cfg.fpu_parent_weight) * parent_utility;
    }

    const bool is_root = (node.parent == nullptr);
    const float fpu_reduction_max = is_root ? cfg.root_fpu_reduction_max : cfg.fpu_reduction_max;
    const float fpu_loss_prop = is_root ? cfg.root_fpu_loss_prop : cfg.fpu_loss_prop;
    const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

}  // namespace skyzero

#endif
