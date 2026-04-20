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

    // Exploration temperature
    int half_life = -1;                  // Python: args.get("half_life", game.board_size)
    float move_temperature_init = 1.1f;
    float move_temperature_final = 1.0f;

    // Virtual-loss multiplier: inflates N(a) in the Gumbel selection rule for
    // in-flight simulations (each pending rollout adds this much to N_eff(a)).
    float num_virtual_losses_per_thread = 3.0f;

    // Uncertainty-Weighted MCTS Backup (KataGo-style; requires value_error head)
    // Weight formula: w = coeff / (u^exp + coeff/max_weight)
    bool enable_uncertainty_weighting = false;
    float uncertainty_coeff = 0.25f;
    float uncertainty_exponent = 1.0f;
    float uncertainty_max_weight = 8.0f;

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

}  // namespace skyzero

#endif
