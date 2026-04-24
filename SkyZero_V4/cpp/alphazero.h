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
#include <vector>

#include <torch/torch.h>

namespace skyzero {

// ---------------------------------------------------------------------------
// Config — only the fields the C++ selfplay path needs.
// Python side owns everything training-related.
// ---------------------------------------------------------------------------
struct AlphaZeroConfig {
    int board_size = 15;

    // Gumbel MCTS
    int num_simulations = 64;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_noise_enabled = true;

    // Exploration temperature
    int half_life = -1;                  // -1 ⇒ use board_size
    float move_temperature_init = 0.8f;
    float move_temperature_final = 0.2f;

    // PUCT / FPU
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.08f;
    float root_fpu_reduction_max = 0.0f;
    float fpu_loss_prop = 0.0f;

    // Stochastic transform / symmetry at inference time
    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;
    bool enable_symmetry_inference_for_root = false;
    bool enable_symmetry_inference_for_child = false;

    // Surprise weighting / value target mixing
    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;
    float value_target_mix_now_factor_constant = 0.2f;

    // Balanced opening (KataGomo-style). Each game samples r ~ U(0,1):
    //   r <  balance_opening_prob → NN-scored random opening;
    //   r >= balance_opening_prob → empty-board start.
    float balance_opening_prob = 0.8f;
    int balanced_opening_max_tries = 20;
    float balanced_opening_avg_dist_factor = 0.8f;
    float balanced_opening_reject_prob = 0.995f;
    float balanced_opening_reject_prob_fallback = 0.8f;

    // Policy-initialization (KataGomo initGamesWithPolicy). After balanced
    // opening, play ~Exp(1)*policy_init_avg_move_num extra moves sampled from
    // the NN policy^(1/temperature) to push the game off the balance plateau.
    // Set avg_move_num <= 0 to disable.
    float policy_init_avg_move_num = 0.0f;
    float policy_init_temperature = 1.0f;

    // Soft resign
    float soft_resign_threshold = 0.9f;
    int soft_resign_step_threshold = 3;
    float soft_resign_prob = 0.7f;
    float soft_resign_sample_weight = 0.1f;
    int min_simulations_in_soft_resign = 8;

    torch::Device device = torch::kCPU;
};

// ---------------------------------------------------------------------------
// Selfplay parallelism config — shared by ParallelMCTS (BatchedLeaf backend)
// and TreeParallelMCTS (SharedTree backend). Lives here so TreeParallelMCTS
// does not need to pull in alphazero_parallel.h just for this struct.
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
    int vloss = 0;

    bool is_expanded() const { return !children.empty(); }

    void update(const std::array<float, 3>& value) {
        v[0] += value[0];
        v[1] += value[1];
        v[2] += value[2];
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
    int gumbel_action = -1;                                 // selected action by Gumbel
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
// (Dynamic variance-scaled cPUCT removed: stdev_factor is fixed at 1.0.)
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
    const AlphaZeroConfig& cfg
) {
    const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));

    const float c_puct = cfg.c_puct + cfg.c_puct_log
        * std::log((total_child_weight + cfg.c_puct_base) / cfg.c_puct_base);

    const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f);

    std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
    if (node.n > 0) {
        parent_q = {node.v[0] / node.n, node.v[1] / node.n, node.v[2] / node.n};
    }
    const float parent_utility = wdl_utility(parent_q);
    const float nn_utility = wdl_utility(node.nn_value_probs);
    const float avg_weight = std::min(1.0f, static_cast<float>(std::pow(visited_policy_mass, cfg.fpu_pow)));
    const float parent_utility_for_fpu = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;

    const float fpu_reduction_max = (node.parent == nullptr) ? cfg.root_fpu_reduction_max : cfg.fpu_reduction_max;
    const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * cfg.fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

}  // namespace skyzero

#endif
