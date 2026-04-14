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

#include "nets.h"
#include "policy_surprise_weighting.h"
#include "subtree_value_bias.h"
#include "utils.h"

namespace skyzero {

// ---------------------------------------------------------------------------
// Config — aligned to Python V3 args dict
// ---------------------------------------------------------------------------
struct AlphaZeroConfig {
    int board_size = 15;
    int num_blocks = 4;
    int num_channels = 128;

    // Gumbel MCTS
    int num_simulations = 512;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_stochastic_eval = false;

    // Exploration temperature
    int half_life = -1;                  // Python: args.get("half_life", game.board_size)
    float move_temperature_init = 1.1f;
    float move_temperature_final = 1.0f;

    // PUCT / FPU
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.2f;
    float root_fpu_reduction_max = 0.1f;
    float fpu_loss_prop = 0.0f;

    // Dynamic Variance-Scaled cPUCT
    float cpuct_utility_stdev_prior = 0.40f;
    float cpuct_utility_stdev_prior_weight = 2.0f;
    float cpuct_utility_stdev_scale = 0.85f;

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
    int n = 0;
    int vloss = 0;

    // Subtree Value Bias bookkeeping
    std::shared_ptr<SubtreeValueBiasEntry> svb_entry;
    float last_svb_delta_sum = 0.0f;
    float last_svb_weight = 0.0f;

    bool is_expanded() const { return !children.empty(); }

    void update(const std::array<float, 3>& value) {
        v[0] += value[0];
        v[1] += value[1];
        v[2] += value[2];
        // parent-perspective utility = child_loss - child_win
        const float u = value[2] - value[0];
        utility_sq_sum += u * u;
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
    if (node.n <= 1) {
        parent_stdev = cfg.cpuct_utility_stdev_prior;
    } else {
        const float effective_n = static_cast<float>(node.n);
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
// effective_parent_n should be (node.n) for single-thread, (node.n + node.vloss) for parallel.
inline SelectParams compute_select_params(
    const MCTSNode& node,
    int effective_parent_n,
    float visited_policy_mass,
    const AlphaZeroConfig& cfg
) {
    const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));

    const float c_puct = cfg.c_puct + cfg.c_puct_log
        * std::log((total_child_weight + cfg.c_puct_base) / cfg.c_puct_base);

    std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
    if (node.n > 0) {
        parent_q = {node.v[0] / node.n, node.v[1] / node.n, node.v[2] / node.n};
    }
    float parent_utility = wdl_utility(parent_q);

    const float stdev_factor = compute_parent_utility_stdev_factor(node, parent_utility, cfg);
    const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f) * stdev_factor;

    const float nn_utility = wdl_utility(node.nn_value_probs);
    const float avg_weight = std::min(1.0f, static_cast<float>(std::pow(visited_policy_mass, cfg.fpu_pow)));
    const float parent_utility_for_fpu = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;

    const float fpu_reduction_max = (node.parent == nullptr) ? cfg.root_fpu_reduction_max : cfg.fpu_reduction_max;
    const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * cfg.fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

// ---------------------------------------------------------------------------
// MCTS — Gumbel AlphaZero (single-threaded, used by playgame / non-parallel)
// Aligned to Python V3 MCTS class
// ---------------------------------------------------------------------------
template <typename Game>
class MCTS {
public:
    MCTS(Game& game, const AlphaZeroConfig& config, ResNet& model)
        : game_(game), cfg_(config), model_(model), rng_(std::random_device{}()) {
        model_->to(cfg_.device);
        model_->eval();
    }

    // ---- search (selfplay mode) ----
    MCTSSearchOutput search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root
    ) {
        if (!root) {
            root.reset(new MCTSNode{state, to_play});
        }

        std::vector<float> nn_policy;
        std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
        if (!root->is_expanded()) {
            auto pair = root_expand(*root);
            nn_policy = pair.first;
            nn_value_probs = pair.second;
            backpropagate(root.get(), nn_value_probs);
        } else {
            nn_policy = root->nn_policy;
            nn_value_probs = root->nn_value_probs;
        }

        auto [mcts_policy, gumbel_action, v_mix] = gumbel_sequential_halving(*root, num_simulations, /*is_eval=*/false);

        MCTSSearchOutput out;
        out.mcts_policy = std::move(mcts_policy);
        out.v_mix = v_mix;
        out.nn_policy = nn_policy;
        out.nn_value_probs = nn_value_probs;
        out.gumbel_action = gumbel_action;
        return out;
    }

    // ---- eval_search (eval / play mode) ----
    MCTSSearchOutput eval_search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root
    ) {
        if (!root) {
            root.reset(new MCTSNode{state, to_play});
        }

        std::vector<float> nn_policy;
        std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
        if (!root->is_expanded()) {
            auto pair = root_expand(*root);
            nn_policy = pair.first;
            nn_value_probs = pair.second;
            backpropagate(root.get(), nn_value_probs);
        } else {
            nn_policy = root->nn_policy;
            nn_value_probs = root->nn_value_probs;
        }

        auto [mcts_policy, gumbel_action, v_mix] = gumbel_sequential_halving(*root, num_simulations, /*is_eval=*/true);

        MCTSSearchOutput out;
        out.mcts_policy = std::move(mcts_policy);
        out.v_mix = v_mix;
        out.nn_policy = nn_policy;
        out.nn_value_probs = nn_value_probs;
        out.gumbel_action = gumbel_action;
        return out;
    }

private:
    // ------------------------------------------------------------------
    // Inference helpers (aligned to Python _inference / _inference_with_stochastic_transform)
    // ------------------------------------------------------------------
    struct InferenceResult {
        std::vector<float> nn_policy;    // softmax over legal
        std::array<float, 3> nn_value_probs;
        std::vector<float> masked_logits; // logits with -inf for illegal
    };

    static std::vector<float> undo_transform_flat(
        const std::vector<float>& transformed,
        int board_size,
        int k,
        bool do_flip
    ) {
        std::vector<float> out(transformed.size(), 0.0f);
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                int rr = r;
                int cc = c;
                for (int t = 0; t < k; ++t) {
                    const int nr = board_size - 1 - cc;
                    const int nc = rr;
                    rr = nr;
                    cc = nc;
                }
                if (do_flip) {
                    cc = board_size - 1 - cc;
                }
                out[r * board_size + c] = transformed[rr * board_size + cc];
            }
        }
        return out;
    }

    InferenceResult inference(const std::vector<int8_t>& state, int to_play) {
        auto encoded = game_.encode_state(state, to_play);
        const int c = game_.num_planes;
        const int h = game_.board_size;
        const int w = game_.board_size;

        auto in_tensor = torch::from_blob(encoded.data(), {1, c, h, w}, torch::kInt8)
                             .to(torch::kFloat32)
                             .clone()
                             .to(cfg_.device);

        torch::NoGradGuard no_grad;
        const auto nn_out = model_->forward(in_tensor);

        auto policy_logits = nn_out.policy_logits;
        auto logits_cpu = policy_logits.flatten().to(torch::kCPU).contiguous();
        std::vector<float> logits(static_cast<size_t>(logits_cpu.numel()));
        std::memcpy(logits.data(), logits_cpu.template data_ptr<float>(), logits.size() * sizeof(float));

        const auto legal = game_.get_is_legal_actions(state, to_play);
        for (size_t i = 0; i < logits.size(); ++i) {
            if (i >= legal.size() || legal[i] == 0) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }

        auto policy = softmax(logits);

        auto value_probs = torch::softmax(nn_out.value_logits, 1).to(torch::kCPU).contiguous();
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        if (value_probs.numel() >= 3) {
            const auto* p = value_probs.template data_ptr<float>();
            value = {p[0], p[1], p[2]};
        }
        return {policy, value, logits};
    }

    InferenceResult inference_with_stochastic_transform(const std::vector<int8_t>& state, int to_play) {
        auto encoded = game_.encode_state(state, to_play);
        const int c = game_.num_planes;
        const int h = game_.board_size;
        const int w = game_.board_size;

        auto in_tensor = torch::from_blob(encoded.data(), {1, c, h, w}, torch::kInt8)
                             .to(torch::kFloat32)
                             .clone()
                             .to(cfg_.device);

        std::uniform_int_distribution<int> dist(0, 7);
        const int transform_type = dist(rng_);
        const int k = transform_type % 4;
        const bool do_flip = transform_type >= 4;

        in_tensor = torch::rot90(in_tensor, k, {2, 3});
        if (do_flip) {
            in_tensor = torch::flip(in_tensor, {3});
        }

        torch::NoGradGuard no_grad;
        const auto nn_out = model_->forward(in_tensor);

        auto policy_logits = nn_out.policy_logits;
        if (do_flip) {
            policy_logits = torch::flip(policy_logits, {3});
        }
        policy_logits = torch::rot90(policy_logits, 4 - (k % 4), {2, 3});

        auto logits_cpu = policy_logits.flatten().to(torch::kCPU).contiguous();
        std::vector<float> logits(static_cast<size_t>(logits_cpu.numel()));
        std::memcpy(logits.data(), logits_cpu.template data_ptr<float>(), logits.size() * sizeof(float));

        const auto legal = game_.get_is_legal_actions(state, to_play);
        for (size_t i = 0; i < logits.size(); ++i) {
            if (i >= legal.size() || legal[i] == 0) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }

        auto policy = softmax(logits);

        auto value_probs = torch::softmax(nn_out.value_logits, 1).to(torch::kCPU).contiguous();
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        if (value_probs.numel() >= 3) {
            const auto* p = value_probs.template data_ptr<float>();
            value = {p[0], p[1], p[2]};
        }
        return {policy, value, logits};
    }

    InferenceResult inference_with_symmetry(const std::vector<int8_t>& state, int to_play) {
        const auto encoded = game_.encode_state(state, to_play);
        const int c = game_.num_planes;
        const int h = game_.board_size;
        const int w = game_.board_size;
        const int area = h * w;

        std::vector<int8_t> sym_encoded;
        sym_encoded.reserve(encoded.size() * 8);
        for (int fi = 0; fi < 2; ++fi) {
            const bool do_flip = (fi == 1);
            for (int k = 0; k < 4; ++k) {
                auto aug = transform_encoded_state(encoded, c, h, k, do_flip);
                sym_encoded.insert(sym_encoded.end(), aug.begin(), aug.end());
            }
        }

        auto in_tensor = torch::from_blob(sym_encoded.data(), {8, c, h, w}, torch::kInt8)
                             .to(torch::kFloat32)
                             .clone()
                             .to(cfg_.device);

        torch::NoGradGuard no_grad;
        const auto nn_out = model_->forward(in_tensor);

        auto policy_cpu = nn_out.policy_logits.reshape({8, area}).to(torch::kCPU).contiguous();
        const float* pp = policy_cpu.template data_ptr<float>();
        std::vector<float> avg_logits(static_cast<size_t>(area), 0.0f);
        for (int i = 0; i < 8; ++i) {
            std::vector<float> logits_i(static_cast<size_t>(area), 0.0f);
            std::memcpy(
                logits_i.data(),
                pp + static_cast<size_t>(i) * static_cast<size_t>(area),
                static_cast<size_t>(area) * sizeof(float)
            );
            const int k = i % 4;
            const bool do_flip = i >= 4;
            const auto untransformed = undo_transform_flat(logits_i, h, k, do_flip);
            for (int j = 0; j < area; ++j) {
                avg_logits[static_cast<size_t>(j)] += untransformed[static_cast<size_t>(j)] / 8.0f;
            }
        }

        const auto legal = game_.get_is_legal_actions(state, to_play);
        for (size_t i = 0; i < avg_logits.size(); ++i) {
            if (i >= legal.size() || legal[i] == 0) {
                avg_logits[i] = -std::numeric_limits<float>::infinity();
            }
        }

        auto policy = softmax(avg_logits);

        auto value_cpu = torch::softmax(nn_out.value_logits, 1).to(torch::kCPU).contiguous();
        std::array<float, 3> value{0.0f, 0.0f, 0.0f};
        if (value_cpu.numel() >= 24) {
            const float* vp = value_cpu.template data_ptr<float>();
            for (int i = 0; i < 8; ++i) {
                const size_t base = static_cast<size_t>(i) * 3;
                value[0] += vp[base] / 8.0f;
                value[1] += vp[base + 1] / 8.0f;
                value[2] += vp[base + 2] / 8.0f;
            }
        } else {
            value = {0.0f, 1.0f, 0.0f};
        }

        return {policy, value, avg_logits};
    }

    // ------------------------------------------------------------------
    // select — PUCT with FPU + dynamic stdev-scaled cPUCT
    // ------------------------------------------------------------------
    MCTSNode* select(MCTSNode& node) {
        if (node.children.empty()) return nullptr;

        float visited_policy_mass = 0.0f;
        for (auto& c : node.children) {
            if (c->n > 0) visited_policy_mass += c->prior;
        }

        const auto sp = compute_select_params(node, node.n, visited_policy_mass, cfg_);

        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        for (auto& c : node.children) {
            float q_value;
            if (c->n == 0) {
                q_value = sp.fpu_value;
            } else {
                const auto child_q_arr = std::array<float, 3>{
                    c->v[0] / c->n, c->v[1] / c->n, c->v[2] / c->n
                };
                q_value = child_q_arr[2] - child_q_arr[0];
            }
            const float u_value = sp.explore_scaling * c->prior / (1.0f + static_cast<float>(c->n));
            const float score = q_value + u_value;
            if (score > best_score) {
                best_score = score;
                best_child = c.get();
            }
        }
        return best_child;
    }

    // ------------------------------------------------------------------
    // expand — child inference
    // ------------------------------------------------------------------
    std::array<float, 3> expand(MCTSNode& node) {
        InferenceResult ir;
        if (cfg_.enable_stochastic_transform_inference_for_child) {
            ir = inference_with_stochastic_transform(node.state, node.to_play);
        } else if (cfg_.enable_symmetry_inference_for_child) {
            ir = inference_with_symmetry(node.state, node.to_play);
        } else {
            ir = inference(node.state, node.to_play);
        }
        node.nn_value_probs = ir.nn_value_probs;
        node.nn_policy = ir.nn_policy;
        node.nn_logits = ir.masked_logits;
        node.children.clear();
        for (int a = 0; a < static_cast<int>(ir.nn_policy.size()); ++a) {
            const float p = ir.nn_policy[a];
            if (p <= 0.0f) continue;
            auto child = std::unique_ptr<MCTSNode>(new MCTSNode{
                game_.get_next_state(node.state, a, node.to_play),
                -node.to_play,
                p,
                &node,
                a
            });
            node.children.push_back(std::move(child));
        }
        return ir.nn_value_probs;
    }

    // ------------------------------------------------------------------
    // root_expand — root inference (no Dirichlet in Gumbel MCTS; Gumbel noise replaces it)
    // ------------------------------------------------------------------
    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& node) {
        InferenceResult ir;
        if (cfg_.enable_stochastic_transform_inference_for_root) {
            ir = inference_with_stochastic_transform(node.state, node.to_play);
        } else if (cfg_.enable_symmetry_inference_for_root) {
            ir = inference_with_symmetry(node.state, node.to_play);
        } else {
            ir = inference(node.state, node.to_play);
        }
        node.nn_value_probs = ir.nn_value_probs;
        node.nn_policy = ir.nn_policy;
        node.nn_logits = ir.masked_logits;

        node.children.clear();
        for (int a = 0; a < static_cast<int>(ir.nn_policy.size()); ++a) {
            const float p = ir.nn_policy[a];
            if (p <= 0.0f) continue;
            auto child = std::unique_ptr<MCTSNode>(new MCTSNode{
                game_.get_next_state(node.state, a, node.to_play),
                -node.to_play,
                p,
                &node,
                a
            });
            node.children.push_back(std::move(child));
        }
        return {ir.nn_policy, ir.nn_value_probs};
    }

    // ------------------------------------------------------------------
    // backpropagate
    // ------------------------------------------------------------------
    static void backpropagate(MCTSNode* node, std::array<float, 3> value) {
        while (node != nullptr) {
            node->update(value);
            value = flip_wdl(value);
            node = node->parent;
        }
    }

    // ------------------------------------------------------------------
    // Gumbel Sequential Halving — aligned to Python V3 MCTS._gumbel_sequential_halving
    // ------------------------------------------------------------------
    struct GumbelResult {
        std::vector<float> improved_policy;
        int gumbel_action;
        std::array<float, 3> v_mix;
    };

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations, bool is_eval) {
        const int action_size = game_.board_size * game_.board_size;

        // Copy logits from root (already masked, -inf for illegal)
        std::vector<float> logits(action_size, -std::numeric_limits<float>::infinity());
        if (!root.nn_logits.empty()) {
            logits = root.nn_logits;
        }
        auto is_legal = game_.get_is_legal_actions(root.state, root.to_play);

        // Gumbel noise
        std::vector<float> g(action_size, 0.0f);
        if (is_eval && !cfg_.gumbel_stochastic_eval) {
            // eval mode: no noise
        } else {
            std::extreme_value_distribution<float> gumbel_dist(0.0f, 1.0f);
            for (int i = 0; i < action_size; ++i) {
                g[i] = gumbel_dist(rng_);
            }
        }

        // scores = logits + g, pick top m
        int m = std::min(num_simulations, cfg_.gumbel_m);
        std::vector<float> legal_scores(action_size, -std::numeric_limits<float>::infinity());
        for (int i = 0; i < action_size; ++i) {
            if (is_legal[i]) {
                legal_scores[i] = logits[i] + g[i];
            }
        }

        // argsort descending, pick top m
        std::vector<int> sorted_actions(action_size);
        std::iota(sorted_actions.begin(), sorted_actions.end(), 0);
        std::sort(sorted_actions.begin(), sorted_actions.end(), [&](int a, int b) {
            return legal_scores[a] > legal_scores[b];
        });

        std::vector<int> surviving_actions;
        surviving_actions.reserve(m);
        for (int i = 0; i < action_size && static_cast<int>(surviving_actions.size()) < m; ++i) {
            if (is_legal[sorted_actions[i]]) {
                surviving_actions.push_back(sorted_actions[i]);
            }
        }
        m = static_cast<int>(surviving_actions.size());

        // Sequential halving
        if (m > 0) {
            const int phases = (m > 1) ? static_cast<int>(std::ceil(std::log2(static_cast<double>(m)))) : 1;
            int sims_budget = num_simulations;

            for (int phase = 0; phase < phases; ++phase) {
                if (sims_budget <= 0) break;
                const int remaining_phases = phases - phase;
                const int sims_this_phase = sims_budget / remaining_phases;
                const int num_actions = static_cast<int>(surviving_actions.size());
                const int sims_per_action = std::max(1, sims_this_phase / num_actions);

                for (int s = 0; s < sims_per_action; ++s) {
                    if (sims_budget <= 0) break;
                    for (int action : surviving_actions) {
                        if (sims_budget <= 0) break;
                        // Find child for this action
                        MCTSNode* child = nullptr;
                        for (auto& c : root.children) {
                            if (c->action_taken == action) {
                                child = c.get();
                                break;
                            }
                        }
                        if (child == nullptr) continue;

                        MCTSNode* node = child;
                        while (node->is_expanded()) {
                            node = select(*node);
                            if (node == nullptr) break;
                        }
                        if (node == nullptr) continue;

                        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
                        if (game_.is_terminal(node->state, node->action_taken, -node->to_play)) {
                            const int result = game_.get_winner(node->state, node->action_taken, -node->to_play) * node->to_play;
                            if (result == 1) value = {1.0f, 0.0f, 0.0f};
                            else if (result == -1) value = {0.0f, 0.0f, 1.0f};
                        } else {
                            value = expand(*node);
                        }
                        backpropagate(node, value);
                        sims_budget -= 1;
                    }
                }

                // Halve surviving actions
                if (sims_budget <= 0) break;
                if (phase < phases - 1) {
                    const float max_n_val = max_child_n(root);
                    const float c_visit = cfg_.gumbel_c_visit;
                    const float c_scale = cfg_.gumbel_c_scale;

                    auto eval_action = [&](int a) -> float {
                        MCTSNode* c = nullptr;
                        for (auto& ch : root.children) {
                            if (ch->action_taken == a) { c = ch.get(); break; }
                        }
                        float q = 0.5f;
                        if (c && c->n > 0) {
                            auto child_wdl = std::array<float, 3>{
                                c->v[0] / c->n, c->v[1] / c->n, c->v[2] / c->n
                            };
                            q = child_wdl[2] - child_wdl[0]; // parent's Q
                            q = (q + 1.0f) / 2.0f; // [0, 1]
                        }
                        return logits[a] + g[a] + (c_visit + max_n_val) * c_scale * q;
                    };

                    std::sort(surviving_actions.begin(), surviving_actions.end(), [&](int a, int b) {
                        return eval_action(a) > eval_action(b);
                    });
                    const int keep = std::max(1, static_cast<int>(surviving_actions.size()) / 2);
                    surviving_actions.resize(keep);
                }
            }
        }

        // ---- Compute improved policy (sigma_q + logits) ----
        const float c_visit = cfg_.gumbel_c_visit;
        const float c_scale = cfg_.gumbel_c_scale;
        const float max_n_val = max_child_n(root);

        // Collect per-action WDL Q and visit counts
        std::vector<std::array<float, 3>> q_wdl(action_size, {0.0f, 0.0f, 0.0f});
        std::vector<float> n_values(action_size, 0.0f);
        for (auto& c : root.children) {
            if (c->n > 0) {
                auto child_wdl = std::array<float, 3>{
                    c->v[0] / c->n, c->v[1] / c->n, c->v[2] / c->n
                };
                // parent perspective: flip child WDL
                q_wdl[c->action_taken] = {child_wdl[2], child_wdl[1], child_wdl[0]};
                n_values[c->action_taken] = static_cast<float>(c->n);
            }
        }

        const float sum_n = std::accumulate(n_values.begin(), n_values.end(), 0.0f);
        const auto& nn_value_wdl = root.nn_value_probs;
        std::array<float, 3> v_mix_wdl;

        if (sum_n > 0.0f) {
            // policy-weighted average of visited Q (WDL)
            std::array<float, 3> weighted_q{0.0f, 0.0f, 0.0f};
            float policy_visited_sum = 1e-12f;
            for (int a = 0; a < action_size; ++a) {
                if (n_values[a] > 0.0f) {
                    const float pw = root.nn_policy.empty() ? 0.0f : root.nn_policy[a];
                    weighted_q[0] += pw * q_wdl[a][0];
                    weighted_q[1] += pw * q_wdl[a][1];
                    weighted_q[2] += pw * q_wdl[a][2];
                    policy_visited_sum += pw;
                }
            }
            weighted_q[0] /= policy_visited_sum;
            weighted_q[1] /= policy_visited_sum;
            weighted_q[2] /= policy_visited_sum;
            v_mix_wdl = {
                (nn_value_wdl[0] + sum_n * weighted_q[0]) / (1.0f + sum_n),
                (nn_value_wdl[1] + sum_n * weighted_q[1]) / (1.0f + sum_n),
                (nn_value_wdl[2] + sum_n * weighted_q[2]) / (1.0f + sum_n),
            };
        } else {
            v_mix_wdl = nn_value_wdl;
        }

        // completed_q: actual Q for visited, v_mix for unvisited
        std::vector<float> completed_q_scalar(action_size, 0.0f);
        for (int a = 0; a < action_size; ++a) {
            std::array<float, 3> cq;
            if (n_values[a] > 0.0f) {
                cq = q_wdl[a];
            } else {
                cq = v_mix_wdl;
            }
            float s = cq[0] - cq[2]; // W - L in [-1, 1]
            s = (s + 1.0f) / 2.0f;   // [0, 1]
            completed_q_scalar[a] = s;
        }

        std::vector<float> sigma_q(action_size, 0.0f);
        for (int a = 0; a < action_size; ++a) {
            sigma_q[a] = (c_visit + max_n_val) * c_scale * completed_q_scalar[a];
        }

        std::vector<float> improved_logits(action_size, 0.0f);
        for (int a = 0; a < action_size; ++a) {
            if (is_legal[a]) {
                improved_logits[a] = logits[a] + sigma_q[a];
            } else {
                improved_logits[a] = -std::numeric_limits<float>::infinity();
            }
        }
        auto improved_policy = softmax(improved_logits);

        // ---- Gumbel action: among surviving, pick most-visited, break ties by final_eval ----
        auto final_eval = [&](int a) -> float {
            return logits[a] + g[a] + sigma_q[a];
        };

        float max_n_surviving = 0.0f;
        for (int a : surviving_actions) {
            max_n_surviving = std::max(max_n_surviving, n_values[a]);
        }
        std::vector<int> most_visited;
        for (int a : surviving_actions) {
            if (n_values[a] == max_n_surviving) {
                most_visited.push_back(a);
            }
        }
        int gumbel_action = -1;
        if (!most_visited.empty()) {
            gumbel_action = most_visited[0];
            float best_eval = final_eval(most_visited[0]);
            for (size_t i = 1; i < most_visited.size(); ++i) {
                const float ev = final_eval(most_visited[i]);
                if (ev > best_eval) {
                    best_eval = ev;
                    gumbel_action = most_visited[i];
                }
            }
        }

        return {improved_policy, gumbel_action, v_mix_wdl};
    }

    // helper: max child N across all root children
    static float max_child_n(const MCTSNode& root) {
        float mx = 0.0f;
        for (auto& c : root.children) {
            mx = std::max(mx, static_cast<float>(c->n));
        }
        return mx;
    }

    Game& game_;
    AlphaZeroConfig cfg_;
    ResNet& model_;
    std::mt19937 rng_;
};


}  // namespace skyzero

#endif
