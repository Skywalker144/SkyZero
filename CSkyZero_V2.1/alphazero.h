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
#include "replaybuffer.h"
#include "utils.h"

namespace skyzero {

struct AlphaZeroConfig {
    int board_size = 15;
    int num_blocks = 4;
    int num_channels = 128;
    float lr = 1e-4f;
    float weight_decay = 3e-5f;

    int full_search_num_simulations = 800;
    int fast_search_num_simulations = 200;
    float full_search_prob = 0.25f;

    float root_temperature_init = 1.25f;
    float root_temperature_final = 1.1f;
    float move_temperature_init = 0.8f;
    float move_temperature_final = 0.2f;

    float total_dirichlet_alpha = 6.75f;
    float dirichlet_epsilon = 0.25f;

    int batch_size = 128;
    int min_buffer_size = 2048;
    int linear_threshold = 200000;
    float replay_alpha = 0.8f;
    int max_buffer_size = 10000000;
    int train_steps_per_generation = 5;
    float target_replay_ratio = 5.0f;

    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.2f;
    float root_fpu_reduction_max = 0.1f;
    float fpu_loss_prop = 0.0f;

    bool enable_forced_playouts = true;
    float forced_playouts_k = 2.0f;

    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;
    bool enable_symmetry_inference_for_root = false;
    bool enable_symmetry_inference_for_child = false;

    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;
    float value_target_mix_now_factor_constant = 0.2f;

    float soft_resign_threshold = 0.9f;
    int soft_resign_step_threshold = 3;
    float soft_resign_prob = 0.7f;
    float soft_resign_sample_weight = 0.1f;

    float policy_loss_weight = 0.93f;
    float opponent_policy_loss_weight = 0.15f;
    float soft_policy_loss_weight = 8.0f;
    float soft_opponent_policy_loss_weight = 0.18f;
    float value_loss_weight = 0.72f;
    float max_grad_norm = 1.0f;

    int savetime_interval = 3600;
    std::string file_name = "model";
    std::string data_dir = "data";
    bool save_on_exit = true;

    torch::Device device = torch::kCPU;
};

struct MCTSNode {
    std::vector<int8_t> state;
    int to_play = 1;
    float prior = 0.0f;
    MCTSNode* parent = nullptr;
    int action_taken = -1;

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<float> nn_policy;
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};

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

struct MCTSSearchOutput {
    std::vector<float> mcts_policy;
    std::array<float, 3> root_value{0.0f, 0.0f, 0.0f};
    std::vector<float> nn_policy;
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
};

inline std::array<float, 3> flip_wdl(const std::array<float, 3>& in) {
    return {in[2], in[1], in[0]};
}

inline float wdl_utility(const std::array<float, 3>& v) {
    return v[0] - v[2];
}

template <typename Game>
class MCTS {
public:
    MCTS(Game& game, const AlphaZeroConfig& config, ResNet& model)
        : game_(game), cfg_(config), model_(model), rng_(std::random_device{}()) {
        model_->to(cfg_.device);
        model_->eval();
    }

    MCTSSearchOutput search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root
    ) {
        const bool is_full_search = (num_simulations == cfg_.full_search_num_simulations);
        if (!root) {
            root.reset(new MCTSNode{state, to_play});
        }

        std::vector<float> nn_policy;
        std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
        if (!root->is_expanded()) {
            auto pair = root_expand(*root, is_full_search);
            nn_policy = pair.first;
            nn_value_probs = pair.second;
            backpropagate(root.get(), nn_value_probs);
        } else {
            nn_policy = root->nn_policy;
            nn_value_probs = root->nn_value_probs;
        }

        for (int sim = 0; sim < num_simulations; ++sim) {
            MCTSNode* node = root.get();
            while (node->is_expanded()) {
                node = select(*node, is_full_search);
                if (node == nullptr) {
                    throw std::runtime_error("MCTS select returned null");
                }
            }

            std::array<float, 3> value{0.0f, 1.0f, 0.0f};
            if (game_.is_terminal(node->state, node->action_taken, -node->to_play)) {
                const int result = game_.get_winner(node->state, node->action_taken, -node->to_play) * node->to_play;
                if (result == 1) value = {1.0f, 0.0f, 0.0f};
                else if (result == -1) value = {0.0f, 0.0f, 1.0f};
            } else {
                auto pair = inference(
                    node->state,
                    node->to_play,
                    cfg_.enable_stochastic_transform_inference_for_child,
                    cfg_.enable_symmetry_inference_for_child
                );
                expand(*node, pair.first);
                node->nn_policy = pair.first;
                node->nn_value_probs = pair.second;
                value = pair.second;
            }

            backpropagate(node, value);
        }

        std::vector<float> mcts_policy(static_cast<size_t>(game_.board_size * game_.board_size), 0.0f);
        for (auto& c : root->children) {
            mcts_policy[static_cast<size_t>(c->action_taken)] = static_cast<float>(c->n);
        }
        const float sum_n = std::accumulate(mcts_policy.begin(), mcts_policy.end(), 0.0f);
        if (sum_n > 0.0f) {
            for (float& p : mcts_policy) {
                p /= sum_n;
            }
        }

        MCTSSearchOutput out;
        out.mcts_policy = std::move(mcts_policy);
        out.nn_policy = std::move(nn_policy);
        out.nn_value_probs = nn_value_probs;
        if (root->n > 0) {
            out.root_value = {
                root->v[0] / static_cast<float>(root->n),
                root->v[1] / static_cast<float>(root->n),
                root->v[2] / static_cast<float>(root->n),
            };
        }
        return out;
    }

private:
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

    std::pair<std::vector<float>, std::array<float, 3>> inference(
        const std::vector<int8_t>& state,
        int to_play,
        bool use_stochastic_transform,
        bool use_symmetry_transform
    ) {
        auto encoded = game_.encode_state(state, to_play);
        const int c = game_.num_planes;
        const int h = game_.board_size;
        const int w = game_.board_size;
        const int area = h * w;

        if (!use_stochastic_transform && use_symmetry_transform) {
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
            const float neg_inf = -1e30f;
            for (size_t i = 0; i < avg_logits.size(); ++i) {
                if (i >= legal.size() || legal[i] == 0) {
                    avg_logits[i] = neg_inf;
                }
            }

            auto policy = softmax(avg_logits);

            auto value_probs = torch::softmax(nn_out.value_logits, 1).to(torch::kCPU).contiguous();
            std::array<float, 3> value{0.0f, 0.0f, 0.0f};
            if (value_probs.numel() >= 24) {
                const float* vp = value_probs.template data_ptr<float>();
                for (int i = 0; i < 8; ++i) {
                    const size_t base = static_cast<size_t>(i) * 3;
                    value[0] += vp[base] / 8.0f;
                    value[1] += vp[base + 1] / 8.0f;
                    value[2] += vp[base + 2] / 8.0f;
                }
            } else {
                value = {0.0f, 1.0f, 0.0f};
            }
            return {policy, value};
        }

        auto in_tensor = torch::from_blob(encoded.data(), {1, c, h, w}, torch::kInt8)
                             .to(torch::kFloat32)
                             .clone()
                             .to(cfg_.device);

        int k = 0;
        bool do_flip = false;
        if (use_stochastic_transform) {
            std::uniform_int_distribution<int> dist(0, 7);
            const int t = dist(rng_);
            k = t % 4;
            do_flip = t >= 4;
            in_tensor = torch::rot90(in_tensor, k, {2, 3});
            if (do_flip) {
                in_tensor = torch::flip(in_tensor, {3});
            }
        }

        torch::NoGradGuard no_grad;
        const auto nn_out = model_->forward(in_tensor);

        auto policy_logits = nn_out.policy_logits;
        if (use_stochastic_transform) {
            if (do_flip) {
                policy_logits = torch::flip(policy_logits, {3});
            }
            policy_logits = torch::rot90(policy_logits, 4 - (k % 4), {2, 3});
        }

        auto logits_cpu = policy_logits.flatten().to(torch::kCPU).contiguous();
        std::vector<float> logits(static_cast<size_t>(logits_cpu.numel()));
        std::memcpy(logits.data(), logits_cpu.template data_ptr<float>(), logits.size() * sizeof(float));

        const auto legal = game_.get_is_legal_actions(state, to_play);
        const float neg_inf = -1e30f;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (i >= legal.size() || legal[i] == 0) {
                logits[i] = neg_inf;
            }
        }

        auto policy = softmax(logits);

        auto value_probs = torch::softmax(nn_out.value_logits, 1).to(torch::kCPU).contiguous();
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        if (value_probs.numel() >= 3) {
            const auto* p = value_probs.template data_ptr<float>();
            value = {p[0], p[1], p[2]};
        }
        return {policy, value};
    }

    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& root, bool is_full_search) {
        auto pair = inference(
            root.state,
            root.to_play,
            cfg_.enable_stochastic_transform_inference_for_root,
            cfg_.enable_symmetry_inference_for_root
        );
        root.nn_policy = pair.first;
        root.nn_value_probs = pair.second;
        expand(root, root.nn_policy);
        if (is_full_search) {
            apply_dirichlet_to_root(root);
        }
        return pair;
    }

    void expand(MCTSNode& node, const std::vector<float>& policy) {
        const auto legal = game_.get_is_legal_actions(node.state, node.to_play);
        node.children.clear();
        for (size_t action = 0; action < legal.size(); ++action) {
            if (!legal[action]) {
                continue;
            }
            auto child = std::make_unique<MCTSNode>();
            child->state = game_.get_next_state(node.state, static_cast<int>(action), node.to_play);
            child->to_play = -node.to_play;
            child->prior = (action < policy.size()) ? policy[action] : 0.0f;
            child->parent = &node;
            child->action_taken = static_cast<int>(action);
            node.children.push_back(std::move(child));
        }
    }

    void apply_dirichlet_to_root(MCTSNode& root) {
        const int n = static_cast<int>(root.children.size());
        if (n <= 0) {
            return;
        }
        const float total_alpha = std::max(1e-6f, cfg_.total_dirichlet_alpha);
        const float alpha = total_alpha / static_cast<float>(n);
        std::gamma_distribution<float> gamma(alpha, 1.0f);

        std::vector<float> noise(static_cast<size_t>(n), 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            noise[static_cast<size_t>(i)] = gamma(rng_);
            sum += noise[static_cast<size_t>(i)];
        }
        if (sum <= 0.0f) {
            return;
        }
        for (float& x : noise) {
            x /= sum;
        }

        const float eps = std::max(0.0f, std::min(1.0f, cfg_.dirichlet_epsilon));
        for (int i = 0; i < n; ++i) {
            auto& child = root.children[static_cast<size_t>(i)];
            child->prior = child->prior * (1.0f - eps) + noise[static_cast<size_t>(i)] * eps;
        }
    }

    MCTSNode* select(MCTSNode& node, bool is_full_search) {
        if (node.children.empty()) {
            return nullptr;
        }

        MCTSNode* best = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();

        const float total_child_n = static_cast<float>(std::max(1, node.n));
        float c_puct = cfg_.c_puct;
        if (is_full_search && node.n > 0) {
            c_puct += cfg_.c_puct_log *
                std::log((total_child_n + cfg_.c_puct_base) / std::max(1e-6f, cfg_.c_puct_base));
        }
        const float sqrt_total = std::sqrt(total_child_n);

        for (auto& c : node.children) {
            const float q = (c->n > 0)
                ? ((c->v[2] - c->v[0]) / static_cast<float>(c->n))
                : -cfg_.fpu_reduction_max;
            const float u = c_puct * c->prior * sqrt_total / (1.0f + static_cast<float>(c->n));
            const float score = q + u;
            if (score > best_score) {
                best_score = score;
                best = c.get();
            }
        }
        return best;
    }

    void backpropagate(MCTSNode* node, std::array<float, 3> value) {
        while (node != nullptr) {
            node->update(value);
            value = flip_wdl(value);
            node = node->parent;
        }
    }

    Game& game_;
    AlphaZeroConfig cfg_;
    ResNet& model_;
    std::mt19937 rng_;
};

template <typename Game>
class AlphaZero {
public:
    AlphaZero(Game& game, ResNet& model, torch::optim::AdamW& optimizer, const AlphaZeroConfig& cfg)
        : game_(game), model_(model), optimizer_(optimizer), cfg_(cfg) {
        model_->to(cfg_.device);
    }

    bool save_checkpoint(const std::string& filepath = "") {
        namespace fs = std::filesystem;
        const fs::path data_dir(cfg_.data_dir);
        const fs::path model_dir = data_dir / "models";
        const fs::path checkpoint_dir = data_dir / "checkpoints";
        fs::create_directories(model_dir);
        fs::create_directories(checkpoint_dir);

        const fs::path model_path = model_dir / (cfg_.file_name + "_model_latest.pth");
        const fs::path checkpoint_path = filepath.empty()
            ? (checkpoint_dir / (cfg_.file_name + "_checkpoint_latest.ckpt"))
            : fs::path(filepath);

        torch::serialize::OutputArchive model_archive;
        model_->save(model_archive);
        model_archive.save_to(model_path.string());

        torch::serialize::OutputArchive checkpoint_archive;
        checkpoint_archive.write("model", model_archive);
        torch::serialize::OutputArchive optimizer_archive;
        optimizer_.save(optimizer_archive);
        checkpoint_archive.write("optimizer", optimizer_archive);
        checkpoint_archive.write("game_count", torch::tensor({int64_t(0)}));
        checkpoint_archive.write("total_samples", torch::tensor({int64_t(0)}));
        checkpoint_archive.save_to(checkpoint_path.string());
        return true;
    }

    bool load_checkpoint(const std::string& filepath = "") {
        namespace fs = std::filesystem;

        fs::path checkpoint_path;
        if (filepath.empty()) {
            const fs::path checkpoint_dir = fs::path(cfg_.data_dir) / "checkpoints";
            if (!fs::exists(checkpoint_dir)) {
                std::cout << "Checkpoint directory not found: " << checkpoint_dir.string() << "\n";
                return false;
            }
            std::filesystem::file_time_type best_time{};
            bool found = false;
            for (const auto& entry : fs::directory_iterator(checkpoint_dir)) {
                if (!entry.is_regular_file() || entry.path().extension() != ".ckpt") {
                    continue;
                }
                const auto t = entry.last_write_time();
                if (!found || t > best_time) {
                    found = true;
                    best_time = t;
                    checkpoint_path = entry.path();
                }
            }
            if (!found) {
                std::cout << "No checkpoint files found in: " << checkpoint_dir.string() << "\n";
                return false;
            }
            std::cout << "Auto-selected latest checkpoint: " << checkpoint_path.string() << "\n";
        } else {
            checkpoint_path = fs::path(filepath);
        }

        if (!fs::exists(checkpoint_path)) {
            std::cout << "Checkpoint file not found: " << checkpoint_path.string() << "\n";
            return false;
        }

        try {
            torch::serialize::InputArchive checkpoint_archive;
            checkpoint_archive.load_from(checkpoint_path.string());

            torch::serialize::InputArchive model_archive;
            checkpoint_archive.read("model", model_archive);
            model_->load(model_archive);

            try {
                torch::serialize::InputArchive optimizer_archive;
                checkpoint_archive.read("optimizer", optimizer_archive);
                optimizer_.load(optimizer_archive);
            } catch (...) {
            }

            std::cout << "Checkpoint loaded from " << checkpoint_path.string() << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cout << "Failed to load checkpoint: " << e.what() << "\n";
            return false;
        }
    }

private:
    Game& game_;
    ResNet& model_;
    torch::optim::AdamW& optimizer_;
    AlphaZeroConfig cfg_;
};

}  // namespace skyzero

#endif
