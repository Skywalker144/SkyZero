#ifndef SKYZERO_ALPHAZERO_PARALLEL_H
#define SKYZERO_ALPHAZERO_PARALLEL_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <functional>
#include <thread>
#include <unordered_map>

#include "alphazero.h"

namespace skyzero {

struct AlphaZeroParallelConfig {
    int num_workers = std::max(1u, std::thread::hardware_concurrency());
    int num_inference_servers = 1;
    int inference_batch_size = 64;
    int inference_batch_wait_us = 250;
    int leaf_batch_size = 8;
    int max_games_to_process_per_tick = 50;
    int idle_sleep_ms = 1;
};

template <typename Game>
class ParallelMCTS {
public:
    using InferenceFn = std::function<std::pair<std::vector<float>, std::array<float, 3>>(const std::vector<int8_t>&)>;
    using BatchInferenceFn = std::function<std::vector<std::pair<std::vector<float>, std::array<float, 3>>>(
        const std::vector<std::vector<int8_t>>&
    )>;

    ParallelMCTS(
        Game& game,
        const AlphaZeroConfig& config,
        int leaf_batch_size,
        InferenceFn infer_fn,
        BatchInferenceFn batch_infer_fn,
        uint64_t seed
    )
        : game_(game),
          cfg_(config),
          leaf_batch_size_(std::max(1, leaf_batch_size)),
          infer_fn_(std::move(infer_fn)),
          batch_infer_fn_(std::move(batch_infer_fn)),
          rng_(seed) {
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
            if (is_full_search) {
                nn_policy = apply_dirichlet_to_root(*root);
            } else {
                nn_policy = root->nn_policy;
            }
            nn_value_probs = root->nn_value_probs;
        }

        struct PendingLeaf {
            MCTSNode* leaf = nullptr;
            std::vector<int8_t> encoded;
            int transform_k = 0;
            bool transform_flip = false;
            std::vector<std::vector<MCTSNode*>> paths;
        };

        int sim = 0;
        while (sim < num_simulations) {
            const int batch_k = std::min(leaf_batch_size_, num_simulations - sim);
            std::vector<PendingLeaf> pending;
            pending.reserve(static_cast<size_t>(batch_k));
            std::unordered_map<MCTSNode*, size_t> pending_lookup;
            pending_lookup.reserve(static_cast<size_t>(batch_k));

            for (int i = 0; i < batch_k; ++i) {
                std::vector<MCTSNode*> path;
                path.reserve(64);

                MCTSNode* node = root.get();
                node->vloss += 1;
                path.push_back(node);

                while (node->is_expanded()) {
                    node = select(*node, is_full_search);
                    if (node == nullptr) {
                        remove_vloss_on_path(path);
                        throw std::runtime_error("MCTS select returned null");
                    }
                    node->vloss += 1;
                    path.push_back(node);
                }

                if (game_.is_terminal(node->state)) {
                    std::array<float, 3> value{0.0f, 1.0f, 0.0f};
                    const int result = game_.get_winner(node->state) * node->to_play;
                    if (result == 1) value = {1.0f, 0.0f, 0.0f};
                    else if (result == -1) value = {0.0f, 0.0f, 1.0f};
                    backpropagate_path_with_vloss(path, value);
                    ++sim;
                    continue;
                }

                auto it = pending_lookup.find(node);
                if (it == pending_lookup.end()) {
                    PendingLeaf pl;
                    pl.leaf = node;
                    pl.encoded = game_.encode_state(node->state, node->to_play);

                    const bool use_stochastic_transform = cfg_.enable_stochastic_transform_inference_for_child;
                    if (use_stochastic_transform) {
                        std::uniform_int_distribution<int> dist(0, 7);
                        const int transform_type = dist(rng_);
                        pl.transform_k = transform_type % 4;
                        pl.transform_flip = transform_type >= 4;
                        pl.encoded = transform_encoded_state(
                            pl.encoded,
                            game_.num_planes,
                            game_.board_size,
                            pl.transform_k,
                            pl.transform_flip
                        );
                    }

                    pl.paths.push_back(std::move(path));
                    pending_lookup.emplace(node, pending.size());
                    pending.push_back(std::move(pl));
                } else {
                    pending[it->second].paths.push_back(std::move(path));
                }
            }

            std::vector<std::pair<std::vector<float>, std::array<float, 3>>> infer_results;
            try {
                std::vector<std::vector<int8_t>> encoded_batch;
                encoded_batch.reserve(pending.size());
                for (const auto& p : pending) {
                    encoded_batch.push_back(p.encoded);
                }

                if (batch_infer_fn_) {
                    infer_results = batch_infer_fn_(encoded_batch);
                } else {
                    infer_results.reserve(encoded_batch.size());
                    for (const auto& e : encoded_batch) {
                        infer_results.push_back(infer_fn_(e));
                    }
                }
            } catch (...) {
                for (const auto& p : pending) {
                    for (const auto& path : p.paths) {
                        remove_vloss_on_path(path);
                    }
                }
                throw;
            }

            if (infer_results.size() != pending.size()) {
                for (const auto& p : pending) {
                    for (const auto& path : p.paths) {
                        remove_vloss_on_path(path);
                    }
                }
                throw std::runtime_error("batch_infer_fn returned unexpected batch size");
            }

            for (size_t i = 0; i < pending.size(); ++i) {
                auto logits = std::move(infer_results[i].first);
                const auto value = infer_results[i].second;

                if (pending[i].transform_k != 0 || pending[i].transform_flip) {
                    logits = undo_transform_flat(logits, game_.board_size, pending[i].transform_k, pending[i].transform_flip);
                }

                const auto legal = game_.get_is_legal_actions(pending[i].leaf->state, pending[i].leaf->to_play);
                for (size_t j = 0; j < logits.size(); ++j) {
                    if (!legal[j]) {
                        logits[j] = -std::numeric_limits<float>::infinity();
                    }
                }

                auto policy = softmax(logits);
                auto& leaf = *pending[i].leaf;
                leaf.nn_policy = policy;
                leaf.nn_value_probs = value;
                leaf.children.clear();
                for (int a = 0; a < static_cast<int>(policy.size()); ++a) {
                    const float p = policy[static_cast<size_t>(a)];
                    if (p <= 0.0f) continue;
                    auto child = std::unique_ptr<MCTSNode>(new MCTSNode{
                        game_.get_next_state(leaf.state, a, leaf.to_play),
                        -leaf.to_play,
                        p,
                        &leaf,
                        a
                    });
                    leaf.children.push_back(std::move(child));
                }

                for (const auto& path : pending[i].paths) {
                    backpropagate_path_with_vloss(path, value);
                    ++sim;
                }
            }
        }

        std::vector<float> mcts_policy(game_.board_size * game_.board_size, 0.0f);
        if (cfg_.enable_forced_playouts && root->is_expanded()) {
            MCTSNode* best_child = nullptr;
            for (auto& c : root->children) {
                if (!best_child || c->n > best_child->n) {
                    best_child = c.get();
                }
            }

            float c_puct = cfg_.c_puct;
            const float total_child_weight = static_cast<float>(std::max(0, root->n - 1));
            if (root->n > 0) {
                c_puct += cfg_.c_puct_log * std::log((total_child_weight + cfg_.c_puct_base) / cfg_.c_puct_base);
            }
            const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f);

            float q_best = 0.0f;
            if (best_child && best_child->n > 0) {
                const std::array<float, 3> bcq{
                    best_child->v[0] / best_child->n,
                    best_child->v[1] / best_child->n,
                    best_child->v[2] / best_child->n,
                };
                q_best = bcq[2] - bcq[0];
            }
            const float u_best = best_child ? (explore_scaling * best_child->prior / (1.0f + static_cast<float>(best_child->n))) : 0.0f;
            const float puct_best = q_best + u_best;

            for (auto& child_ptr : root->children) {
                auto& child = *child_ptr;
                if (&child == best_child) {
                    mcts_policy[child.action_taken] = static_cast<float>(child.n);
                    continue;
                }
                float q_child = 0.0f;
                if (child.n > 0) {
                    const std::array<float, 3> cq{
                        child.v[0] / child.n,
                        child.v[1] / child.n,
                        child.v[2] / child.n,
                    };
                    q_child = cq[2] - cq[0];
                }
                const float puct_gap = puct_best - q_child;
                float max_subtract = 0.0f;
                if (puct_gap > 0.0f) {
                    const float min_denominator = (explore_scaling * child.prior) / puct_gap;
                    max_subtract = (1.0f + child.n) - min_denominator;
                }
                const float new_n = static_cast<float>(child.n) - std::max(0.0f, max_subtract);
                mcts_policy[child.action_taken] = (new_n <= 1.0f) ? 0.0f : new_n;
            }
        } else {
            for (auto& c : root->children) {
                mcts_policy[c->action_taken] = static_cast<float>(c->n);
            }
        }

        const float sum_n = std::accumulate(mcts_policy.begin(), mcts_policy.end(), 0.0f);
        if (sum_n > 0.0f) {
            for (float& p : mcts_policy) p /= sum_n;
        } else if (!root->children.empty()) {
            mcts_policy[root->children[0]->action_taken] = 1.0f;
        }

        MCTSSearchOutput out;
        out.mcts_policy = mcts_policy;
        out.nn_policy = nn_policy;
        out.nn_value_probs = nn_value_probs;
        if (root->n > 0) {
            out.root_value = {root->v[0] / root->n, root->v[1] / root->n, root->v[2] / root->n};
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
        bool use_stochastic_transform
    ) {
        auto encoded = game_.encode_state(state, to_play);

        int k = 0;
        bool do_flip = false;
        if (use_stochastic_transform) {
            std::uniform_int_distribution<int> dist(0, 7);
            const int transform_type = dist(rng_);
            k = transform_type % 4;
            do_flip = transform_type >= 4;
            encoded = transform_encoded_state(encoded, game_.num_planes, game_.board_size, k, do_flip);
        }

        auto pair = infer_fn_(encoded);
        std::vector<float> logits = std::move(pair.first);
        if (use_stochastic_transform) {
            logits = undo_transform_flat(logits, game_.board_size, k, do_flip);
        }

        const auto legal = game_.get_is_legal_actions(state, to_play);
        for (size_t i = 0; i < logits.size(); ++i) {
            if (!legal[i]) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }

        return {softmax(logits), pair.second};
    }

    std::array<float, 3> expand(MCTSNode& node) {
        const auto pair = inference(node.state, node.to_play, cfg_.enable_stochastic_transform_inference_for_child);
        node.nn_policy = pair.first;
        node.nn_value_probs = pair.second;
        node.children.clear();
        for (int a = 0; a < static_cast<int>(node.nn_policy.size()); ++a) {
            const float p = node.nn_policy[a];
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
        return node.nn_value_probs;
    }

    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& node, bool enable_dirichlet) {
        auto pair = inference(node.state, node.to_play, cfg_.enable_stochastic_transform_inference_for_root);
        node.nn_policy = pair.first;
        node.nn_value_probs = pair.second;

        auto root_policy = node.nn_policy;
        if (enable_dirichlet) {
            int current_step = 0;
            for (int8_t v : node.state) {
                if (v != 0) ++current_step;
            }
            root_policy = add_shaped_dirichlet_noise(root_policy, cfg_.total_dirichlet_alpha, cfg_.dirichlet_epsilon, rng_);
            root_policy = root_temperature_transform(
                root_policy,
                current_step,
                cfg_.root_temperature_init,
                cfg_.root_temperature_final,
                game_.board_size
            );
        }

        node.children.clear();
        for (int a = 0; a < static_cast<int>(root_policy.size()); ++a) {
            const float p = root_policy[a];
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
        return {root_policy, node.nn_value_probs};
    }

    std::vector<float> apply_dirichlet_to_root(MCTSNode& node) {
        if (node.nn_policy.empty()) {
            return node.nn_policy;
        }
        int current_step = 0;
        for (int8_t v : node.state) {
            if (v != 0) ++current_step;
        }
        auto root_policy = add_shaped_dirichlet_noise(node.nn_policy, cfg_.total_dirichlet_alpha, cfg_.dirichlet_epsilon, rng_);
        root_policy = root_temperature_transform(
            root_policy,
            current_step,
            cfg_.root_temperature_init,
            cfg_.root_temperature_final,
            game_.board_size
        );

        std::vector<std::unique_ptr<MCTSNode>> new_children;
        new_children.reserve(node.children.size());
        for (int a = 0; a < static_cast<int>(root_policy.size()); ++a) {
            const float p = root_policy[a];
            if (p <= 0.0f) continue;

            MCTSNode* existing = nullptr;
            for (auto& c : node.children) {
                if (c && c->action_taken == a) {
                    existing = c.release();
                    break;
                }
            }
            if (existing) {
                existing->prior = p;
                existing->parent = &node;
                new_children.emplace_back(existing);
            } else {
                new_children.emplace_back(new MCTSNode{
                    game_.get_next_state(node.state, a, node.to_play),
                    -node.to_play,
                    p,
                    &node,
                    a
                });
            }
        }
        node.children = std::move(new_children);
        return root_policy;
    }

    MCTSNode* select(MCTSNode& node, bool is_full_search) {
        if (cfg_.enable_forced_playouts && is_full_search && node.parent == nullptr && node.n > 0) {
            MCTSNode* best_forced_child = nullptr;
            float best_prior = -1.0f;
            const int effective_parent_n = node.n + node.vloss;
            const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));
            const float sqrt_node_n = std::sqrt(total_child_weight);
            for (auto& child_ptr : node.children) {
                auto& child = *child_ptr;
                if (child.prior <= 0.0f) continue;
                const float target_visits = std::sqrt(cfg_.forced_playouts_k * child.prior) * sqrt_node_n;
                if (static_cast<float>(child.n + child.vloss) < target_visits && child.prior > best_prior) {
                    best_prior = child.prior;
                    best_forced_child = &child;
                }
            }
            if (best_forced_child) {
                return best_forced_child;
            }
        }

        float visited_policy_mass = 0.0f;
        for (auto& child_ptr : node.children) {
            if (child_ptr->n > 0 || child_ptr->vloss > 0) {
                visited_policy_mass += child_ptr->prior;
            }
        }

        const int effective_parent_n = node.n + node.vloss;
        const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));
        const float c_puct = cfg_.c_puct + cfg_.c_puct_log * std::log((total_child_weight + cfg_.c_puct_base) / cfg_.c_puct_base);
        const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f);

        std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
        if (node.n > 0) {
            parent_q = {node.v[0] / node.n, node.v[1] / node.n, node.v[2] / node.n};
        }

        float parent_utility = wdl_utility(parent_q);
        const float nn_utility = wdl_utility(node.nn_value_probs);
        const float avg_weight = std::min(1.0f, static_cast<float>(std::pow(visited_policy_mass, cfg_.fpu_pow)));
        parent_utility = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;
        const float fpu_reduction_max = (node.parent == nullptr) ? cfg_.root_fpu_reduction_max : cfg_.fpu_reduction_max;
        const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
        float fpu_value = parent_utility - reduction;
        fpu_value = fpu_value + ((-1.0f) - fpu_value) * cfg_.fpu_loss_prop;

        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        for (auto& child_ptr : node.children) {
            auto& child = *child_ptr;
            const int effective_child_n = child.n + child.vloss;
            float q = fpu_value;
            if (effective_child_n > 0) {
                const float utility_sum = (child.v[2] - child.v[0]) - static_cast<float>(child.vloss);
                q = utility_sum / static_cast<float>(effective_child_n);
            }
            const float u = explore_scaling * child.prior / (1.0f + static_cast<float>(effective_child_n));
            const float score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = &child;
            }
        }
        return best_child;
    }

    static void remove_vloss_on_path(const std::vector<MCTSNode*>& path) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if ((*it)->vloss > 0) {
                (*it)->vloss -= 1;
            }
        }
    }

    static void backpropagate_path_with_vloss(const std::vector<MCTSNode*>& path, std::array<float, 3> value) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->update(value);
            (*it)->vloss -= 1;
            value = flip_wdl(value);
        }
    }

    static void backpropagate(MCTSNode* node, std::array<float, 3> value) {
        while (node != nullptr) {
            node->update(value);
            value = flip_wdl(value);
            node = node->parent;
        }
    }

    Game& game_;
    const AlphaZeroConfig& cfg_;
    int leaf_batch_size_ = 1;
    InferenceFn infer_fn_;
    BatchInferenceFn batch_infer_fn_;
    std::mt19937 rng_;
};

template <typename Game>
class AlphaZeroParallel {
public:
    AlphaZeroParallel(
        Game& game,
        ResNet& model,
        torch::optim::Optimizer& optimizer,
        const AlphaZeroConfig& cfg,
        const AlphaZeroParallelConfig& pcfg = AlphaZeroParallelConfig{}
    )
        : game_(game),
          model_(model),
          optimizer_(optimizer),
          cfg_(cfg),
          pcfg_(pcfg),
          replay_buffer_(game_.board_size, cfg_.min_buffer_size, cfg_.linear_threshold, cfg_.replay_alpha, cfg_.max_buffer_size),
          rng_(std::random_device{}()) {
        model_->to(cfg_.device);
        init_inference_models();
    }

    struct SelfPlayResult {
        std::vector<TrainSample> samples;
        int winner = 0;
        int game_len = 0;
    };

    struct BatchLossStats {
        float total_loss = 0.0f;
        float policy_loss = 0.0f;
        float opponent_policy_loss = 0.0f;
        float soft_policy_loss = 0.0f;
        float soft_opponent_policy_loss = 0.0f;
        float value_loss = 0.0f;
        float full_search_ratio = 0.0f;
    };

    bool save_checkpoint(const std::string& filepath = "") {
        namespace fs = std::filesystem;

        const std::string timestamp = make_timestamp();
        const fs::path data_dir(cfg_.data_dir);
        const fs::path model_dir = data_dir / "models";
        const fs::path checkpoint_dir = data_dir / "checkpoints";
        fs::create_directories(model_dir);
        fs::create_directories(checkpoint_dir);

        const fs::path model_path = model_dir / (cfg_.file_name + "_model_" + timestamp + ".pth");
        const fs::path checkpoint_path = filepath.empty()
            ? (checkpoint_dir / (cfg_.file_name + "_checkpoint_" + timestamp + ".ckpt"))
            : fs::path(filepath);

        torch::serialize::OutputArchive model_archive;
        model_->save(model_archive);
        model_archive.save_to(model_path.string());

        torch::serialize::OutputArchive checkpoint_archive;
        torch::serialize::OutputArchive checkpoint_model_archive;
        torch::serialize::OutputArchive optimizer_archive;
        model_->save(checkpoint_model_archive);
        optimizer_.save(optimizer_archive);
        checkpoint_archive.write("model", checkpoint_model_archive);
        checkpoint_archive.write("optimizer", optimizer_archive);
        checkpoint_archive.write("game_count", torch::tensor({static_cast<int64_t>(game_count_)}));
        checkpoint_archive.write("total_samples", torch::tensor({static_cast<int64_t>(total_samples_)}));

        checkpoint_archive.write("loss_total", vec_to_1d_tensor(total_loss_history_, torch::kFloat32));
        checkpoint_archive.write("loss_policy", vec_to_1d_tensor(policy_loss_history_, torch::kFloat32));
        checkpoint_archive.write("loss_opp_policy", vec_to_1d_tensor(opponent_policy_loss_history_, torch::kFloat32));
        checkpoint_archive.write("loss_soft_policy", vec_to_1d_tensor(soft_policy_loss_history_, torch::kFloat32));
        checkpoint_archive.write("loss_soft_opp_policy", vec_to_1d_tensor(soft_opponent_policy_loss_history_, torch::kFloat32));
        checkpoint_archive.write("loss_value", vec_to_1d_tensor(value_loss_history_, torch::kFloat32));
        checkpoint_archive.write("avg_game_len_history", vec_to_1d_tensor(avg_game_len_history_, torch::kFloat32));

        std::vector<float> winrate_flat;
        winrate_flat.reserve(winrate_history_.size() * 4);
        for (const auto& w : winrate_history_) {
            winrate_flat.push_back(static_cast<float>(w[0]));
            winrate_flat.push_back(w[1]);
            winrate_flat.push_back(w[2]);
            winrate_flat.push_back(w[3]);
        }
        checkpoint_archive.write(
            "winrate_history",
            vec_to_2d_tensor(winrate_flat, static_cast<int64_t>(winrate_history_.size()), 4, torch::kFloat32)
        );

        checkpoint_archive.write("recent_game_lengths", ints_to_1d_tensor(recent_game_lengths_));
        checkpoint_archive.write("recent_sample_lengths", ints_to_1d_tensor(recent_sample_lengths_));
        checkpoint_archive.write("black_win_counts", ints_to_1d_tensor(black_win_counts_));
        checkpoint_archive.write("white_win_counts", ints_to_1d_tensor(white_win_counts_));

        const auto rb = replay_buffer_.get_state();
        torch::serialize::OutputArchive rb_archive;
        rb_archive.write("board_size", torch::tensor({static_cast<int64_t>(rb.board_size)}));
        rb_archive.write("action_size", torch::tensor({static_cast<int64_t>(rb.action_size)}));
        rb_archive.write("min_buffer_size", torch::tensor({static_cast<int64_t>(rb.min_buffer_size)}));
        rb_archive.write("linear_threshold", torch::tensor({static_cast<int64_t>(rb.linear_threshold)}));
        rb_archive.write("alpha", torch::tensor({rb.alpha}));
        rb_archive.write("max_buffer_size", torch::tensor({static_cast<int64_t>(rb.max_buffer_size)}));
        rb_archive.write("ptr", torch::tensor({static_cast<int64_t>(rb.ptr)}));
        rb_archive.write("size", torch::tensor({static_cast<int64_t>(rb.size)}));
        rb_archive.write("total_samples_added", torch::tensor({static_cast<int64_t>(rb.total_samples_added)}));
        rb_archive.write("games_count", torch::tensor({static_cast<int64_t>(rb.games_count)}));

        const int64_t rows = static_cast<int64_t>(rb.size);
        const int64_t board_cells = static_cast<int64_t>(rb.board_size) * static_cast<int64_t>(rb.board_size);
        const int64_t action_cells = static_cast<int64_t>(rb.action_size);
        rb_archive.write("states", vec_to_2d_tensor(rb.states, rows, board_cells, torch::kInt8));
        rb_archive.write("to_play", vec_to_1d_tensor(rb.to_play, torch::kInt8));
        rb_archive.write("policy_targets", vec_to_2d_tensor(rb.policy_targets, rows, action_cells, torch::kFloat32));
        rb_archive.write("opponent_policy_targets", vec_to_2d_tensor(rb.opponent_policy_targets, rows, action_cells, torch::kFloat32));
        rb_archive.write("value_targets", vec_to_2d_tensor(rb.value_targets, rows, 3, torch::kFloat32));
        rb_archive.write("sample_weights", vec_to_1d_tensor(rb.sample_weights, torch::kFloat32));
        rb_archive.write("is_full_search", vec_to_1d_tensor(rb.is_full_search, torch::kUInt8));
        checkpoint_archive.write("replay_buffer", rb_archive);

        checkpoint_archive.save_to(checkpoint_path.string());

        const auto model_bytes = fs::file_size(model_path);
        const auto ckpt_bytes = fs::file_size(checkpoint_path);
        std::cout << "Model saved to " << model_path.string() << " (" << human_size(model_bytes) << ")\n";
        std::cout << "Checkpoint saved to " << checkpoint_path.string() << " (" << human_size(ckpt_bytes) << ")\n";
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

            torch::serialize::InputArchive optimizer_archive;
            checkpoint_archive.read("optimizer", optimizer_archive);
            optimizer_.load(optimizer_archive);

            torch::Tensor t;
            checkpoint_archive.read("game_count", t);
            game_count_ = static_cast<int>(t.template item<int64_t>());
            checkpoint_archive.read("total_samples", t);
            total_samples_ = static_cast<int64_t>(t.template item<int64_t>());

            total_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_total"));
            policy_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_policy"));
            opponent_policy_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_opp_policy"));
            soft_policy_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_soft_policy"));
            soft_opponent_policy_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_soft_opp_policy"));
            value_loss_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "loss_value"));
            avg_game_len_history_ = tensor_to_vec<float>(must_read_tensor(checkpoint_archive, "avg_game_len_history"));

            winrate_history_.clear();
            auto win_t = must_read_tensor(checkpoint_archive, "winrate_history").to(torch::kCPU).contiguous();
            if (win_t.numel() > 0) {
                const auto* p = win_t.template data_ptr<float>();
                const int64_t rows = win_t.size(0);
                for (int64_t i = 0; i < rows; ++i) {
                    winrate_history_.push_back({
                        p[i * 4 + 0],
                        p[i * 4 + 1],
                        p[i * 4 + 2],
                        p[i * 4 + 3]
                    });
                }
            }

            recent_game_lengths_ = tensor_to_ints(must_read_tensor(checkpoint_archive, "recent_game_lengths"));
            recent_sample_lengths_ = tensor_to_ints(must_read_tensor(checkpoint_archive, "recent_sample_lengths"));
            black_win_counts_ = tensor_to_ints(must_read_tensor(checkpoint_archive, "black_win_counts"));
            white_win_counts_ = tensor_to_ints(must_read_tensor(checkpoint_archive, "white_win_counts"));

            torch::serialize::InputArchive rb_archive;
            checkpoint_archive.read("replay_buffer", rb_archive);

            ReplayBufferState rb;
            rb.board_size = static_cast<int>(must_read_tensor(rb_archive, "board_size").template item<int64_t>());
            rb.action_size = static_cast<int>(must_read_tensor(rb_archive, "action_size").template item<int64_t>());
            rb.min_buffer_size = static_cast<int>(must_read_tensor(rb_archive, "min_buffer_size").template item<int64_t>());
            rb.linear_threshold = static_cast<int>(must_read_tensor(rb_archive, "linear_threshold").template item<int64_t>());
            rb.alpha = must_read_tensor(rb_archive, "alpha").template item<float>();
            rb.max_buffer_size = static_cast<int>(must_read_tensor(rb_archive, "max_buffer_size").template item<int64_t>());
            rb.ptr = static_cast<int>(must_read_tensor(rb_archive, "ptr").template item<int64_t>());
            rb.size = static_cast<int>(must_read_tensor(rb_archive, "size").template item<int64_t>());
            rb.total_samples_added = static_cast<int>(must_read_tensor(rb_archive, "total_samples_added").template item<int64_t>());
            rb.games_count = static_cast<int>(must_read_tensor(rb_archive, "games_count").template item<int64_t>());

            rb.states = tensor_to_vec<int8_t>(must_read_tensor(rb_archive, "states"));
            rb.to_play = tensor_to_vec<int8_t>(must_read_tensor(rb_archive, "to_play"));
            rb.policy_targets = tensor_to_vec<float>(must_read_tensor(rb_archive, "policy_targets"));
            rb.opponent_policy_targets = tensor_to_vec<float>(must_read_tensor(rb_archive, "opponent_policy_targets"));
            rb.value_targets = tensor_to_vec<float>(must_read_tensor(rb_archive, "value_targets"));
            rb.sample_weights = tensor_to_vec<float>(must_read_tensor(rb_archive, "sample_weights"));
            rb.is_full_search = tensor_to_vec<uint8_t>(must_read_tensor(rb_archive, "is_full_search"));

            replay_buffer_.load_state(rb);

            // Override LR from current config
            for (auto& pg : optimizer_.param_groups()) {
                static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(cfg_.lr);
            }
            std::cout << "Optimizer LR overridden to " << cfg_.lr << "\n";

            // Override replay buffer parameters from current config
            replay_buffer_.override_params(
                cfg_.min_buffer_size,
                cfg_.linear_threshold,
                cfg_.replay_alpha,
                cfg_.max_buffer_size
            );
            std::cout << "Replay buffer parameters overridden from current config\n";

            sync_all_inference_models();

            std::cout << "Checkpoint loaded from " << checkpoint_path.string() << "\n";
            std::cout << "Replay buffer loaded (" << replay_buffer_.size() << " samples)\n";
            return true;
        } catch (const std::exception& e) {
            std::cout << "Failed to load checkpoint: " << e.what() << "\n";
            return false;
        }
    }

    void learn() {
        int train_game_count = game_count_;
        bool init_flag = true;
        auto last_save_time = std::chrono::steady_clock::now();
        if (session_start_time_.time_since_epoch().count() == 0) {
            session_start_time_ = std::chrono::steady_clock::now();
        }

        start_inference_server();
        start_selfplay_workers();

        try {
            while (!stop_requested) {
                int games_processed = 0;
                while (games_processed < pcfg_.max_games_to_process_per_tick) {
                    SelfPlayResult sp;
                    if (!try_pop_selfplay_result(sp)) {
                        break;
                    }
                    replay_buffer_.add_game(sp.samples);
                    total_samples_ += static_cast<int64_t>(sp.samples.size());
                    recent_game_lengths_.push_back(sp.game_len);
                    recent_sample_lengths_.push_back(static_cast<int>(sp.samples.size()));
                    black_win_counts_.push_back(sp.winner == 1 ? 1 : 0);
                    white_win_counts_.push_back(sp.winner == -1 ? 1 : 0);
                    trim_tail(recent_game_lengths_, 300);
                    trim_tail(recent_sample_lengths_, 300);
                    trim_tail(black_win_counts_, 300);
                    trim_tail(white_win_counts_, 300);
                    game_count_ += 1;
                    games_processed += 1;
                }

                if (game_count_ % 10 == 0 && game_count_ != last_stats_game_count_) {
                    print_selfplay_stats();
                    last_stats_game_count_ = game_count_;
                }

                if (replay_buffer_.size() < cfg_.min_buffer_size) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, pcfg_.idle_sleep_ms)));
                    continue;
                } else if (init_flag) {
                    train_game_count = game_count_;
                    init_flag = false;
                    std::cout << "\n--- Buffer Warmup Complete. Training Started. ---\n";
                }

                if (cfg_.savetime_interval > 0) {
                    const auto now = std::chrono::steady_clock::now();
                    const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - last_save_time).count();
                    if (elapsed_sec >= cfg_.savetime_interval) {
                        save_checkpoint();
                        last_save_time = now;
                    }
                }

                if (game_count_ < train_game_count) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, pcfg_.idle_sleep_ms)));
                    continue;
                }

                std::cout << "\n--- Training Session ---\n";
                std::vector<BatchLossStats> batch_losses;
                batch_losses.reserve(cfg_.train_steps_per_generation);
                for (int step = 0; step < cfg_.train_steps_per_generation; ++step) {
                    const auto batch = replay_buffer_.sample(cfg_.batch_size, rng_);
                    if (batch.empty()) {
                        continue;
                    }
                    batch_losses.push_back(train_batch(batch));
                }

                if (!batch_losses.empty()) {
                    BatchLossStats mean;
                    for (const auto& s : batch_losses) {
                        mean.total_loss += s.total_loss;
                        mean.policy_loss += s.policy_loss;
                        mean.opponent_policy_loss += s.opponent_policy_loss;
                        mean.soft_policy_loss += s.soft_policy_loss;
                        mean.soft_opponent_policy_loss += s.soft_opponent_policy_loss;
                        mean.value_loss += s.value_loss;
                        mean.full_search_ratio += s.full_search_ratio;
                    }
                    const float inv = 1.0f / static_cast<float>(batch_losses.size());
                    mean.total_loss *= inv;
                    mean.policy_loss *= inv;
                    mean.opponent_policy_loss *= inv;
                    mean.soft_policy_loss *= inv;
                    mean.soft_opponent_policy_loss *= inv;
                    mean.value_loss *= inv;
                    mean.full_search_ratio *= inv;

                    total_loss_history_.push_back(mean.total_loss);
                    policy_loss_history_.push_back(mean.policy_loss);
                    opponent_policy_loss_history_.push_back(mean.opponent_policy_loss);
                    soft_policy_loss_history_.push_back(mean.soft_policy_loss);
                    soft_opponent_policy_loss_history_.push_back(mean.soft_opponent_policy_loss);
                    value_loss_history_.push_back(mean.value_loss);

                    std::cout << "  [Training] Full Search Ratio: " << std::fixed << std::setprecision(2)
                              << mean.full_search_ratio << "\n";
                    std::cout << "  [Training] Loss: " << std::fixed << std::setprecision(2)
                              << mean.total_loss << " | Policy Loss: " << mean.policy_loss
                              << " | Value Loss: " << mean.value_loss << "\n";
                }

                sync_all_inference_models();

                float avg_sample_len = 1.0f;
                if (!recent_sample_lengths_.empty()) {
                    const float sum_samples = static_cast<float>(
                        std::accumulate(recent_sample_lengths_.begin(), recent_sample_lengths_.end(), 0)
                    );
                    avg_sample_len = std::max(1.0f, sum_samples / static_cast<float>(recent_sample_lengths_.size()));
                }

                const float replay_ratio = std::max(1e-6f, cfg_.target_replay_ratio);
                int num_next = static_cast<int>(
                    static_cast<float>(cfg_.batch_size * cfg_.train_steps_per_generation) / avg_sample_len / replay_ratio
                );
                num_next = std::max(1, num_next);
                train_game_count = game_count_ + num_next;
                std::cout << "  Next Train after " << num_next << " games\n";
            }

            stop_workers_and_server();
            if (cfg_.save_on_exit) {
                save_checkpoint();
            }
        } catch (...) {
            stop_workers_and_server();
            if (cfg_.save_on_exit) {
                try {
                    save_checkpoint();
                } catch (...) {
                }
            }
            throw;
        }
    }

private:
    struct InferenceRequest {
        std::vector<int8_t> encoded;
        std::promise<std::pair<std::vector<float>, std::array<float, 3>>> promise;
    };

    static void sync_models(ResNet& src, ResNet& dst) {
        torch::NoGradGuard no_grad;

        auto src_params = src->named_parameters(true);
        auto dst_params = dst->named_parameters(true);
        for (const auto& item : src_params) {
            auto* d = dst_params.find(item.key());
            if (d == nullptr) {
                throw std::runtime_error(std::string("sync_models: missing parameter ") + item.key());
            }
            d->copy_(item.value());
        }

        auto src_buffers = src->named_buffers(true);
        auto dst_buffers = dst->named_buffers(true);
        for (const auto& item : src_buffers) {
            auto* d = dst_buffers.find(item.key());
            if (d == nullptr) {
                throw std::runtime_error(std::string("sync_models: missing buffer ") + item.key());
            }
            d->copy_(item.value());
        }
    }

    void init_inference_models() {
        const int num_servers = std::max(1, pcfg_.num_inference_servers);
        inference_models_.clear();
        inference_models_.reserve(static_cast<size_t>(num_servers));
        inference_model_mutexes_.clear();
        inference_model_mutexes_.reserve(static_cast<size_t>(num_servers));

        for (int i = 0; i < num_servers; ++i) {
            inference_models_.push_back(ResNet(game_.board_size, game_.num_planes, cfg_.num_blocks, cfg_.num_channels));
            inference_models_.back()->to(cfg_.device);
            if (cfg_.device.is_cuda()) {
                inference_models_.back()->to(torch::kHalf);
            }
            inference_model_mutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
        }
        sync_all_inference_models();
    }

    void sync_all_inference_models() {
        for (size_t i = 0; i < inference_models_.size(); ++i) {
            std::lock_guard<std::mutex> lk(*inference_model_mutexes_[i]);
            sync_models(model_, inference_models_[i]);
            inference_models_[i]->eval();
            if (cfg_.device.is_cuda()) {
                inference_models_[i]->to(torch::kHalf);
            }
        }
    }

    std::pair<std::vector<float>, std::array<float, 3>> request_inference(const std::vector<int8_t>& encoded) {
        auto req = std::unique_ptr<InferenceRequest>(new InferenceRequest{});
        req->encoded = encoded;
        auto fut = req->promise.get_future();
        {
            std::lock_guard<std::mutex> lk(inference_mutex_);
            inference_queue_.push_back(std::move(req));
        }
        inference_cv_.notify_one();
        return fut.get();
    }

    std::vector<std::pair<std::vector<float>, std::array<float, 3>>>
    request_batch_inference(const std::vector<std::vector<int8_t>>& encoded_batch) {
        std::vector<std::future<std::pair<std::vector<float>, std::array<float, 3>>>> futures;
        futures.reserve(encoded_batch.size());

        {
            std::lock_guard<std::mutex> lk(inference_mutex_);
            for (const auto& encoded : encoded_batch) {
                auto req = std::unique_ptr<InferenceRequest>(new InferenceRequest{});
                req->encoded = encoded;
                futures.push_back(req->promise.get_future());
                inference_queue_.push_back(std::move(req));
            }
        }
        inference_cv_.notify_one();

        std::vector<std::pair<std::vector<float>, std::array<float, 3>>> out;
        out.reserve(futures.size());
        for (auto& fut : futures) {
            out.push_back(fut.get());
        }
        return out;
    }

    void start_inference_server() {
        stop_inference_.store(false);
        const int num_servers = std::max(1, pcfg_.num_inference_servers);
        inference_threads_.clear();
        inference_threads_.reserve(static_cast<size_t>(num_servers));
        for (int i = 0; i < num_servers; ++i) {
            inference_threads_.emplace_back([this, i]() { inference_server_loop(i); });
        }
    }

    void inference_server_loop(int server_idx) {
        const int c = game_.num_planes;
        const int board = game_.board_size;
        const int area = board * board;
        const int max_batch = std::max(1, pcfg_.inference_batch_size);

        while (true) {
            std::vector<std::unique_ptr<InferenceRequest>> batch;
            batch.reserve(max_batch);

            {
                std::unique_lock<std::mutex> lk(inference_mutex_);
                inference_cv_.wait(lk, [&]() {
                    return stop_inference_.load() || !inference_queue_.empty();
                });
                if (stop_inference_.load() && inference_queue_.empty()) {
                    break;
                }

                batch.push_back(std::move(inference_queue_.front()));
                inference_queue_.pop_front();

                if (pcfg_.inference_batch_wait_us > 0) {
                    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(pcfg_.inference_batch_wait_us);
                    while (batch.size() < static_cast<size_t>(max_batch)) {
                        if (inference_queue_.empty()) {
                            if (inference_cv_.wait_until(lk, until) == std::cv_status::timeout) {
                                break;
                            }
                            if (inference_queue_.empty()) {
                                break;
                            }
                        }
                        batch.push_back(std::move(inference_queue_.front()));
                        inference_queue_.pop_front();
                    }
                } else {
                    while (!inference_queue_.empty() && batch.size() < static_cast<size_t>(max_batch)) {
                        batch.push_back(std::move(inference_queue_.front()));
                        inference_queue_.pop_front();
                    }
                }
            }

            try {
                const int bsz = static_cast<int>(batch.size());
                std::vector<float> input_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
                for (int i = 0; i < bsz; ++i) {
                    const auto& enc = batch[i]->encoded;
                    if (enc.size() != static_cast<size_t>(c * area)) {
                        throw std::runtime_error("inference request encoded size mismatch");
                    }
                    const size_t base = static_cast<size_t>(i) * c * area;
                    for (int j = 0; j < c * area; ++j) {
                        input_buf[base + static_cast<size_t>(j)] = static_cast<float>(enc[static_cast<size_t>(j)]);
                    }
                }

                auto input = torch::from_blob(input_buf.data(), {bsz, c, board, board}, torch::kFloat32).to(cfg_.device);
                if (cfg_.device.is_cuda()) {
                    input = input.to(torch::kHalf);
                }
                torch::NoGradGuard no_grad;

                NetworkOutput out;
                {
                    std::lock_guard<std::mutex> mlk(*inference_model_mutexes_[static_cast<size_t>(server_idx)]);
                    out = inference_models_[static_cast<size_t>(server_idx)]->forward(input);
                }

                auto policy = out.policy_logits.reshape({bsz, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
                auto value = torch::softmax(out.value_logits.to(torch::kFloat32), 1).to(torch::kCPU).contiguous();
                const float* pp = policy.template data_ptr<float>();
                const float* vp = value.template data_ptr<float>();

                for (int i = 0; i < bsz; ++i) {
                    std::vector<float> logits(static_cast<size_t>(area), 0.0f);
                    std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area, static_cast<size_t>(area) * sizeof(float));
                    const size_t vi = static_cast<size_t>(i) * 3;
                    std::array<float, 3> v{vp[vi], vp[vi + 1], vp[vi + 2]};
                    batch[i]->promise.set_value({std::move(logits), v});
                }
            } catch (...) {
                for (auto& req : batch) {
                    try {
                        req->promise.set_exception(std::current_exception());
                    } catch (...) {
                    }
                }
            }
        }

        std::deque<std::unique_ptr<InferenceRequest>> pending;
        {
            std::lock_guard<std::mutex> lk(inference_mutex_);
            pending.swap(inference_queue_);
        }
        for (auto& req : pending) {
            try {
                req->promise.set_exception(std::make_exception_ptr(std::runtime_error("inference server stopped")));
            } catch (...) {
            }
        }
    }

    SelfPlayResult selfplay_once(uint64_t seed) {
        struct MemoryStep {
            std::vector<int8_t> state;
            int to_play = 1;
            std::vector<float> mcts_policy;
            std::vector<float> nn_policy;
            std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
            std::array<float, 3> root_value{0.0f, 0.0f, 0.0f};
            uint8_t is_full_search = 1;
            std::vector<float> next_mcts_policy;
            float sample_weight = 1.0f;
        };

        std::mt19937 worker_rng(seed);
        auto infer_fn = [this](const std::vector<int8_t>& encoded) {
            return request_inference(encoded);
        };
        auto batch_infer_fn = [this](const std::vector<std::vector<int8_t>>& encoded_batch) {
            return request_batch_inference(encoded_batch);
        };
        ParallelMCTS<Game> mcts(
            game_,
            cfg_,
            pcfg_.leaf_batch_size,
            infer_fn,
            batch_infer_fn,
            worker_rng()
        );

        std::vector<MemoryStep> memory;
        int to_play = 1;
        auto state = game_.get_initial_state();
        bool in_soft_resign = false;
        std::vector<float> historical_root_value;
        int last_action = -1;
        int last_player = 0;
        std::unique_ptr<MCTSNode> root(new MCTSNode{state, to_play});

        while (!game_.is_terminal(state, last_action, last_player)) {
            int num_simulations = cfg_.fast_search_num_simulations;
            if (!in_soft_resign) {
                std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                num_simulations = (uni(worker_rng) < cfg_.full_search_prob) ? cfg_.full_search_num_simulations : cfg_.fast_search_num_simulations;
            }

            const auto sr = mcts.search(state, to_play, num_simulations, root);
            const float root_value_scalar = sr.root_value[0] - sr.root_value[2];
            historical_root_value.push_back(root_value_scalar);

            const int n = static_cast<int>(historical_root_value.size());
            float absmin_root_value = std::numeric_limits<float>::infinity();
            const int from = std::max(0, n - cfg_.soft_resign_step_threshold);
            for (int i = from; i < n; ++i) {
                absmin_root_value = std::min(absmin_root_value, std::fabs(historical_root_value[i]));
            }
            if (!in_soft_resign) {
                std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                if (absmin_root_value >= cfg_.soft_resign_threshold && uni(worker_rng) < cfg_.soft_resign_prob) {
                    in_soft_resign = true;
                }
            }

            if (!memory.empty()) {
                memory.back().next_mcts_policy = sr.mcts_policy;
            }

            MemoryStep ms;
            ms.state = state;
            ms.to_play = to_play;
            ms.mcts_policy = sr.mcts_policy;
            ms.nn_policy = sr.nn_policy;
            ms.nn_value_probs = sr.nn_value_probs;
            ms.root_value = sr.root_value;
            ms.is_full_search = (num_simulations == cfg_.full_search_num_simulations) ? 1 : 0;
            ms.sample_weight = in_soft_resign ? cfg_.soft_resign_sample_weight : 1.0f;
            memory.push_back(ms);

            const int current_step = static_cast<int>(memory.size());
            const float t = cfg_.move_temperature_final + (cfg_.move_temperature_init - cfg_.move_temperature_final) *
                std::pow(0.5f, static_cast<float>(current_step) / game_.board_size);
            const auto move_probs = temperature_transform(sr.mcts_policy, t);

            std::discrete_distribution<int> action_dist(move_probs.begin(), move_probs.end());
            const int action = action_dist(worker_rng);

            last_action = action;
            last_player = to_play;
            state = game_.get_next_state(state, action, to_play);
            to_play = -to_play;

            std::unique_ptr<MCTSNode> next_root;
            if (root) {
                for (auto& child : root->children) {
                    if (child && child->action_taken == action) {
                        next_root = std::move(child);
                        break;
                    }
                }
            }
            if (next_root) {
                next_root->parent = nullptr;
                root = std::move(next_root);
            } else {
                root.reset(new MCTSNode{state, to_play});
            }
        }

        const int winner = game_.get_winner(state, last_action, last_player);
        std::vector<PolicySurpriseSample> return_memory;
        return_memory.reserve(memory.size());
        for (size_t i = 0; i < memory.size(); ++i) {
            const auto& s = memory[i];
            const int result = winner * s.to_play;

            std::array<float, 3> outcome{0.0f, 1.0f, 0.0f};
            if (result == 1) outcome = {1.0f, 0.0f, 0.0f};
            else if (result == -1) outcome = {0.0f, 0.0f, 1.0f};

            PolicySurpriseSample ps;
            ps.state = s.state;
            ps.to_play = static_cast<int8_t>(s.to_play);
            ps.policy_target = s.mcts_policy;
            ps.opponent_policy_target = s.next_mcts_policy.empty() ? std::vector<float>(s.mcts_policy.size(), 0.0f) : s.next_mcts_policy;
            ps.outcome = outcome;
            ps.nn_policy = s.nn_policy;
            ps.nn_value_probs = s.nn_value_probs;
            ps.root_value = s.root_value;
            ps.is_full_search = s.is_full_search;
            ps.sample_weight = s.sample_weight;
            return_memory.push_back(ps);
        }

        if (!return_memory.empty()) {
            const float now_factor = 1.0f / (1.0f + game_.board_size * game_.board_size * cfg_.value_target_mix_now_factor_constant);
            return_memory.back().value_target = return_memory.back().outcome;
            for (int i = static_cast<int>(return_memory.size()) - 2; i >= 0; --i) {
                const auto next_target = flip_wdl(return_memory[i + 1].value_target);
                return_memory[i].value_target = {
                    (1.0f - now_factor) * next_target[0] + now_factor * return_memory[i].root_value[0],
                    (1.0f - now_factor) * next_target[1] + now_factor * return_memory[i].root_value[1],
                    (1.0f - now_factor) * next_target[2] + now_factor * return_memory[i].root_value[2],
                };
            }
        }

        const auto weights = compute_policy_surprise_weights(return_memory, cfg_.policy_surprise_data_weight, cfg_.value_surprise_data_weight);
        auto weighted_memory = apply_surprise_weighting_to_game(return_memory, weights, worker_rng);

        SelfPlayResult out;
        out.samples = std::move(weighted_memory);
        out.winner = winner;
        out.game_len = static_cast<int>(memory.size());
        return out;
    }

    void start_selfplay_workers() {
        stop_workers_.store(false);
        const int num_workers = std::max(1, pcfg_.num_workers);
        selfplay_threads_.clear();
        selfplay_threads_.reserve(num_workers);

        const uint64_t base_seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        for (int i = 0; i < num_workers; ++i) {
            selfplay_threads_.emplace_back([this, i, base_seed]() {
                uint64_t local_seed = base_seed + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL;
                while (!stop_workers_.load()) {
                    try {
                        auto sp = selfplay_once(local_seed);
                        local_seed += 1;
                        {
                            std::lock_guard<std::mutex> lk(result_mutex_);
                            result_queue_.push_back(std::move(sp));
                        }
                        result_cv_.notify_one();
                    } catch (const std::exception& e) {
                        if (!stop_workers_.load()) {
                            std::cerr << "[SelfPlay Worker " << i << "] " << e.what() << "\n";
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                    }
                }
            });
        }
    }

    bool try_pop_selfplay_result(SelfPlayResult& out) {
        std::lock_guard<std::mutex> lk(result_mutex_);
        if (result_queue_.empty()) {
            return false;
        }
        out = std::move(result_queue_.front());
        result_queue_.pop_front();
        return true;
    }

    void stop_workers_and_server() {
        stop_workers_.store(true);
        for (auto& t : selfplay_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        selfplay_threads_.clear();

        stop_inference_.store(true);
        inference_cv_.notify_all();
        for (auto& t : inference_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        inference_threads_.clear();
    }

    template <typename T>
    static torch::Tensor vec_to_1d_tensor(const std::vector<T>& v, torch::ScalarType dtype) {
        auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        if (v.empty()) {
            return torch::empty({0}, opts);
        }
        return torch::from_blob(const_cast<T*>(v.data()), {static_cast<int64_t>(v.size())}, opts).clone();
    }

    template <typename T>
    static torch::Tensor vec_to_2d_tensor(const std::vector<T>& v, int64_t rows, int64_t cols, torch::ScalarType dtype) {
        auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        if (rows <= 0 || cols <= 0 || v.empty()) {
            return torch::empty({std::max<int64_t>(rows, 0), std::max<int64_t>(cols, 0)}, opts);
        }
        return torch::from_blob(const_cast<T*>(v.data()), {rows, cols}, opts).clone();
    }

    static torch::Tensor ints_to_1d_tensor(const std::vector<int>& v) {
        std::vector<int64_t> out(v.size(), 0);
        for (size_t i = 0; i < v.size(); ++i) {
            out[i] = static_cast<int64_t>(v[i]);
        }
        return vec_to_1d_tensor(out, torch::kInt64);
    }

    static std::vector<int> tensor_to_ints(const torch::Tensor& t) {
        auto cpu = t.to(torch::kCPU).to(torch::kInt64).contiguous();
        std::vector<int> out(static_cast<size_t>(cpu.numel()), 0);
        const auto* p = cpu.template data_ptr<int64_t>();
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<int>(p[i]);
        }
        return out;
    }

    template <typename T>
    static std::vector<T> tensor_to_vec(const torch::Tensor& t) {
        auto cpu = t.to(torch::kCPU).contiguous();
        std::vector<T> out(static_cast<size_t>(cpu.numel()));
        if (!out.empty()) {
            std::memcpy(out.data(), cpu.template data_ptr<T>(), out.size() * sizeof(T));
        }
        return out;
    }

    static torch::Tensor must_read_tensor(torch::serialize::InputArchive& ar, const std::string& key) {
        torch::Tensor t;
        ar.read(key, t);
        return t;
    }

    static void trim_tail(std::vector<int>& v, size_t max_size) {
        if (v.size() > max_size) {
            v.erase(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() - max_size));
        }
    }

    void print_selfplay_stats() {
        float avg_game_len = 0.0f;
        if (!recent_game_lengths_.empty()) {
            const float sum_len = static_cast<float>(
                std::accumulate(recent_game_lengths_.begin(), recent_game_lengths_.end(), 0)
            );
            avg_game_len = sum_len / static_cast<float>(recent_game_lengths_.size());
        }

        float b_rate = 0.0f;
        float w_rate = 0.0f;
        float d_rate = 1.0f;
        const int total_recent = static_cast<int>(black_win_counts_.size());
        if (total_recent > 0) {
            const int b_sum = std::accumulate(black_win_counts_.begin(), black_win_counts_.end(), 0);
            const int w_sum = std::accumulate(white_win_counts_.begin(), white_win_counts_.end(), 0);
            b_rate = static_cast<float>(b_sum) / static_cast<float>(total_recent);
            w_rate = static_cast<float>(w_sum) / static_cast<float>(total_recent);
            d_rate = 1.0f - b_rate - w_rate;
        }

        winrate_history_.push_back({static_cast<float>(game_count_), b_rate, w_rate, d_rate});
        avg_game_len_history_.push_back(avg_game_len);

        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - session_start_time_
        ).count();
        const double sps = (elapsed > 0.0) ? (static_cast<double>(total_samples_) / elapsed) : 0.0;

        std::cout << "Game: " << game_count_
                  << " | Sps: " << std::fixed << std::setprecision(1) << sps
                  << " | BufferSize: " << replay_buffer_.size()
                  << " | WindowSize: " << replay_buffer_.get_window_size()
                  << " | AvgGameLen: " << std::fixed << std::setprecision(1) << avg_game_len
                  << " | BWD: " << std::fixed << std::setprecision(2) << b_rate
                  << " " << w_rate << " " << d_rate << "\n";
    }

    static std::string make_timestamp() {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }

    static std::string human_size(std::uintmax_t bytes) {
        std::ostringstream oss;
        if (bytes >= 1024ULL * 1024ULL) {
            oss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << "MB";
        } else {
            oss << std::fixed << std::setprecision(1) << (static_cast<double>(bytes) / 1024.0) << "KB";
        }
        return oss.str();
    }

    BatchLossStats train_batch(const std::vector<TrainSample>& batch) {
        const int bsz = static_cast<int>(batch.size());
        const int area = game_.board_size * game_.board_size;
        const int c = game_.num_planes;

        std::vector<float> encoded_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
        std::vector<float> policy_target_buf(static_cast<size_t>(bsz) * area, 0.0f);
        std::vector<float> opp_policy_target_buf(static_cast<size_t>(bsz) * area, 0.0f);
        std::vector<float> value_target_buf(static_cast<size_t>(bsz) * 3, 0.0f);
        std::vector<float> sample_weights_buf(static_cast<size_t>(bsz), 1.0f);

        std::uniform_int_distribution<int> aug_dist(0, 7);

        for (int b = 0; b < bsz; ++b) {
            auto encoded = game_.encode_state(batch[b].state, batch[b].to_play);
            auto policy_t = batch[b].policy_target;
            auto opp_policy_t = batch[b].opponent_policy_target;

            const int transform_type = aug_dist(rng_);
            const int k = transform_type % 4;
            const bool do_flip = transform_type >= 4;
            if (k != 0 || do_flip) {
                encoded = transform_encoded_state(encoded, c, game_.board_size, k, do_flip);
                policy_t = reshape_rotate_flip_flat(policy_t, game_.board_size, k, do_flip);
                opp_policy_t = reshape_rotate_flip_flat(opp_policy_t, game_.board_size, k, do_flip);
            }

            const size_t ebase = static_cast<size_t>(b) * c * area;
            for (int i = 0; i < c * area; ++i) {
                encoded_buf[ebase + i] = static_cast<float>(encoded[i]);
            }
            const size_t pbase = static_cast<size_t>(b) * area;
            std::copy(policy_t.begin(), policy_t.end(), policy_target_buf.begin() + pbase);
            std::copy(opp_policy_t.begin(), opp_policy_t.end(), opp_policy_target_buf.begin() + pbase);

            value_target_buf[b * 3 + 0] = batch[b].value_target[0];
            value_target_buf[b * 3 + 1] = batch[b].value_target[1];
            value_target_buf[b * 3 + 2] = batch[b].value_target[2];
            sample_weights_buf[b] = batch[b].sample_weight;
        }

        auto encoded_states = torch::from_blob(encoded_buf.data(), {bsz, c, game_.board_size, game_.board_size}, torch::kFloat32).clone().to(cfg_.device);
        auto policy_targets = torch::from_blob(policy_target_buf.data(), {bsz, area}, torch::kFloat32).clone().to(cfg_.device);
        auto opp_policy_targets = torch::from_blob(opp_policy_target_buf.data(), {bsz, area}, torch::kFloat32).clone().to(cfg_.device);
        auto value_targets = torch::from_blob(value_target_buf.data(), {bsz, 3}, torch::kFloat32).clone().to(cfg_.device);
        auto sample_weights = torch::from_blob(sample_weights_buf.data(), {bsz}, torch::kFloat32).clone().to(cfg_.device);

        auto soft_policy_targets = torch::pow(policy_targets, 0.25);
        soft_policy_targets = soft_policy_targets / (soft_policy_targets.sum(-1, true) + 1e-10);
        auto soft_opp_policy_targets = torch::pow(opp_policy_targets, 0.25);
        soft_opp_policy_targets = soft_opp_policy_targets / (soft_opp_policy_targets.sum(-1, true) + 1e-10);

        model_->train();
        const auto nn_out = model_->forward(encoded_states);

        auto policy_logits = nn_out.policy_logits.view({bsz, -1});
        auto opp_policy_logits = nn_out.opponent_policy_logits.view({bsz, -1});
        auto soft_policy_logits = nn_out.soft_policy_logits.view({bsz, -1});
        auto soft_opp_policy_logits = nn_out.soft_opponent_policy_logits.view({bsz, -1});

        auto weighted_ce = [&](const torch::Tensor& logits, const torch::Tensor& targets) {
            auto loss = -(targets * torch::log_softmax(logits, -1)).sum(-1);
            auto result = (loss * sample_weights).mean();
            if (torch::isnan(result).template item<bool>() || torch::isinf(result).template item<bool>()) {
                return torch::zeros({1}, result.options()).squeeze();
            }
            return result;
        };

        auto policy_loss = weighted_ce(policy_logits, policy_targets);
        auto opp_policy_loss = weighted_ce(opp_policy_logits, opp_policy_targets);
        auto soft_policy_loss = weighted_ce(soft_policy_logits, soft_policy_targets);
        auto soft_opp_policy_loss = weighted_ce(soft_opp_policy_logits, soft_opp_policy_targets);
        auto value_loss = weighted_ce(nn_out.value_logits, value_targets);

        auto total_loss =
            cfg_.policy_loss_weight * policy_loss +
            cfg_.opponent_policy_loss_weight * opp_policy_loss +
            cfg_.soft_policy_loss_weight * soft_policy_loss +
            cfg_.soft_opponent_policy_loss_weight * soft_opp_policy_loss +
            cfg_.value_loss_weight * value_loss;

        float full_search_ratio = 0.0f;
        for (const auto& s : batch) {
            full_search_ratio += static_cast<float>(s.is_full_search ? 1 : 0);
        }
        full_search_ratio /= static_cast<float>(std::max<size_t>(1, batch.size()));

        optimizer_.zero_grad();
        total_loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.max_grad_norm);
        optimizer_.step();

        BatchLossStats out;
        out.total_loss = total_loss.template item<float>();
        out.policy_loss = policy_loss.template item<float>();
        out.opponent_policy_loss = opp_policy_loss.template item<float>();
        out.soft_policy_loss = soft_policy_loss.template item<float>();
        out.soft_opponent_policy_loss = soft_opp_policy_loss.template item<float>();
        out.value_loss = value_loss.template item<float>();
        out.full_search_ratio = full_search_ratio;
        return out;
    }

    Game& game_;
    ResNet& model_;
    torch::optim::Optimizer& optimizer_;
    AlphaZeroConfig cfg_;
    AlphaZeroParallelConfig pcfg_;

    ReplayBuffer replay_buffer_;
    std::vector<ResNet> inference_models_;

    std::vector<std::unique_ptr<std::mutex>> inference_model_mutexes_;
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    std::deque<std::unique_ptr<InferenceRequest>> inference_queue_;
    std::vector<std::thread> inference_threads_;
    std::atomic<bool> stop_inference_{false};

    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::deque<SelfPlayResult> result_queue_;
    std::vector<std::thread> selfplay_threads_;
    std::atomic<bool> stop_workers_{false};

    std::mt19937 rng_;
    int game_count_ = 0;
    int64_t total_samples_ = 0;
    int last_stats_game_count_ = -1;
    std::chrono::steady_clock::time_point session_start_time_{};
    std::vector<float> total_loss_history_;
    std::vector<float> policy_loss_history_;
    std::vector<float> opponent_policy_loss_history_;
    std::vector<float> soft_policy_loss_history_;
    std::vector<float> soft_opponent_policy_loss_history_;
    std::vector<float> value_loss_history_;
    std::vector<std::array<float, 4>> winrate_history_;
    std::vector<float> avg_game_len_history_;
    std::vector<int> recent_game_lengths_;
    std::vector<int> recent_sample_lengths_;
    std::vector<int> black_win_counts_;
    std::vector<int> white_win_counts_;
};

}  // namespace skyzero

#endif
