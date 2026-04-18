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
          rng_(seed)
    {
        if (cfg_.enable_subtree_value_bias) {
            svb_table_ = std::make_shared<SubtreeValueBiasTable>(
                game_.board_size,
                cfg_.subtree_value_bias_table_shards,
                cfg_.subtree_value_bias_pattern_radius
            );
        }
    }

    MCTSSearchOutput search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root,
        bool is_eval = false
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

        auto gumbel = gumbel_sequential_halving(*root, num_simulations, is_eval);

        MCTSSearchOutput out;
        out.mcts_policy = std::move(gumbel.improved_policy);
        out.v_mix = gumbel.v_mix;
        out.nn_policy = std::move(nn_policy);
        out.nn_value_probs = nn_value_probs;
        out.gumbel_action = gumbel.gumbel_action;
        return out;
    }

private:
    struct InferenceResult {
        std::vector<float> policy;
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        std::vector<float> masked_logits;
    };

    struct PendingLeaf {
        MCTSNode* leaf = nullptr;
        std::vector<int8_t> encoded;
        int transform_k = 0;
        bool transform_flip = false;
        int infer_offset = 0;
        int infer_count = 1;
        std::vector<MCTSNode*> path;
    };

    struct GumbelResult {
        std::vector<float> improved_policy;
        int gumbel_action = -1;
        std::array<float, 3> v_mix{0.0f, 1.0f, 0.0f};
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

    InferenceResult inference(
        const std::vector<int8_t>& state,
        int to_play,
        bool use_stochastic_transform,
        bool use_symmetry_transform
    ) {
        auto encoded = game_.encode_state(state, to_play);
        const int area = game_.board_size * game_.board_size;

        if (!use_stochastic_transform && use_symmetry_transform) {
            std::vector<std::vector<int8_t>> encoded_batch;
            encoded_batch.reserve(8);
            for (int fi = 0; fi < 2; ++fi) {
                const bool do_flip = (fi == 1);
                for (int k = 0; k < 4; ++k) {
                    encoded_batch.push_back(transform_encoded_state(encoded, game_.num_planes, game_.board_size, k, do_flip));
                }
            }

            std::vector<std::pair<std::vector<float>, std::array<float, 3>>> infer_results;
            if (batch_infer_fn_) {
                infer_results = batch_infer_fn_(encoded_batch);
            } else {
                infer_results.reserve(encoded_batch.size());
                for (const auto& e : encoded_batch) {
                    infer_results.push_back(infer_fn_(e));
                }
            }
            if (infer_results.size() != 8) {
                throw std::runtime_error("symmetry inference returned unexpected batch size");
            }

            std::vector<float> logits(static_cast<size_t>(area), 0.0f);
            std::array<float, 3> value{0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 8; ++i) {
                const int k = i % 4;
                const bool do_flip = i >= 4;
                auto restored = undo_transform_flat(infer_results[static_cast<size_t>(i)].first, game_.board_size, k, do_flip);
                for (int j = 0; j < area; ++j) {
                    logits[static_cast<size_t>(j)] += restored[static_cast<size_t>(j)] / 8.0f;
                }
                value[0] += infer_results[static_cast<size_t>(i)].second[0] / 8.0f;
                value[1] += infer_results[static_cast<size_t>(i)].second[1] / 8.0f;
                value[2] += infer_results[static_cast<size_t>(i)].second[2] / 8.0f;
            }

            const auto legal = game_.get_is_legal_actions(state, to_play);
            for (size_t i = 0; i < logits.size(); ++i) {
                if (i >= legal.size() || !legal[i]) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
            return {softmax(logits), value, logits};
        }

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
            if (i >= legal.size() || !legal[i]) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }

        return {softmax(logits), pair.second, logits};
    }

    void expand_with(const InferenceResult& ir, MCTSNode& node) {
        node.nn_policy = ir.policy;
        node.nn_value_probs = ir.value;
        node.nn_logits = ir.masked_logits;

        // Determine prev_action for SVB bucketing (grandparent's move)
        const int prev_action = node.parent ? node.parent->action_taken : -1;

        node.children.clear();
        for (int a = 0; a < static_cast<int>(ir.policy.size()); ++a) {
            const float p = ir.policy[a];
            if (p <= 0.0f) continue;
            auto child = std::unique_ptr<MCTSNode>(new MCTSNode{
                game_.get_next_state(node.state, a, node.to_play),
                -node.to_play,
                p,
                &node,
                a
            });
            bind_svb_entry(*child, prev_action, node.state);
            node.children.push_back(std::move(child));
        }
    }

    std::array<float, 3> expand(MCTSNode& node) {
        const auto ir = inference(
            node.state,
            node.to_play,
            cfg_.enable_stochastic_transform_inference_for_child,
            cfg_.enable_symmetry_inference_for_child
        );
        expand_with(ir, node);
        return ir.value;
    }

    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& node) {
        const auto ir = inference(
            node.state,
            node.to_play,
            cfg_.enable_stochastic_transform_inference_for_root,
            cfg_.enable_symmetry_inference_for_root
        );
        expand_with(ir, node);
        return {ir.policy, ir.value};
    }

    MCTSNode* select(MCTSNode& node) {
        // Visited policy mass (vloss counts as "visited")
        float visited_policy_mass = 0.0f;
        for (auto& child_ptr : node.children) {
            if (child_ptr->n > 0 || child_ptr->vloss > 0) {
                visited_policy_mass += child_ptr->prior;
            }
        }

        // Use shared helper for dynamic stdev-scaled cPUCT + FPU
        const int effective_parent_n = node.n + node.vloss;
        const auto sp = compute_select_params(node, effective_parent_n, visited_policy_mass, cfg_);

        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        for (auto& child_ptr : node.children) {
            auto& child = *child_ptr;
            const int effective_child_n = child.n + child.vloss;
            float q = sp.fpu_value;
            if (effective_child_n > 0) {
                const float utility_sum = (child.v[2] - child.v[0]) - static_cast<float>(child.vloss);
                q = utility_sum / static_cast<float>(effective_child_n);
            }
            const float u = sp.explore_scaling * child.prior / (1.0f + static_cast<float>(effective_child_n));
            const float score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = &child;
            }
        }
        return best_child;
    }

    void run_rollouts(MCTSNode& root, const std::vector<int>& actions, int& sims_budget) {
        if (actions.empty() || sims_budget <= 0) {
            return;
        }

        std::vector<PendingLeaf> pending;
        pending.reserve(actions.size());

        for (int action : actions) {
            if (sims_budget <= 0) {
                break;
            }

            MCTSNode* child = nullptr;
            for (auto& c : root.children) {
                if (c && c->action_taken == action) {
                    child = c.get();
                    break;
                }
            }
            if (child == nullptr) {
                continue;
            }

            std::vector<MCTSNode*> path;
            path.reserve(64);
            root.vloss += 1;
            path.push_back(&root);

            MCTSNode* node = child;
            node->vloss += 1;
            path.push_back(node);

            while (node->is_expanded()) {
                node = select(*node);
                if (node == nullptr) {
                    remove_vloss_on_path(path);
                    break;
                }
                node->vloss += 1;
                path.push_back(node);
            }
            if (node == nullptr) {
                continue;
            }

            if (game_.is_terminal(node->state, node->action_taken, -node->to_play)) {
                std::array<float, 3> value{0.0f, 1.0f, 0.0f};
                const int result = game_.get_winner(node->state, node->action_taken, -node->to_play) * node->to_play;
                if (result == 1) value = {1.0f, 0.0f, 0.0f};
                else if (result == -1) value = {0.0f, 0.0f, 1.0f};
                backpropagate_path_with_vloss(path, value);
                sims_budget -= 1;
                continue;
            }

            PendingLeaf pl;
            pl.leaf = node;
            pl.path = std::move(path);
            pl.encoded = game_.encode_state(node->state, node->to_play);

            if (cfg_.enable_stochastic_transform_inference_for_child) {
                std::uniform_int_distribution<int> dist(0, 7);
                const int transform_type = dist(rng_);
                pl.transform_k = transform_type % 4;
                pl.transform_flip = transform_type >= 4;
                pl.encoded = transform_encoded_state(pl.encoded, game_.num_planes, game_.board_size, pl.transform_k, pl.transform_flip);
            } else if (cfg_.enable_symmetry_inference_for_child) {
                pl.infer_count = 8;
            }

            pending.push_back(std::move(pl));
        }

        if (pending.empty()) {
            return;
        }

        std::vector<std::vector<int8_t>> encoded_batch;
        encoded_batch.reserve(pending.size() * 8);
        for (auto& p : pending) {
            p.infer_offset = static_cast<int>(encoded_batch.size());
            if (p.infer_count == 8) {
                for (int fi = 0; fi < 2; ++fi) {
                    const bool do_flip = (fi == 1);
                    for (int k = 0; k < 4; ++k) {
                        encoded_batch.push_back(
                            transform_encoded_state(p.encoded, game_.num_planes, game_.board_size, k, do_flip)
                        );
                    }
                }
            } else {
                encoded_batch.push_back(p.encoded);
            }
        }

        std::vector<std::pair<std::vector<float>, std::array<float, 3>>> infer_results;
        try {
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
                remove_vloss_on_path(p.path);
            }
            throw;
        }

        if (infer_results.size() != encoded_batch.size()) {
            for (const auto& p : pending) {
                remove_vloss_on_path(p.path);
            }
            throw std::runtime_error("batch_infer_fn returned unexpected batch size");
        }

        for (size_t i = 0; i < pending.size(); ++i) {
            std::vector<float> logits;
            std::array<float, 3> value{0.0f, 0.0f, 0.0f};

            if (pending[i].infer_count == 8) {
                const int area = game_.board_size * game_.board_size;
                logits.assign(static_cast<size_t>(area), 0.0f);
                for (int s = 0; s < 8; ++s) {
                    const size_t idx = static_cast<size_t>(pending[i].infer_offset + s);
                    const int k = s % 4;
                    const bool do_flip = s >= 4;
                    auto restored = undo_transform_flat(infer_results[idx].first, game_.board_size, k, do_flip);
                    for (int j = 0; j < area; ++j) {
                        logits[static_cast<size_t>(j)] += restored[static_cast<size_t>(j)] / 8.0f;
                    }
                    value[0] += infer_results[idx].second[0] / 8.0f;
                    value[1] += infer_results[idx].second[1] / 8.0f;
                    value[2] += infer_results[idx].second[2] / 8.0f;
                }
            } else {
                const size_t idx = static_cast<size_t>(pending[i].infer_offset);
                logits = std::move(infer_results[idx].first);
                value = infer_results[idx].second;
                if (pending[i].transform_k != 0 || pending[i].transform_flip) {
                    logits = undo_transform_flat(logits, game_.board_size, pending[i].transform_k, pending[i].transform_flip);
                }
            }

            const auto legal = game_.get_is_legal_actions(pending[i].leaf->state, pending[i].leaf->to_play);
            for (size_t j = 0; j < logits.size(); ++j) {
                if (j >= legal.size() || !legal[j]) {
                    logits[j] = -std::numeric_limits<float>::infinity();
                }
            }

            InferenceResult ir;
            ir.masked_logits = logits;
            ir.policy = softmax(logits);
            ir.value = value;
            expand_with(ir, *pending[i].leaf);
            // Apply SVB correction before backprop
            const auto corrected_value = apply_svb_correction(*pending[i].leaf, ir.value);
            backpropagate_path_with_vloss(pending[i].path, corrected_value);
            sims_budget -= 1;
        }
    }

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations, bool is_eval = false) {
        const int action_size = game_.board_size * game_.board_size;
        std::vector<float> logits = root.nn_logits;
        if (logits.size() != static_cast<size_t>(action_size)) {
            logits.assign(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        }

        const auto is_legal = game_.get_is_legal_actions(root.state, root.to_play);

        // Gumbel noise: disabled in eval mode (unless gumbel_stochastic_eval is set)
        std::vector<float> g(static_cast<size_t>(action_size), 0.0f);
        if (is_eval && !cfg_.gumbel_stochastic_eval) {
            // eval mode: no noise
        } else {
            std::extreme_value_distribution<float> gumbel_dist(0.0f, 1.0f);
            for (int i = 0; i < action_size; ++i) {
                g[static_cast<size_t>(i)] = gumbel_dist(rng_);
            }
        }

        int m = std::min(num_simulations, cfg_.gumbel_m);
        std::vector<int> sorted_actions(static_cast<size_t>(action_size));
        std::iota(sorted_actions.begin(), sorted_actions.end(), 0);
        std::sort(sorted_actions.begin(), sorted_actions.end(), [&](int a, int b) {
            const float sa = (a < static_cast<int>(is_legal.size()) && is_legal[static_cast<size_t>(a)])
                ? (logits[static_cast<size_t>(a)] + g[static_cast<size_t>(a)])
                : -std::numeric_limits<float>::infinity();
            const float sb = (b < static_cast<int>(is_legal.size()) && is_legal[static_cast<size_t>(b)])
                ? (logits[static_cast<size_t>(b)] + g[static_cast<size_t>(b)])
                : -std::numeric_limits<float>::infinity();
            return sa > sb;
        });

        std::vector<int> surviving_actions;
        surviving_actions.reserve(static_cast<size_t>(m));
        for (int a : sorted_actions) {
            if (static_cast<int>(surviving_actions.size()) >= m) {
                break;
            }
            if (a < static_cast<int>(is_legal.size()) && is_legal[static_cast<size_t>(a)]) {
                surviving_actions.push_back(a);
            }
        }

        m = static_cast<int>(surviving_actions.size());
        if (m > 0) {
            const int phases = (m > 1) ? static_cast<int>(std::ceil(std::log2(static_cast<double>(m)))) : 1;
            int sims_budget = num_simulations;

            for (int phase = 0; phase < phases; ++phase) {
                if (sims_budget <= 0 || surviving_actions.empty()) {
                    break;
                }

                const int remaining_phases = phases - phase;
                const int sims_this_phase = sims_budget / remaining_phases;
                const int num_actions = static_cast<int>(surviving_actions.size());
                const int sims_per_action = std::max(1, sims_this_phase / std::max(1, num_actions));

                std::vector<int> rollout_actions;
                rollout_actions.reserve(static_cast<size_t>(std::max(1, sims_per_action * num_actions)));
                for (int s = 0; s < sims_per_action && sims_budget > 0; ++s) {
                    for (int action : surviving_actions) {
                        if (sims_budget <= 0) {
                            break;
                        }
                        rollout_actions.push_back(action);
                    }
                }

                size_t offset = 0;
                while (offset < rollout_actions.size() && sims_budget > 0) {
                    const size_t chunk = std::min(static_cast<size_t>(leaf_batch_size_), rollout_actions.size() - offset);
                    std::vector<int> action_batch;
                    action_batch.reserve(chunk);
                    for (size_t i = 0; i < chunk; ++i) {
                        action_batch.push_back(rollout_actions[offset + i]);
                    }
                    run_rollouts(root, action_batch, sims_budget);
                    offset += chunk;
                }

                if (phase < phases - 1 && !surviving_actions.empty()) {
                    const float max_n = max_child_n(root);
                    const float c_visit = cfg_.gumbel_c_visit;
                    const float c_scale = cfg_.gumbel_c_scale;

                    auto eval_action = [&](int a) {
                        MCTSNode* c = nullptr;
                        for (auto& child : root.children) {
                            if (child && child->action_taken == a) {
                                c = child.get();
                                break;
                            }
                        }
                        float q = 0.5f;
                        if (c && c->n > 0) {
                            const float cw = c->v[0] / static_cast<float>(c->n);
                            const float cl = c->v[2] / static_cast<float>(c->n);
                            q = ((cl - cw) + 1.0f) * 0.5f;
                        }
                        return logits[static_cast<size_t>(a)] + g[static_cast<size_t>(a)] + (c_visit + max_n) * c_scale * q;
                    };

                    std::sort(surviving_actions.begin(), surviving_actions.end(), [&](int a, int b) {
                        return eval_action(a) > eval_action(b);
                    });
                    surviving_actions.resize(static_cast<size_t>(std::max(1, static_cast<int>(surviving_actions.size()) / 2)));
                }
            }
        }

        const float c_visit = cfg_.gumbel_c_visit;
        const float c_scale = cfg_.gumbel_c_scale;
        const float max_n = max_child_n(root);

        std::vector<std::array<float, 3>> q_wdl(static_cast<size_t>(action_size), {0.0f, 0.0f, 0.0f});
        std::vector<float> n_values(static_cast<size_t>(action_size), 0.0f);
        for (auto& c : root.children) {
            if (c && c->n > 0) {
                const float cw = c->v[0] / static_cast<float>(c->n);
                const float cd = c->v[1] / static_cast<float>(c->n);
                const float cl = c->v[2] / static_cast<float>(c->n);
                q_wdl[static_cast<size_t>(c->action_taken)] = {cl, cd, cw};
                n_values[static_cast<size_t>(c->action_taken)] = static_cast<float>(c->n);
            }
        }

        const float sum_n = std::accumulate(n_values.begin(), n_values.end(), 0.0f);
        std::array<float, 3> v_mix = root.nn_value_probs;
        if (sum_n > 0.0f) {
            std::array<float, 3> weighted_q{0.0f, 0.0f, 0.0f};
            float policy_sum = 1e-12f;
            for (int a = 0; a < action_size; ++a) {
                if (n_values[static_cast<size_t>(a)] > 0.0f && a < static_cast<int>(root.nn_policy.size())) {
                    const float p = root.nn_policy[static_cast<size_t>(a)];
                    weighted_q[0] += p * q_wdl[static_cast<size_t>(a)][0];
                    weighted_q[1] += p * q_wdl[static_cast<size_t>(a)][1];
                    weighted_q[2] += p * q_wdl[static_cast<size_t>(a)][2];
                    policy_sum += p;
                }
            }
            weighted_q[0] /= policy_sum;
            weighted_q[1] /= policy_sum;
            weighted_q[2] /= policy_sum;
            v_mix = {
                (root.nn_value_probs[0] + sum_n * weighted_q[0]) / (1.0f + sum_n),
                (root.nn_value_probs[1] + sum_n * weighted_q[1]) / (1.0f + sum_n),
                (root.nn_value_probs[2] + sum_n * weighted_q[2]) / (1.0f + sum_n),
            };
        }

        std::vector<float> sigma_q(static_cast<size_t>(action_size), 0.0f);
        for (int a = 0; a < action_size; ++a) {
            const auto& q = (n_values[static_cast<size_t>(a)] > 0.0f) ? q_wdl[static_cast<size_t>(a)] : v_mix;
            float s = q[0] - q[2];
            s = (s + 1.0f) * 0.5f;
            sigma_q[static_cast<size_t>(a)] = (c_visit + max_n) * c_scale * s;
        }

        std::vector<float> improved_logits(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        for (int a = 0; a < action_size; ++a) {
            if (a < static_cast<int>(is_legal.size()) && is_legal[static_cast<size_t>(a)]) {
                improved_logits[static_cast<size_t>(a)] = logits[static_cast<size_t>(a)] + sigma_q[static_cast<size_t>(a)];
            }
        }
        auto improved_policy = softmax(improved_logits);

        auto final_eval = [&](int a) {
            return logits[static_cast<size_t>(a)] + g[static_cast<size_t>(a)] + sigma_q[static_cast<size_t>(a)];
        };

        int gumbel_action = -1;
        if (!surviving_actions.empty()) {
            float max_n_surviving = -1.0f;
            std::vector<int> most_visited;
            for (int a : surviving_actions) {
                const float nv = n_values[static_cast<size_t>(a)];
                if (nv > max_n_surviving) {
                    max_n_surviving = nv;
                    most_visited.clear();
                    most_visited.push_back(a);
                } else if (nv == max_n_surviving) {
                    most_visited.push_back(a);
                }
            }
            gumbel_action = most_visited[0];
            float best_eval = final_eval(gumbel_action);
            for (size_t i = 1; i < most_visited.size(); ++i) {
                const float ev = final_eval(most_visited[i]);
                if (ev > best_eval) {
                    best_eval = ev;
                    gumbel_action = most_visited[i];
                }
            }
        }

        if (gumbel_action < 0 && !improved_policy.empty()) {
            gumbel_action = static_cast<int>(std::distance(improved_policy.begin(), std::max_element(improved_policy.begin(), improved_policy.end())));
        }
        return {improved_policy, gumbel_action, v_mix};
    }

    static void remove_vloss_on_path(const std::vector<MCTSNode*>& path) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if ((*it)->vloss > 0) {
                (*it)->vloss -= 1;
            }
        }
    }

    void backpropagate_path_with_vloss(const std::vector<MCTSNode*>& path, std::array<float, 3> value) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->update(value);
            (*it)->vloss -= 1;
            if ((*it)->is_expanded()) {
                update_svb_for_node(**it);
            }
            value = flip_wdl(value);
        }
    }

    void backpropagate(MCTSNode* node, std::array<float, 3> value) {
        while (node != nullptr) {
            node->update(value);
            if (node->is_expanded()) {
                update_svb_for_node(*node);
            }
            value = flip_wdl(value);
            node = node->parent;
        }
    }

    static float max_child_n(const MCTSNode& root) {
        float mx = 0.0f;
        for (const auto& c : root.children) {
            if (c) {
                mx = std::max(mx, static_cast<float>(c->n));
            }
        }
        return mx;
    }

public:
    // --- Subtree Value Bias public interface ---
    void cleanup_svb_for_tree(MCTSNode* node) {
        remove_svb_contribution(node);
    }
    void cleanup_unused_svb() {
        if (svb_table_) svb_table_->clear_unused();
    }

private:
    // --- Subtree Value Bias helpers ---

    void bind_svb_entry(MCTSNode& child, int prev_action, const std::vector<int8_t>& parent_state) {
        if (!svb_table_) return;
        const int player = -child.to_play;
        child.svb_entry = svb_table_->get(player, prev_action, child.action_taken, parent_state);
    }

    void update_svb_for_node(MCTSNode& node) {
        if (!svb_table_ || !node.svb_entry || node.n <= 1) return;
        const float nn_utility = wdl_utility(node.nn_value_probs);
        float child_utility_sum = 0.0f;
        float child_weight_sum = 0.0f;
        for (auto& c : node.children) {
            if (c && c->n > 0) {
                const float child_q = (c->v[2] - c->v[0]) / static_cast<float>(c->n);
                const float w = static_cast<float>(c->n);
                child_utility_sum += child_q * w;
                child_weight_sum += w;
            }
        }
        if (child_weight_sum < 1e-6f) return;
        const float children_utility = child_utility_sum / child_weight_sum;
        const float svb_weight = std::pow(child_weight_sum, cfg_.subtree_value_bias_weight_exponent);
        const float svb_delta = (children_utility - nn_utility) * svb_weight;
        auto& entry = *node.svb_entry;
        entry.add(svb_delta - node.last_svb_delta_sum, svb_weight - node.last_svb_weight);
        node.last_svb_delta_sum = svb_delta;
        node.last_svb_weight = svb_weight;
    }

    void remove_svb_contribution(MCTSNode* node) {
        if (!node) return;
        if (node->svb_entry) {
            node->svb_entry->subtract(
                node->last_svb_delta_sum * cfg_.subtree_value_bias_free_prop,
                node->last_svb_weight * cfg_.subtree_value_bias_free_prop
            );
            node->svb_entry.reset();
        }
        for (auto& child : node->children) {
            remove_svb_contribution(child.get());
        }
    }

    std::array<float, 3> apply_svb_correction(const MCTSNode& node, const std::array<float, 3>& value) const {
        if (!svb_table_ || !node.svb_entry) return value;
        const float bias = node.svb_entry->get_bias();
        if (std::fabs(bias) < 1e-8f) return value;
        const float correction = cfg_.subtree_value_bias_factor * bias;
        auto corrected = value;
        corrected[0] = std::max(0.0f, std::min(1.0f, value[0] - correction * 0.5f));
        corrected[2] = std::max(0.0f, std::min(1.0f, value[2] + correction * 0.5f));
        const float sum = corrected[0] + corrected[1] + corrected[2];
        if (sum > 1e-8f) {
            corrected[0] /= sum;
            corrected[1] /= sum;
            corrected[2] /= sum;
        }
        return corrected;
    }

    Game& game_;
    const AlphaZeroConfig& cfg_;
    int leaf_batch_size_ = 1;
    InferenceFn infer_fn_;
    BatchInferenceFn batch_infer_fn_;
    std::mt19937 rng_;
    std::shared_ptr<SubtreeValueBiasTable> svb_table_;
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
        float value_loss = 0.0f;
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

        // Save replay buffer to a separate binary file (streaming, no full-buffer copy)
        const fs::path rb_path = fs::path(checkpoint_path).replace_extension(".rb");
        replay_buffer_.save_to_file(rb_path.string());
        // Store a marker in the archive so load_checkpoint can detect the new format
        checkpoint_archive.write("replay_buffer_file", torch::tensor({1}));

        checkpoint_archive.save_to(checkpoint_path.string());

        const auto model_bytes = fs::file_size(model_path);
        const auto ckpt_bytes = fs::file_size(checkpoint_path);
        const auto rb_bytes = fs::file_size(rb_path);
        std::cout << "Model saved to " << model_path.string() << " (" << human_size(model_bytes) << ")\n";
        std::cout << "Checkpoint saved to " << checkpoint_path.string() << " (" << human_size(ckpt_bytes) << ")\n";
        std::cout << "Replay buffer saved to " << rb_path.string() << " (" << human_size(rb_bytes) << ")\n";
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

            // Load replay buffer: new streaming format (.rb file) or legacy archive format
            torch::Tensor rb_marker;
            const bool has_rb_file = checkpoint_archive.try_read("replay_buffer_file", rb_marker);
            if (has_rb_file) {
                // New format: load from separate binary file
                const fs::path rb_path = fs::path(checkpoint_path).replace_extension(".rb");
                replay_buffer_.load_from_file(rb_path.string());
                std::cout << "Replay buffer loaded from " << rb_path.string() << "\n";
            } else {
                // Legacy format: load from embedded archive
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

                replay_buffer_.load_state(rb);
                std::cout << "Replay buffer loaded from legacy archive format\n";
            }

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
                        mean.value_loss += s.value_loss;
                    }
                    const float inv = 1.0f / static_cast<float>(batch_losses.size());
                    mean.total_loss *= inv;
                    mean.policy_loss *= inv;
                    mean.opponent_policy_loss *= inv;
                    mean.value_loss *= inv;

                    total_loss_history_.push_back(mean.total_loss);
                    policy_loss_history_.push_back(mean.policy_loss);
                    opponent_policy_loss_history_.push_back(mean.opponent_policy_loss);
                    value_loss_history_.push_back(mean.value_loss);

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
            std::array<float, 3> v_mix{0.0f, 1.0f, 0.0f};
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
        auto init = game_.get_initial_state(worker_rng);
        std::vector<int8_t> state = std::move(init.board);
        int to_play = init.to_play;
        bool in_soft_resign = false;
        std::vector<float> historical_v_mix;
        int last_action = -1;
        int last_player = 0;
        std::unique_ptr<MCTSNode> root(new MCTSNode{state, to_play});

        while (!game_.is_terminal(state, last_action, last_player)) {
            int num_simulations = cfg_.num_simulations;
            if (in_soft_resign) {
                num_simulations = std::max(cfg_.num_simulations / 4, cfg_.min_simulations_in_soft_resign);
            }

            const auto sr = mcts.search(state, to_play, num_simulations, root);
            const float v_mix_scalar = sr.v_mix[0] - sr.v_mix[2];
            historical_v_mix.push_back(v_mix_scalar);

            const int n = static_cast<int>(historical_v_mix.size());
            float absmin_v_mix = std::numeric_limits<float>::infinity();
            const int from = std::max(0, n - cfg_.soft_resign_step_threshold);
            for (int i = from; i < n; ++i) {
                absmin_v_mix = std::min(absmin_v_mix, std::fabs(historical_v_mix[i]));
            }
            if (!in_soft_resign) {
                std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                if (absmin_v_mix >= cfg_.soft_resign_threshold && uni(worker_rng) < cfg_.soft_resign_prob) {
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
            ms.v_mix = sr.v_mix;
            ms.sample_weight = in_soft_resign ? cfg_.soft_resign_sample_weight : 1.0f;
            memory.push_back(ms);

            const int move_count = static_cast<int>(memory.size());
            const int half_life = std::max(1, cfg_.half_life);
            const float t = cfg_.move_temperature_init
                - (static_cast<float>(move_count) / static_cast<float>(half_life))
                    * (cfg_.move_temperature_init - cfg_.move_temperature_final);

            int action = sr.gumbel_action;
            if (move_count < half_life) {
                const auto move_probs = temperature_transform(sr.mcts_policy, t);
                std::discrete_distribution<int> action_dist(move_probs.begin(), move_probs.end());
                action = action_dist(worker_rng);
            }
            if (action < 0) {
                std::discrete_distribution<int> action_dist(sr.mcts_policy.begin(), sr.mcts_policy.end());
                action = action_dist(worker_rng);
            }

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
                // Remove SVB contributions from siblings being discarded
                if (cfg_.enable_subtree_value_bias) {
                    for (auto& child : root->children) {
                        if (child) {
                            mcts.cleanup_svb_for_tree(child.get());
                        }
                    }
                    mcts.cleanup_svb_for_tree(root.get());
                }
                next_root->parent = nullptr;
                root = std::move(next_root);
            } else {
                if (cfg_.enable_subtree_value_bias && root) {
                    mcts.cleanup_svb_for_tree(root.get());
                }
                root.reset(new MCTSNode{state, to_play});
            }
        }

        // Clean up unused SVB entries between games
        mcts.cleanup_unused_svb();

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
            ps.v_mix = s.v_mix;
            ps.sample_weight = s.sample_weight;
            return_memory.push_back(ps);
        }

        if (!return_memory.empty()) {
            const float now_factor = 1.0f / (1.0f + game_.board_size * game_.board_size * cfg_.value_target_mix_now_factor_constant);
            return_memory.back().value_target = return_memory.back().outcome;
            for (int i = static_cast<int>(return_memory.size()) - 2; i >= 0; --i) {
                const auto next_target = flip_wdl(return_memory[i + 1].value_target);
                return_memory[i].value_target = {
                    (1.0f - now_factor) * next_target[0] + now_factor * return_memory[i].v_mix[0],
                    (1.0f - now_factor) * next_target[1] + now_factor * return_memory[i].v_mix[1],
                    (1.0f - now_factor) * next_target[2] + now_factor * return_memory[i].v_mix[2],
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

        model_->train();
        const auto nn_out = model_->forward(encoded_states);

        auto policy_logits = nn_out.policy_logits.view({bsz, -1});
        auto opp_policy_logits = nn_out.opponent_policy_logits.view({bsz, -1});

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
        auto value_loss = weighted_ce(nn_out.value_logits, value_targets);

        auto total_loss =
            cfg_.policy_loss_weight * policy_loss +
            cfg_.opponent_policy_loss_weight * opp_policy_loss +
            cfg_.value_loss_weight * value_loss;

        optimizer_.zero_grad();
        total_loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.max_grad_norm);
        optimizer_.step();

        BatchLossStats out;
        out.total_loss = total_loss.template item<float>();
        out.policy_loss = policy_loss.template item<float>();
        out.opponent_policy_loss = opp_policy_loss.template item<float>();
        out.value_loss = value_loss.template item<float>();
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
