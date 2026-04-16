#ifndef SKYZERO_ALPHAZERO_PARALLEL_H
#define SKYZERO_ALPHAZERO_PARALLEL_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <functional>
#include <thread>
#include <unordered_map>

#include <torch/script.h>

#include "alphazero.h"
#include "npz_writer.h"
#include "random_opening.h"
#include "selfplay_manager.h"

namespace skyzero {

struct AlphaZeroParallelConfig {
    int num_workers = std::max(1u, std::thread::hardware_concurrency());
    int num_inference_servers = 1;
    int inference_batch_size = 256;
    int inference_batch_wait_us = 1500;
    int leaf_batch_size = 32;
    int max_games_to_process_per_tick = 50;
    int idle_sleep_ms = 1;
    int model_check_interval_ms = 10000;  // how often to check for new models
};

template <typename Game>
class ParallelMCTS {
public:
    using InferenceFn = std::function<std::pair<std::vector<float>, std::array<float, 4>>(const std::vector<int8_t>&)>;
    using BatchInferenceFn = std::function<std::vector<std::pair<std::vector<float>, std::array<float, 4>>>(
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

    AlphaZeroConfig& config() { return cfg_; }

    MCTSSearchOutput search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root,
        bool is_eval = false,
        int gumbel_m_override = -1
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
            backpropagate(root.get(), nn_value_probs, compute_uncertainty_weight(root->nn_value_error));
        } else {
            nn_policy = root->nn_policy;
            nn_value_probs = root->nn_value_probs;
        }

        auto gumbel = gumbel_sequential_halving(*root, num_simulations, is_eval, gumbel_m_override);

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
        float value_error = 0.0f;  // softplus output of value-error head
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

            std::vector<std::pair<std::vector<float>, std::array<float, 4>>> infer_results;
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
            float value_error = 0.0f;
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
                value_error += infer_results[static_cast<size_t>(i)].second[3] / 8.0f;
            }

            const auto legal = game_.get_is_legal_actions(state, to_play);
            for (size_t i = 0; i < logits.size(); ++i) {
                if (i >= legal.size() || !legal[i]) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
            return {softmax(logits), value, logits, value_error};
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

        std::array<float, 3> wdl{pair.second[0], pair.second[1], pair.second[2]};
        return {softmax(logits), wdl, logits, pair.second[3]};
    }

    void expand_with(const InferenceResult& ir, MCTSNode& node) {
        node.nn_policy = ir.policy;
        node.nn_value_probs = ir.value;
        node.nn_logits = ir.masked_logits;
        node.nn_value_error = ir.value_error;

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
        float visited_policy_mass = 0.0f;
        for (auto& child_ptr : node.children) {
            if (child_ptr->n > 0 || child_ptr->vloss > 0) {
                visited_policy_mass += child_ptr->prior;
            }
        }

        const float effective_parent_weight = node.weighted_n + static_cast<float>(node.vloss);
        const auto sp = compute_select_params(node, effective_parent_weight, visited_policy_mass, cfg_);

        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        for (auto& child_ptr : node.children) {
            auto& child = *child_ptr;
            const float effective_child_weight = child.weighted_n + static_cast<float>(child.vloss);
            float q = sp.fpu_value;
            if (effective_child_weight > 0.0f) {
                const float utility_sum = (child.v[2] - child.v[0]) - static_cast<float>(child.vloss);
                q = utility_sum / effective_child_weight;
            }
            const float u = sp.explore_scaling * child.prior / (1.0f + effective_child_weight);
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
                // Terminal nodes have zero predictive variance — give max weight.
                const float term_weight = cfg_.enable_uncertainty_weighting ? cfg_.uncertainty_max_weight : 1.0f;
                backpropagate_path_with_vloss(path, value, term_weight);
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

        std::vector<std::pair<std::vector<float>, std::array<float, 4>>> infer_results;
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
            float value_error = 0.0f;

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
                    value_error += infer_results[idx].second[3] / 8.0f;
                }
            } else {
                const size_t idx = static_cast<size_t>(pending[i].infer_offset);
                logits = std::move(infer_results[idx].first);
                value[0] = infer_results[idx].second[0];
                value[1] = infer_results[idx].second[1];
                value[2] = infer_results[idx].second[2];
                value_error = infer_results[idx].second[3];
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
            ir.value_error = value_error;
            expand_with(ir, *pending[i].leaf);
            const auto corrected_value = apply_svb_correction(*pending[i].leaf, ir.value);
            const float w = compute_uncertainty_weight(pending[i].leaf->nn_value_error);
            backpropagate_path_with_vloss(pending[i].path, corrected_value, w);
            sims_budget -= 1;
        }
    }

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations, bool is_eval = false, int gumbel_m_override = -1) {
        const int action_size = game_.board_size * game_.board_size;
        std::vector<float> logits = root.nn_logits;
        if (logits.size() != static_cast<size_t>(action_size)) {
            logits.assign(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        }

        const auto is_legal = game_.get_is_legal_actions(root.state, root.to_play);

        std::vector<float> g(static_cast<size_t>(action_size), 0.0f);
        if (is_eval) {
            // eval / cheap-search mode: no noise
        } else {
            std::extreme_value_distribution<float> gumbel_dist(0.0f, 1.0f);
            for (int i = 0; i < action_size; ++i) {
                g[static_cast<size_t>(i)] = gumbel_dist(rng_);
            }
        }

        const int effective_gumbel_m = (gumbel_m_override > 0) ? gumbel_m_override : cfg_.gumbel_m;
        int m = std::min(num_simulations, effective_gumbel_m);
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
                        if (c && c->n > 0 && c->weighted_n > 0.0f) {
                            const float cw = c->v[0] / c->weighted_n;
                            const float cl = c->v[2] / c->weighted_n;
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
            if (c && c->n > 0 && c->weighted_n > 0.0f) {
                const float wn = c->weighted_n;
                const float cw = c->v[0] / wn;
                const float cd = c->v[1] / wn;
                const float cl = c->v[2] / wn;
                q_wdl[static_cast<size_t>(c->action_taken)] = {cl, cd, cw};
                n_values[static_cast<size_t>(c->action_taken)] = wn;
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

    float compute_uncertainty_weight(float value_error_pred) const {
        if (!cfg_.enable_uncertainty_weighting) return 1.0f;
        const float denom = value_error_pred + cfg_.uncertainty_prior;
        if (denom <= 1e-8f) return cfg_.uncertainty_max_weight;
        const float w = std::pow(denom, -cfg_.uncertainty_exponent);
        return std::min(cfg_.uncertainty_max_weight, std::max(0.0f, w));
    }

    void backpropagate_path_with_vloss(const std::vector<MCTSNode*>& path, std::array<float, 3> value, float weight) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            (*it)->update(value, weight);
            (*it)->vloss -= 1;
            if ((*it)->is_expanded()) {
                update_svb_for_node(**it);
            }
            value = flip_wdl(value);
        }
    }

    void backpropagate(MCTSNode* node, std::array<float, 3> value, float weight) {
        while (node != nullptr) {
            node->update(value, weight);
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
                mx = std::max(mx, c->weighted_n);
            }
        }
        return mx;
    }

public:
    void cleanup_svb_for_tree(MCTSNode* node) {
        remove_svb_contribution(node);
    }
    void cleanup_unused_svb() {
        if (svb_table_) svb_table_->clear_unused();
    }

private:
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
            if (c && c->n > 0 && c->weighted_n > 0.0f) {
                const float child_q = (c->v[2] - c->v[0]) / c->weighted_n;
                const float w = c->weighted_n;
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

// ============================================================================
// AlphaZeroParallel: selfplay-only engine with TorchScript inference + NPZ output
// ============================================================================

template <typename Game>
class AlphaZeroParallel {
public:
    AlphaZeroParallel(
        Game& game,
        const AlphaZeroConfig& cfg,
        const AlphaZeroParallelConfig& pcfg = AlphaZeroParallelConfig{},
        const RandomOpeningConfig& opening_cfg = RandomOpeningConfig{}
    )
        : game_(game),
          cfg_(cfg),
          pcfg_(pcfg),
          opening_cfg_(opening_cfg),
          rng_(std::random_device{}()),
          data_writer_(game, cfg.output_dir, cfg.max_rows_per_file),
          model_manager_(cfg.model_dir)
    {
    }

    struct SelfPlayResult {
        std::vector<TrainSample> samples;
        int winner = 0;
        int game_len = 0;
    };

    // Main selfplay loop. Blocks until max_games_total games are played.
    void run() {
        // Wait for initial model
        current_model_path_ = model_manager_.wait_for_model();
        init_inference_models(current_model_path_);

        session_start_time_ = std::chrono::steady_clock::now();

        start_inference_server();
        start_selfplay_workers();
        start_model_watcher();

        try {
            while (!stop_requested && (cfg_.max_games_total < 0 || game_count_ < cfg_.max_games_total)) {
                int games_processed = 0;
                while (games_processed < pcfg_.max_games_to_process_per_tick) {
                    SelfPlayResult sp;
                    if (!try_pop_selfplay_result(sp)) {
                        break;
                    }
                    data_writer_.add_game(sp.samples);
                    total_samples_ += static_cast<int64_t>(sp.samples.size());
                    recent_game_lengths_.push_back(sp.game_len);
                    black_win_counts_.push_back(sp.winner == 1 ? 1 : 0);
                    white_win_counts_.push_back(sp.winner == -1 ? 1 : 0);
                    trim_tail(recent_game_lengths_, 300);
                    trim_tail(black_win_counts_, 300);
                    trim_tail(white_win_counts_, 300);
                    game_count_ += 1;
                    games_processed += 1;
                }

                if (game_count_ % 10 == 0 && game_count_ != last_stats_game_count_) {
                    print_selfplay_stats();
                    last_stats_game_count_ = game_count_;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, pcfg_.idle_sleep_ms)));
            }

            stop_workers_and_server();
            data_writer_.flush();
            std::cout << "Selfplay complete. Games: " << game_count_
                      << " | Total rows written: " << data_writer_.total_rows_written() << std::endl;
        } catch (...) {
            stop_workers_and_server();
            data_writer_.flush();
            throw;
        }
    }

private:
    struct InferenceRequest {
        std::vector<int8_t> encoded;
        std::promise<std::pair<std::vector<float>, std::array<float, 4>>> promise;
    };

    // --- TorchScript model loading ---

    void init_inference_models(const std::string& model_path) {
        const int num_servers = std::max(1, pcfg_.num_inference_servers);
        inference_models_.clear();
        inference_models_.reserve(static_cast<size_t>(num_servers));
        inference_model_mutexes_.clear();
        inference_model_mutexes_.reserve(static_cast<size_t>(num_servers));

        for (int i = 0; i < num_servers; ++i) {
            auto model = torch::jit::load(model_path);
            model.to(cfg_.device);
            if (cfg_.device.is_cuda()) {
                model.to(torch::kHalf);
            }
            model.eval();
            inference_models_.push_back(std::move(model));
            inference_model_mutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
        }
        std::cout << "Loaded TorchScript model: " << model_path << " (" << num_servers << " copies)" << std::endl;
    }

    void reload_inference_models(const std::string& model_path) {
        const int num_servers = static_cast<int>(inference_models_.size());
        for (int i = 0; i < num_servers; ++i) {
            std::lock_guard<std::mutex> lk(*inference_model_mutexes_[static_cast<size_t>(i)]);
            auto model = torch::jit::load(model_path);
            model.to(cfg_.device);
            if (cfg_.device.is_cuda()) {
                model.to(torch::kHalf);
            }
            model.eval();
            inference_models_[static_cast<size_t>(i)] = std::move(model);
        }
        std::cout << "Reloaded TorchScript model: " << model_path << std::endl;
    }

    // --- Model watcher thread ---

    void start_model_watcher() {
        stop_model_watcher_.store(false);
        model_watcher_thread_ = std::thread([this]() {
            while (!stop_model_watcher_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(pcfg_.model_check_interval_ms));
                if (stop_model_watcher_.load()) break;

                if (model_manager_.has_newer_model(current_model_path_)) {
                    auto new_path = model_manager_.get_latest_model();
                    if (!new_path.empty() && new_path != current_model_path_) {
                        try {
                            reload_inference_models(new_path);
                            current_model_path_ = new_path;
                        } catch (const std::exception& e) {
                            std::cerr << "[ModelWatcher] Failed to reload model: " << e.what() << std::endl;
                        }
                    }
                }
            }
        });
    }

    // --- Inference server ---

    std::pair<std::vector<float>, std::array<float, 4>> request_inference(const std::vector<int8_t>& encoded) {
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

    std::vector<std::pair<std::vector<float>, std::array<float, 4>>>
    request_batch_inference(const std::vector<std::vector<int8_t>>& encoded_batch) {
        std::vector<std::future<std::pair<std::vector<float>, std::array<float, 4>>>> futures;
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

        std::vector<std::pair<std::vector<float>, std::array<float, 4>>> out;
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

                // TorchScript forward: returns tuple(policy_logits, opp_policy_logits, value_logits)
                torch::IValue output;
                {
                    std::lock_guard<std::mutex> mlk(*inference_model_mutexes_[static_cast<size_t>(server_idx)]);
                    output = inference_models_[static_cast<size_t>(server_idx)].forward({input});
                }

                auto elements = output.toTuple()->elements();
                auto policy_logits_raw = elements[0].toTensor();  // [B, 1, H, W]
                // elements[1] is opp_policy_logits — not needed for MCTS inference
                auto value_logits_raw = elements[2].toTensor();   // [B, 3]
                // elements[3] is value_error_logit; softplus → predicted squared error
                torch::Tensor value_error_raw;
                bool has_value_error = (elements.size() >= 4);
                if (has_value_error) {
                    value_error_raw = torch::softplus(elements[3].toTensor().to(torch::kFloat32))
                                          .reshape({bsz}).to(torch::kCPU).contiguous();
                }

                auto policy = policy_logits_raw.reshape({bsz, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
                auto value = torch::softmax(value_logits_raw.to(torch::kFloat32), 1).to(torch::kCPU).contiguous();
                const float* pp = policy.data_ptr<float>();
                const float* vp = value.data_ptr<float>();
                const float* vep = has_value_error ? value_error_raw.data_ptr<float>() : nullptr;

                for (int i = 0; i < bsz; ++i) {
                    std::vector<float> logits(static_cast<size_t>(area), 0.0f);
                    std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area, static_cast<size_t>(area) * sizeof(float));
                    const size_t vi = static_cast<size_t>(i) * 3;
                    std::array<float, 4> v{
                        vp[vi], vp[vi + 1], vp[vi + 2],
                        vep ? vep[i] : 0.0f
                    };
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

    // --- Selfplay ---

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
            float policy_weight = 1.0f;  // 0 for cheap-search rows (PCR)
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
        ForkPosition fork_init;
        bool from_fork = fork_queue_.try_pop(fork_init);
        std::vector<int8_t> state;
        int to_play = 1;
        if (from_fork) {
            state = std::move(fork_init.state);
            to_play = fork_init.to_play;
        } else {
            GameInitialState init;
            if (opening_cfg_.enabled) {
                RandomOpeningGenerator opener(game_, opening_cfg_);
                init = opener.generate(worker_rng, infer_fn, batch_infer_fn);
            } else {
                init = game_.get_initial_state(worker_rng);
            }
            state = std::move(init.board);
            to_play = init.to_play;
        }
        bool in_soft_resign = false;
        std::vector<float> historical_v_mix;
        int last_action = -1;
        int last_player = 0;
        std::unique_ptr<MCTSNode> root(new MCTSNode{state, to_play});

        while (!game_.is_terminal(state, last_action, last_player)) {
            // Playout Cap Randomization: full search with prob full_search_prob,
            // cheap search otherwise (lower visits, lower gumbel_m, downweighted
            // sample, policy target excluded from training).
            std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
            const bool is_full_search = (uni01(worker_rng) < cfg_.full_search_prob);

            int num_simulations = is_full_search ? cfg_.num_simulations : cfg_.cheap_simulations;
            int gumbel_m_override = is_full_search ? -1 : cfg_.cheap_gumbel_m;
            if (in_soft_resign) {
                num_simulations = std::max(num_simulations / 4, cfg_.min_simulations_in_soft_resign);
            }

            const float saved_root_fpu = mcts.config().root_fpu_reduction_max;
            if (!is_full_search) {
                mcts.config().root_fpu_reduction_max = mcts.config().fpu_reduction_max;
            }
            const auto sr = mcts.search(state, to_play, num_simulations, root, /*is_eval=*/!is_full_search, gumbel_m_override);
            mcts.config().root_fpu_reduction_max = saved_root_fpu;
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
            float base_weight = in_soft_resign ? cfg_.soft_resign_sample_weight : 1.0f;
            if (!is_full_search) base_weight *= cfg_.cheap_sample_weight;
            ms.sample_weight = base_weight;
            ms.policy_weight = is_full_search ? 1.0f : 0.0f;
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

            // Fork side position: with small probability, sample an alternative
            // legal move (by NN policy, excluding the chosen action) and push
            // the resulting (state, opp_to_play) into the global fork queue.
            // Skip on cheap-search rows: NN policy + Gumbel result less reliable.
            if (cfg_.fork_side_position_prob > 0.0f
                && is_full_search
                && move_count > cfg_.fork_skip_first_n_moves) {
                std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                if (uni(worker_rng) < cfg_.fork_side_position_prob) {
                    int best_alt = -1;
                    float best_p = -1.0f;
                    const auto legal = game_.get_is_legal_actions(state, to_play);
                    for (int a = 0; a < static_cast<int>(sr.nn_policy.size()); ++a) {
                        if (!legal[a] || a == action) continue;
                        if (sr.nn_policy[a] > best_p) {
                            best_p = sr.nn_policy[a];
                            best_alt = a;
                        }
                    }
                    if (best_alt >= 0) {
                        ForkPosition fp;
                        fp.state = game_.get_next_state(state, best_alt, to_play);
                        fp.to_play = -to_play;
                        if (!game_.is_terminal(fp.state, best_alt, to_play)) {
                            fork_queue_.push(std::move(fp), cfg_.max_fork_queue_size);
                        }
                    }
                }
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
            ps.policy_weight = s.policy_weight;
            // KataGo-aligned: opp policy weight only masks the final row (no next move).
            // Cheap-search next-move targets are kept at full weight, since the 0.15
            // global scaling on opp policy loss already makes this head noise-tolerant.
            const bool has_next = (i + 1 < memory.size());
            ps.opp_policy_weight = has_next ? 1.0f : 0.0f;
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

        stop_model_watcher_.store(true);
        if (model_watcher_thread_.joinable()) {
            model_watcher_thread_.join();
        }
    }

    // --- Stats ---

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

        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - session_start_time_
        ).count();
        const double sps = (elapsed > 0.0) ? (static_cast<double>(total_samples_) / elapsed) : 0.0;

        std::cout << "Game: " << game_count_;
        if (cfg_.max_games_total >= 0) std::cout << "/" << cfg_.max_games_total;
        std::cout
                  << " | Sps: " << std::fixed << std::setprecision(1) << sps
                  << " | AvgGameLen: " << std::fixed << std::setprecision(1) << avg_game_len
                  << " | BWD: " << std::fixed << std::setprecision(2) << b_rate
                  << " " << w_rate << " " << d_rate
                  << " | Fork: " << fork_queue_.size()
                  << " (push=" << fork_queue_.push_count()
                  << " pop=" << fork_queue_.pop_count() << ")"
                  << " | Model: " << current_model_path_
                  << "\n";
    }

    // --- Member variables ---

    Game& game_;
    AlphaZeroConfig cfg_;
    AlphaZeroParallelConfig pcfg_;
    RandomOpeningConfig opening_cfg_;

    // TorchScript inference models (one per inference server)
    std::vector<torch::jit::Module> inference_models_;
    std::vector<std::unique_ptr<std::mutex>> inference_model_mutexes_;
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    std::deque<std::unique_ptr<InferenceRequest>> inference_queue_;
    std::vector<std::thread> inference_threads_;
    std::atomic<bool> stop_inference_{false};

    // Selfplay workers
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::deque<SelfPlayResult> result_queue_;
    std::vector<std::thread> selfplay_threads_;
    std::atomic<bool> stop_workers_{false};

    // Model watcher
    SelfplayManager model_manager_;
    std::string current_model_path_;
    std::thread model_watcher_thread_;
    std::atomic<bool> stop_model_watcher_{false};

    // NPZ data writer
    NpzDataWriter<Game> data_writer_;

    // Fork-side-position queue (data diversity)
    ForkQueue fork_queue_;

    // Stats
    std::mt19937 rng_;
    int game_count_ = 0;
    int64_t total_samples_ = 0;
    int last_stats_game_count_ = -1;
    std::chrono::steady_clock::time_point session_start_time_{};
    std::vector<int> recent_game_lengths_;
    std::vector<int> black_win_counts_;
    std::vector<int> white_win_counts_;
};

}  // namespace skyzero

#endif  // SKYZERO_ALPHAZERO_PARALLEL_H
