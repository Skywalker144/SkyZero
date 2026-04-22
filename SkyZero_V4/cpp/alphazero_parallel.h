#ifndef SKYZERO_ALPHAZERO_PARALLEL_H
#define SKYZERO_ALPHAZERO_PARALLEL_H

// Parallel Gumbel MCTS with batched inference.
//
// Ported from CSkyZero_V3/alphazero_parallel.h. Compared to V3:
//   * Subtree Value Bias (SVB) removed — all bind/update/remove/apply
//     helpers and `svb_table_` are gone. Dynamic variance-scaled cPUCT is
//     also handled by the shared helper in alphazero.h (stdev factor == 1).
//   * The in-C++ `AlphaZeroParallel` training-loop class is removed:
//     Python owns training, replay buffer, and checkpointing.
//   * Inference runs against a caller-supplied callback pair (single-state
//     and batch). The selfplay_manager wires these to a shared TorchScript
//     module pool — selfplay here does no torch operations directly.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "alphazero.h"
#include "utils.h"

namespace skyzero {

struct SelfplayParallelConfig {
    int num_workers = 32;
    int num_inference_servers = 2;
    int inference_batch_size = 128;
    int inference_batch_wait_us = 100;
    int leaf_batch_size = 8;
    int max_result_queue_size = 0;  // 0 = auto (2 * num_workers); <0 = unbounded
};

template <typename Game>
class ParallelMCTS {
public:
    using InferenceFn = std::function<std::pair<std::vector<float>, std::array<float, 3>>(const std::vector<int8_t>&)>;
    using BatchInferenceFn = std::function<
        std::vector<std::pair<std::vector<float>, std::array<float, 3>>>(
            const std::vector<std::vector<int8_t>>&
        )>;

    ParallelMCTS(
        Game& game,
        const AlphaZeroConfig& cfg,
        int leaf_batch_size,
        InferenceFn infer_fn,
        BatchInferenceFn batch_infer_fn,
        uint64_t seed
    )
        : game_(game),
          cfg_(cfg),
          leaf_batch_size_(std::max(1, leaf_batch_size)),
          infer_fn_(std::move(infer_fn)),
          batch_infer_fn_(std::move(batch_infer_fn)),
          rng_(seed) {}

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
        auto gumbel = gumbel_sequential_halving(*root, num_simulations);

        MCTSSearchOutput out;
        out.mcts_policy = std::move(gumbel.improved_policy);
        out.v_mix = gumbel.v_mix;
        out.nn_policy = std::move(nn_policy);
        out.nn_value_probs = nn_value_probs;
        out.gumbel_action = gumbel.gumbel_action;
        {
            const int action_size = game_.board_size * game_.board_size;
            out.visit_counts.assign(static_cast<size_t>(action_size), 0.0f);
            for (const auto& c : root->children) {
                if (!c) continue;
                const int a = c->action_taken;
                if (a >= 0 && a < action_size) {
                    out.visit_counts[static_cast<size_t>(a)] = static_cast<float>(c->n);
                }
            }
        }
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
            node.children.push_back(std::move(child));
        }
    }

    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& node) {
        const auto ir = inference(
            node.state, node.to_play,
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
        if (actions.empty() || sims_budget <= 0) return;

        std::vector<PendingLeaf> pending;
        pending.reserve(actions.size());

        for (int action : actions) {
            if (sims_budget <= 0) break;

            MCTSNode* child = nullptr;
            for (auto& c : root.children) {
                if (c && c->action_taken == action) { child = c.get(); break; }
            }
            if (child == nullptr) continue;

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
            if (node == nullptr) continue;

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

        if (pending.empty()) return;

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
            backpropagate_path_with_vloss(pending[i].path, ir.value);
            sims_budget -= 1;
        }
    }

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations) {
        const int action_size = game_.board_size * game_.board_size;
        std::vector<float> logits = root.nn_logits;
        if (logits.size() != static_cast<size_t>(action_size)) {
            logits.assign(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        }

        const auto is_legal = game_.get_is_legal_actions(root.state, root.to_play);

        // Gumbel noise
        std::vector<float> g(static_cast<size_t>(action_size), 0.0f);
        {
            std::extreme_value_distribution<float> gumbel_dist(0.0f, 1.0f);
            for (int i = 0; i < action_size; ++i) {
                g[static_cast<size_t>(i)] = gumbel_dist(rng_);
            }
        }

        int m = std::min(num_simulations, cfg_.gumbel_m);
        std::vector<int> sorted_actions(static_cast<size_t>(action_size));
        std::iota(sorted_actions.begin(), sorted_actions.end(), 0);
        std::sort(sorted_actions.begin(), sorted_actions.end(), [&](int a, int b) {
            const float sa = (a < static_cast<int>(is_legal.size()) && is_legal[a])
                ? (logits[a] + g[a]) : -std::numeric_limits<float>::infinity();
            const float sb = (b < static_cast<int>(is_legal.size()) && is_legal[b])
                ? (logits[b] + g[b]) : -std::numeric_limits<float>::infinity();
            return sa > sb;
        });

        std::vector<int> surviving_actions;
        surviving_actions.reserve(static_cast<size_t>(m));
        for (int a : sorted_actions) {
            if (static_cast<int>(surviving_actions.size()) >= m) break;
            if (a < static_cast<int>(is_legal.size()) && is_legal[a]) surviving_actions.push_back(a);
        }

        m = static_cast<int>(surviving_actions.size());
        if (m > 0) {
            const int phases = (m > 1) ? static_cast<int>(std::ceil(std::log2(static_cast<double>(m)))) : 1;
            int sims_budget = num_simulations;

            for (int phase = 0; phase < phases; ++phase) {
                if (sims_budget <= 0 || surviving_actions.empty()) break;
                const int remaining_phases = phases - phase;
                const int sims_this_phase = sims_budget / remaining_phases;
                const int num_actions = static_cast<int>(surviving_actions.size());
                const int sims_per_action = std::max(1, sims_this_phase / std::max(1, num_actions));

                std::vector<int> rollout_actions;
                rollout_actions.reserve(static_cast<size_t>(std::max(1, sims_per_action * num_actions)));
                for (int s = 0; s < sims_per_action && sims_budget > 0; ++s) {
                    for (int action : surviving_actions) {
                        if (sims_budget <= 0) break;
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
                            if (child && child->action_taken == a) { c = child.get(); break; }
                        }
                        float q = 0.5f;
                        if (c && c->n > 0) {
                            const float cw = c->v[0] / static_cast<float>(c->n);
                            const float cl = c->v[2] / static_cast<float>(c->n);
                            q = ((cl - cw) + 1.0f) * 0.5f;
                        }
                        return logits[a] + g[a] + (c_visit + max_n) * c_scale * q;
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
                if (n_values[a] > 0.0f && a < static_cast<int>(root.nn_policy.size())) {
                    const float p = root.nn_policy[a];
                    weighted_q[0] += p * q_wdl[a][0];
                    weighted_q[1] += p * q_wdl[a][1];
                    weighted_q[2] += p * q_wdl[a][2];
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
            const auto& q = (n_values[a] > 0.0f) ? q_wdl[a] : v_mix;
            float s = q[0] - q[2];
            s = (s + 1.0f) * 0.5f;
            sigma_q[a] = (c_visit + max_n) * c_scale * s;
        }

        std::vector<float> improved_logits(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        for (int a = 0; a < action_size; ++a) {
            if (a < static_cast<int>(is_legal.size()) && is_legal[a]) {
                improved_logits[a] = logits[a] + sigma_q[a];
            }
        }
        auto improved_policy = softmax(improved_logits);

        auto final_eval = [&](int a) { return logits[a] + g[a] + sigma_q[a]; };

        int gumbel_action = -1;
        if (!surviving_actions.empty()) {
            float max_n_surviving = -1.0f;
            std::vector<int> most_visited;
            for (int a : surviving_actions) {
                const float nv = n_values[a];
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
                if (ev > best_eval) { best_eval = ev; gumbel_action = most_visited[i]; }
            }
        }

        if (gumbel_action < 0 && !improved_policy.empty()) {
            gumbel_action = static_cast<int>(std::distance(
                improved_policy.begin(), std::max_element(improved_policy.begin(), improved_policy.end())
            ));
        }
        return {improved_policy, gumbel_action, v_mix};
    }

    static void remove_vloss_on_path(const std::vector<MCTSNode*>& path) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            if ((*it)->vloss > 0) (*it)->vloss -= 1;
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

    static float max_child_n(const MCTSNode& root) {
        float mx = 0.0f;
        for (const auto& c : root.children) {
            if (c) mx = std::max(mx, static_cast<float>(c->n));
        }
        return mx;
    }

    Game& game_;
    const AlphaZeroConfig& cfg_;
    int leaf_batch_size_ = 1;
    InferenceFn infer_fn_;
    BatchInferenceFn batch_infer_fn_;
    std::mt19937 rng_;
};

}  // namespace skyzero

#endif
