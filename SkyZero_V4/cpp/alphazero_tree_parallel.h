#ifndef SKYZERO_ALPHAZERO_TREE_PARALLEL_H
#define SKYZERO_ALPHAZERO_TREE_PARALLEL_H

// Shared-tree multi-threaded Gumbel MCTS. KataGo-style: N search threads
// descend a single tree concurrently, using virtual loss for path diversity.
//
// Public interface mirrors ParallelMCTS<Game> exactly (same constructor +
// search() signature) so selfplay_manager can swap backends via std::variant.
//
// Constraints:
//   * Reuses MCTSNode from alphazero.h as-is (no atomic fields added).
//   * Protects each node via a striped mutex pool (256 stripes) keyed by
//     node pointer hash.
//   * Avoids duplicate NN calls on the same leaf via an in-flight expansion
//     map with condition-variable waiting.
//   * Worker thread pool persists across search() calls.

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "alphazero.h"
#include "alphazero_parallel.h"  // SelfplayParallelConfig
#include "utils.h"

namespace skyzero {

template <typename Game>
class TreeParallelMCTS {
public:
    using InferenceFn = std::function<std::pair<std::vector<float>, std::array<float, 3>>(const std::vector<int8_t>&)>;
    using BatchInferenceFn = std::function<
        std::vector<std::pair<std::vector<float>, std::array<float, 3>>>(
            const std::vector<std::vector<int8_t>>&
        )>;

    TreeParallelMCTS(
        Game& game,
        const AlphaZeroConfig& cfg,
        int search_threads_per_tree,
        InferenceFn infer_fn,
        BatchInferenceFn batch_infer_fn,
        uint64_t seed
    )
        : game_(game),
          cfg_(cfg),
          num_threads_(std::max(1, search_threads_per_tree)),
          infer_fn_(std::move(infer_fn)),
          batch_infer_fn_(std::move(batch_infer_fn)),
          rng_(seed) {
        (void)batch_infer_fn_;  // shared-tree submits one leaf per thread
        start_workers();
    }

    ~TreeParallelMCTS() {
        stop_workers();
    }

    TreeParallelMCTS(const TreeParallelMCTS&) = delete;
    TreeParallelMCTS& operator=(const TreeParallelMCTS&) = delete;
    TreeParallelMCTS(TreeParallelMCTS&&) = delete;
    TreeParallelMCTS& operator=(TreeParallelMCTS&&) = delete;

    // Same signature as ParallelMCTS<Game>::search
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
            backpropagate_from_root(root.get(), nn_value_probs);
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
        out.gumbel_phases = std::move(gumbel.phase_survivors);
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
    // ------------------------------------------------------------------
    // Striped mutex pool for per-node state (n, v[], vloss, children read).
    // ------------------------------------------------------------------
    static constexpr size_t kNumStripes = 256;

    std::mutex& node_mutex(const MCTSNode* node) {
        const auto h = std::hash<const MCTSNode*>()(node);
        return stripe_mutexes_[h % kNumStripes];
    }

    // ------------------------------------------------------------------
    // In-flight expansion coordination.
    // When one thread hits an unexpanded leaf, it registers itself here
    // so other threads wait on the same NN call instead of duplicating it.
    // ------------------------------------------------------------------
    struct ExpansionState {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
    };

    std::shared_ptr<ExpansionState> claim_or_wait_expansion(MCTSNode* leaf) {
        std::lock_guard<std::mutex> lk(inflight_mutex_);
        auto it = inflight_.find(leaf);
        if (it != inflight_.end()) {
            return it->second;  // another thread already claimed
        }
        auto state = std::make_shared<ExpansionState>();
        inflight_.emplace(leaf, state);
        return nullptr;  // caller is the winner
    }

    void finish_expansion(MCTSNode* leaf) {
        std::shared_ptr<ExpansionState> state;
        {
            std::lock_guard<std::mutex> lk(inflight_mutex_);
            auto it = inflight_.find(leaf);
            if (it == inflight_.end()) return;
            state = it->second;
            inflight_.erase(it);
        }
        {
            std::lock_guard<std::mutex> lk(state->m);
            state->done = true;
        }
        state->cv.notify_all();
    }

    void wait_expansion(std::shared_ptr<ExpansionState>& state) {
        std::unique_lock<std::mutex> lk(state->m);
        state->cv.wait(lk, [&]() { return state->done; });
    }

    // ------------------------------------------------------------------
    // Worker thread pool
    // ------------------------------------------------------------------
    struct SimTask {
        int root_action = -1;  // which surviving root action to simulate
    };

    void start_workers() {
        for (int i = 0; i < num_threads_; ++i) {
            worker_rng_seeds_.push_back(rng_());
            workers_.emplace_back([this, i]() {
                std::mt19937 local_rng(worker_rng_seeds_[i]);
                worker_loop(local_rng);
            });
        }
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lk(task_mutex_);
            stop_.store(true);
        }
        task_cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
    }

    void worker_loop(std::mt19937& local_rng) {
        while (true) {
            SimTask task;
            {
                std::unique_lock<std::mutex> lk(task_mutex_);
                task_cv_.wait(lk, [&]() { return stop_.load() || !tasks_.empty(); });
                if (stop_.load() && tasks_.empty()) return;
                task = tasks_.front();
                tasks_.pop_front();
            }
            try {
                run_one_simulation(task, local_rng);
            } catch (...) {
                // swallow per-sim exceptions so the tree stays consistent;
                // the leaf's virtual loss was already removed by the failure path
            }
            {
                std::lock_guard<std::mutex> lk(done_mutex_);
                tasks_outstanding_ -= 1;
                if (tasks_outstanding_ == 0) done_cv_.notify_all();
            }
        }
    }

    // Main-thread: enqueue N tasks and wait until all finish.
    void submit_and_wait(const std::vector<SimTask>& batch) {
        if (batch.empty()) return;
        {
            std::lock_guard<std::mutex> lk(done_mutex_);
            tasks_outstanding_ += static_cast<int>(batch.size());
        }
        {
            std::lock_guard<std::mutex> lk(task_mutex_);
            for (const auto& t : batch) tasks_.push_back(t);
        }
        task_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(done_mutex_);
            done_cv_.wait(lk, [&]() { return tasks_outstanding_ == 0; });
        }
    }

    // ------------------------------------------------------------------
    // Inference path (per leaf, no batching — the external inference
    // server already batches across all concurrent threads).
    // ------------------------------------------------------------------
    struct InferenceResult {
        std::vector<float> policy;
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        std::vector<float> masked_logits;
    };

    InferenceResult inference(
        const std::vector<int8_t>& state,
        int to_play,
        bool use_stochastic_transform,
        std::mt19937* local_rng
    ) {
        auto encoded = game_.encode_state(state, to_play);
        int k = 0;
        bool do_flip = false;
        if (use_stochastic_transform && local_rng != nullptr) {
            std::uniform_int_distribution<int> dist(0, 7);
            const int t = dist(*local_rng);
            k = t % 4;
            do_flip = t >= 4;
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

    // Expand a node under the caller's ownership of its stripe mutex.
    // REQUIRES: caller already locked node_mutex(node) AND node is unexpanded.
    void expand_with_locked(const InferenceResult& ir, MCTSNode& node) {
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
            /*local_rng=*/nullptr
        );
        std::lock_guard<std::mutex> lk(node_mutex(&node));
        expand_with_locked(ir, node);
        return {ir.policy, ir.value};
    }

    // ------------------------------------------------------------------
    // Selection — PUCT + FPU + virtual loss, reads node state under its
    // stripe mutex to get a consistent Q snapshot.
    // ------------------------------------------------------------------
    MCTSNode* select_child(MCTSNode& node) {
        // Snapshot parent counters (need the mutex for coherent n + v[] read).
        int parent_n, parent_vloss;
        std::array<float, 3> parent_v;
        std::array<float, 3> parent_nn_value;
        {
            std::lock_guard<std::mutex> lk(node_mutex(&node));
            parent_n = node.n;
            parent_vloss = node.vloss;
            parent_v = node.v;
            parent_nn_value = node.nn_value_probs;
        }
        // children is stable once populated (we only append during expand,
        // which happens once). Iteration without lock is safe.
        float visited_policy_mass = 0.0f;
        for (auto& child_ptr : node.children) {
            MCTSNode* c = child_ptr.get();
            std::lock_guard<std::mutex> lk(node_mutex(c));
            if (c->n > 0 || c->vloss > 0) {
                visited_policy_mass += c->prior;
            }
        }

        // Recompute compute_select_params using snapshot (we cannot safely
        // pass the raw node reference because we no longer hold its mutex).
        MCTSNode snap;
        snap.n = parent_n;
        snap.v = parent_v;
        snap.nn_value_probs = parent_nn_value;
        snap.parent = node.parent;
        const int effective_parent_n = parent_n + parent_vloss;
        const auto sp = compute_select_params(snap, effective_parent_n, visited_policy_mass, cfg_);

        float best_score = -std::numeric_limits<float>::infinity();
        MCTSNode* best_child = nullptr;
        for (auto& child_ptr : node.children) {
            MCTSNode* c = child_ptr.get();
            int cn, cv_loss;
            std::array<float, 3> cv;
            {
                std::lock_guard<std::mutex> lk(node_mutex(c));
                cn = c->n;
                cv_loss = c->vloss;
                cv = c->v;
            }
            const int eff = cn + cv_loss;
            float q = sp.fpu_value;
            if (eff > 0) {
                const float util_sum = (cv[2] - cv[0]) - static_cast<float>(cv_loss);
                q = util_sum / static_cast<float>(eff);
            }
            const float u = sp.explore_scaling * c->prior / (1.0f + static_cast<float>(eff));
            const float score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = c;
            }
        }
        return best_child;
    }

    void add_vloss(MCTSNode* n) {
        std::lock_guard<std::mutex> lk(node_mutex(n));
        n->vloss += 1;
    }

    void remove_vloss_path(const std::vector<MCTSNode*>& path) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            std::lock_guard<std::mutex> lk(node_mutex(*it));
            if ((*it)->vloss > 0) (*it)->vloss -= 1;
        }
    }

    void backprop_path_with_vloss(
        const std::vector<MCTSNode*>& path, std::array<float, 3> value
    ) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            {
                std::lock_guard<std::mutex> lk(node_mutex(*it));
                (*it)->update(value);
                if ((*it)->vloss > 0) (*it)->vloss -= 1;
            }
            value = flip_wdl(value);
        }
    }

    // Backprop from root after the initial root_expand (no vloss on path).
    void backpropagate_from_root(MCTSNode* root, std::array<float, 3> value) {
        std::lock_guard<std::mutex> lk(node_mutex(root));
        root->update(value);
    }

    // ------------------------------------------------------------------
    // Per-simulation logic. Called by worker threads.
    // ------------------------------------------------------------------
    void run_one_simulation(const SimTask& task, std::mt19937& local_rng) {
        MCTSNode* root = current_root_;
        if (root == nullptr) return;

        // Pick the fixed root child corresponding to task.root_action.
        MCTSNode* child = nullptr;
        for (auto& c : root->children) {
            if (c && c->action_taken == task.root_action) { child = c.get(); break; }
        }
        if (child == nullptr) return;

        std::vector<MCTSNode*> path;
        path.reserve(64);
        add_vloss(root);
        path.push_back(root);
        add_vloss(child);
        path.push_back(child);

        MCTSNode* node = child;
        while (true) {
            // Check expansion under the node's mutex.
            bool expanded;
            {
                std::lock_guard<std::mutex> lk(node_mutex(node));
                expanded = node->is_expanded();
            }
            if (!expanded) break;

            MCTSNode* next = select_child(*node);
            if (next == nullptr) {
                remove_vloss_path(path);
                return;
            }
            add_vloss(next);
            path.push_back(next);
            node = next;
        }

        // `node` is now an unexpanded leaf (or terminal).
        if (game_.is_terminal(node->state, node->action_taken, -node->to_play)) {
            std::array<float, 3> value{0.0f, 1.0f, 0.0f};
            const int result = game_.get_winner(node->state, node->action_taken, -node->to_play) * node->to_play;
            if (result == 1) value = {1.0f, 0.0f, 0.0f};
            else if (result == -1) value = {0.0f, 0.0f, 1.0f};
            backprop_path_with_vloss(path, value);
            return;
        }

        // Expansion: claim or wait.
        auto existing = claim_or_wait_expansion(node);
        if (existing) {
            // Another thread is expanding this leaf. Wait for it.
            wait_expansion(existing);
            // After expansion: backpropagate using the node's NN value
            // (stable after expand finishes).
            std::array<float, 3> value;
            {
                std::lock_guard<std::mutex> lk(node_mutex(node));
                value = node->nn_value_probs;
            }
            backprop_path_with_vloss(path, value);
            return;
        }

        // We are the winner: do the NN call (no stripe lock held).
        InferenceResult ir;
        try {
            ir = inference(
                node->state, node->to_play,
                cfg_.enable_stochastic_transform_inference_for_child,
                &local_rng
            );
        } catch (...) {
            finish_expansion(node);
            remove_vloss_path(path);
            throw;
        }

        // Populate children under the node's stripe mutex.
        {
            std::lock_guard<std::mutex> lk(node_mutex(node));
            if (!node->is_expanded()) {
                expand_with_locked(ir, *node);
            }
        }
        finish_expansion(node);
        backprop_path_with_vloss(path, ir.value);
    }

    // ------------------------------------------------------------------
    // Gumbel Sequential Halving — main-thread driver.
    //   Per phase: build task batch (sims_per_action × num_surviving),
    //   dispatch to workers, wait, then halve surviving actions.
    // ------------------------------------------------------------------
    struct GumbelResult {
        std::vector<float> improved_policy;
        int gumbel_action = -1;
        std::array<float, 3> v_mix{0.0f, 1.0f, 0.0f};
        std::vector<std::vector<int>> phase_survivors;  // snapshots: [0]=initial m, then after each halving
    };

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations) {
        const int action_size = game_.board_size * game_.board_size;
        std::vector<float> logits = root.nn_logits;
        if (logits.size() != static_cast<size_t>(action_size)) {
            logits.assign(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        }
        const auto is_legal = game_.get_is_legal_actions(root.state, root.to_play);

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

        current_root_ = &root;

        std::vector<std::vector<int>> phase_survivors;
        if (m > 0) phase_survivors.push_back(surviving_actions);
        if (m > 0) {
            const int phases = (m > 1) ? static_cast<int>(std::ceil(std::log2(static_cast<double>(m)))) : 1;
            int sims_budget = num_simulations;

            for (int phase = 0; phase < phases; ++phase) {
                if (sims_budget <= 0 || surviving_actions.empty()) break;
                const int remaining_phases = phases - phase;
                const int sims_this_phase = sims_budget / remaining_phases;
                const int num_actions = static_cast<int>(surviving_actions.size());
                const int sims_per_action = std::max(1, sims_this_phase / std::max(1, num_actions));

                std::vector<SimTask> batch;
                batch.reserve(static_cast<size_t>(sims_per_action * num_actions));
                for (int s = 0; s < sims_per_action; ++s) {
                    for (int action : surviving_actions) {
                        if (static_cast<int>(batch.size()) >= sims_budget) break;
                        batch.push_back({action});
                    }
                    if (static_cast<int>(batch.size()) >= sims_budget) break;
                }

                submit_and_wait(batch);
                sims_budget -= static_cast<int>(batch.size());

                if (phase < phases - 1 && surviving_actions.size() > 1) {
                    const float max_n = max_child_n(root);
                    const float c_visit = cfg_.gumbel_c_visit;
                    const float c_scale = cfg_.gumbel_c_scale;

                    auto eval_action = [&](int a) {
                        MCTSNode* c = nullptr;
                        for (auto& child : root.children) {
                            if (child && child->action_taken == a) { c = child.get(); break; }
                        }
                        float q = 0.5f;
                        if (c) {
                            std::lock_guard<std::mutex> lk(node_mutex(c));
                            if (c->n > 0) {
                                const float cw = c->v[0] / static_cast<float>(c->n);
                                const float cl = c->v[2] / static_cast<float>(c->n);
                                q = ((cl - cw) + 1.0f) * 0.5f;
                            }
                        }
                        return logits[a] + g[a] + (c_visit + max_n) * c_scale * q;
                    };

                    std::sort(surviving_actions.begin(), surviving_actions.end(), [&](int a, int b) {
                        return eval_action(a) > eval_action(b);
                    });
                    surviving_actions.resize(static_cast<size_t>(
                        std::max(1, static_cast<int>(surviving_actions.size()) / 2)));
                    phase_survivors.push_back(surviving_actions);
                }
            }
        }

        current_root_ = nullptr;

        // --- Compute v_mix / improved_policy / gumbel_action (same as ParallelMCTS) ---
        const float c_visit = cfg_.gumbel_c_visit;
        const float c_scale = cfg_.gumbel_c_scale;
        const float max_n = max_child_n(root);

        std::vector<std::array<float, 3>> q_wdl(static_cast<size_t>(action_size), {0.0f, 0.0f, 0.0f});
        std::vector<float> n_values(static_cast<size_t>(action_size), 0.0f);
        for (auto& c : root.children) {
            if (!c) continue;
            std::lock_guard<std::mutex> lk(node_mutex(c.get()));
            if (c->n > 0) {
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
        if (gumbel_action >= 0) phase_survivors.push_back({gumbel_action});
        return {improved_policy, gumbel_action, v_mix, std::move(phase_survivors)};
    }

    float max_child_n(const MCTSNode& root) {
        float mx = 0.0f;
        for (const auto& c : root.children) {
            if (!c) continue;
            std::lock_guard<std::mutex> lk(node_mutex(c.get()));
            mx = std::max(mx, static_cast<float>(c->n));
        }
        return mx;
    }

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------
    Game& game_;
    const AlphaZeroConfig& cfg_;
    int num_threads_;
    InferenceFn infer_fn_;
    BatchInferenceFn batch_infer_fn_;
    std::mt19937 rng_;

    std::array<std::mutex, kNumStripes> stripe_mutexes_;

    std::mutex inflight_mutex_;
    std::unordered_map<MCTSNode*, std::shared_ptr<ExpansionState>> inflight_;

    // Worker pool
    std::vector<std::thread> workers_;
    std::vector<uint64_t> worker_rng_seeds_;
    std::atomic<bool> stop_{false};

    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::deque<SimTask> tasks_;

    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    int tasks_outstanding_ = 0;

    // Shared by main-thread during a search() call, read by workers.
    // Only set before submit_and_wait and cleared after, so no races.
    MCTSNode* current_root_ = nullptr;
};

}  // namespace skyzero

#endif
