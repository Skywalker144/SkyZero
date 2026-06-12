#ifndef SKYZERO_ALPHAZERO_TREE_PARALLEL_H
#define SKYZERO_ALPHAZERO_TREE_PARALLEL_H

// Shared-tree multi-threaded Gumbel MCTS. KataGo-style: N search threads
// descend a single tree concurrently, using virtual loss for path diversity.
//
// Public interface mirrors ParallelMCTS<Game> exactly (same constructor +
// search() signature) so selfplay_manager can swap backends via std::variant.
//
// Constraints:
//   * Reuses MCTSNode from skyzero.h as-is (no atomic fields added).
//   * Protects each node via a striped mutex pool (256 stripes) keyed by
//     node pointer hash.
//   * Avoids duplicate NN calls on the same leaf via an in-flight expansion
//     map with condition-variable waiting.
//   * Worker thread pool persists across search() calls.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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

#include "skyzero.h"
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

    // Legacy constructor: per-leaf inference via single-state callback. Used by
    // gomoku_elo / gomoku_ab / mcts_bench. NN forward is whatever the callback
    // does (typically batch=1 under a mutex), so throughput caps at one forward
    // at a time.
    TreeParallelMCTS(
        Game& game,
        const SkyZeroConfig& cfg,
        int search_threads_per_tree,
        InferenceFn infer_fn,
        uint64_t seed
    )
        : game_(game),
          cfg_(cfg),
          num_threads_(std::max(1, search_threads_per_tree)),
          infer_fn_(std::move(infer_fn)),
          rng_(seed) {
        start_workers();
    }

    // Batched constructor: workers submit encoded states to a queue, a single
    // batcher thread accumulates up to `leaf_batch_size` requests (with a
    // `batch_timeout_us` cap) and runs them through `batch_infer_fn` in one
    // forward call. Lets a single GPU saturate.
    TreeParallelMCTS(
        Game& game,
        const SkyZeroConfig& cfg,
        int search_threads_per_tree,
        int leaf_batch_size,
        int batch_timeout_us,
        BatchInferenceFn batch_infer_fn,
        uint64_t seed
    )
        : game_(game),
          cfg_(cfg),
          num_threads_(std::max(1, search_threads_per_tree)),
          batch_infer_fn_(std::move(batch_infer_fn)),
          leaf_batch_size_(std::max(1, leaf_batch_size)),
          batch_timeout_us_(std::max(0, batch_timeout_us)),
          rng_(seed) {
        start_batcher();
        start_workers();
    }

    ~TreeParallelMCTS() {
        stop_workers();
        stop_batcher();
    }

    TreeParallelMCTS(const TreeParallelMCTS&) = delete;
    TreeParallelMCTS& operator=(const TreeParallelMCTS&) = delete;
    TreeParallelMCTS(TreeParallelMCTS&&) = delete;
    TreeParallelMCTS& operator=(TreeParallelMCTS&&) = delete;

    // Same signature as ParallelMCTS<Game>::search.
    // fast_search: this move is a fastSearch (KataGo cheap search) — the
    // caller already capped num_simulations; here it disables root
    // exploration (Gumbel noise / Dirichlet noise / forced playouts / root
    // FPU & policy temperature) for this one search.
    MCTSSearchOutput search(
        const std::vector<int8_t>& state,
        int to_play,
        int num_simulations,
        std::unique_ptr<MCTSNode>& root,
        bool fast_search = false
    ) {
        fast_search_ = fast_search;
        if (!root) {
            root.reset(new MCTSNode(state, to_play));
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
        MCTSSearchOutput out;
        out.nn_policy = std::move(nn_policy);
        out.nn_value_probs = nn_value_probs;
        if (cfg_.root_search_algo == SkyZeroConfig::RootSearchAlgo::kPuct) {
            puct_root_search(*root, num_simulations, out);
        } else {
            auto gumbel = gumbel_sequential_halving(*root, num_simulations);
            out.mcts_policy = std::move(gumbel.improved_policy);
            out.v_mix = gumbel.v_mix;
            out.gumbel_action = gumbel.gumbel_action;
            out.gumbel_phases = std::move(gumbel.phase_survivors);
            out.root_child_wdl = std::move(gumbel.q_wdl);
        }
        {
            const int action_size = Game::MAX_AREA;
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

    // ---- Analysis / ponder API ------------------------------------------
    // search() above runs a fixed Gumbel sequential-halving budget and is what
    // *selects* moves. The methods below instead let a caller accumulate plain
    // PUCT simulations on a persistent tree and read the current estimate
    // between chunks — i.e. KataGo-style ponder until the user stops. Play uses
    // search()/Gumbel; analysis uses these.
    void ensure_root_expanded(MCTSNode& root) {
        if (!root.is_expanded()) {
            auto pair = root_expand(root);
            backpropagate_from_root(&root, pair.second);
        }
    }

    // Run `n` PUCT-root simulations, accumulating into the existing tree.
    void run_puct_sims(MCTSNode& root, int n) {
        if (n <= 0) return;
        fast_search_ = false;  // analysis always uses the full root settings
        current_root_ = &root;
        std::vector<SimTask> batch(static_cast<size_t>(n));
        for (auto& t : batch) t.root_action = -1;  // -1 => select root child by PUCT
        submit_and_wait(batch);
        current_root_ = nullptr;
    }

    // Snapshot the current tree as an MCTSSearchOutput: visit distribution,
    // per-child root-perspective WDL, root value (visit-weighted), and synthetic
    // Gumbel phases ([all visited, best]) so the existing front-end overlay
    // renders the analysis unchanged.
    MCTSSearchOutput report_analysis(MCTSNode& root) {
        const int action_size = Game::MAX_AREA;
        MCTSSearchOutput out;
        out.nn_policy = root.nn_policy;
        out.nn_value_probs = root.nn_value_probs;
        out.visit_counts.assign(static_cast<size_t>(action_size), 0.0f);
        out.root_child_wdl.assign(static_cast<size_t>(action_size), {0.0f, 0.0f, 0.0f});
        std::vector<float> n_values(static_cast<size_t>(action_size), 0.0f);
        for (auto& c : root.children) {
            if (!c) continue;
            std::lock_guard<std::mutex> lk(node_mutex(c.get()));
            const int a = c->action_taken;
            if (a < 0 || a >= action_size || c->n <= 0) continue;
            const float cw = c->v[0] / static_cast<float>(c->n);
            const float cd = c->v[1] / static_cast<float>(c->n);
            const float cl = c->v[2] / static_cast<float>(c->n);
            out.root_child_wdl[static_cast<size_t>(a)] = {cl, cd, cw};  // root-perspective W,D,L
            n_values[static_cast<size_t>(a)] = static_cast<float>(c->n);
            out.visit_counts[static_cast<size_t>(a)] = static_cast<float>(c->n);
        }
        const float sum_n = std::accumulate(n_values.begin(), n_values.end(), 0.0f);
        out.mcts_policy.assign(static_cast<size_t>(action_size), 0.0f);
        std::array<float, 3> vroot = root.nn_value_probs;
        if (sum_n > 0.0f) {
            std::array<float, 3> acc{0.0f, 0.0f, 0.0f};
            for (int a = 0; a < action_size; ++a) {
                if (n_values[a] <= 0.0f) continue;
                out.mcts_policy[static_cast<size_t>(a)] = n_values[a] / sum_n;
                acc[0] += n_values[a] * out.root_child_wdl[static_cast<size_t>(a)][0];
                acc[1] += n_values[a] * out.root_child_wdl[static_cast<size_t>(a)][1];
                acc[2] += n_values[a] * out.root_child_wdl[static_cast<size_t>(a)][2];
            }
            vroot = {acc[0] / sum_n, acc[1] / sum_n, acc[2] / sum_n};
        }
        out.v_mix = vroot;
        std::vector<int> visited;
        int best = -1;
        float best_n = -1.0f;
        for (int a = 0; a < action_size; ++a) {
            if (n_values[a] <= 0.0f) continue;
            visited.push_back(a);
            if (n_values[a] > best_n) { best_n = n_values[a]; best = a; }
        }
        if (!visited.empty()) {
            out.gumbel_phases.push_back(std::move(visited));
            if (best >= 0) out.gumbel_phases.push_back(std::vector<int>{best});
        }
        out.gumbel_action = best;
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
            // Seed by value: capturing a reference into a member vector races
            // with later push_back reallocations (worker threads start reading
            // while the constructor is still appending).
            const uint64_t seed = rng_();
            workers_.emplace_back([this, seed]() {
                std::mt19937 local_rng(seed);
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
    // Inference path.
    //   * Batched mode (batch_infer_fn_ set): worker submits the encoded
    //     state to infer_queue_ and blocks on a per-request cv. A single
    //     batcher thread accumulates up to leaf_batch_size_ requests with
    //     batch_timeout_us_ slack, runs one batched forward, and notifies
    //     all waiters. This is what saturates the GPU.
    //   * Legacy mode (infer_fn_ set): worker calls infer_fn_ directly,
    //     batch=1 forward serialized by whatever mutex the callback owns.
    // ------------------------------------------------------------------
    struct InferenceResult {
        std::vector<float> policy;
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        std::vector<float> masked_logits;
    };

    struct InferRequest {
        std::vector<int8_t> encoded;
        std::vector<float> policy;
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};
        bool done = false;
        bool failed = false;
        std::mutex m;
        std::condition_variable cv;
    };

    InferenceResult inference(
        const std::vector<int8_t>& state,
        int to_play,
        bool use_stochastic_transform,
        std::mt19937* local_rng,
        bool is_root
    ) {
        auto encoded = game_.encode_state_v5(state, to_play);   // V5: 5-plane padded
        int k = 0;
        bool do_flip = false;
        if (use_stochastic_transform && local_rng != nullptr) {
            std::uniform_int_distribution<int> dist(0, 7);
            const int t = dist(*local_rng);
            k = t % 4;
            do_flip = t >= 4;
            encoded = transform_encoded_state(encoded, game_.num_planes, Game::MAX_BOARD_SIZE, k, do_flip);
        }

        std::vector<float> logits;
        std::array<float, 3> value{0.0f, 1.0f, 0.0f};

        if (batch_infer_fn_) {
            auto req = std::make_shared<InferRequest>();
            req->encoded = std::move(encoded);
            {
                std::lock_guard<std::mutex> lk(queue_mu_);
                infer_queue_.push_back(req);
            }
            queue_cv_.notify_one();
            {
                std::unique_lock<std::mutex> lk(req->m);
                req->cv.wait(lk, [&]() { return req->done; });
            }
            if (req->failed) throw std::runtime_error("batched inference failed");
            logits = std::move(req->policy);
            value = req->value;
        } else {
            auto pair = infer_fn_(encoded);
            logits = std::move(pair.first);
            value = pair.second;
        }

        if (use_stochastic_transform) {
            logits = undo_transform_flat(logits, Game::MAX_BOARD_SIZE, k, do_flip);
        }

        const auto legal = (is_root && cfg_.root_symmetry_pruning)
            ? game_.get_canonical_legal_actions_canvas(state, to_play)
            : game_.get_is_legal_actions_canvas(state, to_play);
        for (size_t i = 0; i < logits.size(); ++i) {
            if (i >= legal.size() || !legal[i]) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
        return {softmax(logits), value, logits};
    }

    // ------------------------------------------------------------------
    // Batcher thread: accumulates requests up to leaf_batch_size_ or
    // batch_timeout_us_ and runs one batched NN forward.
    // ------------------------------------------------------------------
    void start_batcher() {
        if (!batch_infer_fn_) return;
        batcher_thread_ = std::thread([this]() { batcher_loop(); });
    }

    void stop_batcher() {
        if (!batcher_thread_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            batcher_stop_.store(true);
        }
        queue_cv_.notify_all();
        batcher_thread_.join();
    }

    void batcher_loop() {
        while (true) {
            std::vector<std::shared_ptr<InferRequest>> batch;
            {
                std::unique_lock<std::mutex> lk(queue_mu_);
                queue_cv_.wait(lk, [&]() {
                    return batcher_stop_.load() || !infer_queue_.empty();
                });
                if (batcher_stop_.load() && infer_queue_.empty()) return;

                // Try to accumulate up to leaf_batch_size_ within the timeout.
                if (batch_timeout_us_ > 0 &&
                    static_cast<int>(infer_queue_.size()) < leaf_batch_size_) {
                    queue_cv_.wait_for(
                        lk, std::chrono::microseconds(batch_timeout_us_),
                        [&]() {
                            return batcher_stop_.load() ||
                                static_cast<int>(infer_queue_.size()) >= leaf_batch_size_;
                        });
                }

                const int take = std::min(
                    static_cast<int>(infer_queue_.size()), leaf_batch_size_);
                for (int i = 0; i < take; ++i) {
                    batch.push_back(std::move(infer_queue_.front()));
                    infer_queue_.pop_front();
                }
            }

            if (batch.empty()) continue;

            std::vector<std::vector<int8_t>> encoded_batch;
            encoded_batch.reserve(batch.size());
            for (auto& req : batch) {
                encoded_batch.push_back(std::move(req->encoded));
            }

            try {
                auto results = batch_infer_fn_(encoded_batch);
                if (results.size() != batch.size()) {
                    throw std::runtime_error("batch_infer_fn size mismatch");
                }
                for (size_t i = 0; i < batch.size(); ++i) {
                    {
                        std::lock_guard<std::mutex> lk(batch[i]->m);
                        batch[i]->policy = std::move(results[i].first);
                        batch[i]->value = results[i].second;
                        batch[i]->done = true;
                    }
                    batch[i]->cv.notify_one();
                }
            } catch (...) {
                for (auto& req : batch) {
                    {
                        std::lock_guard<std::mutex> lk(req->m);
                        req->failed = true;
                        req->done = true;
                    }
                    req->cv.notify_one();
                }
            }
        }
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
            auto child = std::unique_ptr<MCTSNode>(new MCTSNode(
                game_.get_next_state_canvas(node.state, a, node.to_play),
                -node.to_play,
                p,
                &node,
                a
            ));
            node.children.push_back(std::move(child));
        }
    }

    std::pair<std::vector<float>, std::array<float, 3>> root_expand(MCTSNode& node) {
        const auto ir = inference(
            node.state, node.to_play,
            cfg_.enable_stochastic_transform_inference_for_root,
            /*local_rng=*/nullptr,
            /*is_root=*/true
        );
        std::lock_guard<std::mutex> lk(node_mutex(&node));
        expand_with_locked(ir, node);
        return {ir.policy, ir.value};
    }

    // ------------------------------------------------------------------
    // Gumbel deterministic non-root selection (eq. 14). Mirrors
    // select_child's striped-mutex snapshotting: each child's (n, vloss, v) is
    // read under its own stripe; nn_logits / nn_value_probs are immutable once
    // the node is expanded. N(a) uses effective counts so virtual loss
    // diversifies concurrent descents, exactly as PUCT does here.
    // ------------------------------------------------------------------
    MCTSNode* gumbel_select_child(MCTSNode& node) {
        std::array<float, 3> node_nn_value;
        {
            std::lock_guard<std::mutex> lk(node_mutex(&node));
            node_nn_value = node.nn_value_probs;
        }
        const auto& logits = node.nn_logits;  // stable after expand
        std::vector<GumbelChildStat> stats;
        stats.reserve(node.children.size());
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
            float u = 0.0f;
            if (eff > 0) {
                u = ((cv[2] - cv[0]) - static_cast<float>(cv_loss)) / static_cast<float>(eff);
            }
            const float lg = (c->action_taken >= 0 && c->action_taken < static_cast<int>(logits.size()))
                ? logits[static_cast<size_t>(c->action_taken)] : 0.0f;
            stats.push_back({c->prior, eff, u, lg});
        }
        const int idx = gumbel_deterministic_select(
            stats, wdl_utility(node_nn_value), cfg_.gumbel_c_visit, cfg_.gumbel_c_scale);
        return (idx >= 0) ? node.children[static_cast<size_t>(idx)].get() : nullptr;
    }

    // ------------------------------------------------------------------
    // Selection — PUCT + FPU + virtual loss, reads node state under its
    // stripe mutex to get a consistent Q snapshot.
    // ------------------------------------------------------------------
    MCTSNode* select_child(MCTSNode& node) {
        if (cfg_.non_root_search_algo == SkyZeroConfig::NonRootSearchAlgo::kGumbel) {
            return gumbel_select_child(node);
        }
        return puct_select_child(node, /*is_root=*/false);
    }

    // PUCT + FPU + virtual-loss selection: the default non-root policy, also
    // used to drive the root for kPuct root search and during analysis/ponder
    // (forced PUCT regardless of non_root_search_algo). is_root switches to
    // root FPU and enables KataGo forced playouts
    // (searchexplorehelpers.cpp:166-169) when configured.
    MCTSNode* puct_select_child(MCTSNode& node, bool is_root) {
        // Snapshot parent counters (need the mutex for coherent n + v[] read).
        int parent_n, parent_vloss;
        float parent_q_sum_sq;
        std::array<float, 3> parent_v;
        std::array<float, 3> parent_nn_value;
        {
            std::lock_guard<std::mutex> lk(node_mutex(&node));
            parent_n = node.n;
            parent_vloss = node.vloss;
            parent_q_sum_sq = node.q_sum_sq;
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
        snap.q_sum_sq = parent_q_sum_sq;
        snap.nn_value_probs = parent_nn_value;
        snap.parent = node.parent;
        const int effective_parent_n = parent_n + parent_vloss;
        // fastSearch: root FPU falls back to the non-root FPU and forced
        // playouts are off (KataGo play.cpp:1201-1203).
        const auto sp = compute_select_params(
            snap, effective_parent_n, visited_policy_mass, cfg_, is_root && !fast_search_);

        const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));
        const bool forced_enabled = is_root && !fast_search_
            && cfg_.root_desired_per_child_visits_coeff > 0.0f;

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
            float score;
            if (forced_enabled && c->prior > 0.0f
                && static_cast<float>(eff) < std::sqrt(
                    c->prior * total_child_weight * cfg_.root_desired_per_child_visits_coeff)) {
                score = 1e20f;
            } else {
                float q = sp.fpu_value;
                if (eff > 0) {
                    const float util_sum = (cv[2] - cv[0]) - static_cast<float>(cv_loss);
                    q = util_sum / static_cast<float>(eff);
                }
                const float u = sp.explore_scaling * c->prior / (1.0f + static_cast<float>(eff));
                score = q + u;
            }
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

        std::vector<MCTSNode*> path;
        path.reserve(64);
        add_vloss(root);
        path.push_back(root);

        // root_action >= 0: Gumbel SH dictates which root child to simulate.
        // root_action < 0: PUCT root (kPuct search or analysis/ponder) —
        // select the root child by PUCT so the tree deepens like a normal
        // AlphaZero search.
        MCTSNode* child = nullptr;
        if (task.root_action < 0) {
            child = puct_select_child(*root, /*is_root=*/true);
        } else {
            for (auto& c : root->children) {
                if (c && c->action_taken == task.root_action) { child = c.get(); break; }
            }
        }
        if (child == nullptr) { remove_vloss_path(path); return; }
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
        if (game_.is_terminal_canvas(node->state, node->action_taken, -node->to_play)) {
            std::array<float, 3> value{0.0f, 1.0f, 0.0f};
            const int result = game_.get_winner_canvas(node->state, node->action_taken, -node->to_play) * node->to_play;
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
                &local_rng,
                /*is_root=*/false
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
        std::vector<std::array<float, 3>> q_wdl;        // per-action root-perspective mean WDL ({0,0,0} if unvisited)
    };

    GumbelResult gumbel_sequential_halving(MCTSNode& root, int num_simulations) {
        const int action_size = Game::MAX_AREA;
        std::vector<float> logits = root.nn_logits;
        if (logits.size() != static_cast<size_t>(action_size)) {
            logits.assign(static_cast<size_t>(action_size), -std::numeric_limits<float>::infinity());
        }
        // Derive legal mask from masked logits so root_symmetry_pruning (which
        // sets non-canonical positions to -inf during root_expand) is honored
        // here too, and so this mask never disagrees with what expand_with
        // actually built children for.
        std::vector<uint8_t> is_legal(static_cast<size_t>(action_size), 0);
        for (int a = 0; a < action_size && a < static_cast<int>(logits.size()); ++a) {
            if (logits[a] > -std::numeric_limits<float>::infinity()) {
                is_legal[a] = 1;
            }
        }

        std::vector<float> g(static_cast<size_t>(action_size), 0.0f);
        if (cfg_.gumbel_noise_enabled && !fast_search_) {
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
        return {improved_policy, gumbel_action, v_mix, std::move(phase_survivors), std::move(q_wdl)};
    }

    // KataGo-style PUCT root: noise/temperature the root priors (main thread,
    // workers idle between search() calls), run the whole budget as
    // root_action=-1 simulations, then build the pruned-visit policy target
    // and temperature-sample the move via the shared helpers.
    void puct_root_search(MCTSNode& root, int num_simulations, MCTSSearchOutput& out) {
        const int action_size = Game::MAX_AREA;
        const int turn_number = count_stones(root.state);
        const int board_area = game_.board_size * game_.board_size;
        if (!fast_search_) {
            apply_root_policy_noise_and_temperature(root, turn_number, board_area, cfg_, rng_);
        }

        current_root_ = &root;
        std::vector<SimTask> batch(static_cast<size_t>(std::max(0, num_simulations)));
        for (auto& t : batch) t.root_action = -1;
        submit_and_wait(batch);
        current_root_ = nullptr;

        std::vector<PuctRootChildStat> stats;
        stats.reserve(root.children.size());
        std::vector<std::array<float, 3>> q_wdl(
            static_cast<size_t>(action_size), {0.0f, 0.0f, 0.0f});
        float visited_policy_mass = 0.0f;
        for (auto& child_ptr : root.children) {
            MCTSNode* c = child_ptr.get();
            if (!c) continue;
            PuctRootChildStat s;
            s.action = c->action_taken;
            s.prior = c->prior;
            {
                std::lock_guard<std::mutex> lk(node_mutex(c));
                s.n = c->n;
                if (c->n > 0) {
                    const float cw = c->v[0] / static_cast<float>(c->n);
                    const float cd = c->v[1] / static_cast<float>(c->n);
                    const float cl = c->v[2] / static_cast<float>(c->n);
                    s.q = cl - cw;            // root-perspective W−L
                    s.wdl = {cl, cd, cw};     // root-perspective W,D,L
                }
            }
            if (s.n > 0) {
                if (s.action >= 0 && s.action < action_size) {
                    q_wdl[static_cast<size_t>(s.action)] = s.wdl;
                }
                visited_policy_mass += s.prior;
            }
            stats.push_back(s);
        }

        MCTSNode snap;
        {
            std::lock_guard<std::mutex> lk(node_mutex(&root));
            snap.n = root.n;
            snap.v = root.v;
            snap.q_sum_sq = root.q_sum_sq;
            snap.nn_value_probs = root.nn_value_probs;
            snap.parent = root.parent;
        }
        const auto sp = compute_select_params(
            snap, snap.n, visited_policy_mass, cfg_, /*is_root=*/!fast_search_);
        auto pr = puct_root_assemble(
            stats, action_size, sp.explore_scaling, turn_number, board_area, cfg_, rng_);

        out.mcts_policy = std::move(pr.target_policy);
        out.gumbel_action = pr.chosen_action;
        out.gumbel_phases = std::move(pr.phases);
        out.root_child_wdl = std::move(q_wdl);
        // Root value: average over all backups through the root, which already
        // blends the root NN value (backpropagated once at expansion) with the
        // child returns — same construction as the Gumbel v_mix.
        out.v_mix = snap.nn_value_probs;
        if (snap.n > 0) {
            out.v_mix = {
                snap.v[0] / static_cast<float>(snap.n),
                snap.v[1] / static_cast<float>(snap.n),
                snap.v[2] / static_cast<float>(snap.n),
            };
        }
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
    const SkyZeroConfig& cfg_;
    int num_threads_;
    InferenceFn infer_fn_;
    BatchInferenceFn batch_infer_fn_;
    int leaf_batch_size_ = 1;
    int batch_timeout_us_ = 0;
    std::mt19937 rng_;

    // Batcher: queue + cv + thread. Only populated when batch_infer_fn_ is set.
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<InferRequest>> infer_queue_;
    std::atomic<bool> batcher_stop_{false};
    std::thread batcher_thread_;

    std::array<std::mutex, kNumStripes> stripe_mutexes_;

    std::mutex inflight_mutex_;
    std::unordered_map<MCTSNode*, std::shared_ptr<ExpansionState>> inflight_;

    // Worker pool
    std::vector<std::thread> workers_;
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
    // Same write-before-submit discipline as current_root_: set at the top of
    // search() / run_puct_sims, read by workers in puct_select_child.
    bool fast_search_ = false;
};

}  // namespace skyzero

#endif
