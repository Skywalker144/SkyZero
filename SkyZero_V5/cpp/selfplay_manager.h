#ifndef SKYZERO_SELFPLAY_MANAGER_H
#define SKYZERO_SELFPLAY_MANAGER_H

// SelfplayEngine: owns the TorchScript inference pool, inference servers,
// and selfplay workers. Each game produces a PolicySurpriseSample vector
// that the caller post-processes (KataGo-style value bootstrap,
// surprise weighting, stochastic replication → TrainSamples → NpzWriter).

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "alphazero.h"
#include "alphazero_parallel.h"
#include "game_initializer.h"
#include "policy_init.h"
#include "policy_surprise_weighting.h"
#include "random_opening.h"
#include "utils.h"

namespace skyzero {

template <typename Game>
class SelfplayEngine {
public:
    struct SelfplayResult {
        std::vector<PolicySurpriseSample> samples;  // one per (state, to_play) step
        int winner = 0;                             // +1 black, -1 white, 0 draw
        int game_len = 0;
        std::vector<int8_t> final_state;            // board after last move was played
        std::vector<int8_t> initial_state;          // board after opening setup, before first MCTS
        bool balanced_opening = false;              // true=random balanced opening, false=empty
        int initial_to_play = 1;                    // side to move after opening setup
        int board_size = 0;                         // per-game board size (canvas-aware consumers)
        RuleType rule = RuleType::RENJU;            // per-game rule
    };

    SelfplayEngine(
        GameInitializer& game_init,
        const AlphaZeroConfig& cfg,
        const SelfplayParallelConfig& pcfg,
        const std::string& model_path,
        std::vector<torch::Device> devices
    )
        : game_init_(game_init), cfg_(cfg), pcfg_(pcfg), devices_(std::move(devices)) {
        const int num_servers = std::max(1, pcfg_.num_inference_servers);
        if (static_cast<int>(devices_.size()) != num_servers) {
            throw std::runtime_error(
                "SelfplayEngine: devices.size()=" + std::to_string(devices_.size())
                + " must equal num_inference_servers=" + std::to_string(num_servers));
        }
        inference_models_.reserve(num_servers);
        inference_model_mutexes_.reserve(num_servers);
        for (int i = 0; i < num_servers; ++i) {
            const auto& dev = devices_[i];
            auto mod = torch::jit::load(model_path, dev);
            mod.eval();
            if (dev.is_cuda()) {
                mod.to(torch::kHalf);
            }
            inference_models_.push_back(std::move(mod));
            inference_model_mutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
        }
    }

    ~SelfplayEngine() {
        stop();
    }

    void start() {
        stop_inference_.store(false);
        stop_workers_.store(false);
        // Resolve result-queue capacity. 0 = auto, <0 = unbounded.
        if (pcfg_.max_result_queue_size == 0) {
            result_queue_cap_ = std::max(1, pcfg_.num_workers * 2);
        } else if (pcfg_.max_result_queue_size < 0) {
            result_queue_cap_ = 0;  // unbounded
        } else {
            result_queue_cap_ = pcfg_.max_result_queue_size;
        }
        for (int i = 0; i < pcfg_.num_inference_servers; ++i) {
            inference_threads_.emplace_back([this, i]() { inference_server_loop(i); });
        }
        const uint64_t base_seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        for (int i = 0; i < pcfg_.num_workers; ++i) {
            selfplay_threads_.emplace_back([this, i, base_seed]() {
                uint64_t seed = base_seed + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL;
                while (!stop_workers_.load()) {
                    try {
                        auto result = selfplay_once(seed++);
                        {
                            std::unique_lock<std::mutex> lk(result_mutex_);
                            if (result_queue_cap_ > 0) {
                                result_space_cv_.wait(lk, [this]() {
                                    return stop_workers_.load()
                                        || result_queue_.size() < static_cast<size_t>(result_queue_cap_);
                                });
                                if (stop_workers_.load()) break;
                            }
                            result_queue_.push_back(std::move(result));
                        }
                        result_cv_.notify_one();
                    } catch (const std::exception& e) {
                        if (!stop_workers_.load()) {
                            std::cerr << "[worker " << i << "] " << e.what() << "\n";
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                    }
                }
            });
        }
    }

    bool try_pop_result(SelfplayResult& out, int wait_ms = 100) {
        std::unique_lock<std::mutex> lk(result_mutex_);
        if (result_queue_.empty()) {
            result_cv_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                                [&]() { return !result_queue_.empty() || stop_workers_.load(); });
        }
        if (result_queue_.empty()) return false;
        out = std::move(result_queue_.front());
        result_queue_.pop_front();
        result_space_cv_.notify_one();
        return true;
    }

    void stop() {
        stop_workers_.store(true);
        result_cv_.notify_all();
        result_space_cv_.notify_all();
        for (auto& t : selfplay_threads_) if (t.joinable()) t.join();
        selfplay_threads_.clear();

        stop_inference_.store(true);
        inference_cv_.notify_all();
        for (auto& t : inference_threads_) if (t.joinable()) t.join();
        inference_threads_.clear();
    }

    // Daemon hot-reload: replace each inference server's TorchScript module
    // in place. Worker threads keep producing search requests during the
    // swap; pending forward() calls block on the per-server mutex and pick
    // up the new module when released. Lock all mutexes in index order
    // (workers only ever take one) so there's no deadlock.
    void reload_model(const std::string& path) {
        const int num_servers = static_cast<int>(inference_models_.size());
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(num_servers);
        for (int i = 0; i < num_servers; ++i) {
            locks.emplace_back(*inference_model_mutexes_[i]);
        }
        for (int i = 0; i < num_servers; ++i) {
            const auto& dev = devices_[i];
            auto mod = torch::jit::load(path, dev);
            mod.eval();
            if (dev.is_cuda()) {
                mod.to(torch::kHalf);
            }
            inference_models_[i] = std::move(mod);
        }
    }

private:
    // ---- Inference request plumbing (V5) ----
    struct InferenceRequest {
        std::vector<int8_t> encoded;                   // V5: 5*MAX_AREA = 1125 int8
        std::array<float, 12> globals{};               // V5: 12-dim global features
        std::promise<std::pair<std::vector<float>, std::array<float, 3>>> promise;
    };

    // V5: derive globals from encoded state given the *per-game* Game ref.
    // Static so the worker thread can pass its own per-game `game_` (each
    // self-play game has its own rule / board_size / forbidden_plane).
    // V5 plane layout: 0=mask, 1=own, 2=opp, 3=fb_b, 4=fb_w. ply is recovered
    // from total stones on planes 1+2 (own + opp), which is mask-correct
    // because off-board cells are 0 there.
    static std::array<float, 12> derive_globals_from_encoded(
        const Game& g, const std::vector<int8_t>& encoded
    ) {
        constexpr int A = Game::MAX_AREA;   // 225
        int ply = 0;
        for (size_t i = A; i < 3 * static_cast<size_t>(A); ++i) ply += encoded[i];
        const int to_play = (ply % 2 == 0) ? 1 : -1;
        auto gf = g.compute_global_features(ply, to_play);
        std::array<float, 12> out{};
        for (int i = 0; i < 12; ++i) out[i] = gf.data[i];
        return out;
    }

    // Caller (worker thread) has computed globals using its per-game Game.
    std::pair<std::vector<float>, std::array<float, 3>> request_inference(
        const std::vector<int8_t>& encoded,
        const std::array<float, 12>& globals
    ) {
        auto req = std::unique_ptr<InferenceRequest>(new InferenceRequest{});
        req->encoded = encoded;
        req->globals = globals;
        auto fut = req->promise.get_future();
        {
            std::lock_guard<std::mutex> lk(inference_mutex_);
            inference_queue_.push_back(std::move(req));
        }
        inference_cv_.notify_one();
        return fut.get();
    }

    std::vector<std::pair<std::vector<float>, std::array<float, 3>>>
    request_batch_inference(
        const std::vector<std::vector<int8_t>>& encoded_batch,
        const std::vector<std::array<float, 12>>& globals_batch
    ) {
        std::vector<std::future<std::pair<std::vector<float>, std::array<float, 3>>>> futures;
        futures.reserve(encoded_batch.size());
        {
            std::lock_guard<std::mutex> lk(inference_mutex_);
            for (size_t i = 0; i < encoded_batch.size(); ++i) {
                auto req = std::unique_ptr<InferenceRequest>(new InferenceRequest{});
                req->encoded = encoded_batch[i];
                req->globals = globals_batch[i];
                futures.push_back(req->promise.get_future());
                inference_queue_.push_back(std::move(req));
            }
        }
        inference_cv_.notify_one();

        std::vector<std::pair<std::vector<float>, std::array<float, 3>>> out;
        out.reserve(futures.size());
        for (auto& fut : futures) out.push_back(fut.get());
        return out;
    }

    void inference_server_loop(int server_idx) {
        // V5: state is 5-plane padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE (15×15).
        // Network output is (B, 6, 225) for policy (we take main head idx 0)
        // and (B, 3) for value_wdl from a flat dict.
        const int c = Game::NUM_SPATIAL_PLANES_V5;     // 5
        const int board = Game::MAX_BOARD_SIZE;        // 15
        const int area = board * board;                // 225
        constexpr int g_dim = 12;                      // num_global_features
        const int max_batch = std::max(1, pcfg_.inference_batch_size);
        const torch::Device device = devices_[server_idx];
        const bool on_cuda = device.is_cuda();

        // Per-server pinned host buffers (allocated once, reused every batch).
        auto pinned_i8 = torch::TensorOptions().dtype(torch::kInt8).pinned_memory(on_cuda);
        auto pinned_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(on_cuda);
        torch::Tensor pinned_input = torch::empty({max_batch, c, board, board}, pinned_i8);
        torch::Tensor pinned_global = torch::empty({max_batch, g_dim}, pinned_f32);   // V5: globals
        torch::Tensor pinned_policy = torch::empty({max_batch, area}, pinned_f32);
        torch::Tensor pinned_value = torch::empty({max_batch, 3}, pinned_f32);

        while (true) {
            std::vector<std::unique_ptr<InferenceRequest>> batch;
            batch.reserve(max_batch);

            {
                std::unique_lock<std::mutex> lk(inference_mutex_);
                inference_cv_.wait(lk, [&]() {
                    return stop_inference_.load() || !inference_queue_.empty();
                });
                if (stop_inference_.load() && inference_queue_.empty()) break;

                batch.push_back(std::move(inference_queue_.front()));
                inference_queue_.pop_front();

                if (pcfg_.inference_batch_wait_us > 0) {
                    const auto until = std::chrono::steady_clock::now()
                        + std::chrono::microseconds(pcfg_.inference_batch_wait_us);
                    while (batch.size() < static_cast<size_t>(max_batch)) {
                        if (inference_queue_.empty()) {
                            if (inference_cv_.wait_until(lk, until) == std::cv_status::timeout) break;
                            if (inference_queue_.empty()) break;
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

                // V5: Pack int8 spatial encodings AND float32 globals into pinned buffers.
                auto pinned_input_view = pinned_input.narrow(0, 0, bsz);
                auto pinned_global_view = pinned_global.narrow(0, 0, bsz);
                int8_t* in_ptr = pinned_input_view.data_ptr<int8_t>();
                float* g_ptr = pinned_global_view.data_ptr<float>();
                const size_t per_sample = static_cast<size_t>(c) * area;
                for (int i = 0; i < bsz; ++i) {
                    const auto& enc = batch[i]->encoded;
                    if (enc.size() != per_sample) {
                        throw std::runtime_error("inference request encoded size mismatch");
                    }
                    std::memcpy(in_ptr + static_cast<size_t>(i) * per_sample,
                                enc.data(), per_sample);
                    std::memcpy(g_ptr + static_cast<size_t>(i) * g_dim,
                                batch[i]->globals.data(), g_dim * sizeof(float));
                }

                // Async H2D + dtype cast on GPU.
                torch::Tensor input_gpu, global_gpu;
                if (on_cuda) {
                    input_gpu = pinned_input_view.to(device, torch::kHalf, /*non_blocking=*/true);
                    global_gpu = pinned_global_view.to(device, torch::kHalf, /*non_blocking=*/true);
                } else {
                    input_gpu = pinned_input_view.to(torch::kFloat32);
                    global_gpu = pinned_global_view;   // already float32
                }

                torch::NoGradGuard no_grad;
                torch::jit::IValue out_iv;
                {
                    std::lock_guard<std::mutex> mlk(*inference_model_mutexes_[server_idx]);
                    // V5: model.forward(state, global) → flat Dict[str, Tensor]
                    out_iv = inference_models_[server_idx].forward({input_gpu, global_gpu});
                }

                // V5: dict output. Keys: policy (B, 6, area), value_wdl (B, 3).
                // Take main head (idx 0 of 6 policy outputs per v15 spec).
                auto out_dict = out_iv.toGenericDict();
                auto policy_all = out_dict.at("policy").toTensor();      // (B, 6, area)
                auto policy_logits = policy_all.select(1, 0).contiguous();   // (B, area)
                auto value_logits = out_dict.at("value_wdl").toTensor();     // (B, 3)

                // Stay on GPU for reshape / cast / softmax; one async D2H per output.
                auto policy_gpu = policy_logits.reshape({bsz, area});
                if (policy_gpu.scalar_type() != torch::kFloat32) {
                    policy_gpu = policy_gpu.to(torch::kFloat32);
                }
                auto value_prob_gpu = torch::softmax(value_logits.to(torch::kFloat32), 1);

                auto pinned_policy_view = pinned_policy.narrow(0, 0, bsz);
                auto pinned_value_view = pinned_value.narrow(0, 0, bsz);
                if (on_cuda) {
                    pinned_policy_view.copy_(policy_gpu, /*non_blocking=*/true);
                    pinned_value_view.copy_(value_prob_gpu, /*non_blocking=*/true);
                    torch::cuda::synchronize(device.index());
                } else {
                    pinned_policy_view.copy_(policy_gpu);
                    pinned_value_view.copy_(value_prob_gpu);
                }

                const float* pp = pinned_policy_view.data_ptr<float>();
                const float* vp = pinned_value_view.data_ptr<float>();
                for (int i = 0; i < bsz; ++i) {
                    std::vector<float> logits(static_cast<size_t>(area), 0.0f);
                    std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area,
                                static_cast<size_t>(area) * sizeof(float));
                    const size_t vi = static_cast<size_t>(i) * 3;
                    std::array<float, 3> v{vp[vi], vp[vi + 1], vp[vi + 2]};
                    batch[i]->promise.set_value({std::move(logits), v});
                }
            } catch (...) {
                for (auto& req : batch) {
                    try { req->promise.set_exception(std::current_exception()); } catch (...) {}
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
                req->promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("inference server stopped"))
                );
            } catch (...) {}
        }
    }

    // ---- One self-play game ----
    SelfplayResult selfplay_once(uint64_t seed) {
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

        // Per-game (size, rule) sample. KataGomo-style: shared mutex-guarded
        // GameInitializer hands back a fresh Gomoku each game; this game
        // object lives on the stack for the duration of selfplay_once and
        // must outlive the MCTS engines below (declared after this), since
        // they hold a reference into it.
        Game game = game_init_.create_game();
        Game& game_ = game;

        std::mt19937 worker_rng(seed);
        // The lambdas capture the per-game Game by reference so each request's
        // global-feature one-hot (rule + ply) reflects this game's rule, not a
        // shared one. Inference server itself is rule-agnostic.
        auto infer_fn = [this, &game_](const std::vector<int8_t>& encoded) {
            const auto globals = derive_globals_from_encoded(game_, encoded);
            return request_inference(encoded, globals);
        };
        auto batch_infer_fn = [this, &game_](const std::vector<std::vector<int8_t>>& batch) {
            std::vector<std::array<float, 12>> globals_batch;
            globals_batch.reserve(batch.size());
            for (const auto& enc : batch) {
                globals_batch.push_back(derive_globals_from_encoded(game_, enc));
            }
            return request_batch_inference(batch, globals_batch);
        };
        ParallelMCTS<Game> mcts(
            game_, cfg_, pcfg_.leaf_batch_size, infer_fn, batch_infer_fn, worker_rng());
        const bool reuse_enabled = cfg_.enable_tree_reuse;

        std::vector<MemoryStep> memory;
        auto init = game_.get_initial_state(worker_rng);
        std::vector<int8_t> state = std::move(init.board);
        int to_play = init.to_play;

        bool used_balanced_opening = false;
        {
            std::uniform_real_distribution<float> u01(0.0f, 1.0f);
            if (u01(worker_rng) < cfg_.balance_opening_prob) {
                RandomOpening<Game> ro(game_, infer_fn, cfg_, worker_rng());
                ro.initialize(state, to_play);
                used_balanced_opening = true;
            }
        }

        // KataGomo initGamesWithPolicy (play.cpp:1271-1282): after the balanced
        // opening, play a few extra policy-sampled moves to shake the position
        // off the "both sides look balanced" plateau. If the sampled trajectory
        // ends the game, drop this game and produce an empty result — consistent
        // with how a balanced_opening terminal would be handled.
        if (cfg_.policy_init_avg_move_num > 0.0f) {
            PolicyInit<Game> pi(game_, infer_fn, cfg_, worker_rng());
            if (!pi.initialize(state, to_play)) {
                SelfplayResult empty;
                return empty;
            }
        }

        std::vector<int8_t> initial_state_snapshot = state;
        const int initial_to_play_snapshot = to_play;

        bool in_soft_resign = false;
        std::vector<float> historical_v_mix;
        int last_action = -1;
        int last_player = 0;
        std::unique_ptr<MCTSNode> root(new MCTSNode{state, to_play});

        // >0 = global override, -1 = per-game board_size, 0 = disabled (greedy from move 0).
        const int half_life =
            cfg_.half_life > 0 ? cfg_.half_life :
            cfg_.half_life < 0 ? game_.board_size :
            0;

        while (!game_.is_terminal_canvas(state, last_action, last_player)) {
            if (stop_workers_.load()) {
                SelfplayResult empty;
                return empty;
            }

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

            int action = sr.gumbel_action;
            if (move_count < half_life) {
                const float inv_t = 1.0f / std::max(cfg_.move_temperature, 1e-6f);
                std::vector<float> w(sr.visit_counts.size());
                float sum_w = 0.0f;
                for (size_t i = 0; i < w.size(); ++i) {
                    w[i] = (sr.visit_counts[i] > 0.0f)
                        ? std::pow(sr.visit_counts[i], inv_t) : 0.0f;
                    sum_w += w[i];
                }
                if (sum_w > 0.0f) {
                    std::discrete_distribution<int> action_dist(w.begin(), w.end());
                    action = action_dist(worker_rng);
                }
            }
            if (action < 0) {
                std::discrete_distribution<int> action_dist(sr.mcts_policy.begin(), sr.mcts_policy.end());
                action = action_dist(worker_rng);
            }

            last_action = action;
            last_player = to_play;
            state = game_.get_next_state_canvas(state, action, to_play);
            to_play = -to_play;

            // Tree reuse: navigate to the child for `action` instead of
            // rebuilding the tree. Gumbel state (g[a], surviving_actions,
            // sigma_q, v_mix, improved_policy) is search-local in
            // gumbel_sequential_halving, so nothing on the node needs reset.
            // vloss is already 0 on retained nodes (synchronous backprop on
            // every path before search() returns).
            std::unique_ptr<MCTSNode> next_root;
            if (reuse_enabled) {
                for (auto& c : root->children) {
                    if (c && c->action_taken == action) {
                        next_root = std::move(c);
                        break;
                    }
                }
            }
            if (next_root) {
                next_root->parent = nullptr;
                // Children are constructed in expand_with() via
                // game_.get_next_state(parent.state, action, parent.to_play),
                // so the child's stored state must equal the just-applied state.
                assert(next_root->state == state && next_root->to_play == to_play);
                root = std::move(next_root);
                // Old root + sibling subtrees freed by the unique_ptr move.
            } else {
                // Reuse disabled, or chosen action wasn't expanded (can happen
                // when the improved-policy fallback at the action selection
                // above picks an unvisited action).
                root.reset(new MCTSNode{state, to_play});
            }
        }

        const int winner = game_.get_winner_canvas(state, last_action, last_player);
        SelfplayResult result;
        result.winner = winner;
        result.final_state = state;
        result.initial_state = std::move(initial_state_snapshot);
        result.initial_to_play = initial_to_play_snapshot;
        result.balanced_opening = used_balanced_opening;
        result.board_size = game_.board_size;
        result.rule = game_.rule;
        int total_moves = 0;
        for (int8_t v : state) if (v != 0) ++total_moves;
        result.game_len = total_moves;
        result.samples.reserve(memory.size());
        for (size_t i = 0; i < memory.size(); ++i) {
            const auto& s = memory[i];
            const int winner_from_side = winner * s.to_play;
            std::array<float, 3> outcome{0.0f, 1.0f, 0.0f};
            if (winner_from_side == 1) outcome = {1.0f, 0.0f, 0.0f};
            else if (winner_from_side == -1) outcome = {0.0f, 0.0f, 1.0f};

            // V5: canvas-pad encoded state and per-step global features here,
            // using the *per-game* `game_` (with this game's rule + size).
            // selfplay_main.cpp must NOT re-encode — by the time samples leave
            // selfplay_once, state is already 5*MAX_AREA = 1125 int8.
            int ply_for_step = 0;
            for (int8_t v : s.state) if (v != 0) ++ply_for_step;
            auto gf = game_.compute_global_features(ply_for_step, s.to_play);

            PolicySurpriseSample ps;
            ps.state = game_.encode_state_v5(s.state, s.to_play);
            for (int j = 0; j < 12; ++j) ps.global_features[j] = gf.data[j];
            ps.to_play = static_cast<int8_t>(s.to_play);
            ps.policy_target = s.mcts_policy;
            ps.opponent_policy_target = s.next_mcts_policy.empty()
                ? std::vector<float>(s.mcts_policy.size(), 0.0f)
                : s.next_mcts_policy;
            ps.has_opponent_policy = !s.next_mcts_policy.empty();
            ps.outcome = outcome;
            ps.nn_policy = s.nn_policy;
            ps.nn_value_probs = s.nn_value_probs;
            ps.v_mix = s.v_mix;
            ps.sample_weight = s.sample_weight;
            result.samples.push_back(std::move(ps));
        }

        // KataGomo-aligned main value target: pure game outcome propagated
        // backward with per-step perspective flip. Equivalent to KataGomo's
        // fillValueTDTargets(..., nowFactor=0.0, rowGlobal[0:3]) — the geometric
        // mixture collapses to all-weight-on-outcome.
        if (!result.samples.empty()) {
            const int N = static_cast<int>(result.samples.size());
            auto cur = result.samples[N - 1].outcome;
            result.samples[N - 1].value_target = cur;
            for (int i = N - 2; i >= 0; --i) {
                cur = flip_wdl(cur);
                result.samples[i].value_target = cur;
            }
        }

        // KataGomo-aligned TD(λ) value targets and futurepos targets.
        // TD: 3 horizons (long/mid/short) × WLD. Reverse-recursion equivalent
        //     to KataGomo's fillValueTDTargets:
        //       T(anchor) = outcome  (virtual extra slot beyond N-1)
        //       T(i)      = nf*v_mix[i] + (1-nf)*flip(T(i+1))   for i ∈ [0, N-1]
        //     nowFactor = 1/(1 + boardArea * c) for c ∈ {0.176, 0.056, 0.016}.
        //     Layout: long[0:3], mid[3:6], short[6:9]; each = (W,D,L) from
        //     samples[i].to_play perspective.
        //     NOTE: the LAST sample (i=N-1) also mixes its v_mix with outcome,
        //     matching KataGomo. Earlier impl hard-anchored T(N-1)=outcome
        //     (skipping v_mix[N-1]) which shifted targets by 1 step vs KataGomo.
        // Futurepos: cells at +8 / +32 steps clamped to game end, encoded from
        //     samples[i].to_play perspective (+1 own, -1 opp, 0 empty),
        //     padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE (off-board = 0).
        if (!result.samples.empty()) {
            const int N = static_cast<int>(result.samples.size());
            const double board_area = static_cast<double>(game_.board_size) * game_.board_size;
            const double now_factors[3] = {
                1.0 / (1.0 + board_area * 0.176),  // long
                1.0 / (1.0 + board_area * 0.056),  // mid
                1.0 / (1.0 + board_area * 0.016),  // short
            };

            for (int h = 0; h < 3; ++h) {
                const double nf = now_factors[h];
                const double rd = 1.0 - nf;
                // anchor: T(virtual N) = outcome, in samples[N-1].to_play view.
                const auto& last_outcome = result.samples[N - 1].outcome;
                double nx_w = last_outcome[0];
                double nx_d = last_outcome[1];
                double nx_l = last_outcome[2];
                for (int i = N - 1; i >= 0; --i) {
                    const auto& v = result.samples[i].v_mix;
                    const double cur_w = nf * v[0] + rd * nx_w;
                    const double cur_d = nf * v[1] + rd * nx_d;
                    const double cur_l = nf * v[2] + rd * nx_l;
                    result.samples[i].td_value_target[3 * h + 0] = static_cast<float>(cur_w);
                    result.samples[i].td_value_target[3 * h + 1] = static_cast<float>(cur_d);
                    result.samples[i].td_value_target[3 * h + 2] = static_cast<float>(cur_l);
                    // Prepare for i-1: perspective flips by one move, so swap W/L.
                    nx_w = cur_l;
                    nx_d = cur_d;
                    nx_l = cur_w;
                }
            }

            const int M = Game::MAX_BOARD_SIZE;
            const int A = Game::MAX_AREA;
            const int B = game_.board_size;
            const int OFFSETS[2] = {8, 32};
            for (int idx = 0; idx < N; ++idx) {
                auto& fp = result.samples[idx].futurepos_target;
                fp.assign(2 * A, 0);
                const int8_t pla = result.samples[idx].to_play;
                for (int chan = 0; chan < 2; ++chan) {
                    // KataGomo trainingwrite.cpp:937-955 builds posHistForFutureBoards
                    // of size N+1: index k = board after k moves played, index N =
                    // terminal board (after the final move). Mirror that layout with
                    // memory[0..N-1] for k<N and the post-loop `state` variable for
                    // k=N — the while loop above leaves `state` at the terminal
                    // position (cf. result.final_state = state). Without the k=N
                    // slot, samples within OFFSET of the end miss the final stone.
                    // Use raw board (stride B, values {-1,0,+1}). result.samples[j].state
                    // is encode_state_v5 output (5*M*M, values {0,1}) — wrong stride and
                    // wrong value domain; would produce a constant target independent
                    // of the actual future board.
                    const int j = std::min(N, idx + OFFSETS[chan]);
                    const auto& s = (j < N) ? memory[j].state : state;
                    int8_t* out = fp.data() + chan * A;
                    for (int r = 0; r < B; ++r) {
                        for (int c = 0; c < B; ++c) {
                            const int8_t v = s[r * B + c];
                            if (v == pla) out[r * M + c] = 1;
                            else if (v == -pla) out[r * M + c] = -1;
                        }
                    }
                }
            }
        }

        return result;
    }

    GameInitializer& game_init_;
    const AlphaZeroConfig& cfg_;
    SelfplayParallelConfig pcfg_;
    std::vector<torch::Device> devices_;

    std::vector<torch::jit::script::Module> inference_models_;
    std::vector<std::unique_ptr<std::mutex>> inference_model_mutexes_;

    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    std::deque<std::unique_ptr<InferenceRequest>> inference_queue_;
    std::vector<std::thread> inference_threads_;
    std::atomic<bool> stop_inference_{false};

    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    std::condition_variable result_space_cv_;
    std::deque<SelfplayResult> result_queue_;
    int result_queue_cap_ = 0;  // 0 means unbounded
    std::vector<std::thread> selfplay_threads_;
    std::atomic<bool> stop_workers_{false};
};

}  // namespace skyzero

#endif
