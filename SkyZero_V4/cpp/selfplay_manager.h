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
#include "alphazero_tree_parallel.h"
#include "policy_init.h"
#include "policy_surprise_weighting.h"
#include "random_opening.h"
#include "utils.h"

namespace skyzero {

// MCTS backend selector (run.cfg MCTS_BACKEND key).
//   BatchedLeaf: existing ParallelMCTS — batched leaf parallelism within
//                one search, single-threaded selection, VL for path diversity.
//   SharedTree : new TreeParallelMCTS — KataGo-style, multiple search
//                threads descend the same tree concurrently, VL for
//                concurrent path diversity.
struct MCTSBackendConfig {
    enum Kind { BatchedLeaf = 0, SharedTree = 1 };
    Kind kind = BatchedLeaf;
    int search_threads_per_tree = 4;
};

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
    };

    SelfplayEngine(
        Game& game,
        const AlphaZeroConfig& cfg,
        const SelfplayParallelConfig& pcfg,
        const MCTSBackendConfig& bcfg,
        const std::string& model_path,
        torch::Device device
    )
        : game_(game), cfg_(cfg), pcfg_(pcfg), bcfg_(bcfg), device_(device) {
        const int num_servers = std::max(1, pcfg_.num_inference_servers);
        inference_models_.reserve(num_servers);
        inference_model_mutexes_.reserve(num_servers);
        for (int i = 0; i < num_servers; ++i) {
            auto mod = torch::jit::load(model_path, device_);
            mod.eval();
            if (device_.is_cuda()) {
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

private:
    // ---- Inference request plumbing ----
    struct InferenceRequest {
        std::vector<int8_t> encoded;
        std::promise<std::pair<std::vector<float>, std::array<float, 3>>> promise;
    };

    std::pair<std::vector<float>, std::array<float, 3>> request_inference(
        const std::vector<int8_t>& encoded
    ) {
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
        for (auto& fut : futures) out.push_back(fut.get());
        return out;
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
                std::vector<float> input_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
                for (int i = 0; i < bsz; ++i) {
                    const auto& enc = batch[i]->encoded;
                    if (enc.size() != static_cast<size_t>(c * area)) {
                        throw std::runtime_error("inference request encoded size mismatch");
                    }
                    const size_t base = static_cast<size_t>(i) * c * area;
                    for (int j = 0; j < c * area; ++j) {
                        input_buf[base + j] = static_cast<float>(enc[j]);
                    }
                }

                auto input = torch::from_blob(input_buf.data(), {bsz, c, board, board}, torch::kFloat32)
                                 .clone().to(device_);
                if (device_.is_cuda()) input = input.to(torch::kHalf);

                torch::NoGradGuard no_grad;
                torch::jit::IValue out_iv;
                {
                    std::lock_guard<std::mutex> mlk(*inference_model_mutexes_[server_idx]);
                    out_iv = inference_models_[server_idx].forward({input});
                }

                auto tuple = out_iv.toTuple();
                auto policy_logits = tuple->elements()[0].toTensor();
                auto value_logits = tuple->elements()[2].toTensor();

                auto policy = policy_logits.reshape({bsz, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
                auto value = torch::softmax(value_logits.to(torch::kFloat32), 1).to(torch::kCPU).contiguous();
                const float* pp = policy.data_ptr<float>();
                const float* vp = value.data_ptr<float>();

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

        std::mt19937 worker_rng(seed);
        auto infer_fn = [this](const std::vector<int8_t>& encoded) {
            return request_inference(encoded);
        };
        auto batch_infer_fn = [this](const std::vector<std::vector<int8_t>>& batch) {
            return request_batch_inference(batch);
        };
        // Dispatch to the configured MCTS backend. Both classes expose the
        // same public search() signature; we use a small lambda to forward.
        std::unique_ptr<ParallelMCTS<Game>> mcts_batched;
        std::unique_ptr<TreeParallelMCTS<Game>> mcts_tree;
        std::function<MCTSSearchOutput(const std::vector<int8_t>&, int, int, std::unique_ptr<MCTSNode>&)> search_fn;
        if (bcfg_.kind == MCTSBackendConfig::SharedTree) {
            mcts_tree.reset(new TreeParallelMCTS<Game>(
                game_, cfg_, bcfg_.search_threads_per_tree, infer_fn, batch_infer_fn, worker_rng()));
            search_fn = [&](const std::vector<int8_t>& s, int tp, int nsim,
                            std::unique_ptr<MCTSNode>& rp) {
                return mcts_tree->search(s, tp, nsim, rp);
            };
        } else {
            mcts_batched.reset(new ParallelMCTS<Game>(
                game_, cfg_, pcfg_.leaf_batch_size, infer_fn, batch_infer_fn, worker_rng()));
            search_fn = [&](const std::vector<int8_t>& s, int tp, int nsim,
                            std::unique_ptr<MCTSNode>& rp) {
                return mcts_batched->search(s, tp, nsim, rp);
            };
        }

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

        const int half_life = std::max(1, cfg_.half_life > 0 ? cfg_.half_life : game_.board_size);

        while (!game_.is_terminal(state, last_action, last_player)) {
            if (stop_workers_.load()) {
                SelfplayResult empty;
                return empty;
            }

            int num_simulations = cfg_.num_simulations;
            if (in_soft_resign) {
                num_simulations = std::max(cfg_.num_simulations / 4, cfg_.min_simulations_in_soft_resign);
            }

            const auto sr = search_fn(state, to_play, num_simulations, root);
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
                next_root->parent = nullptr;
                root = std::move(next_root);
            } else {
                root.reset(new MCTSNode{state, to_play});
            }
        }

        const int winner = game_.get_winner(state, last_action, last_player);
        SelfplayResult result;
        result.winner = winner;
        result.final_state = state;
        result.initial_state = std::move(initial_state_snapshot);
        result.initial_to_play = initial_to_play_snapshot;
        result.balanced_opening = used_balanced_opening;
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

            PolicySurpriseSample ps;
            ps.state = s.state;
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

        // KataGo-style value bootstrap (tail → head with mixing factor)
        if (!result.samples.empty()) {
            const float now_factor = 1.0f / (1.0f + static_cast<float>(game_.board_size)
                * game_.board_size * cfg_.value_target_mix_now_factor_constant);
            result.samples.back().value_target = result.samples.back().outcome;
            for (int i = static_cast<int>(result.samples.size()) - 2; i >= 0; --i) {
                const auto next_target = flip_wdl(result.samples[i + 1].value_target);
                result.samples[i].value_target = {
                    (1.0f - now_factor) * next_target[0] + now_factor * result.samples[i].v_mix[0],
                    (1.0f - now_factor) * next_target[1] + now_factor * result.samples[i].v_mix[1],
                    (1.0f - now_factor) * next_target[2] + now_factor * result.samples[i].v_mix[2],
                };
            }
        }

        return result;
    }

    Game& game_;
    const AlphaZeroConfig& cfg_;
    SelfplayParallelConfig pcfg_;
    MCTSBackendConfig bcfg_;
    torch::Device device_;

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
