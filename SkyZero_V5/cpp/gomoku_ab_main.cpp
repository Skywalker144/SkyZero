// gomoku_elo — headless model-A-vs-model-B match runner.
//
// Loads two TorchScript models, plays N games alternating colors (A-black on
// even game indices, B-black on odd), and appends one JSON line per game to
// the output file. Designed to feed python/elo.py.
//
// Shares config/inference scaffolding with gomoku_play_main.cpp.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "alphazero.h"
#include "alphazero_tree_parallel.h"
#include "envs/gomoku.h"

using namespace skyzero;

static std::unordered_map<std::string, std::string> parse_cfg(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::string line;
    while (std::getline(f, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        const auto a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos) continue;
        const auto b = line.find_last_not_of(" \t\r");
        line = line.substr(a, b - a + 1);
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (!val.empty() && (val.front() == '"' || val.front() == '\'')) {
            const char q = val.front();
            if (val.size() >= 2 && val.back() == q) val = val.substr(1, val.size() - 2);
        }
        out[std::move(key)] = std::move(val);
    }
    return out;
}

template <typename T>
static T cfg_get(const std::unordered_map<std::string, std::string>& c,
                 const std::string& key, T fallback) {
    auto it = c.find(key);
    if (it == c.end() || it->second.empty()) return fallback;
    std::istringstream ss(it->second);
    T v;
    ss >> v;
    if (ss.fail()) return fallback;
    return v;
}

static bool cfg_get_bool(const std::unordered_map<std::string, std::string>& c,
                         const std::string& key, bool fallback) {
    auto it = c.find(key);
    if (it == c.end() || it->second.empty()) return fallback;
    const auto& v = it->second;
    if (v == "0" || v == "false" || v == "False" || v == "no") return false;
    if (v == "1" || v == "true" || v == "True" || v == "yes") return true;
    return fallback;
}

struct CliArgs {
    std::string model_a;
    std::string model_b;
    std::string config;
    std::string output;
    int num_games = 40;
    int num_simulations_override = -1;
    uint64_t seed = 0;
    bool seed_set = false;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (k == "--model-a") a.model_a = need("--model-a");
        else if (k == "--model-b") a.model_b = need("--model-b");
        else if (k == "--config") a.config = need("--config");
        else if (k == "--output") a.output = need("--output");
        else if (k == "--num-games") a.num_games = std::stoi(need("--num-games"));
        else if (k == "--num-simulations") a.num_simulations_override = std::stoi(need("--num-simulations"));
        else if (k == "--seed") { a.seed = std::stoull(need("--seed")); a.seed_set = true; }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model_a.empty() || a.model_b.empty() || a.config.empty() || a.output.empty()) {
        throw std::runtime_error(
            "usage: gomoku_elo --model-a PATH --model-b PATH --config PATH --output PATH "
            "[--num-games N] [--num-simulations N] [--seed S]");
    }
    return a;
}

// Load a TorchScript model and return an (infer_fn, batch_infer_fn) pair plus
// a holder that keeps the module alive for the lifetime of the closures.
struct ModelHandle {
    torch::jit::script::Module module;
    std::mutex mu;
};

static std::unique_ptr<ModelHandle> load_model(const std::string& path, const torch::Device& device) {
    auto h = std::make_unique<ModelHandle>();
    h->module = torch::jit::load(path, device);
    h->module.eval();
    if (device.is_cuda()) h->module.to(torch::kHalf);
    return h;
}

// Batched inference server: one thread that drains a request queue, builds a
// tensor batch (capped at batch_size, waiting up to wait_us for fill-up), runs
// one forward pass under the model's mutex, and scatters results back via
// per-request promises. Thread-safe; multiple game threads may submit
// concurrently. Owns one ModelHandle (caller-supplied).
class BatchedInferenceServer {
public:
    using Output = std::pair<std::vector<float>, std::array<float, 3>>;

    BatchedInferenceServer(ModelHandle* h, torch::Device device, int channels,
                           int board_size, int batch_size, int wait_us,
                           const Gomoku* game = nullptr)
        : h_(h), device_(device), c_(channels), board_(board_size),
          area_(board_size * board_size),
          batch_size_(std::max(1, batch_size)),
          wait_us_(std::max(0, wait_us)),
          game_(game) {
        thread_ = std::thread([this] { loop(); });
    }

    ~BatchedInferenceServer() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    BatchedInferenceServer(const BatchedInferenceServer&) = delete;
    BatchedInferenceServer& operator=(const BatchedInferenceServer&) = delete;

    Output infer(const std::vector<int8_t>& encoded) {
        auto req = std::make_unique<Request>();
        req->encoded = encoded;
        auto fut = req->p.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back(std::move(req));
        }
        cv_.notify_one();
        return fut.get();
    }

    std::vector<Output> infer_batch(const std::vector<std::vector<int8_t>>& batch) {
        std::vector<std::future<Output>> futs;
        futs.reserve(batch.size());
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& enc : batch) {
                auto req = std::make_unique<Request>();
                req->encoded = enc;
                futs.push_back(req->p.get_future());
                queue_.push_back(std::move(req));
            }
        }
        cv_.notify_all();
        std::vector<Output> out;
        out.reserve(futs.size());
        for (auto& f : futs) out.push_back(f.get());
        return out;
    }

private:
    struct Request {
        std::vector<int8_t> encoded;
        std::promise<Output> p;
    };

    void loop() {
        for (;;) {
            std::vector<std::unique_ptr<Request>> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
                // Briefly wait for more requests to fill the batch.
                if (wait_us_ > 0 && static_cast<int>(batch.size()) < batch_size_) {
                    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::microseconds(wait_us_);
                    while (static_cast<int>(batch.size()) < batch_size_) {
                        while (!queue_.empty()
                               && static_cast<int>(batch.size()) < batch_size_) {
                            batch.push_back(std::move(queue_.front()));
                            queue_.pop_front();
                        }
                        if (static_cast<int>(batch.size()) >= batch_size_) break;
                        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) break;
                    }
                } else {
                    while (!queue_.empty()
                           && static_cast<int>(batch.size()) < batch_size_) {
                        batch.push_back(std::move(queue_.front()));
                        queue_.pop_front();
                    }
                }
            }
            try {
                run_forward(batch);
            } catch (...) {
                auto exc = std::current_exception();
                for (auto& r : batch) r->p.set_exception(exc);
            }
        }
    }

    void run_forward(std::vector<std::unique_ptr<Request>>& batch) {
        const int bsz = static_cast<int>(batch.size());
        constexpr int g_dim = 12;
        std::vector<float> input_buf(static_cast<size_t>(bsz) * c_ * area_, 0.0f);
        std::vector<float> global_buf(static_cast<size_t>(bsz) * g_dim, 0.0f);
        for (int i = 0; i < bsz; ++i) {
            const auto& enc = batch[i]->encoded;
            if (enc.size() != static_cast<size_t>(c_ * area_)) {
                throw std::runtime_error("encoded size mismatch");
            }
            const size_t base = static_cast<size_t>(i) * c_ * area_;
            for (int j = 0; j < c_ * area_; ++j) {
                input_buf[base + j] = static_cast<float>(enc[j]);
            }
            // V5: derive globals from encoded (ply via own+opp planes, to_play parity).
            if (game_) {
                int ply = 0;
                for (size_t j = area_; j < 3 * static_cast<size_t>(area_); ++j) ply += enc[j];
                const int to_play = (ply % 2 == 0) ? 1 : -1;
                auto gf = game_->compute_global_features(ply, to_play);
                std::memcpy(global_buf.data() + i * g_dim, gf.data, g_dim * sizeof(float));
            }
        }
        auto input = torch::from_blob(input_buf.data(), {bsz, c_, board_, board_},
                                      torch::kFloat32)
                         .clone()
                         .to(device_);
        auto global_t = torch::from_blob(global_buf.data(), {bsz, g_dim}, torch::kFloat32)
                            .clone()
                            .to(device_);
        if (device_.is_cuda()) {
            input = input.to(torch::kHalf);
            global_t = global_t.to(torch::kHalf);
        }

        torch::jit::IValue out_iv;
        {
            std::lock_guard<std::mutex> lk(h_->mu);
            torch::NoGradGuard no_grad2;
            out_iv = h_->module.forward({input, global_t});   // V5
        }
        // V5: dict output
        auto out_dict = out_iv.toGenericDict();
        auto policy_all = out_dict.at("policy").toTensor();
        auto policy_logits = policy_all.select(1, 0).contiguous();   // main head
        auto value_logits = out_dict.at("value_wdl").toTensor();
        auto policy = policy_logits.reshape({bsz, area_})
                          .to(torch::kFloat32)
                          .to(torch::kCPU)
                          .contiguous();
        auto value = torch::softmax(value_logits.to(torch::kFloat32), 1)
                         .to(torch::kCPU)
                         .contiguous();
        const float* pp = policy.data_ptr<float>();
        const float* vp = value.data_ptr<float>();
        for (int i = 0; i < bsz; ++i) {
            std::vector<float> logits(static_cast<size_t>(area_), 0.0f);
            std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area_,
                        static_cast<size_t>(area_) * sizeof(float));
            const size_t vi = static_cast<size_t>(i) * 3;
            std::array<float, 3> v{vp[vi], vp[vi + 1], vp[vi + 2]};
            batch[i]->p.set_value({std::move(logits), v});
        }
    }

    ModelHandle* h_;
    torch::Device device_;
    int c_;
    int board_;
    int area_;
    int batch_size_;
    int wait_us_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<Request>> queue_;
    bool stop_ = false;
    std::thread thread_;
    const Gomoku* game_ = nullptr;
};


int main(int argc, char** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);

        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);
        const auto cfg_map = parse_cfg(cli.config);

        AlphaZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 800);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        // Force Gumbel noise OFF for evaluation regardless of cfg — we want
        // deterministic-ish play so Elo reflects strength, not sampling luck.
        cfg.gumbel_noise_enabled = false;
        cfg.half_life = cfg_get<int>(cfg_map, "HALF_LIFE", 0);
        cfg.c_puct = cfg_get<float>(cfg_map, "C_PUCT", 1.1f);
        cfg.c_puct_log = cfg_get<float>(cfg_map, "C_PUCT_LOG", 0.45f);
        cfg.c_puct_base = cfg_get<float>(cfg_map, "C_PUCT_BASE", 500.0f);
        cfg.fpu_pow = cfg_get<float>(cfg_map, "FPU_POW", 1.0f);
        cfg.fpu_reduction_max = cfg_get<float>(cfg_map, "FPU_REDUCTION_MAX", 0.25f);
        cfg.root_fpu_reduction_max = cfg_get<float>(cfg_map, "ROOT_FPU_REDUCTION_MAX", 0.0f);
        cfg.fpu_loss_prop = cfg_get<float>(cfg_map, "FPU_LOSS_PROP", 0.0f);
        cfg.lcb_k = cfg_get<float>(cfg_map, "LCB_K", 4.0f);
        cfg.cpuct_utility_stdev_prior = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR", 0.25f);
        cfg.cpuct_utility_stdev_prior_weight = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR_WEIGHT", 1.0f);
        cfg.cpuct_utility_stdev_scale = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_SCALE", 0.0f);
        cfg.enable_stochastic_transform_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", false);
        cfg.enable_stochastic_transform_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", false);
        cfg.enable_symmetry_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_ROOT", true);
        cfg.enable_symmetry_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_CHILD", true);
        cfg.root_symmetry_pruning =
            cfg_get_bool(cfg_map, "ROOT_SYMMETRY_PRUNING", true);

        if (cli.num_simulations_override > 0) cfg.num_simulations = cli.num_simulations_override;

        // V5: 5-plane padded encoding + 12-dim global features
        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 5);
        const std::string rule_str = ([&]() -> std::string {
            auto it = cfg_map.find("RULE");
            return (it != cfg_map.end()) ? it->second : "renju";
        })();
        const RuleType rule = rule_from_string(rule_str);
        Gomoku game(cfg.board_size, rule, /*forbidden_plane=*/rule != RuleType::FREESTYLE);
        if (cfg.half_life < 0) cfg.half_life = game.board_size;

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);
        cfg.device = device;

        auto ha = load_model(cli.model_a, device);
        auto hb = load_model(cli.model_b, device);

        // V5: hardcoded c=5, board=15, regardless of game.board_size (padded)
        const int c = Gomoku::NUM_SPATIAL_PLANES_V5;
        const int board = Gomoku::MAX_BOARD_SIZE;

        const int infer_batch = cfg_get<int>(cfg_map, "INFERENCE_BATCH_SIZE", 64);
        const int infer_wait_us = cfg_get<int>(cfg_map, "INFERENCE_WAIT_US", 200);
        const int num_concurrent = std::max(
            1, cfg_get<int>(cfg_map, "NUM_CONCURRENT_GAMES", 4));
        const int default_search_threads =
            cfg_get<int>(cfg_map, "SEARCH_THREADS_PER_TREE", 8);
        const int search_threads = std::max(
            1, cfg_get<int>(cfg_map, "ELO_SEARCH_THREADS_PER_TREE",
                             default_search_threads));

        BatchedInferenceServer server_a(ha.get(), device, c, board, infer_batch, infer_wait_us, &game);
        BatchedInferenceServer server_b(hb.get(), device, c, board, infer_batch, infer_wait_us, &game);

        auto infer_a = [&](const std::vector<int8_t>& e) { return server_a.infer(e); };
        auto infer_b = [&](const std::vector<int8_t>& e) { return server_b.infer(e); };
        auto fwd_a = [&](const std::vector<std::vector<int8_t>>& b) { return server_a.infer_batch(b); };
        auto fwd_b = [&](const std::vector<std::vector<int8_t>>& b) { return server_b.infer_batch(b); };

        const uint64_t seed = cli.seed_set ? cli.seed : std::random_device{}();

        std::ofstream out(cli.output, std::ios::app);
        if (!out) throw std::runtime_error("cannot open output: " + cli.output);

        std::cerr << "[gomoku_elo] A=" << cli.model_a << "\n"
                  << "              B=" << cli.model_b << "\n"
                  << "              device=" << (use_cuda ? "cuda" : "cpu")
                  << " sims=" << cfg.num_simulations
                  << " threads=" << search_threads
                  << " concurrent=" << num_concurrent
                  << " infer_batch=" << infer_batch
                  << " games=" << cli.num_games
                  << " seed=" << seed << "\n";

        std::atomic<int> next_game_idx{0};
        std::atomic<int> a_wins{0}, b_wins{0}, draws{0};
        std::mutex out_mu, log_mu;
        std::atomic<bool> abort_flag{false};
        std::mutex err_mu;
        std::exception_ptr first_err;

        auto worker = [&](int tid) {
            try {
                std::mt19937 rng(seed
                    + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(tid + 1));
                TreeParallelMCTS<Gomoku> mcts_a(game, cfg, search_threads, infer_a, fwd_a, rng());
                TreeParallelMCTS<Gomoku> mcts_b(game, cfg, search_threads, infer_b, fwd_b, rng());
                std::unique_ptr<MCTSNode> root_a, root_b;

                while (!abort_flag.load()) {
                    const int g = next_game_idx.fetch_add(1);
                    if (g >= cli.num_games) break;

                    // Even g: A plays black (to_play=1). Odd g: B plays black.
                    const bool a_is_black = (g % 2 == 0);
                    auto init = game.get_initial_state(rng);
                    std::vector<int8_t> state = std::move(init.board);
                    int to_play = init.to_play;
                    int last_action = -1;
                    int last_player = 0;
                    int plies = 0;

                    root_a.reset(new MCTSNode{state, to_play});
                    root_b.reset(new MCTSNode{state, to_play});

                    while (!game.is_terminal(state, last_action, last_player)) {
                        const bool a_to_move = (a_is_black && to_play == 1)
                                            || (!a_is_black && to_play == -1);
                        auto& mcts = a_to_move ? mcts_a : mcts_b;
                        auto& root = a_to_move ? root_a : root_b;

                        root.reset(new MCTSNode{state, to_play});
                        const auto res = mcts.search(state, to_play, cfg.num_simulations, root);
                        // Elo: prefer LCB-selected move; fall back to Gumbel
                        // when no child has enough visits for a variance estimate.
                        int action = res.lcb_action >= 0 ? res.lcb_action : res.gumbel_action;
                        if (action < 0) {
                            const auto legal = game.get_is_legal_actions(state, to_play);
                            for (int i = 0; i < static_cast<int>(legal.size()); ++i) {
                                if (legal[i]) { action = i; break; }
                            }
                        }
                        if (action < 0) break;

                        state = game.get_next_state(state, action, to_play);
                        last_action = action;
                        last_player = to_play;
                        to_play = -to_play;
                        ++plies;
                    }

                    const int winner = game.get_winner(state, last_action, last_player);
                    int winner_a = 0;
                    if (winner != 0) {
                        const int a_side = a_is_black ? 1 : -1;
                        winner_a = (winner == a_side) ? 1 : -1;
                    }
                    if (winner_a > 0) a_wins.fetch_add(1);
                    else if (winner_a < 0) b_wins.fetch_add(1);
                    else draws.fetch_add(1);
                    const int aw = a_wins.load();
                    const int bw = b_wins.load();
                    const int dr = draws.load();

                    {
                        std::lock_guard<std::mutex> lk(out_mu);
                        out << "{\"a\":\"" << cli.model_a << "\","
                            << "\"b\":\"" << cli.model_b << "\","
                            << "\"a_black\":" << (a_is_black ? "true" : "false") << ","
                            << "\"winner_a\":" << winner_a << ","
                            << "\"plies\":" << plies << "}\n";
                        out.flush();
                    }
                    {
                        std::lock_guard<std::mutex> lk(log_mu);
                        std::cerr << "[gomoku_elo] game " << (g + 1) << "/" << cli.num_games
                                  << " a_black=" << (a_is_black ? 1 : 0)
                                  << " winner_a=" << winner_a
                                  << " plies=" << plies
                                  << " | A:" << aw << " D:" << dr << " B:" << bw << "\n";
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!first_err) first_err = std::current_exception();
                abort_flag.store(true);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(num_concurrent);
        for (int t = 0; t < num_concurrent; ++t) workers.emplace_back(worker, t);
        for (auto& w : workers) w.join();
        if (first_err) std::rethrow_exception(first_err);

        const int aw = a_wins.load();
        const int bw = b_wins.load();
        const int dr = draws.load();
        const int played = aw + bw + dr;
        const float n = static_cast<float>(std::max(1, played));
        const float score = (aw + 0.5f * dr) / n;
        std::cerr << "[gomoku_elo] done. A score=" << std::fixed << std::setprecision(3) << score
                  << " (" << aw << "W " << dr << "D " << bw << "L)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gomoku_elo] fatal: " << e.what() << "\n";
        return 2;
    }
}
