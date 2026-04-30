// gomoku_ab — headless A-vs-B MCTS-config match runner for a single model.
//
// Loads ONE TorchScript model and plays N games against itself alternating
// colors (A-black on even game indices, B-black on odd). A and B differ only
// in their MCTS / inference search configs (scripts/ab/{a,b}.cfg). Both
// sides share a single BatchedInferenceServer to halve GPU memory and
// improve batch fill-rate. Appends one JSON line per game (model, cfg_a,
// cfg_b, a_black, winner_a, plies) to the output file. Designed to feed
// python/ab.py.
//
// Shares config/inference scaffolding with gomoku_elo_main.cpp.

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

// Populate an AlphaZeroConfig from a parsed cfg map. Used twice in main()
// (once for A side, once for B side). gumbel_noise_enabled is forced false
// because evaluation must be deterministic-ish — we want Elo to reflect
// strength, not sampling luck.
static AlphaZeroConfig build_mcts_cfg(
        const std::unordered_map<std::string, std::string>& m,
        int board_size,
        const torch::Device& device) {
    AlphaZeroConfig c;
    c.board_size = board_size;
    c.num_simulations = cfg_get<int>(m, "NUM_SIMULATIONS", 800);
    c.gumbel_m = cfg_get<int>(m, "GUMBEL_M", 16);
    c.gumbel_c_visit = cfg_get<float>(m, "GUMBEL_C_VISIT", 50.0f);
    c.gumbel_c_scale = cfg_get<float>(m, "GUMBEL_C_SCALE", 1.0f);
    c.gumbel_noise_enabled = false;
    c.half_life = cfg_get<int>(m, "HALF_LIFE", 0);
    c.c_puct = cfg_get<float>(m, "C_PUCT", 1.1f);
    c.c_puct_log = cfg_get<float>(m, "C_PUCT_LOG", 0.45f);
    c.c_puct_base = cfg_get<float>(m, "C_PUCT_BASE", 500.0f);
    c.fpu_pow = cfg_get<float>(m, "FPU_POW", 1.0f);
    c.fpu_reduction_max = cfg_get<float>(m, "FPU_REDUCTION_MAX", 0.16f);
    c.fpu_loss_prop = cfg_get<float>(m, "FPU_LOSS_PROP", 0.0f);
    c.cpuct_utility_stdev_prior = cfg_get<float>(m, "CPUCT_UTILITY_STDEV_PRIOR", 0.40f);
    c.cpuct_utility_stdev_prior_weight = cfg_get<float>(m, "CPUCT_UTILITY_STDEV_PRIOR_WEIGHT", 2.0f);
    c.cpuct_utility_stdev_scale = cfg_get<float>(m, "CPUCT_UTILITY_STDEV_SCALE", 0.85f);
    c.enable_stochastic_transform_inference_for_root =
        cfg_get_bool(m, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", false);
    c.enable_stochastic_transform_inference_for_child =
        cfg_get_bool(m, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", false);
    c.enable_symmetry_inference_for_root =
        cfg_get_bool(m, "ENABLE_SYMMETRY_ROOT", true);
    c.enable_symmetry_inference_for_child =
        cfg_get_bool(m, "ENABLE_SYMMETRY_CHILD", true);
    c.root_symmetry_pruning =
        cfg_get_bool(m, "ROOT_SYMMETRY_PRUNING", true);
    c.device = device;
    return c;
}

struct CliArgs {
    std::string model;
    std::string config_ab;
    std::string config_a;
    std::string config_b;
    std::string output;
    int num_games = 200;
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
        if (k == "--model") a.model = need("--model");
        else if (k == "--config-ab") a.config_ab = need("--config-ab");
        else if (k == "--config-a") a.config_a = need("--config-a");
        else if (k == "--config-b") a.config_b = need("--config-b");
        else if (k == "--output") a.output = need("--output");
        else if (k == "--num-games") a.num_games = std::stoi(need("--num-games"));
        else if (k == "--seed") { a.seed = std::stoull(need("--seed")); a.seed_set = true; }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.config_ab.empty() || a.config_a.empty()
        || a.config_b.empty() || a.output.empty()) {
        throw std::runtime_error(
            "usage: gomoku_ab --model PATH --config-ab PATH --config-a PATH --config-b PATH "
            "--output PATH [--num-games N] [--seed S]");
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
        const auto cfg_ab_map = parse_cfg(cli.config_ab);
        const auto cfg_a_map  = parse_cfg(cli.config_a);
        const auto cfg_b_map  = parse_cfg(cli.config_b);

        const int board_size = cfg_get<int>(cfg_ab_map, "BOARD_SIZE", 15);
        // NUM_PLANES is documented in ab.cfg but the binary uses the hardcoded
        // V5 value (Gomoku::NUM_SPATIAL_PLANES_V5) — the cfg key only exists
        // so users can sanity-check it matches the trained model.
        const std::string rule_str = ([&]() -> std::string {
            auto it = cfg_ab_map.find("RULE");
            return (it != cfg_ab_map.end()) ? it->second : "renju";
        })();
        const RuleType rule = rule_from_string(rule_str);
        Gomoku game(board_size, rule, /*forbidden_plane=*/rule != RuleType::FREESTYLE);

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);

        AlphaZeroConfig cfg_a = build_mcts_cfg(cfg_a_map, board_size, device);
        AlphaZeroConfig cfg_b = build_mcts_cfg(cfg_b_map, board_size, device);
        if (cfg_a.half_life < 0) cfg_a.half_life = game.board_size;
        if (cfg_b.half_life < 0) cfg_b.half_life = game.board_size;

        // Single shared model + inference server. Both sides use the same
        // weights, so loading twice would just waste GPU memory and split
        // the inference batch.
        auto h = load_model(cli.model, device);

        const int c = Gomoku::NUM_SPATIAL_PLANES_V5;
        const int board = Gomoku::MAX_BOARD_SIZE;
        const int infer_batch = cfg_get<int>(cfg_ab_map, "INFERENCE_BATCH_SIZE", 64);
        const int infer_wait_us = cfg_get<int>(cfg_ab_map, "INFERENCE_WAIT_US", 200);
        const int num_concurrent = std::max(
            1, cfg_get<int>(cfg_ab_map, "NUM_CONCURRENT_GAMES", 4));
        const int threads_a = std::max(
            1, cfg_get<int>(cfg_a_map, "SEARCH_THREADS_PER_TREE", 8));
        const int threads_b = std::max(
            1, cfg_get<int>(cfg_b_map, "SEARCH_THREADS_PER_TREE", 8));

        BatchedInferenceServer server(h.get(), device, c, board, infer_batch,
                                      infer_wait_us, &game);

        auto infer = [&](const std::vector<int8_t>& e) { return server.infer(e); };
        auto fwd   = [&](const std::vector<std::vector<int8_t>>& b) { return server.infer_batch(b); };

        const uint64_t seed = cli.seed_set ? cli.seed : std::random_device{}();

        std::ofstream out(cli.output, std::ios::app);
        if (!out) throw std::runtime_error("cannot open output: " + cli.output);

        std::cerr << "[gomoku_ab] model=" << cli.model << "\n"
                  << "             cfg-ab=" << cli.config_ab
                  << " cfg-a="  << cli.config_a
                  << " cfg-b="  << cli.config_b << "\n"
                  << "             device=" << (use_cuda ? "cuda" : "cpu")
                  << " sims_a=" << cfg_a.num_simulations
                  << " sims_b=" << cfg_b.num_simulations
                  << " threads_a=" << threads_a
                  << " threads_b=" << threads_b
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
                TreeParallelMCTS<Gomoku> mcts_a(game, cfg_a, threads_a, infer, fwd, rng());
                TreeParallelMCTS<Gomoku> mcts_b(game, cfg_b, threads_b, infer, fwd, rng());
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
                        const int sims = a_to_move ? cfg_a.num_simulations : cfg_b.num_simulations;
                        const auto res = mcts.search(state, to_play, sims, root);
                        int action = res.gumbel_action;
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
                        out << "{\"model\":\"" << cli.model << "\","
                            << "\"cfg_a\":\"" << cli.config_a << "\","
                            << "\"cfg_b\":\"" << cli.config_b << "\","
                            << "\"a_black\":" << (a_is_black ? "true" : "false") << ","
                            << "\"winner_a\":" << winner_a << ","
                            << "\"plies\":" << plies << "}\n";
                        out.flush();
                    }
                    {
                        std::lock_guard<std::mutex> lk(log_mu);
                        std::cerr << "[gomoku_ab] game " << (g + 1) << "/" << cli.num_games
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
        std::cerr << "[gomoku_ab] done. A score=" << std::fixed << std::setprecision(3) << score
                  << " (" << aw << "W " << dr << "D " << bw << "L)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gomoku_ab] fatal: " << e.what() << "\n";
        return 2;
    }
}
