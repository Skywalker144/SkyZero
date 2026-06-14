// mcts_bench — head-to-head wall-clock comparison of root-parallel
// (ParallelMCTS, leaf batching) vs tree-parallel (TreeParallelMCTS, shared
// tree with virtual loss) on a SINGLE MCTS instance.
//
// What "parallel" means here:
//   * Root-parallel  = single thread, Gumbel SH picks K leaves per halving
//                      round; all K leaves go in one NN batch.
//                      Knob: --root-batch (= leaf_batch_size).
//   * Tree-parallel  = N worker threads descend a shared tree using virtual
//                      loss; each thread queries the NN on its own.
//                      Knob: --tree-threads.
//
// Each (knob, value) sweep runs --searches fresh searches of --sims
// simulations from an empty board. The model is JIT-warmed up first so the
// first measured iteration doesn't pay CUDA init.
//
// Output (TSV to stdout):
//   variant    knob    value   mean_ms total_sims  sims_per_sec
//
// Usage:
//   mcts_bench --model models/latest.pt --config scripts/run.cfg
//              [--sims 400] [--searches 16] [--warmup 2]
//              [--root-batch 1,4,8,16,32] [--tree-threads 1,2,4,8,16,32]

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "skyzero.h"
#include "skyzero_parallel.h"
#include "skyzero_tree_parallel.h"
#include "envs/gomoku.h"

using namespace skyzero;

// ---- Config parsing (subset of selfplay_main.cpp) -------------------------
static std::unordered_map<std::string, std::string> parse_cfg(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    auto load = [&out](std::istream& f) {
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
    };
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    load(f);
    // Server-local override (matches scripts/run.sh's run.cfg.local handling).
    std::ifstream lf(path + ".local");
    if (lf) load(lf);
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

// ---- CLI ------------------------------------------------------------------
struct CliArgs {
    std::string model;
    std::string config;
    int sims = 400;
    int searches = 16;
    int warmup = 2;
    std::vector<int> root_batches;
    std::vector<int> tree_threads;
};

static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs a;
    a.root_batches = {1, 4, 8, 16, 32};
    a.tree_threads = {1, 2, 4, 8, 16, 32};
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if      (k == "--model")        a.model = need("--model");
        else if (k == "--config")       a.config = need("--config");
        else if (k == "--sims")         a.sims = std::stoi(need("--sims"));
        else if (k == "--searches")     a.searches = std::stoi(need("--searches"));
        else if (k == "--warmup")       a.warmup = std::stoi(need("--warmup"));
        else if (k == "--root-batch")   a.root_batches = parse_csv_ints(need("--root-batch"));
        else if (k == "--tree-threads") a.tree_threads = parse_csv_ints(need("--tree-threads"));
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.config.empty()) {
        throw std::runtime_error(
            "usage: mcts_bench --model PATH --config PATH "
            "[--sims N] [--searches N] [--warmup N] "
            "[--root-batch 1,4,8,16,32] [--tree-threads 1,2,4,8,16,32]");
    }
    return a;
}

// ---- Main -----------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);
        const auto cfg_map = parse_cfg(cli.config);

        auto require_str = [&](const std::string& k) -> std::string {
            auto it = cfg_map.find(k);
            if (it == cfg_map.end() || it->second.empty()) {
                throw std::runtime_error("missing required key in config: " + k);
            }
            return it->second;
        };

        // --- Build SkyZeroConfig (mirrors mcts_probe; non-relevant knobs at defaults) ---
        SkyZeroConfig cfg;
        cfg.board_size = std::stoi(require_str("MAIN_BOARD_SIZE"));
        cfg.num_simulations = cli.sims;
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.non_root_search_algo = SkyZeroConfig::parse_non_root_search_algo(
            cfg_get<std::string>(cfg_map, "NON_ROOT_SEARCH_ALGO", "puct"));
        cfg.root_search_algo = SkyZeroConfig::parse_root_search_algo(
            cfg_get<std::string>(cfg_map, "ROOT_SEARCH_ALGO", "gumbel"));
        cfg.lcb_for_selection = cfg_get_bool(cfg_map, "ROOT_LCB_SELECTION", true);
        cfg.validate();
        cfg.gumbel_noise_enabled = false;   // bench → deterministic
        cfg.c_puct = cfg_get<float>(cfg_map, "C_PUCT", 1.1f);
        cfg.c_puct_log = cfg_get<float>(cfg_map, "C_PUCT_LOG", 0.45f);
        cfg.c_puct_base = cfg_get<float>(cfg_map, "C_PUCT_BASE", 500.0f);
        cfg.fpu_pow = cfg_get<float>(cfg_map, "FPU_POW", 1.0f);
        cfg.fpu_reduction_max = cfg_get<float>(cfg_map, "FPU_REDUCTION_MAX", 0.08f);
        cfg.fpu_loss_prop = cfg_get<float>(cfg_map, "FPU_LOSS_PROP", 0.0f);
        cfg.cpuct_utility_stdev_prior =
            cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR", 0.40f);
        cfg.cpuct_utility_stdev_prior_weight =
            cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR_WEIGHT", 2.0f);
        cfg.cpuct_utility_stdev_scale =
            cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_SCALE", 0.85f);
        cfg.enable_stochastic_transform_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", true);
        cfg.enable_stochastic_transform_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", true);

        const std::string rule_str = require_str("MAIN_RULE");
        const RuleType rule = rule_from_string(rule_str);
        Gomoku game(cfg.board_size, rule, /*forbidden_plane=*/rule != RuleType::FREESTYLE);

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);
        cfg.device = device;

        auto model = torch::jit::load(cli.model, device);
        model.eval();
        if (use_cuda) model.to(torch::kHalf);
        std::mutex model_mu;   // serialize NN forward — both variants call into the same module

        const int c = Gomoku::NUM_SPATIAL_PLANES_V5;
        const int board = Gomoku::MAX_BOARD_SIZE;
        const int area = board * board;
        constexpr int g_dim = 12;

        auto derive_globals = [&](const std::vector<int8_t>& encoded) -> std::array<float, 12> {
            int ply = 0;
            for (size_t i = area; i < 3 * static_cast<size_t>(area); ++i) ply += encoded[i];
            const int to_play = (ply % 2 == 0) ? 1 : -1;
            auto gf = game.compute_global_features(ply, to_play);
            std::array<float, 12> out{};
            for (int i = 0; i < 12; ++i) out[i] = gf.data[i];
            return out;
        };

        auto run_forward = [&](const std::vector<std::vector<int8_t>>& batch)
                               -> std::vector<std::pair<std::vector<float>, std::array<float, 3>>> {
            const int bsz = static_cast<int>(batch.size());
            std::vector<float> input_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
            std::vector<float> global_buf(static_cast<size_t>(bsz) * g_dim, 0.0f);
            for (int i = 0; i < bsz; ++i) {
                const auto& enc = batch[i];
                if (enc.size() != static_cast<size_t>(c * area)) {
                    throw std::runtime_error("encoded size mismatch");
                }
                const size_t base = static_cast<size_t>(i) * c * area;
                for (int j = 0; j < c * area; ++j) {
                    input_buf[base + j] = static_cast<float>(enc[j]);
                }
                auto g = derive_globals(enc);
                std::memcpy(global_buf.data() + i * g_dim, g.data(), g_dim * sizeof(float));
            }
            auto input = torch::from_blob(input_buf.data(), {bsz, c, board, board}, torch::kFloat32)
                             .clone().to(device);
            auto global_t = torch::from_blob(global_buf.data(), {bsz, g_dim}, torch::kFloat32)
                                .clone().to(device);
            if (device.is_cuda()) {
                input = input.to(torch::kHalf);
                global_t = global_t.to(torch::kHalf);
            }
            torch::jit::IValue out_iv;
            {
                std::lock_guard<std::mutex> lk(model_mu);
                out_iv = model.forward({input, global_t});
            }
            auto out_dict = out_iv.toGenericDict();
            auto policy_all = out_dict.at("policy").toTensor();
            auto policy_logits = policy_all.select(1, 0).contiguous();
            auto value_logits = out_dict.at("value_wdl").toTensor();
            auto policy = policy_logits.reshape({bsz, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
            auto value = torch::softmax(value_logits.to(torch::kFloat32), 1).to(torch::kCPU).contiguous();
            const float* pp = policy.data_ptr<float>();
            const float* vp = value.data_ptr<float>();
            std::vector<std::pair<std::vector<float>, std::array<float, 3>>> out;
            out.reserve(bsz);
            for (int i = 0; i < bsz; ++i) {
                std::vector<float> logits(static_cast<size_t>(area), 0.0f);
                std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area,
                            static_cast<size_t>(area) * sizeof(float));
                const size_t vi = static_cast<size_t>(i) * 3;
                std::array<float, 3> v{vp[vi], vp[vi + 1], vp[vi + 2]};
                out.emplace_back(std::move(logits), v);
            }
            return out;
        };

        auto infer_fn = [&](const std::vector<int8_t>& encoded) {
            auto r = run_forward({encoded});
            return r.front();
        };
        auto batch_infer_fn = [&](const std::vector<std::vector<int8_t>>& batch) {
            return run_forward(batch);
        };

        // --- Warm up the model: first inference on CUDA pays kernel-init cost ---
        std::cerr << "[mcts_bench] warming up model...\n";
        std::mt19937 rng0(0);
        {
            auto init = game.get_initial_state(rng0);
            std::vector<std::vector<int8_t>> dummy(8, init.board);
            for (int i = 0; i < cli.warmup; ++i) run_forward(dummy);
        }

        const auto initial = game.get_initial_state(rng0);
        std::cout << "[mcts_bench] device=" << (use_cuda ? "cuda" : "cpu")
                  << " sims=" << cli.sims
                  << " searches=" << cli.searches
                  << " warmup=" << cli.warmup
                  << " board=" << cfg.board_size
                  << " rule=" << rule_str << "\n";
        std::cout << "variant\tknob\tvalue\tmean_ms\ttotal_sims\tsims_per_sec\n";
        std::cout << std::fixed << std::setprecision(2);

        auto bench_one = [&](const std::string& variant, const std::string& knob, int value,
                             const std::function<void()>& run_searches) {
            const auto t0 = std::chrono::steady_clock::now();
            run_searches();
            if (use_cuda) torch::cuda::synchronize();
            const auto t1 = std::chrono::steady_clock::now();
            const double total_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double mean_ms = total_ms / std::max(1, cli.searches);
            const long long total_sims = static_cast<long long>(cli.searches) * cli.sims;
            const double sims_per_sec = total_sims / (total_ms / 1000.0);
            std::cout << variant << "\t" << knob << "\t" << value
                      << "\t" << mean_ms
                      << "\t" << total_sims
                      << "\t" << sims_per_sec << "\n" << std::flush;
        };

        // ---- Root-parallel sweep (ParallelMCTS, leaf batching) ----
        for (int batch_size : cli.root_batches) {
            std::mt19937 rng(12345);
            ParallelMCTS<Gomoku> mcts(game, cfg, batch_size, infer_fn, batch_infer_fn, rng());
            bench_one("root_parallel", "leaf_batch", batch_size, [&]() {
                for (int i = 0; i < cli.searches; ++i) {
                    std::unique_ptr<MCTSNode> root(new MCTSNode(initial.board, initial.to_play));
                    (void)mcts.search(initial.board, initial.to_play, cli.sims, root);
                }
            });
        }

        // ---- Tree-parallel sweep (TreeParallelMCTS, shared tree + vloss) ----
        for (int n_threads : cli.tree_threads) {
            std::mt19937 rng(12345);
            TreeParallelMCTS<Gomoku> mcts(game, cfg, n_threads, infer_fn, rng());
            bench_one("tree_parallel", "threads", n_threads, [&]() {
                for (int i = 0; i < cli.searches; ++i) {
                    std::unique_ptr<MCTSNode> root(new MCTSNode(initial.board, initial.to_play));
                    (void)mcts.search(initial.board, initial.to_play, cli.sims, root);
                }
            });
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mcts_bench] error: " << e.what() << "\n";
        return 1;
    }
}
