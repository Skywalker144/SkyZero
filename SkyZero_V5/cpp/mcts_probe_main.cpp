// mcts_probe — loads a TorchScript model, runs one MCTS search on an empty
// Gomoku board, and prints the root value (v_mix). Used as a post-export
// diagnostic in the training loop.

#include <array>
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

#include "alphazero.h"
#include "alphazero_parallel.h"
#include "envs/gomoku.h"

using namespace skyzero;

// ---- Config parsing (copied from selfplay_main.cpp) -----------------------
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

// ---- CLI ------------------------------------------------------------------
struct CliArgs {
    std::string model;
    std::string config;
    int num_simulations_override = -1;
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
        else if (k == "--config") a.config = need("--config");
        else if (k == "--num-simulations") a.num_simulations_override = std::stoi(need("--num-simulations"));
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.config.empty()) {
        throw std::runtime_error("usage: mcts_probe --model PATH --config PATH [--num-simulations N]");
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);
        const auto cfg_map = parse_cfg(cli.config);

        AlphaZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 64);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.half_life = cfg_get<int>(cfg_map, "HALF_LIFE", 0);
        cfg.c_puct = cfg_get<float>(cfg_map, "C_PUCT", 1.1f);
        cfg.c_puct_log = cfg_get<float>(cfg_map, "C_PUCT_LOG", 0.45f);
        cfg.c_puct_base = cfg_get<float>(cfg_map, "C_PUCT_BASE", 500.0f);
        cfg.fpu_pow = cfg_get<float>(cfg_map, "FPU_POW", 1.0f);
        cfg.fpu_reduction_max = cfg_get<float>(cfg_map, "FPU_REDUCTION_MAX", 0.08f);
        cfg.root_fpu_reduction_max = cfg_get<float>(cfg_map, "ROOT_FPU_REDUCTION_MAX", 0.0f);
        cfg.fpu_loss_prop = cfg_get<float>(cfg_map, "FPU_LOSS_PROP", 0.0f);
        cfg.enable_stochastic_transform_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", true);
        cfg.enable_stochastic_transform_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", true);
        cfg.enable_symmetry_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_ROOT", false);
        cfg.enable_symmetry_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_CHILD", false);

        // Probe-specific simulation budget: PROBE_NUM_SIMULATIONS in run.cfg
        // (falls back to NUM_SIMULATIONS). CLI --num-simulations still wins.
        cfg.num_simulations = cfg_get<int>(cfg_map, "PROBE_NUM_SIMULATIONS", cfg.num_simulations);
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

        auto model = torch::jit::load(cli.model, device);
        model.eval();
        if (use_cuda) model.to(torch::kHalf);
        std::mutex model_mu;

        // V5: hardcoded c=5, board=15, regardless of game.board_size (padded)
        const int c = Gomoku::NUM_SPATIAL_PLANES_V5;
        const int board = Gomoku::MAX_BOARD_SIZE;
        const int area = board * board;
        constexpr int g_dim = 12;

        // V5: derive globals from encoded for forward
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
                out_iv = model.forward({input, global_t});   // V5 double input
            }
            // V5: dict output
            auto out_dict = out_iv.toGenericDict();
            auto policy_all = out_dict.at("policy").toTensor();
            auto policy_logits = policy_all.select(1, 0).contiguous();   // main head
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

        std::mt19937 rng(std::random_device{}());
        ParallelMCTS<Gomoku> mcts(game, cfg, /*leaf_batch_size=*/1, infer_fn, batch_infer_fn, rng());

        auto init = game.get_initial_state(rng);
        std::unique_ptr<MCTSNode> root(new MCTSNode{init.board, init.to_play});
        const auto sr = mcts.search(init.board, init.to_play, cfg.num_simulations, root);

        std::cout << "[mcts_probe] simulations=" << cfg.num_simulations
                  << " device=" << (use_cuda ? "cuda" : "cpu") << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[mcts_probe] v_mix W=" << sr.v_mix[0]
                  << " D=" << sr.v_mix[1]
                  << " L=" << sr.v_mix[2]
                  << " scalar=" << (sr.v_mix[0] - sr.v_mix[2]) << "\n";
        std::cout << "[mcts_probe] nn_value W=" << sr.nn_value_probs[0]
                  << " D=" << sr.nn_value_probs[1]
                  << " L=" << sr.nn_value_probs[2] << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[mcts_probe] fatal: " << e.what() << "\n";
        return 2;
    }
}
