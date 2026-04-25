// gomoku_eval — headless model-A-vs-model-B match runner.
//
// Loads two TorchScript models, plays N games alternating colors (A-black on
// even game indices, B-black on odd), and appends one JSON line per game to
// the output file. Designed to feed python/elo.py.
//
// Shares config/inference scaffolding with gomoku_play_main.cpp.

#include <algorithm>
#include <array>
#include <cstdint>
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
            "usage: gomoku_eval --model-a PATH --model-b PATH --config PATH --output PATH "
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

        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 4);
        Gomoku game(cfg.board_size, /*renju=*/true, /*forbidden_plane=*/num_planes >= 4);
        if (cfg.half_life <= 0) cfg.half_life = game.board_size;

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);
        cfg.device = device;

        auto ha = load_model(cli.model_a, device);
        auto hb = load_model(cli.model_b, device);

        const int c = game.num_planes;
        const int board = game.board_size;
        const int area = board * board;

        auto make_run_forward = [&](ModelHandle* h) {
            return [&, h](const std::vector<std::vector<int8_t>>& batch)
                       -> std::vector<std::pair<std::vector<float>, std::array<float, 3>>> {
                const int bsz = static_cast<int>(batch.size());
                std::vector<float> input_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
                for (int i = 0; i < bsz; ++i) {
                    const auto& enc = batch[i];
                    if (enc.size() != static_cast<size_t>(c * area)) {
                        throw std::runtime_error("encoded size mismatch");
                    }
                    const size_t base = static_cast<size_t>(i) * c * area;
                    for (int j = 0; j < c * area; ++j) {
                        input_buf[base + j] = static_cast<float>(enc[j]);
                    }
                }
                auto input = torch::from_blob(input_buf.data(), {bsz, c, board, board}, torch::kFloat32)
                                 .clone().to(device);
                if (device.is_cuda()) input = input.to(torch::kHalf);

                torch::jit::IValue out_iv;
                {
                    std::lock_guard<std::mutex> lk(h->mu);
                    torch::NoGradGuard no_grad2;
                    out_iv = h->module.forward({input});
                }
                auto tuple = out_iv.toTuple();
                auto policy_logits = tuple->elements()[0].toTensor();
                auto value_logits = tuple->elements()[2].toTensor();
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
        };

        auto fwd_a = make_run_forward(ha.get());
        auto fwd_b = make_run_forward(hb.get());
        auto infer_a = [&](const std::vector<int8_t>& e) { return fwd_a({e}).front(); };
        auto infer_b = [&](const std::vector<int8_t>& e) { return fwd_b({e}).front(); };

        const int search_threads = cfg_get<int>(cfg_map, "SEARCH_THREADS_PER_TREE", 8);
        const uint64_t seed = cli.seed_set ? cli.seed : std::random_device{}();
        std::mt19937 rng(seed);

        TreeParallelMCTS<Gomoku> mcts_a(game, cfg, search_threads, infer_a, fwd_a, rng());
        TreeParallelMCTS<Gomoku> mcts_b(game, cfg, search_threads, infer_b, fwd_b, rng());

        std::ofstream out(cli.output, std::ios::app);
        if (!out) throw std::runtime_error("cannot open output: " + cli.output);

        std::cerr << "[gomoku_eval] A=" << cli.model_a << "\n"
                  << "              B=" << cli.model_b << "\n"
                  << "              device=" << (use_cuda ? "cuda" : "cpu")
                  << " sims=" << cfg.num_simulations
                  << " threads=" << search_threads
                  << " games=" << cli.num_games
                  << " seed=" << seed << "\n";

        int a_wins = 0, b_wins = 0, draws = 0;

        for (int g = 0; g < cli.num_games; ++g) {
            // Even g: A plays black (to_play=1). Odd g: B plays black.
            const bool a_is_black = (g % 2 == 0);
            auto init = game.get_initial_state(rng);
            std::vector<int8_t> state = std::move(init.board);
            int to_play = init.to_play;
            int last_action = -1;
            int last_player = 0;
            int plies = 0;

            std::unique_ptr<MCTSNode> root_a(new MCTSNode{state, to_play});
            std::unique_ptr<MCTSNode> root_b(new MCTSNode{state, to_play});

            while (!game.is_terminal(state, last_action, last_player)) {
                // Which model moves? Black = to_play==1.
                const bool a_to_move = (a_is_black && to_play == 1) || (!a_is_black && to_play == -1);
                auto& mcts = a_to_move ? mcts_a : mcts_b;
                auto& root = a_to_move ? root_a : root_b;

                root.reset(new MCTSNode{state, to_play});
                const auto res = mcts.search(state, to_play, cfg.num_simulations, root);
                int action = res.gumbel_action;
                if (action < 0) {
                    // Fallback: first legal.
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
            // winner_a: +1 if A won, -1 if B won, 0 draw.
            int winner_a = 0;
            if (winner != 0) {
                const int a_side = a_is_black ? 1 : -1;
                winner_a = (winner == a_side) ? 1 : -1;
            }
            if (winner_a > 0) ++a_wins;
            else if (winner_a < 0) ++b_wins;
            else ++draws;

            out << "{\"a\":\"" << cli.model_a << "\","
                << "\"b\":\"" << cli.model_b << "\","
                << "\"a_black\":" << (a_is_black ? "true" : "false") << ","
                << "\"winner_a\":" << winner_a << ","
                << "\"plies\":" << plies << "}\n";
            out.flush();

            std::cerr << "[gomoku_eval] game " << (g + 1) << "/" << cli.num_games
                      << " a_black=" << (a_is_black ? 1 : 0)
                      << " winner_a=" << winner_a
                      << " plies=" << plies
                      << " | A:" << a_wins << " D:" << draws << " B:" << b_wins << "\n";
        }

        const float n = static_cast<float>(cli.num_games);
        const float score = (a_wins + 0.5f * draws) / n;
        std::cerr << "[gomoku_eval] done. A score=" << std::fixed << std::setprecision(3) << score
                  << " (" << a_wins << "W " << draws << "D " << b_wins << "L)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gomoku_eval] fatal: " << e.what() << "\n";
        return 2;
    }
}
