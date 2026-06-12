// gomoku_play — interactive human-vs-AI Gomoku.
//
// Loads a TorchScript model + play.cfg, runs MCTS synchronously on each AI
// turn, and prints policy / value diagnostics. Mirrors the inference
// scaffolding in mcts_probe_main.cpp and the interactive loop from
// CSkyZero_V3/playgame.h.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poll.h>  // non-blocking stdin check during analysis ponder

#include <torch/script.h>
#include <torch/torch.h>

#include "skyzero_tree_parallel.h"  // transitively provides skyzero.h
#include "envs/gomoku.h"

using namespace skyzero;

// ---- Config parsing (copied from mcts_probe_main.cpp) --------------------
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

// ---- UI helpers (ported from CSkyZero_V3/playgame.h) ---------------------
static void print_board(const std::vector<int8_t>& board, int size, int last_action) {
    std::cout << "   ";
    for (int c = 0; c < size; ++c) {
        std::cout << (c < 10 ? " " : "") << c << ' ';
    }
    std::cout << '\n';

    for (int r = 0; r < size; ++r) {
        std::cout << (r < 10 ? " " : "") << r << ' ';
        for (int c = 0; c < size; ++c) {
            const int loc = r * size + c;
            const int8_t v = board[loc];
            char ch = '.';
            if (v == 1) ch = 'X';
            if (v == -1) ch = 'O';
            if (loc == last_action) {
                std::cout << '[' << ch << ']';
            } else {
                std::cout << ' ' << ch << ' ';
            }
        }
        std::cout << '\n';
    }
}

static int argmax_index(const std::vector<float>& p) {
    if (p.empty()) return -1;
    return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
}

// `p` is canvas-stride (length MAX_AREA, indexed r*stride+c). We display only
// the in-board sub-grid [0, board_size) x [0, board_size).
static void print_policy_grid(const std::vector<float>& p, int board_size, int stride,
                              const std::string& title) {
    std::cout << title << "\n";
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            const int idx = r * stride + c;
            const float v = (idx >= 0 && idx < static_cast<int>(p.size())) ? p[idx] : 0.0f;
            if (v < 0.01f) {
                std::cout << "     . ";  // aligned with "%6.2f "
            } else {
                std::cout << std::setw(6) << std::fixed << std::setprecision(2) << v << ' ';
            }
        }
        std::cout << '\n';
    }
}

// Signed variant for futurepos (values in [-1, +1] after tanh; sign encodes
// own (+) vs opponent (-) future stone). Width matches print_policy_grid so
// the Python-side whitespace-tokenized parser handles both.
static void print_signed_grid(const std::vector<float>& p, int board_size, int stride,
                              const std::string& title, float thresh = 0.05f) {
    std::cout << title << "\n";
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            const int idx = r * stride + c;
            const float v = (idx >= 0 && idx < static_cast<int>(p.size())) ? p[idx] : 0.0f;
            if (std::fabs(v) < thresh) {
                std::cout << "     . ";  // 7 chars total, matches "%+6.2f "
            } else {
                std::cout << std::setw(6) << std::showpos << std::fixed << std::setprecision(2)
                          << v << std::noshowpos << ' ';
            }
        }
        std::cout << '\n';
    }
}

static bool parse_row_col(const std::string& text, int& row, int& col) {
    std::istringstream iss(text);
    if (!(iss >> row >> col)) return false;
    std::string remain;
    if (iss >> remain) return false;
    return true;
}

static bool parse_human_side(const std::string& text, int& side) {
    std::istringstream iss(text);
    if (!(iss >> side)) return false;
    std::string remain;
    if (iss >> remain) return false;
    return side == 1 || side == -1;
}

// ---- CLI -----------------------------------------------------------------
struct CliArgs {
    std::string model;
    std::string config;
    int num_simulations_override = -1;
    int human_side_override = 0;  // 0 = ask interactively
    int board_size_override = -1;
    std::string rule_override;  // empty = use cfg
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
        else if (k == "--human-side") a.human_side_override = std::stoi(need("--human-side"));
        else if (k == "--board-size") a.board_size_override = std::stoi(need("--board-size"));
        else if (k == "--rule") a.rule_override = need("--rule");
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.config.empty()) {
        throw std::runtime_error("usage: gomoku_play --model PATH --config PATH "
                                 "[--num-simulations N] [--human-side {1,-1}] "
                                 "[--board-size N] [--rule {renju,standard,freestyle}]");
    }
    return a;
}

// ---- Game loop state -----------------------------------------------------
struct Snapshot {
    std::vector<int8_t> state;
    int to_play = 1;
    int last_action = -1;
    int last_player = 0;
};

int main(int argc, char** argv) {
    try {
        // Line-buffer stdout so GUI front-ends see prompts/boards promptly.
        std::setvbuf(stdout, nullptr, _IOLBF, 0);

        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);
        const auto cfg_map = parse_cfg(cli.config);

        SkyZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 800);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.non_root_search_algo = SkyZeroConfig::parse_non_root_search_algo(
            cfg_get<std::string>(cfg_map, "NON_ROOT_SEARCH_ALGO", "puct"));
        cfg.root_search_algo = SkyZeroConfig::parse_root_search_algo(
            cfg_get<std::string>(cfg_map, "ROOT_SEARCH_ALGO", "gumbel"));
        // PUCT-root knobs, match-mode defaults: no Dirichlet noise, no forced
        // playouts, argmax move selection (temperature 0).
        cfg.root_noise_enabled = cfg_get_bool(cfg_map, "ROOT_NOISE_ENABLED", false);
        cfg.root_fpu_reduction_max =
            cfg_get<float>(cfg_map, "ROOT_FPU_REDUCTION_MAX", 0.0f);
        cfg.root_desired_per_child_visits_coeff =
            cfg_get<float>(cfg_map, "ROOT_DESIRED_PER_CHILD_VISITS_COEFF", 0.0f);
        cfg.chosen_move_temperature =
            cfg_get<float>(cfg_map, "CHOSEN_MOVE_TEMPERATURE", 0.0f);
        cfg.chosen_move_temperature_early =
            cfg_get<float>(cfg_map, "CHOSEN_MOVE_TEMPERATURE_EARLY", 0.0f);
        cfg.chosen_move_temperature_halflife =
            cfg_get<float>(cfg_map, "CHOSEN_MOVE_TEMPERATURE_HALFLIFE", 19.0f);
        cfg.validate();
        cfg.gumbel_noise_enabled = cfg_get_bool(cfg_map, "GUMBEL_NOISE_ENABLED", true);
        cfg.c_puct = cfg_get<float>(cfg_map, "C_PUCT", 1.1f);
        cfg.c_puct_log = cfg_get<float>(cfg_map, "C_PUCT_LOG", 0.45f);
        cfg.c_puct_base = cfg_get<float>(cfg_map, "C_PUCT_BASE", 500.0f);
        cfg.fpu_pow = cfg_get<float>(cfg_map, "FPU_POW", 1.0f);
        cfg.fpu_reduction_max = cfg_get<float>(cfg_map, "FPU_REDUCTION_MAX", 0.08f);
        cfg.fpu_loss_prop = cfg_get<float>(cfg_map, "FPU_LOSS_PROP", 0.0f);
        cfg.cpuct_utility_stdev_prior = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR", 0.40f);
        cfg.cpuct_utility_stdev_prior_weight = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR_WEIGHT", 2.0f);
        cfg.cpuct_utility_stdev_scale = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_SCALE", 0.85f);
        cfg.enable_stochastic_transform_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", false);
        cfg.enable_stochastic_transform_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", false);
        // Default false, matching every shipped cfg (all set 0 explicitly).
        cfg.enable_symmetry_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_ROOT", false);
        cfg.enable_symmetry_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_CHILD", false);
        cfg.root_symmetry_pruning =
            cfg_get_bool(cfg_map, "ROOT_SYMMETRY_PRUNING", true);

        if (cli.num_simulations_override >= 0) cfg.num_simulations = cli.num_simulations_override;
        if (cli.board_size_override > 0) {
            if (cli.board_size_override > Gomoku::MAX_BOARD_SIZE) {
                throw std::runtime_error("--board-size exceeds compile-time MAX_BOARD_SIZE");
            }
            cfg.board_size = cli.board_size_override;
        }

        const std::string rule_str = ([&]() -> std::string {
            if (!cli.rule_override.empty()) return cli.rule_override;
            auto it = cfg_map.find("RULE");
            return (it != cfg_map.end()) ? it->second : "renju";
        })();
        const RuleType rule = rule_from_string(rule_str);
        Gomoku game(cfg.board_size, rule, /*forbidden_plane=*/rule != RuleType::FREESTYLE);

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);
        cfg.device = device;

        auto model = torch::jit::load(cli.model, device);
        model.eval();
        if (use_cuda) model.to(torch::kHalf);
        std::mutex model_mu;

        // V5: NUM_SPATIAL_PLANES_V5 planes on a MAX_BOARD_SIZE canvas (MAX_AREA cells).
        const int c = Gomoku::NUM_SPATIAL_PLANES_V5;
        const int board = Gomoku::MAX_BOARD_SIZE;
        const int area = board * board;
        constexpr int g_dim = 12;

        // V5 helper: derive globals from encoded state (game.rule + ply parity).
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
                torch::NoGradGuard no_grad;
                out_iv = model.forward({input, global_t});   // V5: double input
            }
            // V5: dict output. policy (B, 4, area), value_wdl (B, 3).
            auto out_dict = out_iv.toGenericDict();
            auto policy_all = out_dict.at("policy").toTensor();      // (B, 4, area)
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

        // Single forward to extract the opponent-policy head (elements()[1]).
        // Run independently from MCTS so the search engine's infer_fn signature
        // stays unchanged. Applies legal-move masking + softmax, mirroring the
        // policy-head pipeline in inference(). Optionally also fills the two
        // futurepos planes (+8 / +32 step occupancy, tanh-mapped to [-1, +1]
        // from to_play perspective; nets.py:178, train.py:141) when out
        // pointers are non-null — saves a second forward pass.
        auto forward_opp_policy = [&](const std::vector<int8_t>& state, int to_play,
                                       std::vector<float>* fp8_out = nullptr,
                                       std::vector<float>* fp32_out = nullptr) {
            auto encoded = game.encode_state_v5(state, to_play);   // V5
            std::vector<float> input_buf(static_cast<size_t>(c) * area, 0.0f);
            for (int j = 0; j < c * area; ++j) {
                input_buf[j] = static_cast<float>(encoded[j]);
            }
            auto input = torch::from_blob(input_buf.data(), {1, c, board, board}, torch::kFloat32)
                             .clone().to(device);
            // V5: build globals tensor from state
            int ply = 0;
            for (auto v : state) if (v != 0) ++ply;
            auto gf = game.compute_global_features(ply, to_play);
            auto global_t = torch::from_blob(gf.data, {1, g_dim}, torch::kFloat32).clone().to(device);
            if (device.is_cuda()) {
                input = input.to(torch::kHalf);
                global_t = global_t.to(torch::kHalf);
            }

            torch::jit::IValue out_iv;
            {
                std::lock_guard<std::mutex> lk(model_mu);
                torch::NoGradGuard no_grad;
                out_iv = model.forward({input, global_t});   // V5
            }
            // V5: opp policy = policy[:, 1, :] (slim 4-output head in nets.py:70-83;
            // idx layout is main / opp / soft_main / soft_opp).
            auto out_dict = out_iv.toGenericDict();
            auto policy_all = out_dict.at("policy").toTensor();   // (1, 4, area)
            auto opp_logits = policy_all.select(1, 1).contiguous();
            auto opp = opp_logits.reshape({1, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
            const float* op = opp.data_ptr<float>();

            std::vector<float> logits(static_cast<size_t>(area), 0.0f);
            std::memcpy(logits.data(), op, static_cast<size_t>(area) * sizeof(float));
            // logits is canvas-stride (MAX_AREA); use the canvas legal mask.
            const auto legal = game.get_is_legal_actions_canvas(state, to_play);
            for (size_t i = 0; i < logits.size(); ++i) {
                if (i >= legal.size() || !legal[i]) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
            float maxv = -std::numeric_limits<float>::infinity();
            for (float v : logits) if (v > maxv) maxv = v;
            std::vector<float> probs(logits.size(), 0.0f);
            float sum = 0.0f;
            if (std::isfinite(maxv)) {
                for (size_t i = 0; i < logits.size(); ++i) {
                    probs[i] = std::isfinite(logits[i]) ? std::exp(logits[i] - maxv) : 0.0f;
                    sum += probs[i];
                }
                if (sum > 0.0f) for (auto& v : probs) v /= sum;
            }

            // Futurepos: pre-tanh logits at "value_futurepos", shape (1, 2, H, W).
            // Loss applies tanh; mirror that here so the displayed values are
            // in the same [-1, +1] domain as the int8 targets.
            if (fp8_out != nullptr || fp32_out != nullptr) {
                auto fp_t = out_dict.at("value_futurepos").toTensor()
                              .to(torch::kFloat32).to(torch::kCPU).contiguous();
                auto fp_tanh = torch::tanh(fp_t).contiguous();
                const float* fpp = fp_tanh.data_ptr<float>();
                if (fp8_out)  fp8_out->assign(fpp,             fpp + area);
                if (fp32_out) fp32_out->assign(fpp + area,     fpp + 2 * area);
            }
            return probs;
        };

        std::mt19937 rng(std::random_device{}());
        const int search_threads = cfg_get<int>(cfg_map, "SEARCH_THREADS_PER_TREE", 8);
        const int leaf_batch_size = cfg_get<int>(cfg_map, "LEAF_BATCH_SIZE", 1);
        const int batch_timeout_us = cfg_get<int>(cfg_map, "BATCH_TIMEOUT_US", 0);

        // run_forward already takes a batched encoded list and returns N
        // (policy, value) pairs — its signature matches BatchInferenceFn
        // exactly, so we can hand it to the batched MCTS constructor as-is.
        TreeParallelMCTS<Gomoku>::BatchInferenceFn batch_infer_fn = run_forward;
        std::unique_ptr<TreeParallelMCTS<Gomoku>> mcts_ptr;
        if (leaf_batch_size > 1) {
            mcts_ptr.reset(new TreeParallelMCTS<Gomoku>(
                game, cfg, search_threads, leaf_batch_size, batch_timeout_us,
                batch_infer_fn, rng()));
        } else {
            mcts_ptr.reset(new TreeParallelMCTS<Gomoku>(
                game, cfg, search_threads, infer_fn, rng()));
        }
        TreeParallelMCTS<Gomoku>& mcts = *mcts_ptr;

        std::cout << "[gomoku_play] model=" << cli.model
                  << " device=" << (use_cuda ? "cuda" : "cpu")
                  << " simulations=" << cfg.num_simulations
                  << " search_threads=" << search_threads
                  << " leaf_batch_size=" << leaf_batch_size
                  << " batch_timeout_us=" << batch_timeout_us
                  << " symmetry_root=" << cfg.enable_symmetry_inference_for_root
                  << " symmetry_child=" << cfg.enable_symmetry_inference_for_child
                  << "\n";

        // --- Prompt human side ------------------------------------------------
        int human_side = cli.human_side_override;
        while (human_side != 1 && human_side != -1) {
            std::cout
                << "1 for Human first move (Black) and -1 for Human second move (White)\n"
                << "Enter: ";
            std::string line;
            if (!std::getline(std::cin, line)) return 0;
            if (!parse_human_side(line, human_side)) {
                std::cout << "Invalid input. Please enter 1 or -1.\n";
                human_side = 0;
            }
        }

        // --- Game state (persists across `newgame` so the MCTS engine,
        //     model, and batcher threads survive between rounds) -------------
        std::vector<int8_t> state;
        int to_play = 1;
        int last_action = -1;
        int last_player = 0;
        std::unique_ptr<MCTSNode> root;
        std::vector<Snapshot> history;

        auto reset_game = [&]() {
            auto init = game.get_initial_state(rng);
            state = std::move(init.board);
            to_play = init.to_play;
            last_action = -1;
            last_player = 0;
            root.reset(new MCTSNode(state, to_play));
            history.clear();
        };

        reset_game();
        print_board(state, game.board_size, last_action);

        auto push_history = [&]() {
            history.push_back(Snapshot{state, to_play, last_action, last_player});
        };

        // Returns true if `input` was a recognised setting command (handled).
        auto try_handle_setting = [&](const std::string& input) -> bool {
            std::istringstream iss(input);
            std::string kw;
            if (!(iss >> kw)) return false;
            if (kw == "sims") {
                int n = -1;
                if (!(iss >> n) || n < 0) {
                    std::cout << "Invalid input: sims requires N >= 0 (0 = pure NN argmax).\n";
                    return true;
                }
                cfg.num_simulations = n;
                std::cout << "[setting] num_simulations=" << n << "\n";
                return true;
            }
            if (kw == "gm") {
                int m = 0;
                if (!(iss >> m) || m < 1) {
                    std::cout << "Invalid input: gm requires M >= 1.\n";
                    return true;
                }
                cfg.gumbel_m = m;
                std::cout << "[setting] gumbel_m=" << m << "\n";
                return true;
            }
            if (kw == "noise") {
                int v = -1;
                if (!(iss >> v) || (v != 0 && v != 1)) {
                    std::cout << "Invalid input: noise requires 0 or 1.\n";
                    return true;
                }
                cfg.gumbel_noise_enabled = (v == 1);
                std::cout << "[setting] gumbel_noise_enabled=" << v << "\n";
                return true;
            }
            if (kw == "prune") {
                int v = -1;
                if (!(iss >> v) || (v != 0 && v != 1)) {
                    std::cout << "Invalid input: prune requires 0 or 1.\n";
                    return true;
                }
                cfg.root_symmetry_pruning = (v == 1);
                std::cout << "[setting] root_symmetry_pruning=" << v << "\n";
                return true;
            }
            return false;
        };

        auto undo_two_moves = [&]() -> bool {
            if (history.size() < 2) return false;
            history.pop_back();
            Snapshot restore = history.back();
            history.pop_back();
            state = std::move(restore.state);
            to_play = restore.to_play;
            last_action = restore.last_action;
            last_player = restore.last_player;
            root.reset(new MCTSNode(state, to_play));
            return true;
        };

        // `newgame [1|-1]` — reset the board (optionally swapping human side)
        // without tearing down the process. Lets play_web start a fresh game
        // for ~150ms instead of paying ~2.5s to respawn gomoku_play.
        auto try_handle_newgame = [&](const std::string& input) -> bool {
            std::istringstream iss(input);
            std::string kw;
            if (!(iss >> kw) || kw != "newgame") return false;
            int s = human_side;
            if (iss >> s) {
                if (s != 1 && s != -1) {
                    std::cout << "Invalid newgame: side must be 1 or -1.\n";
                    return true;
                }
                human_side = s;
            }
            reset_game();
            std::cout << "[setting] newgame human_side=" << human_side << "\n";
            print_board(state, game.board_size, last_action);
            return true;
        };

        // --- Diagnostic printing shared by the AI move and analysis paths ---
        constexpr int MA = Gomoku::MAX_BOARD_SIZE;
        auto print_value_table = [&](const MCTSSearchOutput& o) {
            const float root_value = o.v_mix[0] - o.v_mix[2];
            const float nn_value = o.nn_value_probs[0] - o.nn_value_probs[2];
            std::cout
                << "          "
                << std::setw(7) << "Win" << "  " << std::setw(7) << "Draw" << "  "
                << std::setw(7) << "Loss" << "  " << std::setw(7) << "W-L" << '\n'
                << "  root:   "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.v_mix[0] * 100.0f) << "%  "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.v_mix[1] * 100.0f) << "%  "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.v_mix[2] * 100.0f) << "%  "
                << std::showpos << std::fixed << std::setprecision(2) << root_value << std::noshowpos << '\n'
                << "  nn:     "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.nn_value_probs[0] * 100.0f) << "%  "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.nn_value_probs[1] * 100.0f) << "%  "
                << std::setw(6) << std::fixed << std::setprecision(2) << (o.nn_value_probs[2] * 100.0f) << "%  "
                << std::showpos << std::fixed << std::setprecision(2) << nn_value << std::noshowpos << '\n';
        };
        auto print_phases = [&](const MCTSSearchOutput& o) {
            for (size_t i = 0; i < o.gumbel_phases.size(); ++i) {
                std::cout << "Gumbel Phase " << i << " (" << o.gumbel_phases[i].size() << "):";
                for (int a : o.gumbel_phases[i]) {
                    std::cout << ' ' << (a / MA) << ',' << (a % MA);
                }
                std::cout << '\n';
            }
        };
        auto print_winrate = [&](const MCTSSearchOutput& o) {
            // Per-candidate win rate (expected score, draws = 0.5) from the root
            // player's view, canvas-stride. Unvisited cells print blank.
            std::vector<float> winrate(o.root_child_wdl.size(), 0.0f);
            for (size_t i = 0; i < o.root_child_wdl.size(); ++i) {
                const auto& w = o.root_child_wdl[i];
                if (w[0] + w[1] + w[2] > 0.5f) winrate[i] = (w[0] - w[2] + 1.0f) * 0.5f;
            }
            print_policy_grid(winrate, game.board_size, MA, "Gumbel WinRate:");
        };

        // --- Analysis / ponder: continuous PUCT search on the current position
        //     (side to move) until the front-end sends `stop`. Does NOT play a
        //     move. Returns 1 if the client asked to quit, else 0. ----------
        const int analyze_chunk = cfg_get<int>(cfg_map, "ANALYZE_CHUNK_SIMS", 128);
        // Command received mid-analyze; replayed by the human-step loop after
        // the analyze loop exits, as if freshly typed.
        std::string pending_cmd;
        auto stdin_has_input = []() -> bool {
            // Lines already slurped into the stdio buffer (e.g. a client that
            // pipelined "analyze\nstop\n" in one write) are invisible to
            // poll(2) on the fd — check the buffer first.
            if (std::cin.rdbuf()->in_avail() > 0) return true;
            struct pollfd pfd;
            pfd.fd = 0; pfd.events = POLLIN; pfd.revents = 0;
            return ::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN);
        };
        auto run_analyze = [&]() -> int {
            root.reset(new MCTSNode(state, to_play));
            mcts.ensure_root_expanded(*root);
            std::cout << "Analyze start\n";
            // NN-only views don't change as the search deepens: emit them once.
            {
                std::vector<float> fp8, fp32;
                auto opp_policy = forward_opp_policy(state, to_play, &fp8, &fp32);
                print_policy_grid(root->nn_policy, game.board_size, MA, "NN Strategy:");
                print_policy_grid(opp_policy, game.board_size, MA, "NN Opp Strategy:");
                print_signed_grid(fp8,  game.board_size, MA, "NN Futurepos +8:");
                print_signed_grid(fp32, game.board_size, MA, "NN Futurepos +32:");
            }
            std::cout.flush();
            while (true) {
                if (stdin_has_input()) {
                    std::string cmd;
                    if (!std::getline(std::cin, cmd)) return 1;
                    if (cmd == "stop" || cmd == "s") break;
                    if (cmd == "q" || cmd == "Q") return 1;
                    // Any other command (side / newgame / a move) means "stop
                    // analyzing, then do that". Dropping it would silently
                    // desync the client, which gets no reply otherwise.
                    if (!cmd.empty()) { pending_cmd = cmd; break; }
                }
                mcts.run_puct_sims(*root, analyze_chunk);
                auto out = mcts.report_analysis(*root);
                print_policy_grid(out.mcts_policy, game.board_size, MA, "MCTS Strategy (improved policy):");
                std::vector<float> visit_dist(out.visit_counts.size(), 0.0f);
                {
                    const float sum_n = std::accumulate(out.visit_counts.begin(), out.visit_counts.end(), 0.0f);
                    if (sum_n > 0.0f) {
                        for (size_t i = 0; i < out.visit_counts.size(); ++i) {
                            visit_dist[i] = out.visit_counts[i] / sum_n;
                        }
                    }
                }
                print_policy_grid(visit_dist, game.board_size, MA, "MCTS Visits (N(s,a)/sum):");
                print_value_table(out);
                print_phases(out);
                print_winrate(out);
                std::cout.flush();
            }
            std::cout << "Analyze stopped\n";
            root.reset(new MCTSNode(state, to_play));
            print_board(state, game.board_size, last_action);
            return 0;
        };

        while (true) {
            if (game.is_terminal(state, last_action, last_player)) {
                const int winner = game.get_winner(state, last_action, last_player);
                if (winner == 1) std::cout << "Black wins!\n";
                else if (winner == -1) std::cout << "White wins!\n";
                else std::cout << "Draw!\n";

                bool resume = false;
                while (!resume) {
                    std::cout << "Game Over. 'u' to undo, 'newgame [1|-1]' for new game, 'q' to quit: ";
                    std::string resp;
                    if (!std::getline(std::cin, resp)) return 0;
                    if (resp == "u" || resp == "U") {
                        if (undo_two_moves()) {
                            std::cout << "Undo successful.\n";
                            print_board(state, game.board_size, last_action);
                            resume = true;
                            break;
                        }
                        std::cout << "Nothing to undo.\n";
                        continue;
                    }
                    if (try_handle_newgame(resp)) {
                        resume = true;
                        break;
                    }
                    if (resp == "q" || resp == "Q") return 0;
                    std::cout << "Unrecognized command.\n";
                }
                continue;
            }

            if (to_play == human_side) {
                while (true) {
                    std::string input;
                    if (!pending_cmd.empty()) {
                        input = pending_cmd;
                        pending_cmd.clear();
                    } else {
                        std::cout << "Human step (row col / 'u' for undo / 'q' for quit):\n";
                        std::cout.flush();
                        if (!std::getline(std::cin, input)) return 0;
                    }

                    if (input == "u" || input == "U") {
                        if (undo_two_moves()) {
                            std::cout << "Undo successful.\n";
                            print_board(state, game.board_size, last_action);
                        } else {
                            std::cout << "Nothing to undo.\n";
                        }
                        continue;
                    }
                    if (input == "q" || input == "Q") {
                        std::cout << "Exiting game.\n";
                        return 0;
                    }
                    if (try_handle_setting(input)) {
                        continue;
                    }
                    if (try_handle_newgame(input)) {
                        break;  // back to top of play loop, re-check whose turn
                    }
                    {
                        std::istringstream iss(input);
                        std::string kw;
                        if ((iss >> kw) && kw == "side") {
                            int s = 0;
                            if (!(iss >> s) || (s != 1 && s != -1)) {
                                std::cout << "Invalid input: side requires 1 or -1.\n";
                                continue;
                            }
                            human_side = s;
                            std::cout << "[setting] human_side=" << s << "\n";
                            break;  // re-check whose turn it is
                        }
                    }

                    if (input == "analyze" || input == "a") {
                        if (run_analyze() == 1) { std::cout << "Exiting game.\n"; return 0; }
                        continue;  // back to the Human step prompt
                    }

                    int row = -1, col = -1;
                    if (!parse_row_col(input, row, col)) {
                        std::cout << "Invalid input format. Please enter 'row col' (e.g., '7 7').\n";
                        continue;
                    }
                    if (row < 0 || row >= game.board_size || col < 0 || col >= game.board_size) {
                        std::cout << "Invalid move: out of board range.\n";
                        continue;
                    }

                    const int action = row * game.board_size + col;
                    const auto legal = game.get_is_legal_actions(state, to_play);
                    if (!legal[action]) {
                        std::cout << "Invalid move: (" << row << ", " << col
                                  << ") is forbidden or occupied.\n";
                        continue;
                    }

                    push_history();
                    state = game.get_next_state(state, action, to_play);
                    last_action = action;
                    last_player = to_play;
                    to_play = -to_play;
                    root.reset(new MCTSNode(state, to_play));
                    break;
                }
            } else {
                push_history();
                std::cout << "SkyZero thinking...\n";

                const auto out = mcts.search(state, to_play, cfg.num_simulations, root);
                // MCTS / NN outputs are canvas-stride (length MAX_AREA, indexed
                // r*MAX_BOARD_SIZE+c). game state stays board-stride. Translate
                // at the boundary, mirroring selfplay_manager.h:577.
                constexpr int M = Gomoku::MAX_BOARD_SIZE;
                int canvas_action = out.gumbel_action;
                if (canvas_action < 0) canvas_action = argmax_index(out.mcts_policy);
                if (canvas_action < 0) {
                    std::cout << "No legal action found. Exiting game.\n";
                    return 1;
                }
                const int action = Gomoku::canvas_pos_to_loc(canvas_action, game.board_size);
                if (action < 0) {
                    std::cout << "AI returned off-board canvas action " << canvas_action
                              << ". Exiting game.\n";
                    return 1;
                }
                const int row = action / game.board_size;
                const int col = action % game.board_size;

                print_policy_grid(out.mcts_policy, game.board_size, M, "MCTS Strategy (improved policy):");
                std::vector<float> visit_dist(out.visit_counts.size(), 0.0f);
                {
                    const float sum_n = std::accumulate(out.visit_counts.begin(), out.visit_counts.end(), 0.0f);
                    if (sum_n > 0.0f) {
                        for (size_t i = 0; i < out.visit_counts.size(); ++i) {
                            visit_dist[i] = out.visit_counts[i] / sum_n;
                        }
                    }
                }
                print_policy_grid(visit_dist, game.board_size, M, "MCTS Visits (N(s,a)/sum):");
                print_policy_grid(out.nn_policy, game.board_size, M, "NN Strategy:");
                {
                    std::vector<float> fp8, fp32;
                    auto opp_policy = forward_opp_policy(state, to_play, &fp8, &fp32);
                    print_policy_grid(opp_policy, game.board_size, M, "NN Opp Strategy:");
                    print_signed_grid(fp8,  game.board_size, M, "NN Futurepos +8:");
                    print_signed_grid(fp32, game.board_size, M, "NN Futurepos +32:");
                }

                print_value_table(out);
                print_phases(out);
                print_winrate(out);
                std::cout << "AI move: (" << row << ", " << col << ")\n";

                state = game.get_next_state(state, action, to_play);
                last_action = action;
                last_player = to_play;
                to_play = -to_play;
                root.reset(new MCTSNode(state, to_play));
            }

            print_board(state, game.board_size, last_action);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gomoku_play] fatal: " << e.what() << "\n";
        return 2;
    }
}
