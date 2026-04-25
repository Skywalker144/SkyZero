// gomoku_play — interactive human-vs-AI Gomoku.
//
// Loads a TorchScript model + play.cfg, runs MCTS synchronously on each AI
// turn, and prints policy / value diagnostics. Mirrors the inference
// scaffolding in mcts_probe_main.cpp and the interactive loop from
// CSkyZero_V3/playgame.h.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
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

#include <torch/script.h>
#include <torch/torch.h>

#include "alphazero_tree_parallel.h"  // transitively provides alphazero.h
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

static void print_policy_grid(const std::vector<float>& p, int board_size, const std::string& title) {
    std::cout << title << "\n";
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            const int idx = r * board_size + c;
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
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.config.empty()) {
        throw std::runtime_error("usage: gomoku_play --model PATH --config PATH "
                                 "[--num-simulations N] [--human-side {1,-1}]");
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

        AlphaZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 800);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.gumbel_noise_enabled = cfg_get_bool(cfg_map, "GUMBEL_NOISE_ENABLED", true);
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

        if (cli.num_simulations_override > 0) cfg.num_simulations = cli.num_simulations_override;

        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 4);
        Gomoku game(cfg.board_size, /*renju=*/true, /*forbidden_plane=*/num_planes >= 4);
        if (cfg.half_life <= 0) cfg.half_life = game.board_size;

        const bool use_cuda = torch::cuda::is_available();
        const torch::Device device = use_cuda ? torch::Device(torch::kCUDA, 0)
                                              : torch::Device(torch::kCPU);
        cfg.device = device;

        auto model = torch::jit::load(cli.model, device);
        model.eval();
        if (use_cuda) model.to(torch::kHalf);
        std::mutex model_mu;

        const int c = game.num_planes;
        const int board = game.board_size;
        const int area = board * board;

        auto run_forward = [&](const std::vector<std::vector<int8_t>>& batch)
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
                std::lock_guard<std::mutex> lk(model_mu);
                torch::NoGradGuard no_grad;
                out_iv = model.forward({input});
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

        auto infer_fn = [&](const std::vector<int8_t>& encoded) {
            auto r = run_forward({encoded});
            return r.front();
        };
        auto batch_infer_fn = [&](const std::vector<std::vector<int8_t>>& batch) {
            return run_forward(batch);
        };

        std::mt19937 rng(std::random_device{}());
        const int search_threads = cfg_get<int>(cfg_map, "SEARCH_THREADS_PER_TREE", 8);
        TreeParallelMCTS<Gomoku> mcts(game, cfg, search_threads, infer_fn, batch_infer_fn, rng());

        std::cout << "[gomoku_play] model=" << cli.model
                  << " device=" << (use_cuda ? "cuda" : "cpu")
                  << " simulations=" << cfg.num_simulations
                  << " search_threads=" << search_threads
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

        // --- Game state ------------------------------------------------------
        auto init = game.get_initial_state(rng);
        std::vector<int8_t> state = std::move(init.board);
        int to_play = init.to_play;
        int last_action = -1;
        int last_player = 0;

        std::unique_ptr<MCTSNode> root(new MCTSNode{state, to_play});
        std::vector<Snapshot> history;
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
                int n = 0;
                if (!(iss >> n) || n < 1) {
                    std::cout << "Invalid input: sims requires N >= 1.\n";
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
            root.reset(new MCTSNode{state, to_play});
            return true;
        };

        while (true) {
            if (game.is_terminal(state, last_action, last_player)) {
                const int winner = game.get_winner(state, last_action, last_player);
                if (winner == 1) std::cout << "Black wins!\n";
                else if (winner == -1) std::cout << "White wins!\n";
                else std::cout << "Draw!\n";

                std::cout << "Game Over. 'u' to undo, 'q' to quit: ";
                std::string resp;
                if (!std::getline(std::cin, resp)) return 0;
                if (resp == "u" || resp == "U") {
                    if (undo_two_moves()) {
                        std::cout << "Undo successful.\n";
                        print_board(state, game.board_size, last_action);
                        continue;
                    }
                    std::cout << "Nothing to undo.\n";
                }
                break;
            }

            if (to_play == human_side) {
                while (true) {
                    std::cout << "Human step (row col / 'u' for undo / 'q' for quit):\n";
                    std::cout.flush();
                    std::string input;
                    if (!std::getline(std::cin, input)) return 0;

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
                    root.reset(new MCTSNode{state, to_play});
                    break;
                }
            } else {
                push_history();
                std::cout << "AlphaZero thinking...\n";

                const auto out = mcts.search(state, to_play, cfg.num_simulations, root);
                int action = out.gumbel_action;
                if (action < 0) action = argmax_index(out.mcts_policy);
                if (action < 0) {
                    std::cout << "No legal action found. Exiting game.\n";
                    return 1;
                }
                const int row = action / game.board_size;
                const int col = action % game.board_size;

                print_policy_grid(out.mcts_policy, game.board_size, "MCTS Strategy (improved policy):");
                std::vector<float> visit_dist(out.visit_counts.size(), 0.0f);
                {
                    const float sum_n = std::accumulate(out.visit_counts.begin(), out.visit_counts.end(), 0.0f);
                    if (sum_n > 0.0f) {
                        for (size_t i = 0; i < out.visit_counts.size(); ++i) {
                            visit_dist[i] = out.visit_counts[i] / sum_n;
                        }
                    }
                }
                print_policy_grid(visit_dist, game.board_size, "MCTS Visits (N(s,a)/sum):");
                print_policy_grid(out.nn_policy, game.board_size, "NN Strategy:");

                const float root_value = out.v_mix[0] - out.v_mix[2];
                const float nn_value = out.nn_value_probs[0] - out.nn_value_probs[2];
                std::cout
                    << "          "
                    << std::setw(7) << "Win" << "  "
                    << std::setw(7) << "Draw" << "  "
                    << std::setw(7) << "Loss" << "  "
                    << std::setw(7) << "W-L" << '\n'
                    << "  root:   "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.v_mix[0] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.v_mix[1] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.v_mix[2] * 100.0f) << "%  "
                    << std::showpos << std::fixed << std::setprecision(2) << root_value << std::noshowpos << '\n'
                    << "  nn:     "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[0] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[1] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[2] * 100.0f) << "%  "
                    << std::showpos << std::fixed << std::setprecision(2) << nn_value << std::noshowpos << '\n';
                for (size_t i = 0; i < out.gumbel_phases.size(); ++i) {
                    std::cout << "Gumbel Phase " << i
                              << " (" << out.gumbel_phases[i].size() << "):";
                    for (int a : out.gumbel_phases[i]) {
                        std::cout << ' ' << (a / game.board_size)
                                  << ',' << (a % game.board_size);
                    }
                    std::cout << '\n';
                }
                std::cout << "AI move: (" << row << ", " << col << ")\n";

                state = game.get_next_state(state, action, to_play);
                last_action = action;
                last_player = to_play;
                to_play = -to_play;
                root.reset(new MCTSNode{state, to_play});
            }

            print_board(state, game.board_size, last_action);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[gomoku_play] fatal: " << e.what() << "\n";
        return 2;
    }
}
