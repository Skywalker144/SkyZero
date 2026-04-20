// Interactive terminal driver for SkyZero V4.
// Reuses ParallelMCTS + Gomoku with a synchronous TorchScript inference lambda.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "alphazero.h"
#include "alphazero_parallel.h"
#include "envs/gomoku.h"

using skyzero::AlphaZeroConfig;
using skyzero::Gomoku;
using skyzero::MCTSNode;
using skyzero::MCTSSearchOutput;
using skyzero::ParallelMCTS;

namespace {

namespace fs = std::filesystem;

// Default model directory: <exe_dir>/../../data/models
// (cpp/build/play -> ../../data/models = project_root/data/models)
std::string default_model_dir(const char* argv0) {
    fs::path exe(argv0);
    fs::path exe_dir = exe.parent_path();
    if (exe_dir.empty()) exe_dir = fs::current_path();
    return (exe_dir / ".." / ".." / "data" / "models").lexically_normal().string();
}

// Pick newest model in `dir`. Prefers files matching skyzero-s<S>-e<E>.pt by
// (epoch desc, step desc); falls back to random_init.pt; returns "" if none.
std::string find_latest_model(const std::string& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return "";
    static const std::regex re("^skyzero-s(\\d+)-e(\\d+)\\.pt$");
    long long best_e = -1, best_s = -1;
    fs::path best;
    fs::path random_init;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name == "random_init.pt") { random_init = entry.path(); continue; }
        std::smatch m;
        if (!std::regex_match(name, m, re)) continue;
        long long s = std::stoll(m[1].str());
        long long e = std::stoll(m[2].str());
        if (e > best_e || (e == best_e && s > best_s)) {
            best_e = e; best_s = s; best = entry.path();
        }
    }
    if (!best.empty()) return best.string();
    if (!random_init.empty()) return random_init.string();
    return "";
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--model PATH | --model-dir DIR] [options]\n"
              << "\nModel selection (if --model omitted, latest skyzero-s*-e*.pt in --model-dir is used):\n"
              << "  --model PATH           TorchScript model (.pt) to load\n"
              << "  --model-dir DIR        Directory to search (default: <exe>/../../data/models)\n"
              << "\nGame options:\n"
              << "  --board-size N         Board size (default: 15)\n"
              << "  --no-renju             Disable renju rules (default: renju on)\n"
              << "\nMCTS options:\n"
              << "  --num-simulations N    MCTS simulations per move (default: 800)\n"
              << "  --gumbel-m N           Gumbel top-k actions (default: 16)\n"
              << "  --enable-symmetry      Average over 8 symmetries at root/child\n"
              << "  --disable-stochastic   Turn off random symmetry transform in inference\n"
              << "\nNN options:\n"
              << "  --device DEVICE        cpu or cuda (default: cuda if available)\n"
              << "  --fp16                 Run model in half precision (auto-on for cuda)\n"
              << "  --fp32                 Force float32 inference\n"
              << "\nPlay options:\n"
              << "  --human-side N         1 = human plays black, -1 = white, 0 = ask (default)\n"
              << std::endl;
}

void print_board(const std::vector<int8_t>& board, int bs, int last_action) {
    std::cout << "   ";
    for (int c = 0; c < bs; ++c) {
        std::cout << std::setw(2) << c << " ";
    }
    std::cout << "\n";
    for (int r = 0; r < bs; ++r) {
        std::cout << std::setw(2) << r << " ";
        for (int c = 0; c < bs; ++c) {
            const int loc = r * bs + c;
            const int8_t v = board[loc];
            const bool is_last = (loc == last_action);
            const char* open = is_last ? "[" : " ";
            const char* close = is_last ? "]" : " ";
            if (v == 1) std::cout << open << "X" << close;
            else if (v == -1) std::cout << open << "O" << close;
            else std::cout << open << "." << close;
        }
        std::cout << "\n";
    }
}

void print_policy_grid(const std::vector<float>& pol, int bs, const char* title) {
    std::cout << title << "\n";
    std::cout << "      ";
    for (int c = 0; c < bs; ++c) {
        std::cout << std::setw(5) << c << " ";
    }
    std::cout << "\n";
    for (int r = 0; r < bs; ++r) {
        std::cout << std::setw(3) << r << "  ";
        for (int c = 0; c < bs; ++c) {
            const float p = pol[r * bs + c];
            if (p < 0.0001f) {
                std::cout << "     . ";
            } else {
                std::cout << std::setw(5) << std::fixed << std::setprecision(2) << p << " ";
            }
        }
        std::cout << "\n";
    }
}

void print_value_summary(const std::array<float, 3>& v_mix,
                         const std::array<float, 3>& nn_val) {
    const float v_mix_wl = v_mix[0] - v_mix[2];
    const float nn_wl = nn_val[0] - nn_val[2];
    std::cout << "           "
              << std::setw(7) << "Win"
              << std::setw(7) << "Draw"
              << std::setw(7) << "Loss"
              << std::setw(8) << "W-L" << "\n";
    std::cout << "  v_mix: "
              << std::setw(7) << std::fixed << std::setprecision(2) << (v_mix[0] * 100.0f) << "%"
              << std::setw(6) << std::fixed << std::setprecision(2) << (v_mix[1] * 100.0f) << "%"
              << std::setw(6) << std::fixed << std::setprecision(2) << (v_mix[2] * 100.0f) << "%"
              << std::setw(7) << std::showpos << std::fixed << std::setprecision(2) << v_mix_wl << std::noshowpos << "\n";
    std::cout << "  NN:    "
              << std::setw(7) << std::fixed << std::setprecision(2) << (nn_val[0] * 100.0f) << "%"
              << std::setw(6) << std::fixed << std::setprecision(2) << (nn_val[1] * 100.0f) << "%"
              << std::setw(6) << std::fixed << std::setprecision(2) << (nn_val[2] * 100.0f) << "%"
              << std::setw(7) << std::showpos << std::fixed << std::setprecision(2) << nn_wl << std::noshowpos << "\n";
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// Snapshot of game state for undo. Tree is not preserved.
struct HistoryEntry {
    std::vector<int8_t> state;
    int to_play;
    int last_action;
    int last_player;
};

}  // namespace

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string model_dir;
    int board_size = 15;
    bool use_renju = true;
    bool enable_forbidden_plane = true;
    int num_simulations = 800;
    int gumbel_m = 16;
    bool enable_symmetry = false;
    bool disable_stochastic = false;
    int human_side = 0;  // 0 => ask

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    int fp16_mode = -1;  // -1 = auto, 0 = fp32, 1 = fp16

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--model") model_path = next();
        else if (arg == "--model-dir") model_dir = next();
        else if (arg == "--board-size") board_size = std::stoi(next());
        else if (arg == "--no-renju") { use_renju = false; enable_forbidden_plane = false; }
        else if (arg == "--renju") { use_renju = true; enable_forbidden_plane = true; }
        else if (arg == "--num-simulations") num_simulations = std::stoi(next());
        else if (arg == "--gumbel-m") gumbel_m = std::stoi(next());
        else if (arg == "--enable-symmetry") enable_symmetry = true;
        else if (arg == "--disable-stochastic") disable_stochastic = true;
        else if (arg == "--human-side") human_side = std::stoi(next());
        else if (arg == "--device") {
            std::string dev = next();
            if (dev == "cpu") device = torch::Device(torch::kCPU);
            else if (dev == "cuda") device = torch::Device(torch::kCUDA);
            else { std::cerr << "Unknown device: " << dev << std::endl; return 1; }
        }
        else if (arg == "--fp16") fp16_mode = 1;
        else if (arg == "--fp32") fp16_mode = 0;
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        const std::string search_dir = model_dir.empty() ? default_model_dir(argv[0]) : model_dir;
        model_path = find_latest_model(search_dir);
        if (model_path.empty()) {
            std::cerr << "No model found in '" << search_dir
                      << "'. Pass --model PATH or --model-dir DIR.\n\n";
            print_usage(argv[0]);
            return 1;
        }
        std::cout << "Auto-selected latest model from " << search_dir << "\n";
    }

    const bool use_fp16 = (fp16_mode == 1) || (fp16_mode == -1 && device.is_cuda());

    Gomoku game(board_size, use_renju, enable_forbidden_plane);

    std::cout << "=== SkyZero V4 Play ===\n";
    std::cout << "Board: " << board_size << "x" << board_size
              << " | Renju: " << (use_renju ? "yes" : "no")
              << " | Planes: " << game.num_planes << "\n";
    std::cout << "Simulations: " << num_simulations
              << " | Gumbel-m: " << gumbel_m
              << " | Device: " << device
              << " | FP16: " << (use_fp16 ? "yes" : "no") << "\n";

    // Load TorchScript model
    torch::jit::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model '" << model_path << "': " << e.what() << std::endl;
        return 1;
    }
    model.to(device);
    if (use_fp16) model.to(torch::kHalf);
    model.eval();
    std::cout << "Loaded model: " << model_path << "\n";

    const int area = board_size * board_size;
    const int c = game.num_planes;
    std::mutex model_mtx;

    auto run_batch = [&](const std::vector<std::vector<int8_t>>& batch)
        -> std::vector<std::pair<std::vector<float>, std::array<float, 4>>> {
        const int bsz = static_cast<int>(batch.size());
        std::vector<float> input_buf(static_cast<size_t>(bsz) * c * area, 0.0f);
        for (int i = 0; i < bsz; ++i) {
            const auto& enc = batch[i];
            if (static_cast<int>(enc.size()) != c * area) {
                throw std::runtime_error("encoded size mismatch");
            }
            const size_t base = static_cast<size_t>(i) * c * area;
            for (int j = 0; j < c * area; ++j) {
                input_buf[base + j] = static_cast<float>(enc[j]);
            }
        }
        auto input = torch::from_blob(
            input_buf.data(),
            {bsz, c, board_size, board_size},
            torch::kFloat32
        ).to(device);
        if (use_fp16) input = input.to(torch::kHalf);

        torch::NoGradGuard no_grad;
        torch::IValue output;
        {
            std::lock_guard<std::mutex> lk(model_mtx);
            output = model.forward({input});
        }
        auto elements = output.toTuple()->elements();
        auto policy_logits_raw = elements[0].toTensor();  // [B, 1, H, W]
        auto value_logits_raw = elements[2].toTensor();   // [B, 3]
        // elements[3] is predicted short-term squared value error
        // (model already applies softplus-with-gradient-floor + 0.25 multiplier)
        torch::Tensor value_error_raw;
        bool has_value_error = (elements.size() >= 4);
        if (has_value_error) {
            value_error_raw = elements[3].toTensor().to(torch::kFloat32)
                                  .reshape({bsz}).to(torch::kCPU).contiguous();
        }
        auto policy = policy_logits_raw.reshape({bsz, area}).to(torch::kFloat32).to(torch::kCPU).contiguous();
        auto value = torch::softmax(value_logits_raw.to(torch::kFloat32), 1).to(torch::kCPU).contiguous();
        const float* pp = policy.data_ptr<float>();
        const float* vp = value.data_ptr<float>();
        const float* vep = has_value_error ? value_error_raw.data_ptr<float>() : nullptr;

        std::vector<std::pair<std::vector<float>, std::array<float, 4>>> out;
        out.reserve(bsz);
        for (int i = 0; i < bsz; ++i) {
            std::vector<float> logits(area, 0.0f);
            std::memcpy(logits.data(), pp + static_cast<size_t>(i) * area, area * sizeof(float));
            std::array<float, 4> v{
                vp[i * 3], vp[i * 3 + 1], vp[i * 3 + 2],
                vep ? vep[i] : 0.0f
            };
            out.push_back({std::move(logits), v});
        }
        return out;
    };

    auto infer_fn = [&](const std::vector<int8_t>& encoded)
        -> std::pair<std::vector<float>, std::array<float, 4>> {
        auto batch = run_batch({encoded});
        return std::move(batch[0]);
    };
    auto batch_infer_fn = [&](const std::vector<std::vector<int8_t>>& batch) {
        return run_batch(batch);
    };

    // MCTS config
    AlphaZeroConfig cfg;
    cfg.board_size = board_size;
    cfg.num_simulations = num_simulations;
    cfg.gumbel_m = gumbel_m;
    cfg.enable_symmetry_inference_for_root = enable_symmetry;
    cfg.enable_symmetry_inference_for_child = enable_symmetry;
    cfg.enable_stochastic_transform_inference_for_root = !disable_stochastic;
    cfg.enable_stochastic_transform_inference_for_child = !disable_stochastic;
    cfg.device = device;

    const uint64_t seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    ParallelMCTS<Gomoku> mcts(game, cfg, /*leaf_batch_size=*/1, infer_fn, batch_infer_fn, seed);

    // Human side
    if (human_side == 0) {
        std::cout << "\nEnter 1 for Human-first (play Black) or -1 for Human-second (play White): ";
        std::string line;
        if (!std::getline(std::cin, line)) return 0;
        try {
            human_side = std::stoi(trim(line));
        } catch (...) {
            std::cerr << "Invalid side, defaulting to 1 (Black).\n";
            human_side = 1;
        }
    }
    if (human_side != 1 && human_side != -1) {
        std::cerr << "Invalid human-side, must be 1 or -1.\n";
        return 1;
    }
    std::cout << "Human plays " << (human_side == 1 ? "Black (X)" : "White (O)") << ".\n";
    std::cout << "Input format: 'row col' (e.g. '7 7'). 'u' = undo last full move, 'q' = quit.\n\n";

    std::mt19937 rng(seed ^ 0xabcdef);
    (void)rng;

    std::vector<int8_t> state(area, 0);
    int to_play = 1;
    int last_action = -1;
    int last_player = 0;
    std::unique_ptr<MCTSNode> root;
    std::vector<HistoryEntry> history;

    print_board(state, board_size, last_action);

    while (true) {
        if (game.is_terminal(state, last_action, last_player)) {
            const int winner = game.get_winner(state, last_action, last_player);
            if (winner == 1) std::cout << "Black wins!\n";
            else if (winner == -1) std::cout << "White wins!\n";
            else std::cout << "Draw!\n";

            std::cout << "Game Over. 'u' to undo, 'q' or Enter to quit: " << std::flush;
            std::string line;
            if (!std::getline(std::cin, line)) break;
            line = trim(line);
            if (line == "u") {
                // Undo both AI and human moves (two entries).
                if (history.size() >= 2) {
                    auto a = history.back(); history.pop_back();
                    auto b = history.back(); history.pop_back();
                    state = b.state;
                    to_play = b.to_play;
                    last_action = b.last_action;
                    last_player = b.last_player;
                    root.reset();
                    std::cout << "Undo successful.\n";
                    print_board(state, board_size, last_action);
                    continue;
                } else {
                    std::cout << "Nothing to undo.\n";
                    break;
                }
            }
            break;
        }

        if (to_play == human_side) {
            while (true) {
                std::cout << "Human step (row col / 'u' for undo / 'q' for quit): " << std::flush;
                std::string line;
                if (!std::getline(std::cin, line)) return 0;
                line = trim(line);
                if (line == "q") {
                    std::cout << "Exiting game.\n";
                    return 0;
                }
                if (line == "u") {
                    if (history.size() >= 2) {
                        auto a = history.back(); history.pop_back();
                        auto b = history.back(); history.pop_back();
                        state = b.state;
                        to_play = b.to_play;
                        last_action = b.last_action;
                        last_player = b.last_player;
                        root.reset();
                        std::cout << "Undo successful.\n";
                        print_board(state, board_size, last_action);
                        continue;
                    } else {
                        std::cout << "Nothing to undo.\n";
                        continue;
                    }
                }

                int row, col;
                std::stringstream ss(line);
                if (!(ss >> row >> col)) {
                    std::cout << "Invalid input. Enter 'row col', e.g. '7 7'.\n";
                    continue;
                }
                if (row < 0 || row >= board_size || col < 0 || col >= board_size) {
                    std::cout << "Out of bounds.\n";
                    continue;
                }
                const int action = row * board_size + col;
                const auto legal = game.get_is_legal_actions(state, to_play);
                if (!legal[action]) {
                    std::cout << "Illegal move (occupied, too far from stones, or forbidden for empty board center).\n";
                    continue;
                }

                history.push_back({state, to_play, last_action, last_player});
                state = game.get_next_state(state, action, to_play);
                last_action = action;
                last_player = to_play;

                // Tree reuse after human move.
                if (root) {
                    std::unique_ptr<MCTSNode> next_root;
                    for (auto& child : root->children) {
                        if (child && child->action_taken == action) {
                            next_root = std::move(child);
                            break;
                        }
                    }
                    if (next_root) {
                        next_root->parent = nullptr;
                        root = std::move(next_root);
                    } else {
                        root.reset();
                    }
                }
                break;
            }

            to_play = -to_play;
            print_board(state, board_size, last_action);
            continue;
        }

        // AI turn
        history.push_back({state, to_play, last_action, last_player});
        std::cout << "AlphaZero thinking (" << num_simulations << " sims)...\n";
        const auto t0 = std::chrono::steady_clock::now();
        auto sr = mcts.search(state, to_play, num_simulations, root, /*is_eval=*/true, /*gumbel_m_override=*/-1);
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

        const int action = sr.gumbel_action;
        const int ar = action / board_size;
        const int ac = action % board_size;

        std::cout << "AlphaZero step: (" << ar << ", " << ac << ")  [" << std::fixed
                  << std::setprecision(2) << elapsed << "s]\n";
        print_policy_grid(sr.mcts_policy, board_size, "MCTS Policy:");
        print_policy_grid(sr.nn_policy, board_size, "NN Policy:");
        print_value_summary(sr.v_mix, sr.nn_value_probs);

        state = game.get_next_state(state, action, to_play);
        last_action = action;
        last_player = to_play;

        // Tree reuse after AI move.
        if (root) {
            std::unique_ptr<MCTSNode> next_root;
            for (auto& child : root->children) {
                if (child && child->action_taken == action) {
                    next_root = std::move(child);
                    break;
                }
            }
            if (next_root) {
                next_root->parent = nullptr;
                root = std::move(next_root);
            } else {
                root.reset();
            }
        }

        to_play = -to_play;
        print_board(state, board_size, last_action);
    }

    return 0;
}
