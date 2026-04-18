// generate_openings.cpp
// Balanced opening generator for Gomoku (AlphaZero).
//
// Phase 1: Random target move count N in [3, 10].
// Phase 2: Heuristic scatter -- compact, clustered random placement.
// Phase 3: (Skipped) Tactical rejection via VCF/VCT solver.
// Phase 4: Value-based balancing -- MCTS evaluation to find a move that
//          pulls the position toward 50% win rate.
//
// Output: envs/gomoku_openings.txt in the existing 3-line-per-opening format
//         (weight / black coords / white coords), with weight derived from
//         the balance quality of the position.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../envs/gomoku.h"
#include "../alphazero.h"

namespace {

// -------------------------------------------------------------------------
// Configuration defaults (overridable via command-line args)
// -------------------------------------------------------------------------
struct GenConfig {
    int count                 = 200;     // number of openings to generate
    int simulations           = 400;     // MCTS playouts per candidate point
    int balance_k             = 4;       // exponent in W = (1 - V^2)^k
    float reject_threshold    = 0.20f;   // |V| > this => discard the board
    int board_size            = 15;
    int num_blocks            = 4;
    int num_channels          = 128;
    std::string data_dir      = "data/gomoku";
    std::string file_name     = "gomoku";
    std::string output_path   = "envs/gomoku_openings.txt";
    int scatter_dist          = 2;       // Chebyshev distance for candidate gen
    int min_N                 = 3;       // min random scatter moves
    int max_N                 = 10;      // max random scatter moves
    int max_retries           = 10000;   // safety cap on retries
};

// -------------------------------------------------------------------------
// Parse simple --key value command-line arguments
// -------------------------------------------------------------------------
GenConfig parse_args(int argc, char* argv[]) {
    GenConfig cfg;
    for (int i = 1; i < argc - 1; i += 2) {
        std::string key = argv[i];
        std::string val = argv[i + 1];
        if (key == "--count")            cfg.count = std::stoi(val);
        else if (key == "--simulations") cfg.simulations = std::stoi(val);
        else if (key == "--k")           cfg.balance_k = std::stoi(val);
        else if (key == "--reject")      cfg.reject_threshold = std::stof(val);
        else if (key == "--board_size")  cfg.board_size = std::stoi(val);
        else if (key == "--num_blocks")  cfg.num_blocks = std::stoi(val);
        else if (key == "--num_channels") cfg.num_channels = std::stoi(val);
        else if (key == "--data_dir")    cfg.data_dir = val;
        else if (key == "--file_name")   cfg.file_name = val;
        else if (key == "--output")      cfg.output_path = val;
        else if (key == "--min_N")       cfg.min_N = std::stoi(val);
        else if (key == "--max_N")       cfg.max_N = std::stoi(val);
        else {
            std::cerr << "Unknown argument: " << key << "\n";
        }
    }
    return cfg;
}

// -------------------------------------------------------------------------
// Collect candidate points within Chebyshev distance [1, dist] of any stone
// -------------------------------------------------------------------------
std::vector<int> get_nearby_empty(
    const std::vector<int8_t>& board, int board_size, int dist
) {
    std::vector<int> result;
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            const int loc = r * board_size + c;
            if (board[loc] != 0) continue;

            bool near = false;
            for (int dr = -dist; dr <= dist && !near; ++dr) {
                for (int dc = -dist; dc <= dist && !near; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    const int nr = r + dr;
                    const int nc = c + dc;
                    if (nr < 0 || nr >= board_size || nc < 0 || nc >= board_size) continue;
                    if (board[nr * board_size + nc] != 0) near = true;
                }
            }
            if (near) result.push_back(loc);
        }
    }
    return result;
}

// -------------------------------------------------------------------------
// Check if placing `color` at `loc` is a forbidden point (Renju rules)
// -------------------------------------------------------------------------
bool is_forbidden_move(
    const std::vector<int8_t>& board, int board_size, int loc, int color
) {
    if (color != 1) return false;  // only Black has forbidden points

    // Reuse the ForbiddenPointFinder from Gomoku
    // We need to construct a temporary Gomoku instance for this.
    // Actually, we can directly use the board to build the finder.
    // The ForbiddenPointFinder is a private nested struct inside Gomoku,
    // so we use Gomoku's get_is_legal_actions which already handles it,
    // but that uses distance-3 filtering.  Instead, we'll use a simpler
    // approach: temporarily place the stone and check via get_winner.
    // If get_winner returns -1 (white wins) when black just played, it
    // means black played on a forbidden point.

    // Actually, we can just build a temp Gomoku and check.
    // But simpler: build the next state, call get_winner.
    auto next = board;
    next[loc] = static_cast<int8_t>(color);

    skyzero::Gomoku temp_game(board_size, true, true);
    int winner = temp_game.get_winner(next, loc, color);
    // If black played at a forbidden point, get_winner returns -1
    return (winner == -1);
}

// -------------------------------------------------------------------------
// Check if the board already has a winner (five in a row, etc.)
// -------------------------------------------------------------------------
bool board_is_terminal(
    const std::vector<int8_t>& board, int board_size,
    int last_action, int last_player
) {
    skyzero::Gomoku temp_game(board_size, true, true);
    return temp_game.is_terminal(board, last_action, last_player);
}

// -------------------------------------------------------------------------
// Simple shape heuristic: count how many same-color neighbors exist
// around `loc` for the player `color` (for weighting scatter placement).
// Returns a weight >= 1.0.
// -------------------------------------------------------------------------
float shape_weight(
    const std::vector<int8_t>& board, int board_size, int loc, int color
) {
    const int r = loc / board_size;
    const int c = loc % board_size;
    int count = 0;
    // Check 8 directions for adjacent same-color stones
    static const int dirs[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (const auto& d : dirs) {
        const int nr = r + d[0];
        const int nc = c + d[1];
        if (nr >= 0 && nr < board_size && nc >= 0 && nc < board_size) {
            if (board[nr * board_size + nc] == static_cast<int8_t>(color)) {
                ++count;
            }
        }
    }
    // Also check for "two-in-a-line" patterns (distance 2 along a line)
    static const int line_dirs[4][2] = {{1,0},{0,1},{1,1},{1,-1}};
    for (const auto& d : line_dirs) {
        for (int sign : {1, -1}) {
            const int r1 = r + d[0] * sign;
            const int c1 = c + d[1] * sign;
            const int r2 = r + d[0] * sign * 2;
            const int c2 = c + d[1] * sign * 2;
            if (r1 >= 0 && r1 < board_size && c1 >= 0 && c1 < board_size &&
                r2 >= 0 && r2 < board_size && c2 >= 0 && c2 < board_size) {
                if (board[r1 * board_size + c1] == static_cast<int8_t>(color) &&
                    board[r2 * board_size + c2] == static_cast<int8_t>(color)) {
                    count += 2;  // bonus for extending a line
                }
            }
        }
    }
    return 1.0f + 0.5f * static_cast<float>(count);
}

// -------------------------------------------------------------------------
// Format a single opening in the 3-line format
// Coords are relative to center (dx, dy)
// -------------------------------------------------------------------------
struct GeneratedOpening {
    std::vector<int8_t> board;
    int to_play;
    float weight;    // balance quality weight
    float best_absV; // diagnostic: |V| of the best balanced point
};

std::string format_opening(
    const GeneratedOpening& op, int board_size
) {
    const int center = board_size / 2;
    std::ostringstream oss;

    // Line 1: weight
    oss << std::fixed << std::setprecision(6) << op.weight << "\n";

    // Line 2: black stone coordinates (relative to center)
    bool first_black = true;
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            if (op.board[r * board_size + c] == 1) {
                if (!first_black) oss << " ";
                oss << (r - center) << "," << (c - center);
                first_black = false;
            }
        }
    }
    oss << "\n";

    // Line 3: white stone coordinates (relative to center)
    bool first_white = true;
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            if (op.board[r * board_size + c] == -1) {
                if (!first_white) oss << " ";
                oss << (r - center) << "," << (c - center);
                first_white = false;
            }
        }
    }
    // If no white stones, still emit empty line
    oss << "\n";

    return oss.str();
}

// -------------------------------------------------------------------------
// Print a board to stdout for diagnostics
// -------------------------------------------------------------------------
void print_board(const std::vector<int8_t>& board, int board_size) {
    std::cout << "   ";
    for (int c = 0; c < board_size; ++c) {
        std::cout << (c < 10 ? " " : "") << c << ' ';
    }
    std::cout << '\n';
    for (int r = 0; r < board_size; ++r) {
        std::cout << (r < 10 ? " " : "") << r << ' ';
        for (int c = 0; c < board_size; ++c) {
            const int8_t v = board[r * board_size + c];
            char ch = '.';
            if (v == 1) ch = 'X';
            if (v == -1) ch = 'O';
            std::cout << ' ' << ch << ' ';
        }
        std::cout << '\n';
    }
}

}  // anonymous namespace

// =========================================================================
// Main
// =========================================================================
int main(int argc, char* argv[]) {
    auto gen_cfg = parse_args(argc, argv);

    std::cout << "=== Balanced Opening Generator ===\n"
              << "  count:       " << gen_cfg.count << "\n"
              << "  simulations: " << gen_cfg.simulations << "\n"
              << "  k:           " << gen_cfg.balance_k << "\n"
              << "  reject |V|:  " << gen_cfg.reject_threshold << "\n"
              << "  board_size:  " << gen_cfg.board_size << "\n"
              << "  output:      " << gen_cfg.output_path << "\n"
              << "  N range:     [" << gen_cfg.min_N << ", " << gen_cfg.max_N << "]\n"
              << std::endl;

    // -----------------------------------------------------------------
    // Set up AlphaZero config, model, optimizer, MCTS
    // -----------------------------------------------------------------
    skyzero::AlphaZeroConfig az_cfg;
    az_cfg.board_size = gen_cfg.board_size;
    az_cfg.num_blocks = gen_cfg.num_blocks;
    az_cfg.num_channels = gen_cfg.num_channels;
    az_cfg.lr = 1e-4f;
    az_cfg.weight_decay = 3e-5f;
    az_cfg.file_name = gen_cfg.file_name;
    az_cfg.data_dir = gen_cfg.data_dir;

    // Evaluation config: no dirichlet noise, no forced playouts
    az_cfg.dirichlet_epsilon = 0.0f;
    az_cfg.enable_forced_playouts = false;
    az_cfg.full_search_num_simulations = gen_cfg.simulations;
    az_cfg.fast_search_num_simulations = gen_cfg.simulations;

    // Use symmetry averaging for more accurate evaluation
    az_cfg.enable_stochastic_transform_inference_for_root = false;
    az_cfg.enable_symmetry_inference_for_root = true;
    az_cfg.enable_stochastic_transform_inference_for_child = true;
    az_cfg.enable_symmetry_inference_for_child = false;

    // Dynamic Variance-Scaled cPUCT (same as play config)
    az_cfg.cpuct_utility_stdev_prior = 0.40f;
    az_cfg.cpuct_utility_stdev_prior_weight = 2.0f;
    az_cfg.cpuct_utility_stdev_scale = 0.85f;

    // SVB disabled for simplicity in opening generation
    az_cfg.enable_subtree_value_bias = false;

    az_cfg.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    skyzero::Gomoku game(az_cfg.board_size, true, true);
    auto model = skyzero::ResNet(game.board_size, game.num_planes, az_cfg.num_blocks, az_cfg.num_channels);
    model->to(az_cfg.device);
    model->eval();

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(az_cfg.lr).weight_decay(az_cfg.weight_decay)
    );

    skyzero::AlphaZero<skyzero::Gomoku> az(game, model, optimizer, az_cfg);
    if (!az.load_checkpoint()) {
        std::cerr << "ERROR: Could not load checkpoint. Aborting.\n";
        return 1;
    }
    model->eval();

    skyzero::MCTS<skyzero::Gomoku> mcts(game, az_cfg, model);

    // -----------------------------------------------------------------
    // Random engine
    // -----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_N(gen_cfg.min_N, gen_cfg.max_N);

    const int board_size = gen_cfg.board_size;
    const int center = board_size / 2;
    const int area = board_size * board_size;

    // -----------------------------------------------------------------
    // Generate openings
    // -----------------------------------------------------------------
    std::vector<GeneratedOpening> openings;
    openings.reserve(static_cast<size_t>(gen_cfg.count));

    int attempts = 0;
    int rejected_terminal = 0;
    int rejected_balance = 0;

    while (static_cast<int>(openings.size()) < gen_cfg.count) {
        if (++attempts > gen_cfg.max_retries) {
            std::cerr << "WARNING: Hit max retry limit (" << gen_cfg.max_retries
                      << "). Generated " << openings.size() << " openings.\n";
            break;
        }

        // ============================================================
        // Phase 1: Determine target scatter move count N
        // ============================================================
        const int N = dist_N(rng);

        // ============================================================
        // Phase 2: Heuristic scatter
        // ============================================================
        std::vector<int8_t> board(static_cast<size_t>(area), 0);
        int to_play = 1;   // Black first
        int last_action = -1;
        int last_player = 0;
        bool scatter_ok = true;

        // Move 1: center 3x3 preference
        {
            std::vector<int> center_cells;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    const int r = center + dr;
                    const int c = center + dc;
                    if (r >= 0 && r < board_size && c >= 0 && c < board_size) {
                        center_cells.push_back(r * board_size + c);
                    }
                }
            }
            std::uniform_int_distribution<int> d(0, static_cast<int>(center_cells.size()) - 1);
            const int loc = center_cells[static_cast<size_t>(d(rng))];
            board[loc] = static_cast<int8_t>(to_play);
            last_action = loc;
            last_player = to_play;
            to_play = -to_play;
        }

        // Moves 2..N
        for (int move = 1; move < N; ++move) {
            auto candidates = get_nearby_empty(board, board_size, gen_cfg.scatter_dist);
            if (candidates.empty()) {
                scatter_ok = false;
                break;
            }

            // Forbidden point filtering for Black
            if (to_play == 1) {
                std::vector<int> filtered;
                for (int loc : candidates) {
                    if (!is_forbidden_move(board, board_size, loc, 1)) {
                        filtered.push_back(loc);
                    }
                }
                candidates = std::move(filtered);
                if (candidates.empty()) {
                    scatter_ok = false;
                    break;
                }
            }

            // Shape-weighted random selection
            std::vector<float> weights;
            weights.reserve(candidates.size());
            for (int loc : candidates) {
                weights.push_back(shape_weight(board, board_size, loc, to_play));
            }
            std::discrete_distribution<int> pick(weights.begin(), weights.end());
            const int chosen_loc = candidates[static_cast<size_t>(pick(rng))];

            board[chosen_loc] = static_cast<int8_t>(to_play);
            last_action = chosen_loc;
            last_player = to_play;
            to_play = -to_play;

            // Check if board became terminal
            if (board_is_terminal(board, board_size, last_action, last_player)) {
                scatter_ok = false;
                break;
            }
        }

        if (!scatter_ok) {
            ++rejected_terminal;
            continue;
        }

        // Quick sanity: board should not already be terminal
        if (board_is_terminal(board, board_size, last_action, last_player)) {
            ++rejected_terminal;
            continue;
        }

        // ============================================================
        // Phase 4: Value-based balancing
        // ============================================================
        // Generate candidate points for the N+1-th move
        auto balance_candidates = get_nearby_empty(board, board_size, gen_cfg.scatter_dist);
        if (balance_candidates.empty()) {
            ++rejected_terminal;
            continue;
        }

        // Forbidden point filtering for current player
        if (to_play == 1) {
            std::vector<int> filtered;
            for (int loc : balance_candidates) {
                if (!is_forbidden_move(board, board_size, loc, 1)) {
                    filtered.push_back(loc);
                }
            }
            balance_candidates = std::move(filtered);
        }

        if (balance_candidates.empty()) {
            ++rejected_terminal;
            continue;
        }

        // Evaluate each candidate with MCTS
        struct CandidateEval {
            int loc;
            float V;       // scalar value = W - L, from the side-to-move's perspective AFTER placing
            float absV;
        };
        std::vector<CandidateEval> evals;
        evals.reserve(balance_candidates.size());

        std::cout << "[" << (openings.size() + 1) << "/" << gen_cfg.count
                  << "] Evaluating " << balance_candidates.size()
                  << " candidates (N=" << N << ", to_play=" << (to_play == 1 ? "Black" : "White")
                  << ") ..." << std::flush;

        for (int loc : balance_candidates) {
            // Place the stone
            auto next_board = board;
            next_board[loc] = static_cast<int8_t>(to_play);
            const int next_to_play = -to_play;

            // Check terminal
            if (board_is_terminal(next_board, board_size, loc, to_play)) {
                // This is a winning/losing move -- skip it
                continue;
            }

            // Run MCTS from the opponent's perspective
            std::unique_ptr<skyzero::MCTSNode> root;
            auto sr = mcts.search(next_board, next_to_play, gen_cfg.simulations, root);

            // root_value is from next_to_play's perspective: [win, draw, loss]
            // V = win - loss (for next_to_play)
            // We want the position to be balanced, meaning V close to 0
            const float V_opponent = sr.root_value[0] - sr.root_value[2];

            CandidateEval ce;
            ce.loc = loc;
            ce.V = V_opponent;  // from opponent's (next_to_play's) perspective
            ce.absV = std::fabs(V_opponent);
            evals.push_back(ce);
        }

        if (evals.empty()) {
            std::cout << " no valid candidates.\n";
            ++rejected_terminal;
            continue;
        }

        // Find minimum |V|
        float min_absV = std::numeric_limits<float>::infinity();
        for (const auto& e : evals) {
            min_absV = std::min(min_absV, e.absV);
        }

        // Rejection: if best candidate still too unbalanced
        if (min_absV > gen_cfg.reject_threshold) {
            std::cout << " rejected (min|V|=" << std::fixed << std::setprecision(3) << min_absV << ")\n";
            ++rejected_balance;
            continue;
        }

        // Compute selection weights: W_i = (1 - V_i^2)^k
        std::vector<float> sel_weights;
        sel_weights.reserve(evals.size());
        float sum_w = 0.0f;
        for (const auto& e : evals) {
            const float v2 = e.V * e.V;
            float w = std::pow(std::max(0.0f, 1.0f - v2), gen_cfg.balance_k);
            sel_weights.push_back(w);
            sum_w += w;
        }

        if (sum_w < 1e-20f) {
            std::cout << " all weights zero.\n";
            ++rejected_balance;
            continue;
        }

        // Sample according to the distribution
        std::discrete_distribution<int> balance_pick(sel_weights.begin(), sel_weights.end());
        const int chosen_idx = balance_pick(rng);
        const auto& chosen = evals[static_cast<size_t>(chosen_idx)];

        // Place the balancing move
        auto final_board = board;
        final_board[chosen.loc] = static_cast<int8_t>(to_play);
        const int final_to_play = -to_play;

        // Compute opening weight based on balance quality
        // Weight = (1 - V^2)^k, normalized later, but store raw for now
        const float balance_quality = std::pow(
            std::max(0.0f, 1.0f - chosen.V * chosen.V),
            gen_cfg.balance_k
        );

        GeneratedOpening op;
        op.board = final_board;
        op.to_play = final_to_play;
        op.weight = balance_quality;
        op.best_absV = chosen.absV;
        openings.push_back(op);

        const int r = chosen.loc / board_size;
        const int c = chosen.loc % board_size;
        std::cout << " OK  balance_move=(" << r << "," << c
                  << ")  V=" << std::showpos << std::fixed << std::setprecision(3) << chosen.V
                  << std::noshowpos
                  << "  quality=" << std::fixed << std::setprecision(4) << balance_quality
                  << "\n";
    }

    std::cout << "\n=== Generation complete ===\n"
              << "  Generated:         " << openings.size() << "\n"
              << "  Total attempts:    " << attempts << "\n"
              << "  Rejected terminal: " << rejected_terminal << "\n"
              << "  Rejected balance:  " << rejected_balance << "\n"
              << std::endl;

    if (openings.empty()) {
        std::cerr << "No openings generated. Exiting.\n";
        return 1;
    }

    // -----------------------------------------------------------------
    // Normalize weights so they sum to 1.0
    // -----------------------------------------------------------------
    float total_weight = 0.0f;
    for (const auto& op : openings) {
        total_weight += op.weight;
    }
    if (total_weight < 1e-20f) {
        // Fallback: uniform
        total_weight = static_cast<float>(openings.size());
        for (auto& op : openings) {
            op.weight = 1.0f;
        }
    }

    // -----------------------------------------------------------------
    // Write output file
    // -----------------------------------------------------------------
    std::ofstream ofs(gen_cfg.output_path);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Cannot open output file: " << gen_cfg.output_path << "\n";
        return 1;
    }

    for (auto& op : openings) {
        op.weight /= total_weight;
        ofs << format_opening(op, board_size);
    }
    ofs.close();

    std::cout << "Wrote " << openings.size() << " openings to " << gen_cfg.output_path << "\n";

    // Print a few samples for visual inspection
    const int show_count = std::min(3, static_cast<int>(openings.size()));
    for (int i = 0; i < show_count; ++i) {
        std::cout << "\n--- Sample opening #" << (i + 1)
                  << " (weight=" << std::fixed << std::setprecision(6) << openings[i].weight
                  << ", |V|=" << std::fixed << std::setprecision(3) << openings[i].best_absV
                  << ") ---\n";
        print_board(openings[i].board, board_size);
    }

    return 0;
}
