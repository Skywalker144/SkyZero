#include <csignal>
#include <iostream>
#include <string>

#include "alphazero.h"
#include "alphazero_parallel.h"
#include "envs/gomoku.h"
#include "random_opening.h"

// Use signal handler from utils.h (sets skyzero::stop_requested)

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "\nSelfplay options:\n"
              << "  --model-dir DIR        Directory to watch for .pt models (default: data/models)\n"
              << "  --output-dir DIR       Directory for NPZ output (default: data/selfplay)\n"
              << "  --max-games N          Max games to play (default: 4000)\n"
              << "  --max-rows-per-file N  Max rows per NPZ file (default: 25000)\n"
              << "\nGame options:\n"
              << "  --board-size N         Board size (default: 15)\n"
              << "  --renju               Enable renju rules (default: on)\n"
              << "  --no-renju            Disable renju rules\n"
              << "  --openings FILE        Opening book file\n"
              << "  --empty-board-prob F   Probability of empty board vs opening (default: 0.0)\n"
              << "  --online-openings      Enable online balanced opening generation\n"
              << "  --opening-min-moves N  Min random scatter moves (default: 0)\n"
              << "  --opening-max-moves N  Max random scatter moves (default: 11)\n"
              << "  --opening-balance-power F  Exponent k in (1-V^2)^k (default: 4.0)\n"
              << "  --opening-reject-prob F  Probabilistic reject strength (default: 0.995)\n"
              << "  --opening-reject-prob-fallback F  Reject prob after max_retries (default: 0.8)\n"
              << "  --opening-max-retries N  Retries before reject_prob decays (default: 20)\n"
              << "\nMCTS options:\n"
              << "  --num-simulations N    MCTS simulations per move (default: 32)\n"
              << "  --gumbel-m N           Gumbel top-k actions (default: 16)\n"
              << "  --c-puct F             Exploration constant (default: 1.1)\n"
              << "\nNN options:\n"
              << "  --device DEVICE        torch device: cpu or cuda (default: cuda if available)\n"
              << "\nParallel options:\n"
              << "  --num-workers N        Selfplay worker threads (default: auto)\n"
              << "  --num-servers N        Inference server threads (default: 1)\n"
              << "  --inference-batch N    Max inference batch size (default: 256)\n"
              << "  --leaf-batch N         Leaf batch size for MCTS (default: 32)\n"
              << "  --model-check-ms N     Model check interval in ms (default: 10000)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, skyzero::signal_handler);
    std::signal(SIGTERM, skyzero::signal_handler);

    skyzero::AlphaZeroConfig cfg;
    skyzero::AlphaZeroParallelConfig pcfg;
    bool use_renju = true;
    bool enable_forbidden_plane = true;
    std::string openings_file;
    float empty_board_prob = 0.0f;
    skyzero::RandomOpeningConfig opening_cfg;

    // Default device
    if (torch::cuda::is_available()) {
        cfg.device = torch::kCUDA;
    }

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
        else if (arg == "--model-dir") cfg.model_dir = next();
        else if (arg == "--output-dir") cfg.output_dir = next();
        else if (arg == "--max-games") cfg.max_games_total = std::stoi(next());
        else if (arg == "--max-rows-per-file") cfg.max_rows_per_file = std::stoi(next());
        else if (arg == "--board-size") cfg.board_size = std::stoi(next());
        else if (arg == "--renju") use_renju = true;
        else if (arg == "--no-renju") { use_renju = false; enable_forbidden_plane = false; }
        else if (arg == "--openings") openings_file = next();
        else if (arg == "--empty-board-prob") empty_board_prob = std::stof(next());
        else if (arg == "--num-simulations") cfg.num_simulations = std::stoi(next());
        else if (arg == "--gumbel-m") cfg.gumbel_m = std::stoi(next());
        else if (arg == "--device") {
            std::string dev = next();
            if (dev == "cpu") cfg.device = torch::kCPU;
            else if (dev == "cuda") cfg.device = torch::kCUDA;
            else { std::cerr << "Unknown device: " << dev << std::endl; return 1; }
        }
        else if (arg == "--num-workers") pcfg.num_workers = std::stoi(next());
        else if (arg == "--num-servers") pcfg.num_inference_servers = std::stoi(next());
        else if (arg == "--inference-batch") pcfg.inference_batch_size = std::stoi(next());
        else if (arg == "--leaf-batch") pcfg.leaf_batch_size = std::stoi(next());
        else if (arg == "--model-check-ms") pcfg.model_check_interval_ms = std::stoi(next());
        // Additional MCTS params
        else if (arg == "--gumbel-c-visit") cfg.gumbel_c_visit = std::stof(next());
        else if (arg == "--gumbel-c-scale") cfg.gumbel_c_scale = std::stof(next());
        else if (arg == "--half-life") cfg.half_life = std::stoi(next());
        else if (arg == "--move-temp-init") cfg.move_temperature_init = std::stof(next());
        else if (arg == "--move-temp-final") cfg.move_temperature_final = std::stof(next());
        else if (arg == "--policy-surprise-weight") cfg.policy_surprise_data_weight = std::stof(next());
        else if (arg == "--value-surprise-weight") cfg.value_surprise_data_weight = std::stof(next());
        else if (arg == "--soft-resign-threshold") cfg.soft_resign_threshold = std::stof(next());
        else if (arg == "--soft-resign-prob") cfg.soft_resign_prob = std::stof(next());
        else if (arg == "--inference-batch-wait-us") pcfg.inference_batch_wait_us = std::stoi(next());
        else if (arg == "--enable-symmetry-root") cfg.enable_symmetry_inference_for_root = true;
        else if (arg == "--enable-symmetry-child") cfg.enable_symmetry_inference_for_child = true;
        else if (arg == "--disable-stochastic-root") cfg.enable_stochastic_transform_inference_for_root = false;
        else if (arg == "--disable-stochastic-child") cfg.enable_stochastic_transform_inference_for_child = false;
        // Online balanced opening generation
        else if (arg == "--online-openings") opening_cfg.enabled = true;
        else if (arg == "--opening-min-moves") opening_cfg.min_moves = std::stoi(next());
        else if (arg == "--opening-max-moves") opening_cfg.max_moves = std::stoi(next());
        else if (arg == "--opening-balance-power") opening_cfg.balance_power = std::stof(next());
        else if (arg == "--opening-reject-prob") opening_cfg.reject_prob = std::stof(next());
        else if (arg == "--opening-reject-prob-fallback") opening_cfg.reject_prob_fallback = std::stof(next());
        else if (arg == "--opening-max-retries") opening_cfg.max_retries = std::stoi(next());
        // Fork side positions
        else if (arg == "--fork-side-prob") cfg.fork_side_position_prob = std::stof(next());
        else if (arg == "--max-fork-queue") cfg.max_fork_queue_size = std::stoi(next());
        else if (arg == "--fork-skip-first-n") cfg.fork_skip_first_n_moves = std::stoi(next());
        // Uncertainty-Weighted MCTS Backup
        else if (arg == "--enable-uncertainty-weighting") cfg.enable_uncertainty_weighting = true;
        else if (arg == "--uncertainty-coeff") cfg.uncertainty_coeff = std::stof(next());
        else if (arg == "--uncertainty-exponent") cfg.uncertainty_exponent = std::stof(next());
        else if (arg == "--uncertainty-max-weight") cfg.uncertainty_max_weight = std::stof(next());
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Defaults
    if (cfg.half_life < 0) {
        cfg.half_life = cfg.board_size;
    }

    skyzero::Gomoku game(cfg.board_size, use_renju, enable_forbidden_plane);
    if (!openings_file.empty()) {
        game.load_openings(openings_file, empty_board_prob);
    }

    std::cout << "=== SkyZero V4 Selfplay ===" << std::endl;
    std::cout << "Board: " << cfg.board_size << "x" << cfg.board_size
              << " | Renju: " << (use_renju ? "yes" : "no")
              << " | Planes: " << game.num_planes << std::endl;
    std::cout << "Simulations: " << cfg.num_simulations
              << " | Gumbel-m: " << cfg.gumbel_m
              << " | Workers: " << pcfg.num_workers
              << " | Servers: " << pcfg.num_inference_servers << std::endl;
    std::cout << "Model dir: " << cfg.model_dir
              << " | Output dir: " << cfg.output_dir << std::endl;
    std::cout << "Max games: " << cfg.max_games_total
              << " | Device: " << cfg.device << std::endl;
    if (opening_cfg.enabled) {
        std::cout << "Online openings: ON (moves=" << opening_cfg.min_moves
                  << "-" << opening_cfg.max_moves
                  << ", power=" << opening_cfg.balance_power
                  << ", reject_prob=" << opening_cfg.reject_prob
                  << "->" << opening_cfg.reject_prob_fallback
                  << ", retries=" << opening_cfg.max_retries << ")" << std::endl;
    }

    skyzero::AlphaZeroParallel<skyzero::Gomoku> engine(game, cfg, pcfg, opening_cfg);
    engine.run();

    if (skyzero::stop_requested) {
        std::cout << "Selfplay interrupted by user." << std::endl;
        return 130;  // 128 + SIGINT(2), conventional shell exit code for Ctrl+C
    }
    std::cout << "Done." << std::endl;
    return 0;
}
