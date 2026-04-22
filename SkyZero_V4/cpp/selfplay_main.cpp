// selfplay_main — C++ self-play entry point.
//
// Loads a TorchScript model, spins up parallel MCTS workers + inference
// servers, and writes NPZ files of training samples until --max-games is
// reached. Reads the same run.cfg that Python uses so both sides stay in
// lockstep on hyperparameters.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "alphazero.h"
#include "alphazero_parallel.h"
#include "envs/gomoku.h"
#include "npz_writer.h"
#include "policy_surprise_weighting.h"
#include "selfplay_manager.h"
#include "utils.h"

namespace fs = std::filesystem;
using namespace skyzero;

// ---------------------------------------------------------------------------
// Config parser (KEY=VALUE, shell-style) with bash-compatible quoting-lite.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, std::string> parse_cfg(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open config: " + path);
    }
    std::string line;
    while (std::getline(f, line)) {
        // strip comment
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        // trim
        const auto a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos) continue;
        const auto b = line.find_last_not_of(" \t\r");
        line = line.substr(a, b - a + 1);
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // trim value / strip surrounding quotes
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
    std::string model;
    std::string output_dir;
    std::string log_dir;
    std::string config;
    int iter = 0;
    int max_games = 0;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (k == "--model") a.model = need("--model");
        else if (k == "--output-dir") a.output_dir = need("--output-dir");
        else if (k == "--log-dir") a.log_dir = need("--log-dir");
        else if (k == "--config") a.config = need("--config");
        else if (k == "--iter") a.iter = std::stoi(need("--iter"));
        else if (k == "--max-games") a.max_games = std::stoi(need("--max-games"));
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.output_dir.empty() || a.config.empty() || a.max_games <= 0) {
        throw std::runtime_error(
            "usage: selfplay_main --model PATH --output-dir DIR --config PATH "
            "--iter N --max-games N [--log-dir DIR]"
        );
    }
    if (a.log_dir.empty()) a.log_dir = a.output_dir + "/../logs";
    return a;
}

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, signal_handler);
        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);
        const auto cfg_map = parse_cfg(cli.config);

        // --- AlphaZeroConfig ---
        AlphaZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 64);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.half_life = cfg_get<int>(cfg_map, "HALF_LIFE", -1);
        cfg.move_temperature_init = cfg_get<float>(cfg_map, "MOVE_TEMPERATURE_INIT", 0.8f);
        cfg.move_temperature_final = cfg_get<float>(cfg_map, "MOVE_TEMPERATURE_FINAL", 0.2f);
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
        cfg.policy_surprise_data_weight = cfg_get<float>(cfg_map, "POLICY_SURPRISE_DATA_WEIGHT", 0.5f);
        cfg.value_surprise_data_weight = cfg_get<float>(cfg_map, "VALUE_SURPRISE_DATA_WEIGHT", 0.1f);
        cfg.value_target_mix_now_factor_constant =
            cfg_get<float>(cfg_map, "VALUE_TARGET_MIX_NOW_FACTOR_CONSTANT", 0.2f);
        cfg.balance_opening_prob = cfg_get<float>(cfg_map, "BALANCE_OPENING_PROB", 0.8f);
        cfg.balanced_opening_max_tries = cfg_get<int>(cfg_map, "BALANCED_OPENING_MAX_TRIES", 20);
        cfg.balanced_opening_avg_dist_factor =
            cfg_get<float>(cfg_map, "BALANCED_OPENING_AVG_DIST_FACTOR", 0.8f);
        cfg.balanced_opening_reject_prob =
            cfg_get<float>(cfg_map, "BALANCED_OPENING_REJECT_PROB", 0.995f);
        cfg.balanced_opening_reject_prob_fallback =
            cfg_get<float>(cfg_map, "BALANCED_OPENING_REJECT_PROB_FALLBACK", 0.8f);
        cfg.soft_resign_threshold = cfg_get<float>(cfg_map, "SOFT_RESIGN_THRESHOLD", 0.9f);
        cfg.soft_resign_step_threshold = cfg_get<int>(cfg_map, "SOFT_RESIGN_STEP_THRESHOLD", 3);
        cfg.soft_resign_prob = cfg_get<float>(cfg_map, "SOFT_RESIGN_PROB", 0.7f);
        cfg.soft_resign_sample_weight = cfg_get<float>(cfg_map, "SOFT_RESIGN_SAMPLE_WEIGHT", 0.1f);
        cfg.min_simulations_in_soft_resign = cfg_get<int>(cfg_map, "MIN_SIMS_IN_SOFT_RESIGN", 8);

        const bool use_cuda = torch::cuda::is_available();
        cfg.device = use_cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);

        // --- SelfplayParallelConfig ---
        SelfplayParallelConfig pcfg;
        pcfg.num_workers = cfg_get<int>(cfg_map, "NUM_WORKERS", 32);
        pcfg.num_inference_servers = cfg_get<int>(cfg_map, "NUM_INFERENCE_SERVERS", 2);
        pcfg.inference_batch_size = cfg_get<int>(cfg_map, "INFERENCE_BATCH_SIZE", 128);
        pcfg.inference_batch_wait_us = cfg_get<int>(cfg_map, "INFERENCE_WAIT_US", 100);
        pcfg.leaf_batch_size = cfg_get<int>(cfg_map, "LEAF_BATCH_SIZE", 8);
        pcfg.max_result_queue_size = cfg_get<int>(cfg_map, "MAX_RESULT_QUEUE_SIZE", 0);

        // --- MCTSBackendConfig ---
        MCTSBackendConfig bcfg;
        {
            auto it = cfg_map.find("MCTS_BACKEND");
            const std::string s = (it != cfg_map.end()) ? it->second : "batched_leaf";
            if (s == "shared_tree" || s == "tree" || s == "1") {
                bcfg.kind = MCTSBackendConfig::SharedTree;
            } else {
                bcfg.kind = MCTSBackendConfig::BatchedLeaf;
            }
        }
        bcfg.search_threads_per_tree = cfg_get<int>(cfg_map, "SEARCH_THREADS_PER_TREE", 4);

        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 4);
        const bool forbidden_plane = (num_planes >= 4);
        Gomoku game(cfg.board_size, /*renju=*/true, forbidden_plane);
        if (cfg.half_life <= 0) cfg.half_life = game.board_size;

        const int max_rows_per_npz = cfg_get<int>(cfg_map, "MAX_ROWS_PER_NPZ", 25000);
        const int64_t linear_threshold = cfg_get<int64_t>(cfg_map, "LINEAR_THRESHOLD", 2000000);
        const double replay_alpha = cfg_get<double>(cfg_map, "REPLAY_ALPHA", 0.8);

        // --- Writers & engine ---
        fs::create_directories(cli.output_dir);
        fs::create_directories(cli.log_dir);

        // Aggregate cumulative games/rows from last_run.tsv (history up to this iter).
        int64_t cum_games = 0;
        int64_t cum_rows = 0;
        {
            std::ifstream hf(fs::path(cli.log_dir) / "last_run.tsv");
            std::string line;
            bool first = true;
            while (std::getline(hf, line)) {
                if (first) { first = false; if (line.rfind("iter", 0) == 0) continue; }
                std::istringstream ls(line);
                int it; int64_t g, r;
                if (ls >> it >> g >> r) { cum_games += g; cum_rows += r; }
            }
        }
        const int64_t window_size = (cum_rows <= linear_threshold)
            ? cum_rows
            : static_cast<int64_t>(linear_threshold *
                std::pow(static_cast<double>(cum_rows) / linear_threshold, replay_alpha));

        std::ostringstream prefix_os;
        prefix_os << "iter_";
        prefix_os.width(6); prefix_os.fill('0'); prefix_os << cli.iter;
        const std::string npz_prefix = prefix_os.str();

        // Clean any leftover parts from a previously interrupted run of this
        // same iter, so NpzWriter's part counter re-aligns with disk state.
        // last_run.tsv is only appended on clean finish, so these orphans are
        // not yet accounted for in cum_rows/cum_games and must be discarded.
        {
            const std::string part_prefix = npz_prefix + "_part_";
            int removed = 0;
            for (const auto& p : fs::directory_iterator(cli.output_dir)) {
                const auto name = p.path().filename().string();
                if (name.rfind(part_prefix, 0) == 0) {
                    std::error_code ec;
                    fs::remove(p.path(), ec);
                    if (!ec) ++removed;
                }
            }
            if (removed > 0) {
                std::cout << "[selfplay] removed " << removed
                          << " leftover part file(s) from prior run of iter "
                          << cli.iter << "\n";
            }
        }

        NpzWriter writer(cli.output_dir, npz_prefix, game.board_size, game.num_planes, max_rows_per_npz);

        std::cout << "[selfplay] model=" << cli.model
                  << " iter=" << cli.iter
                  << " max_games=" << cli.max_games
                  << " workers=" << pcfg.num_workers
                  << " servers=" << pcfg.num_inference_servers
                  << " device=" << (use_cuda ? "cuda" : "cpu")
                  << "\n";
        std::cout << "[selfplay] TotalGames=" << cum_games
                  << " TotalSamples=" << cum_rows
                  << " WindowSize=" << window_size << "\n";

        std::cout << "[selfplay] mcts_backend="
                  << (bcfg.kind == MCTSBackendConfig::SharedTree ? "shared_tree" : "batched_leaf")
                  << " search_threads_per_tree=" << bcfg.search_threads_per_tree << "\n";
        SelfplayEngine<Gomoku> engine(game, cfg, pcfg, bcfg, cli.model, cfg.device);
        engine.start();

        // --- Main collection loop ---
        std::mt19937 rng(std::random_device{}());
        int games_done = 0;
        int64_t total_rows = 0;
        int black_wins = 0, white_wins = 0, draws = 0;
        const auto t0 = std::chrono::steady_clock::now();
        int last_report = 0;

        while (games_done < cli.max_games && !stop_requested.load()) {
            SelfplayEngine<Gomoku>::SelfplayResult r;
            if (!engine.try_pop_result(r, 200)) continue;
            if (r.samples.empty()) continue;

            const auto weights = compute_policy_surprise_weights(
                r.samples, cfg.policy_surprise_data_weight, cfg.value_surprise_data_weight);
            auto weighted = apply_surprise_weighting_to_game(r.samples, weights, rng);

            for (auto& ts : weighted) {
                // Expand raw board state (H*W int8, {-1,0,1}) to the 4-plane
                // encoded state that Python training consumes directly.
                ts.state = game.encode_state(ts.state, ts.to_play);
                writer.append(ts);
                total_rows += 1;
            }
            games_done += 1;
            if (r.winner == 1) ++black_wins;
            else if (r.winner == -1) ++white_wins;
            else ++draws;

            if (games_done - last_report >= 100 || games_done == cli.max_games) {
                last_report = games_done;
                const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - t0).count();
                const double sps = (dt > 0.0) ? (static_cast<double>(total_rows) / dt) : 0.0;
                const double avg_len = static_cast<double>(total_rows) / games_done;
                const double b = static_cast<double>(black_wins) / games_done;
                const double w = static_cast<double>(white_wins) / games_done;
                const double d = static_cast<double>(draws) / games_done;
                std::cout << "[selfplay] games=" << games_done << "/" << cli.max_games
                          << " sps=" << std::fixed << std::setprecision(1) << sps
                          << " AvgGameLen=" << std::fixed << std::setprecision(1) << avg_len
                          << " BWD=" << std::fixed << std::setprecision(2) << b
                          << "/" << std::fixed << std::setprecision(2) << w
                          << "/" << std::fixed << std::setprecision(2) << d
                          << "\n";
            }
        }

        engine.stop();
        writer.flush();

        // Append last_run.tsv
        const fs::path last_run = fs::path(cli.log_dir) / "last_run.tsv";
        const bool had_header = fs::exists(last_run);
        std::ofstream lf(last_run, std::ios::app);
        if (!had_header) lf << "iter\tgames\trows\tseconds\n";
        const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t0).count();
        lf << cli.iter << "\t" << games_done << "\t" << total_rows << "\t" << dt << "\n";

        std::cout << "[selfplay] done. games=" << games_done
                  << " rows=" << total_rows
                  << " t=" << std::fixed << std::setprecision(1) << dt << "s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[selfplay] fatal: " << e.what() << "\n";
        return 2;
    }
}
