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
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

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
    std::string stats_file;       // daemon mode: per-version stats append target
    int iter = 0;
    int max_games = 0;
    bool daemon = false;
    int model_watch_poll_ms = 2000;
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
        else if (k == "--daemon") a.daemon = true;
        else if (k == "--model-watch-poll-ms") a.model_watch_poll_ms = std::stoi(need("--model-watch-poll-ms"));
        else if (k == "--stats-file") a.stats_file = need("--stats-file");
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.output_dir.empty() || a.config.empty()) {
        throw std::runtime_error(
            "usage: selfplay_main --model PATH --output-dir DIR --config PATH "
            "(--iter N --max-games N | --daemon [--model-watch-poll-ms MS] [--stats-file PATH]) "
            "[--log-dir DIR]"
        );
    }
    if (!a.daemon && a.max_games <= 0) {
        throw std::runtime_error("--max-games required in non-daemon mode");
    }
    if (a.log_dir.empty()) a.log_dir = a.output_dir + "/../logs";
    if (a.daemon && a.stats_file.empty()) {
        a.stats_file = a.log_dir + "/daemon_stats.tsv";
    }
    return a;
}

// stat() wrappers for the daemon's mtime-watch loop. Returns 0 / -1 on error
// (callers treat 0/-1 as "skip this poll tick"). Linux has nanosecond mtime
// in struct stat::st_mtim; we collapse to a single int64 for easy compare.
static long long file_mtime_ns(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
#ifdef __linux__
    return static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL
         + static_cast<long long>(st.st_mtim.tv_nsec);
#else
    return static_cast<long long>(st.st_mtime) * 1000000000LL;
#endif
}

static long long file_size(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return -1;
    return static_cast<long long>(st.st_size);
}

// Naive "iter": N lookup in latest.meta.json. The file is generated by
// python/export_model.py with a known schema, so we don't pull in a JSON
// dependency — we just locate the key and read the integer that follows.
// Returns -1 on any parse error or missing file.
static int read_meta_iter(const std::string& meta_path) {
    std::ifstream f(meta_path);
    if (!f) return -1;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto p = s.find("\"iter\"");
    if (p == std::string::npos) return -1;
    const auto colon = s.find(':', p);
    if (colon == std::string::npos) return -1;
    int v = -1;
    std::istringstream ss(s.substr(colon + 1));
    ss >> v;
    return ss.fail() ? -1 : v;
}

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        torch::NoGradGuard no_grad;
        c10::InferenceMode im;

        const auto cli = parse_cli(argc, argv);

        // Cold-start: in daemon mode, latest.pt may not exist yet (the train
        // loop hasn't run init_model.py + first export). Block here rather
        // than crash inside torch::jit::load.
        if (cli.daemon) {
            while (!stop_requested.load() && !fs::exists(cli.model)) {
                std::cout << "[Daemon] waiting for " << cli.model << " ...\n";
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            if (stop_requested.load()) return 0;
        }
        auto cfg_map = parse_cfg(cli.config);
        // Allow env var to override the cfg file (used by selfplay.sh's GPU_NUM
        // wrapper, which derives this list at run time).
        if (const char* env = std::getenv("INFERENCE_SERVER_DEVICES")) {
            cfg_map["INFERENCE_SERVER_DEVICES"] = env;
        }

        // --- AlphaZeroConfig ---
        AlphaZeroConfig cfg;
        cfg.board_size = cfg_get<int>(cfg_map, "BOARD_SIZE", 15);
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 64);
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.half_life = cfg_get<int>(cfg_map, "HALF_LIFE", -1);
        cfg.move_temperature = cfg_get<float>(cfg_map, "MOVE_TEMPERATURE", 1.0f);
        cfg.c_puct = cfg_get<float>(cfg_map, "C_PUCT", 1.1f);
        cfg.c_puct_log = cfg_get<float>(cfg_map, "C_PUCT_LOG", 0.45f);
        cfg.c_puct_base = cfg_get<float>(cfg_map, "C_PUCT_BASE", 500.0f);
        cfg.fpu_pow = cfg_get<float>(cfg_map, "FPU_POW", 1.0f);
        cfg.fpu_reduction_max = cfg_get<float>(cfg_map, "FPU_REDUCTION_MAX", 0.08f);
        cfg.root_fpu_reduction_max = cfg_get<float>(cfg_map, "ROOT_FPU_REDUCTION_MAX", 0.0f);
        cfg.fpu_loss_prop = cfg_get<float>(cfg_map, "FPU_LOSS_PROP", 0.0f);
        cfg.cpuct_utility_stdev_prior = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR", 0.25f);
        cfg.cpuct_utility_stdev_prior_weight = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_PRIOR_WEIGHT", 1.0f);
        cfg.cpuct_utility_stdev_scale = cfg_get<float>(cfg_map, "CPUCT_UTILITY_STDEV_SCALE", 0.0f);
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
        cfg.policy_init_avg_move_num =
            cfg_get<float>(cfg_map, "POLICY_INIT_AVG_MOVE_NUM", 0.0f);
        cfg.policy_init_temperature =
            cfg_get<float>(cfg_map, "POLICY_INIT_TEMPERATURE", 1.0f);
        cfg.soft_resign_threshold = cfg_get<float>(cfg_map, "SOFT_RESIGN_THRESHOLD", 0.9f);
        cfg.soft_resign_step_threshold = cfg_get<int>(cfg_map, "SOFT_RESIGN_STEP_THRESHOLD", 3);
        cfg.soft_resign_prob = cfg_get<float>(cfg_map, "SOFT_RESIGN_PROB", 0.7f);
        cfg.soft_resign_sample_weight = cfg_get<float>(cfg_map, "SOFT_RESIGN_SAMPLE_WEIGHT", 0.1f);
        cfg.min_simulations_in_soft_resign = cfg_get<int>(cfg_map, "MIN_SIMS_IN_SOFT_RESIGN", 8);
        cfg.enable_tree_reuse = cfg_get_bool(cfg_map, "ENABLE_TREE_REUSE", true);

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

        // --- Per-inference-server device assignment (KataGo-style multi-GPU) ---
        // Empty / unset INFERENCE_SERVER_DEVICES preserves single-GPU behavior:
        // every server lands on cuda:0 (or CPU if CUDA unavailable). A non-empty
        // comma-separated list of GPU indices must have exactly num_inference_servers
        // entries. e.g. "0,1,0,1" with NUM_INFERENCE_SERVERS=4 -> 2-GPU round-robin.
        std::vector<torch::Device> server_devices;
        {
            const int n = pcfg.num_inference_servers;
            std::string s;
            if (auto it = cfg_map.find("INFERENCE_SERVER_DEVICES"); it != cfg_map.end()) {
                s = it->second;
            }
            // trim
            const auto a = s.find_first_not_of(" \t\r");
            if (a == std::string::npos) s.clear();
            else { const auto b = s.find_last_not_of(" \t\r"); s = s.substr(a, b - a + 1); }

            if (!use_cuda) {
                server_devices.assign(n, torch::Device(torch::kCPU));
            } else if (s.empty()) {
                server_devices.assign(n, torch::Device(torch::kCUDA, 0));
            } else {
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    const auto ta = tok.find_first_not_of(" \t\r");
                    if (ta == std::string::npos) continue;
                    const auto tb = tok.find_last_not_of(" \t\r");
                    tok = tok.substr(ta, tb - ta + 1);
                    if (tok.empty()) continue;
                    server_devices.emplace_back(torch::kCUDA, std::stoi(tok));
                }
                if (static_cast<int>(server_devices.size()) != n) {
                    throw std::runtime_error(
                        "INFERENCE_SERVER_DEVICES has " + std::to_string(server_devices.size())
                        + " entries but NUM_INFERENCE_SERVERS=" + std::to_string(n));
                }
            }
            pcfg.inference_server_devices.clear();
            pcfg.inference_server_devices.reserve(server_devices.size());
            for (const auto& d : server_devices) {
                pcfg.inference_server_devices.push_back(d.is_cuda() ? d.index() : -1);
            }
        }

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

        // V5: num_planes=5 (mask + own + opp + fb_b + fb_w), padded to MAX_BOARD_SIZE.
        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 5);
        const int num_global_features = cfg_get<int>(cfg_map, "NUM_GLOBAL_FEATURES", 12);
        const std::string rule_str = ([&]() -> std::string {
            auto it = cfg_map.find("RULE");
            return (it != cfg_map.end()) ? it->second : "renju";
        })();
        const RuleType rule = rule_from_string(rule_str);
        const bool forbidden_plane = (rule != RuleType::FREESTYLE);
        Gomoku game(cfg.board_size, rule, forbidden_plane);
        if (cfg.half_life <= 0) cfg.half_life = game.board_size;

        const int max_rows_per_npz = cfg_get<int>(cfg_map, "MAX_ROWS_PER_NPZ", 25000);
        const int64_t linear_threshold = cfg_get<int64_t>(cfg_map, "LINEAR_THRESHOLD", 2000000);
        const double replay_alpha = cfg_get<double>(cfg_map, "REPLAY_ALPHA", 0.8);

        // --- Writers & engine ---
        fs::create_directories(cli.output_dir);
        fs::create_directories(cli.log_dir);

        // Aggregate cumulative games/rows from last_run.tsv (history up to this iter).
        // Daemon mode skips this — last_run.tsv is per-iter, the daemon has no iter
        // boundary, and the readout is purely cosmetic anyway (shuffle.py recomputes
        // the window from live NPZ row counts).
        int64_t cum_games = 0;
        int64_t cum_rows = 0;
        int64_t window_size = 0;
        if (!cli.daemon) {
            std::ifstream hf(fs::path(cli.log_dir) / "last_run.tsv");
            std::string line;
            bool first = true;
            while (std::getline(hf, line)) {
                if (first) { first = false; if (line.rfind("iter", 0) == 0) continue; }
                std::istringstream ls(line);
                int it; int64_t g, r;
                if (ls >> it >> g >> r) { cum_games += g; cum_rows += r; }
            }
            window_size = (cum_rows <= linear_threshold)
                ? cum_rows
                : static_cast<int64_t>(linear_threshold *
                    std::pow(static_cast<double>(cum_rows) / linear_threshold, replay_alpha));
        }

        // Initial NPZ filename prefix. Daemon picks a placeholder here and
        // rotates to the real "daemon_v<iter>_pid<pid>" prefix after reading
        // latest.meta.json (which we can only do once latest.pt exists; the
        // cold-start wait above guarantees that).
        std::string npz_prefix;
        if (cli.daemon) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "daemon_pending_pid%d", static_cast<int>(::getpid()));
            npz_prefix = buf;
        } else {
            std::ostringstream prefix_os;
            prefix_os << "iter_";
            prefix_os.width(6); prefix_os.fill('0'); prefix_os << cli.iter;
            npz_prefix = prefix_os.str();
        }

        // Clean any leftover parts from a previously interrupted run of this
        // same iter, so NpzWriter's part counter re-aligns with disk state.
        // last_run.tsv is only appended on clean finish, so these orphans are
        // not yet accounted for in cum_rows/cum_games and must be discarded.
        // Daemon mode skips this (its files include pid + monotonic version
        // and stay valid across restarts).
        if (!cli.daemon) {
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
                std::cout << "[SelfPlay] removed " << removed
                          << " leftover part file(s) from prior run of iter "
                          << cli.iter << "\n";
            }
        }

        // V5: state is padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE = 15×15
        // regardless of game.board_size, so state_row_override = 5*225 = 1125.
        const int state_row = Gomoku::NUM_SPATIAL_PLANES_V5 * Gomoku::MAX_AREA;
        NpzWriter writer(cli.output_dir, npz_prefix, Gomoku::MAX_BOARD_SIZE, game.num_planes,
                         max_rows_per_npz, /*max_pending_jobs=*/4,
                         num_global_features, state_row);

        std::cout << "[SelfPlay] model=" << cli.model
                  << (cli.daemon ? " mode=daemon" : "")
                  << " iter=" << cli.iter
                  << " max_games=" << cli.max_games
                  << " workers=" << pcfg.num_workers
                  << " servers=" << pcfg.num_inference_servers
                  << " devices=[";
        for (size_t i = 0; i < server_devices.size(); ++i) {
            if (i > 0) std::cout << ",";
            if (server_devices[i].is_cuda()) std::cout << "cuda:" << static_cast<int>(server_devices[i].index());
            else std::cout << "cpu";
        }
        std::cout << "]\n";
        if (!cli.daemon) {
            std::cout << "[SelfPlay] TotalGames=" << cum_games
                      << " TotalSamples=" << cum_rows
                      << " WindowSize=" << window_size << "\n";
        }

        std::cout << "[SelfPlay] mcts_backend="
                  << (bcfg.kind == MCTSBackendConfig::SharedTree ? "shared_tree" : "batched_leaf")
                  << " search_threads_per_tree=" << bcfg.search_threads_per_tree << "\n";
        SelfplayEngine<Gomoku> engine(game, cfg, pcfg, bcfg, cli.model, server_devices);
        engine.start();

        std::mt19937 rng(std::random_device{}());

        if (cli.daemon) {
            // --- Daemon main loop ---
            // Read initial model version from latest.meta.json and rotate the
            // writer to the real prefix. If meta is missing or malformed we
            // fall back to version 0; the next successful export will replace
            // the prefix anyway.
            const std::string meta_path = (fs::path(cli.model).parent_path() / "latest.meta.json").string();
            int cur_version = std::max(0, read_meta_iter(meta_path));
            auto make_prefix = [](int v) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "daemon_v%06d_pid%d", v, static_cast<int>(::getpid()));
                return std::string(buf);
            };
            writer.rotate(make_prefix(cur_version));

            long long last_mtime_ns = file_mtime_ns(cli.model);
            const auto t_global = std::chrono::steady_clock::now();
            int total_games = 0;
            int64_t total_rows = 0;
            int last_log_games = 0;

            // Per-version stats (reset after each reload).
            auto version_t0 = std::chrono::steady_clock::now();
            int v_games = 0;
            int64_t v_rows = 0;
            int v_black = 0, v_white = 0, v_draw = 0;
            double v_sum_len = 0.0;

            auto append_stats_row = [&](int version) {
                if (cli.stats_file.empty()) return;
                const bool had_header = fs::exists(cli.stats_file);
                std::ofstream sf(cli.stats_file, std::ios::app);
                if (!sf) return;
                if (!had_header) {
                    sf << "model_version\tgames\trows\tseconds"
                          "\tavg_len\tbwr\twwr\tdwr\tstart_unix\tend_unix\n";
                }
                const auto t1 = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(t1 - version_t0).count();
                const double avg_len = v_games > 0 ? v_sum_len / v_games : 0.0;
                const double bwr = v_games > 0 ? static_cast<double>(v_black) / v_games : 0.0;
                const double wwr = v_games > 0 ? static_cast<double>(v_white) / v_games : 0.0;
                const double dwr = v_games > 0 ? static_cast<double>(v_draw) / v_games : 0.0;
                const auto end_unix = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                const auto start_unix = end_unix - static_cast<long long>(dt);
                sf << version << "\t" << v_games << "\t" << v_rows
                   << "\t" << std::fixed << std::setprecision(2) << dt
                   << "\t" << std::fixed << std::setprecision(2) << avg_len
                   << "\t" << std::fixed << std::setprecision(4) << bwr
                   << "\t" << std::fixed << std::setprecision(4) << wwr
                   << "\t" << std::fixed << std::setprecision(4) << dwr
                   << "\t" << start_unix << "\t" << end_unix << "\n";
            };
            auto reset_version_counters = [&]() {
                version_t0 = std::chrono::steady_clock::now();
                v_games = 0; v_rows = 0; v_black = 0; v_white = 0; v_draw = 0;
                v_sum_len = 0.0;
            };

            std::cout << "[Daemon] starting at v=" << cur_version
                      << " prefix=" << make_prefix(cur_version)
                      << " poll_ms=" << cli.model_watch_poll_ms << "\n";

            auto last_poll = std::chrono::steady_clock::now();

            while (!stop_requested.load()) {
                // Model-mtime watch.
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_poll).count()
                    >= cli.model_watch_poll_ms) {
                    last_poll = now;
                    const long long m1 = file_mtime_ns(cli.model);
                    if (m1 != 0 && m1 != last_mtime_ns) {
                        // Stability: re-stat after a short wait, require mtime
                        // and size both unchanged. Combined with export.sh's
                        // os.replace this is overkill but cheap and protects
                        // against any future non-atomic writer.
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        const long long m2 = file_mtime_ns(cli.model);
                        const long long sz1 = file_size(cli.model);
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        const long long m3 = file_mtime_ns(cli.model);
                        const long long sz2 = file_size(cli.model);
                        if (m2 == m3 && sz1 == sz2 && sz1 > 0) {
                            try {
                                engine.reload_model(cli.model);
                                const int new_version = std::max(0, read_meta_iter(meta_path));
                                append_stats_row(cur_version);
                                cur_version = new_version;
                                writer.rotate(make_prefix(cur_version));
                                reset_version_counters();
                                last_mtime_ns = m3;
                                std::cout << "[Daemon] reload_model: v=" << cur_version
                                          << " prefix=" << make_prefix(cur_version) << "\n";
                            } catch (const std::exception& e) {
                                std::cerr << "[Daemon] reload failed: " << e.what() << "\n";
                                last_mtime_ns = m3;  // don't retry the same broken file
                            }
                        }
                    }
                }

                SelfplayEngine<Gomoku>::SelfplayResult r;
                if (!engine.try_pop_result(r, 200)) continue;
                if (r.samples.empty()) continue;

                const auto weights = compute_policy_surprise_weights(
                    r.samples, cfg.policy_surprise_data_weight, cfg.value_surprise_data_weight);
                auto weighted = apply_surprise_weighting_to_game(r.samples, weights, rng);
                for (auto& ts : weighted) {
                    // V5: derive ply from raw board (count non-zero) BEFORE encode replaces it.
                    int ply = 0;
                    for (auto v : ts.state) if (v != 0) ++ply;
                    auto gf = game.compute_global_features(ply, ts.to_play);
                    for (int i = 0; i < 12; ++i) ts.global_features[i] = gf.data[i];
                    ts.state = game.encode_state_v5(ts.state, ts.to_play);
                    writer.append(ts);
                    v_rows += 1;
                    total_rows += 1;
                }
                v_sum_len += r.game_len;
                v_games += 1;
                total_games += 1;
                if (r.winner == 1) ++v_black;
                else if (r.winner == -1) ++v_white;
                else ++v_draw;

                if (total_games - last_log_games >= 100) {
                    last_log_games = total_games;
                    const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_global).count();
                    const double sps = dt > 0.0 ? total_rows / dt : 0.0;
                    std::cout << "[Daemon] v=" << cur_version
                              << " total_games=" << total_games
                              << " total_rows=" << total_rows
                              << " sps=" << std::fixed << std::setprecision(1) << sps
                              << "\n";
                }
            }

            engine.stop();
            writer.flush();
            append_stats_row(cur_version);
            std::cout << "[Daemon] shutdown clean. total_games=" << total_games
                      << " total_rows=" << total_rows << "\n";
            return 0;
        }

        // --- Legacy main collection loop (one-shot, --max-games) ---
        int games_done = 0;
        int64_t total_rows = 0;
        int black_wins = 0, white_wins = 0, draws = 0;
        int min_len = std::numeric_limits<int>::max();
        int max_len = 0;
        double sum_len = 0.0, sum_sq_len = 0.0;
        SelfplayEngine<Gomoku>::SelfplayResult min_game, max_game;
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
                // V5: encode to 5-plane padded layout (mask + own + opp + fb_b + fb_w)
                // and capture per-step global features (rule one-hot, ply, etc.).
                int ply = 0;
                for (auto v : ts.state) if (v != 0) ++ply;
                auto gf = game.compute_global_features(ply, ts.to_play);
                for (int i = 0; i < 12; ++i) ts.global_features[i] = gf.data[i];
                ts.state = game.encode_state_v5(ts.state, ts.to_play);
                writer.append(ts);
                total_rows += 1;
            }
            const int L = r.game_len;
            sum_len += L;
            sum_sq_len += static_cast<double>(L) * L;
            if (L < min_len) { min_len = L; min_game = r; }
            if (L > max_len) { max_len = L; max_game = r; }

            games_done += 1;
            if (r.winner == 1) ++black_wins;
            else if (r.winner == -1) ++white_wins;
            else ++draws;

            if (games_done - last_report >= 100 || games_done == cli.max_games) {
                last_report = games_done;
                const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - t0).count();
                const double sps = (dt > 0.0) ? (static_cast<double>(total_rows) / dt) : 0.0;
                const double avg_len = sum_len / games_done;
                const double var_len = std::max(0.0, sum_sq_len / games_done - avg_len * avg_len);
                const double std_len = std::sqrt(var_len);
                const double b = static_cast<double>(black_wins) / games_done;
                const double w = static_cast<double>(white_wins) / games_done;
                const double d = static_cast<double>(draws) / games_done;
                const int gw = static_cast<int>(std::to_string(cli.max_games).size());
                const int bp = static_cast<int>(std::round(b * 100.0));
                const int wp = static_cast<int>(std::round(w * 100.0));
                const int dp = static_cast<int>(std::round(d * 100.0));
                std::cout << "[SelfPlay] Games=" << std::setw(gw) << std::setfill('0') << games_done
                          << std::setfill(' ') << "/" << cli.max_games
                          << " Sps=" << std::fixed << std::setprecision(1) << sps
                          << " GameLen:Avg=" << std::fixed << std::setprecision(1) << avg_len
                          << " Min=" << min_len
                          << " Max=" << max_len
                          << " Std=" << static_cast<int>(std::round(std_len))
                          << " BWD=" << std::setw(2) << std::setfill('0') << bp
                          << "/" << std::setw(2) << std::setfill('0') << wp
                          << "/" << std::setw(2) << std::setfill('0') << dp
                          << std::setfill(' ') << "\n";
            }
        }

        engine.stop();
        writer.flush();

        // Append last_run.tsv
        const fs::path last_run = fs::path(cli.log_dir) / "last_run.tsv";
        const bool had_header = fs::exists(last_run);
        std::ofstream lf(last_run, std::ios::app);
        if (!had_header) lf << "iter\tgames\trows\tseconds"
                               "\tmin_len\tmax_len\tavg_len"
                               "\tblack_win_rate\twhite_win_rate\tdraw_rate\n";
        const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t0).count();
        lf << cli.iter << "\t" << games_done << "\t" << total_rows << "\t" << dt;
        if (games_done > 0) {
            const double avg_len = sum_len / games_done;
            const double bwr = static_cast<double>(black_wins) / games_done;
            const double wwr = static_cast<double>(white_wins) / games_done;
            const double dwr = static_cast<double>(draws) / games_done;
            lf << "\t" << min_len << "\t" << max_len
               << "\t" << std::fixed << std::setprecision(3) << avg_len
               << "\t" << std::fixed << std::setprecision(4) << bwr
               << "\t" << std::fixed << std::setprecision(4) << wwr
               << "\t" << std::fixed << std::setprecision(4) << dwr;
        } else {
            lf << "\t\t\t\t\t\t";
        }
        lf << "\n";

        std::cout << "[SelfPlay] done. games=" << games_done
                  << " rows=" << total_rows
                  << " t=" << std::fixed << std::setprecision(1) << dt << "s\n";

        auto print_board = [&](const char* tag, const SelfplayEngine<Gomoku>::SelfplayResult& r) {
            const char* opening = r.balanced_opening ? "balanced" : "empty";
            std::cout << "[SelfPlay] " << tag
                      << " opening=" << opening
                      << " game_len=" << r.game_len
                      << " winner=" << r.winner << "\n";
            const int N = game.board_size;
            auto dump = [&](const std::vector<int8_t>& board) {
                for (int i = 0; i < N; ++i) {
                    std::cout << "  ";
                    for (int j = 0; j < N; ++j) {
                        const int8_t v = board[i * N + j];
                        const char c = (v == 1) ? 'X' : (v == -1) ? 'O' : '.';
                        std::cout << c << ' ';
                    }
                    std::cout << "\n";
                }
            };
            if (r.balanced_opening && !r.initial_state.empty()) {
                std::cout << "  initial (to_play=" << r.initial_to_play << "):\n";
                dump(r.initial_state);
                std::cout << "  final:\n";
            }
            dump(r.final_state);
        };
        if (games_done > 0) {
            print_board("min-len game:", min_game);
            print_board("max-len game:", max_game);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[SelfPlay] fatal: " << e.what() << "\n";
        return 2;
    }
}
