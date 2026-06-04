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
#include <cstdio>

#include <torch/script.h>
#include <torch/torch.h>

#include "skyzero.h"
#include "skyzero_parallel.h"
#include "envs/gomoku.h"
#include "game_initializer.h"
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
    auto load = [&out](std::istream& f) {
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
    };
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open config: " + path);
    }
    load(f);
    // Server-local override: same precedence as scripts/run.sh's
    // `source run.cfg.local`. Optional — silently skipped if absent.
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

// ---------------------------------------------------------------------------
// List parsers (comma-separated). Used for BOARD_SIZES / BOARD_SIZE_RELPROBS /
// RULES / RULE_RELPROBS, modeled after KataGomo's bSizes / bSizeRelProbs cfg.
// Empty / missing → fallback list. Whitespace tolerated.
// ---------------------------------------------------------------------------
static std::vector<std::string> cfg_split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    for (auto& v : out) {
        const auto a = v.find_first_not_of(" \t\r");
        const auto b = v.find_last_not_of(" \t\r");
        v = (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
    }
    return out;
}

template <typename T>
static std::vector<T> cfg_get_num_list(const std::unordered_map<std::string, std::string>& c,
                                       const std::string& key,
                                       const std::vector<T>& fallback) {
    auto it = c.find(key);
    if (it == c.end() || it->second.empty()) return fallback;
    const auto parts = cfg_split_csv(it->second);
    std::vector<T> out;
    out.reserve(parts.size());
    for (const auto& p : parts) {
        if (p.empty()) continue;
        std::istringstream ss(p);
        T v;
        ss >> v;
        if (ss.fail()) {
            throw std::runtime_error("bad numeric in " + key + ": '" + p + "'");
        }
        out.push_back(v);
    }
    if (out.empty()) return fallback;
    return out;
}

static std::vector<std::string> cfg_get_string_list(
    const std::unordered_map<std::string, std::string>& c,
    const std::string& key,
    const std::vector<std::string>& fallback
) {
    auto it = c.find(key);
    if (it == c.end() || it->second.empty()) return fallback;
    auto parts = cfg_split_csv(it->second);
    parts.erase(std::remove_if(parts.begin(), parts.end(),
                               [](const std::string& s) { return s.empty(); }),
                parts.end());
    if (parts.empty()) return fallback;
    return parts;
}

struct CliArgs {
    std::string model;
    std::string output_dir;
    std::string log_dir;
    std::string config;
    int iter = 0;
    int max_games = 0;
    bool daemon = false;
    int model_watch_poll_ms = 2000;
    // Per-iter warmup override for NUM_SIMULATIONS (computed by
    // python/compute_num_simulations.py). <= 0 means "use cfg value".
    int num_simulations_override = -1;
    // Daemon-only: shell command (e.g. "python warmup.py num-simulations
    // --data-dir DIR") executed at startup and on every model reload, whose
    // stdout integer overrides cfg.num_simulations. Empty disables.
    std::string sims_warmup_cmd;
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
        else if (k == "--num-simulations") a.num_simulations_override = std::stoi(need("--num-simulations"));
        else if (k == "--sims-warmup-cmd") a.sims_warmup_cmd = need("--sims-warmup-cmd");
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.model.empty() || a.output_dir.empty() || a.config.empty()) {
        throw std::runtime_error(
            "usage: selfplay_main --model PATH --output-dir DIR --config PATH "
            "(--iter N --max-games N | --daemon [--model-watch-poll-ms MS]) "
            "[--log-dir DIR]"
        );
    }
    if (!a.daemon && a.max_games <= 0) {
        throw std::runtime_error("--max-games required in non-daemon mode");
    }
    if (a.log_dir.empty()) a.log_dir = a.output_dir + "/../logs";
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

// Run a shell command and parse its first stdout line as an int. Returns
// -1 on any failure (empty cmd, popen error, non-zero exit, non-numeric
// stdout). Used by the daemon to consult python/warmup.py for staged
// hyperparameters (currently only num_simulations).
static int popen_int(const std::string& cmd) {
    if (cmd.empty()) return -1;
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) return -1;
    char buf[256];
    std::string out;
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
    const int rc = ::pclose(pipe);
    if (rc != 0) return -1;
    try {
        return std::stoi(out);
    } catch (...) {
        return -1;
    }
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
        // Allow env var to override the cfg file. selfplay.sh and
        // run_selfplay_daemon.sh resolve per-GPU counts to absolute values
        // here and export them via env.
        for (const char* k : {"INFERENCE_SERVER_DEVICES",
                              "NUM_INFERENCE_SERVERS",
                              "NUM_WORKERS"}) {
            if (const char* env = std::getenv(k)) {
                cfg_map[k] = env;
            }
        }

        // --- SkyZeroConfig ---
        SkyZeroConfig cfg;
        // canvas size is the compile-time constant in cpp/envs/gomoku.h:65.
        cfg.board_size = Gomoku::MAX_BOARD_SIZE;
        cfg.num_simulations = cfg_get<int>(cfg_map, "NUM_SIMULATIONS", 64);
        if (cli.num_simulations_override > 0) {
            cfg.num_simulations = cli.num_simulations_override;
        }
        // Daemon-only: consult warmup script for staged sims on startup.
        // engine_'s cfg_ is a const reference into this very cfg, so we
        // also re-poll on every model reload (see daemon main loop below).
        if (cli.daemon && !cli.sims_warmup_cmd.empty()) {
            const int v = popen_int(cli.sims_warmup_cmd);
            if (v > 0) {
                cfg.num_simulations = v;
                std::cout << "[Daemon] startup num_simulations=" << v
                          << " (from warmup cmd)\n";
            } else {
                std::cerr << "[Daemon] warmup cmd failed; keeping num_simulations="
                          << cfg.num_simulations << "\n";
            }
        }
        cfg.gumbel_m = cfg_get<int>(cfg_map, "GUMBEL_M", 16);
        cfg.gumbel_c_visit = cfg_get<float>(cfg_map, "GUMBEL_C_VISIT", 50.0f);
        cfg.gumbel_c_scale = cfg_get<float>(cfg_map, "GUMBEL_C_SCALE", 1.0f);
        cfg.non_root_search_algo = SkyZeroConfig::parse_non_root_search_algo(
            cfg_get<std::string>(cfg_map, "NON_ROOT_SEARCH_ALGO", "puct"));
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
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_ROOT", true);
        cfg.enable_stochastic_transform_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_STOCHASTIC_TRANSFORM_CHILD", true);
        cfg.enable_symmetry_inference_for_root =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_ROOT", false);
        cfg.enable_symmetry_inference_for_child =
            cfg_get_bool(cfg_map, "ENABLE_SYMMETRY_CHILD", false);
        cfg.policy_surprise_data_weight = cfg_get<float>(cfg_map, "POLICY_SURPRISE_DATA_WEIGHT", 0.5f);
        cfg.value_surprise_data_weight = cfg_get<float>(cfg_map, "VALUE_SURPRISE_DATA_WEIGHT", 0.1f);
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
        cfg.soft_resign_sample_weight = cfg_get<float>(cfg_map, "SOFT_RESIGN_SAMPLE_WEIGHT", 0.1f);
        cfg.reduced_visits_fraction = cfg_get<float>(cfg_map, "REDUCED_VISITS_FRACTION", 0.25f);
        cfg.reduced_visits_min_floor = cfg_get<int>(cfg_map, "REDUCED_VISITS_MIN_FLOOR", 16);
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
        }

        // V5: num_planes=5 (mask + own + opp + fb_b + fb_w), padded to MAX_BOARD_SIZE.
        const int num_planes = cfg_get<int>(cfg_map, "NUM_PLANES", 5);
        const int num_global_features = cfg_get<int>(cfg_map, "NUM_GLOBAL_FEATURES", 12);

        // ---- Per-game (size, rule) sampling, KataGomo bSizes / bSizeRelProbs ----
        // BOARD_SIZES / BOARD_SIZE_RELPROBS / RULES / RULE_RELPROBS are required
        // lists. MAIN_BOARD_SIZE / MAIN_RULE are also required and must appear
        // in the lists with relprob > 0; they pick out the (size, rule) pair
        // used as the headline for selfplay-stat logging.
        auto require_str = [&](const std::string& k) -> std::string {
            auto it = cfg_map.find(k);
            if (it == cfg_map.end() || it->second.empty()) {
                throw std::runtime_error("missing required key in run.cfg: " + k);
            }
            return it->second;
        };
        const int main_board_size = std::stoi(require_str("MAIN_BOARD_SIZE"));
        const std::string main_rule_str = require_str("MAIN_RULE");
        const RuleType main_rule = rule_from_string(main_rule_str);

        auto require_list_present = [&](const std::string& k) {
            auto it = cfg_map.find(k);
            if (it == cfg_map.end() || it->second.empty()) {
                throw std::runtime_error("missing required key in run.cfg: " + k);
            }
        };
        require_list_present("BOARD_SIZES");
        require_list_present("BOARD_SIZE_RELPROBS");
        require_list_present("RULES");
        require_list_present("RULE_RELPROBS");
        const auto sizes      = cfg_get_num_list<int>(cfg_map, "BOARD_SIZES", {});
        const auto size_probs = cfg_get_num_list<float>(cfg_map, "BOARD_SIZE_RELPROBS", {});
        const auto rule_strs  = cfg_get_string_list(cfg_map, "RULES", {});
        const auto rule_probs = cfg_get_num_list<float>(cfg_map, "RULE_RELPROBS", {});

        // Validate MAIN_* is in lists with relprob > 0.
        int msi = -1;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] == main_board_size) { msi = static_cast<int>(i); break; }
        }
        if (msi < 0) {
            throw std::runtime_error("MAIN_BOARD_SIZE=" + std::to_string(main_board_size)
                                     + " not present in BOARD_SIZES");
        }
        if (msi >= static_cast<int>(size_probs.size()) || size_probs[msi] <= 0.0f) {
            throw std::runtime_error("MAIN_BOARD_SIZE=" + std::to_string(main_board_size)
                                     + " has zero relprob in BOARD_SIZE_RELPROBS");
        }
        int mri = -1;
        for (size_t i = 0; i < rule_strs.size(); ++i) {
            if (rule_strs[i] == main_rule_str) { mri = static_cast<int>(i); break; }
        }
        if (mri < 0) {
            throw std::runtime_error("MAIN_RULE='" + main_rule_str + "' not present in RULES");
        }
        if (mri >= static_cast<int>(rule_probs.size()) || rule_probs[mri] <= 0.0f) {
            throw std::runtime_error("MAIN_RULE='" + main_rule_str
                                     + "' has zero relprob in RULE_RELPROBS");
        }

        std::vector<RuleType> rules;
        rules.reserve(rule_strs.size());
        for (const auto& s : rule_strs) rules.push_back(rule_from_string(s));

        // Seed: derive from --iter + a stable salt so daemon and main runs
        // diverge sharply even if started in the same wall-clock second.
        const uint64_t init_seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        ) ^ (static_cast<uint64_t>(cli.iter) * 0x9E3779B97F4A7C15ULL);
        GameInitializer game_init(sizes, size_probs, rules, rule_probs, init_seed);
        game_init.log_distribution(std::cout);

        const int max_rows_per_npz = cfg_get<int>(cfg_map, "MAX_ROWS_PER_NPZ", 25000);

        // --- Writers & engine ---
        fs::create_directories(cli.output_dir);
        fs::create_directories(cli.log_dir);

        // Aggregate cumulative games/rows from selfplay.tsv. Counts BOTH
        // producers (main + daemon) — see python/selfplay_log.py for the
        // same logic. Used only for the diagnostic print below.
        //
        // Resume hygiene: if a prior run finished selfplay (so it appended a
        // producer=main row) but train/export got interrupted, state.json
        // still points at the previous iter; resume reruns this same iter,
        // and the previously-written main row would otherwise double-count.
        // Drop any main rows with iter >= cli.iter. Daemon rows are not
        // dropped — they have no iter/resume boundary.
        int64_t cum_games = 0;
        int64_t cum_rows = 0;
        if (!cli.daemon) {
            const fs::path tsv_path = fs::path(cli.log_dir) / "selfplay.tsv";
            std::vector<std::string> kept_lines;
            std::string header_line;
            bool had_header = false;
            int dropped = 0;
            {
                std::ifstream hf(tsv_path);
                std::string line;
                bool first = true;
                while (std::getline(hf, line)) {
                    if (first) {
                        first = false;
                        if (line.rfind("producer", 0) == 0) {
                            header_line = line;
                            had_header = true;
                            continue;
                        }
                    }
                    std::istringstream ls(line);
                    std::string producer;
                    int it; int64_t g, r;
                    if (!(ls >> producer >> it >> g >> r)) {
                        kept_lines.push_back(line);
                        continue;
                    }
                    if (producer == "main" && it >= cli.iter) {
                        ++dropped;
                        continue;
                    }
                    cum_games += g;
                    cum_rows += r;
                    kept_lines.push_back(line);
                }
            }
            if (dropped > 0) {
                std::ofstream of(tsv_path, std::ios::trunc);
                if (had_header) of << header_line << "\n";
                for (const auto& l : kept_lines) of << l << "\n";
                std::cout << "[SelfPlay] dropped " << dropped
                          << " stale main row(s) with iter >= "
                          << cli.iter << " (resume after interrupted train)\n";
            }
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

        // Clean leftover part files from a previously interrupted run.
        //
        // Main loop: orphans of *this same* iter, so NpzWriter's part counter
        // re-aligns. selfplay.tsv (producer="main") is only appended on clean
        // finish, so these orphans are not yet accounted for in cum_rows.
        //
        // Daemon: orphans named daemon_pending_pid*_part_*.npz from prior
        // daemon crashes that died before rotating to a real version prefix.
        // Real daemon files (daemon_v<NNNNNN>_pid<pid>_part_*.npz) are
        // intentionally kept across restarts.
        {
            const std::string sweep_prefix = cli.daemon
                ? std::string("daemon_pending_pid")
                : (npz_prefix + "_part_");
            int removed = 0;
            for (const auto& p : fs::directory_iterator(cli.output_dir)) {
                const auto name = p.path().filename().string();
                if (name.rfind(sweep_prefix, 0) == 0) {
                    std::error_code ec;
                    fs::remove(p.path(), ec);
                    if (!ec) ++removed;
                }
            }
            if (removed > 0) {
                std::cout << "[SelfPlay] removed " << removed
                          << " leftover " << (cli.daemon ? "daemon_pending" : "iter part")
                          << " file(s)\n";
            }
        }

        // V5: state is padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE
        // regardless of game.board_size, so
        // state_row_override = NUM_SPATIAL_PLANES_V5 * MAX_AREA.
        const int state_row = Gomoku::NUM_SPATIAL_PLANES_V5 * Gomoku::MAX_AREA;
        NpzWriter writer(cli.output_dir, npz_prefix, Gomoku::MAX_BOARD_SIZE, num_planes,
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
            // Cumulative across both producers (main + daemon). Shuffle.py
            // computes the real window from disk-scanned row counts.
            std::cout << "[SelfPlay] TotalGames=" << cum_games
                      << " TotalSamples=" << cum_rows << "\n";
        }

        SelfplayEngine<Gomoku> engine(game_init, cfg, pcfg, cli.model, server_devices);
        engine.start();

        std::mt19937 rng(std::random_device{}());

        // Unified selfplay.tsv writer. Both the main loop (one row per iter,
        // producer="main") and the daemon (one row per model version,
        // producer="daemon") append through here, so consumers
        // (bucket.py, view_loss.py) have a single source of truth.
        auto write_selfplay_row = [&](
            const std::string& producer, int iter_or_version,
            int games, int64_t rows, double seconds,
            int min_len, int max_len, double sum_len,
            int black, int white, int draw,
            int main_games, int main_min, int main_max, double main_sum_len,
            int main_black, int main_white, int main_draw,
            long long start_unix, long long end_unix)
        {
            const fs::path tsv = fs::path(cli.log_dir) / "selfplay.tsv";
            const bool had_header = fs::exists(tsv);
            std::ofstream of(tsv, std::ios::app);
            if (!of) return;
            if (!had_header) {
                of << "producer\titer_or_version\tgames\trows\tseconds"
                      "\tmin_len\tmax_len\tavg_len\tbwr\twwr\tdwr"
                      "\tmain_games\tmain_min_len\tmain_max_len\tmain_avg_len"
                      "\tmain_bwr\tmain_wwr\tmain_dwr"
                      "\tstart_unix\tend_unix\n";
            }
            auto rate = [](int n, int d) { return d > 0 ? static_cast<double>(n) / d : 0.0; };
            auto avg = [](double s, int n) { return n > 0 ? s / n : 0.0; };
            of << producer << "\t" << iter_or_version
               << "\t" << games << "\t" << rows
               << "\t" << std::fixed << std::setprecision(2) << seconds
               << "\t" << (games > 0 ? min_len : 0)
               << "\t" << (games > 0 ? max_len : 0)
               << "\t" << std::fixed << std::setprecision(3) << avg(sum_len, games)
               << "\t" << std::fixed << std::setprecision(4) << rate(black, games)
               << "\t" << std::fixed << std::setprecision(4) << rate(white, games)
               << "\t" << std::fixed << std::setprecision(4) << rate(draw, games)
               << "\t" << main_games
               << "\t" << (main_games > 0 ? main_min : 0)
               << "\t" << (main_games > 0 ? main_max : 0)
               << "\t" << std::fixed << std::setprecision(3) << avg(main_sum_len, main_games)
               << "\t" << std::fixed << std::setprecision(4) << rate(main_black, main_games)
               << "\t" << std::fixed << std::setprecision(4) << rate(main_white, main_games)
               << "\t" << std::fixed << std::setprecision(4) << rate(main_draw, main_games)
               << "\t" << start_unix << "\t" << end_unix << "\n";
        };

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

            // Per-version stats (reset after each reload). Two parallel sets:
            //   v_*       — full set (all sizes/rules)
            //   v_main_*  — main pair only (MAIN_BOARD_SIZE × MAIN_RULE)
            auto version_t0 = std::chrono::steady_clock::now();
            int v_games = 0;
            int64_t v_rows = 0;
            int v_black = 0, v_white = 0, v_draw = 0;
            int v_min_len = std::numeric_limits<int>::max();
            int v_max_len = 0;
            double v_sum_len = 0.0;
            int v_main_games = 0;
            int v_main_black = 0, v_main_white = 0, v_main_draw = 0;
            int v_main_min_len = std::numeric_limits<int>::max();
            int v_main_max_len = 0;
            double v_main_sum_len = 0.0;

            auto append_stats_row = [&](int version) {
                const auto t1 = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(t1 - version_t0).count();
                const auto end_unix = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                const auto start_unix = end_unix - static_cast<long long>(dt);
                write_selfplay_row(
                    "daemon", version,
                    v_games, v_rows, dt,
                    v_min_len, v_max_len, v_sum_len,
                    v_black, v_white, v_draw,
                    v_main_games, v_main_min_len, v_main_max_len, v_main_sum_len,
                    v_main_black, v_main_white, v_main_draw,
                    start_unix, end_unix);
            };
            auto reset_version_counters = [&]() {
                version_t0 = std::chrono::steady_clock::now();
                v_games = 0; v_rows = 0; v_black = 0; v_white = 0; v_draw = 0;
                v_min_len = std::numeric_limits<int>::max(); v_max_len = 0;
                v_sum_len = 0.0;
                v_main_games = 0; v_main_black = 0; v_main_white = 0; v_main_draw = 0;
                v_main_min_len = std::numeric_limits<int>::max(); v_main_max_len = 0;
                v_main_sum_len = 0.0;
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
                                // Re-poll warmup. Engine's cfg_ is a ref to
                                // this cfg, so workers pick up the new value
                                // on their next selfplay_once iteration.
                                if (!cli.sims_warmup_cmd.empty()) {
                                    const int v = popen_int(cli.sims_warmup_cmd);
                                    if (v > 0 && v != cfg.num_simulations) {
                                        std::cout << "[Daemon] num_simulations "
                                                  << cfg.num_simulations
                                                  << " -> " << v << " (warmup)\n";
                                        cfg.num_simulations = v;
                                    }
                                }
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
                    // state and global_features are already populated by
                    // selfplay_once (per-game game_ encodes V5 + computes
                    // rule one-hot + ply). Just write the row.
                    writer.append(ts);
                    v_rows += 1;
                    total_rows += 1;
                }
                v_sum_len += r.game_len;
                v_min_len = std::min(v_min_len, r.game_len);
                v_max_len = std::max(v_max_len, r.game_len);
                v_games += 1;
                total_games += 1;
                if (r.winner == 1) ++v_black;
                else if (r.winner == -1) ++v_white;
                else ++v_draw;
                if (r.board_size == main_board_size && r.rule == main_rule) {
                    v_main_sum_len += r.game_len;
                    v_main_min_len = std::min(v_main_min_len, r.game_len);
                    v_main_max_len = std::max(v_main_max_len, r.game_len);
                    v_main_games += 1;
                    if (r.winner == 1) ++v_main_black;
                    else if (r.winner == -1) ++v_main_white;
                    else ++v_main_draw;
                }

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

        // --- Main collection loop (one-shot, --max-games) ---
        // Two parallel stat sets, both emitted as one selfplay.tsv row at end:
        //   full set (all games)        — fills the unprefixed cols (games/avg_len/...).
        //   main pair (MAIN_* filtered) — fills the main_* cols. Also drives
        //                                 the per-100-games progress print below.
        int games_done = 0;
        int64_t total_rows = 0;
        int black_wins = 0, white_wins = 0, draws = 0;
        int min_len = std::numeric_limits<int>::max();
        int max_len = 0;
        double sum_len = 0.0, sum_sq_len = 0.0;
        SelfplayEngine<Gomoku>::SelfplayResult min_game, max_game;

        int main_games = 0;
        int main_black = 0, main_white = 0, main_draw = 0;
        int main_min_len = std::numeric_limits<int>::max();
        int main_max_len = 0;
        double main_sum_len = 0.0, main_sum_sq_len = 0.0;
        SelfplayEngine<Gomoku>::SelfplayResult main_min_game, main_max_game;

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
                // state + global_features are already populated by selfplay_once
                // (per-game game_ encodes V5 canvas-padded + rule one-hot + ply).
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

            const bool is_main = (r.board_size == main_board_size && r.rule == main_rule);
            if (is_main) {
                main_sum_len += L;
                main_sum_sq_len += static_cast<double>(L) * L;
                if (L < main_min_len) { main_min_len = L; main_min_game = r; }
                if (L > main_max_len) { main_max_len = L; main_max_game = r; }
                main_games += 1;
                if (r.winner == 1) ++main_black;
                else if (r.winner == -1) ++main_white;
                else ++main_draw;
            }

            if (games_done - last_report >= 100 || games_done == cli.max_games) {
                last_report = games_done;
                const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - t0).count();
                const double sps = (dt > 0.0) ? (static_cast<double>(total_rows) / dt) : 0.0;
                const int gw = static_cast<int>(std::to_string(cli.max_games).size());

                std::cout << "[SelfPlay] Games=" << std::setw(gw) << std::setfill('0') << games_done
                          << std::setfill(' ') << "/" << cli.max_games
                          << " Sps=" << std::fixed << std::setprecision(1) << sps
                          << " main(" << main_board_size << "×" << rule_to_string(main_rule)
                          << ", n=" << main_games << ")";

                if (main_games > 0) {
                    const double m_avg = main_sum_len / main_games;
                    const double m_var = std::max(0.0, main_sum_sq_len / main_games - m_avg * m_avg);
                    const double m_std = std::sqrt(m_var);
                    const double mb = static_cast<double>(main_black) / main_games;
                    const double mw = static_cast<double>(main_white) / main_games;
                    const double md = static_cast<double>(main_draw) / main_games;
                    const int mbp = static_cast<int>(std::round(mb * 100.0));
                    const int mwp = static_cast<int>(std::round(mw * 100.0));
                    const int mdp = static_cast<int>(std::round(md * 100.0));
                    std::cout << " GameLen:Avg=" << std::fixed << std::setprecision(1) << m_avg
                              << " Min=" << main_min_len
                              << " Max=" << main_max_len
                              << " Std=" << static_cast<int>(std::round(m_std))
                              << " BWD=" << std::setw(2) << std::setfill('0') << mbp
                              << "/" << std::setw(2) << std::setfill('0') << mwp
                              << "/" << std::setw(2) << std::setfill('0') << mdp
                              << std::setfill(' ');
                } else {
                    std::cout << " GameLen=N/A BWD=N/A";
                }
                std::cout << "\n";
            }
        }

        engine.stop();
        writer.flush();

        const auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t0).count();
        const auto end_unix = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto start_unix = end_unix - static_cast<long long>(dt);
        write_selfplay_row(
            "main", cli.iter,
            games_done, total_rows, dt,
            min_len, max_len, sum_len,
            black_wins, white_wins, draws,
            main_games, main_min_len, main_max_len, main_sum_len,
            main_black, main_white, main_draw,
            start_unix, end_unix);

        std::cout << "[SelfPlay] done. games=" << games_done
                  << " rows=" << total_rows
                  << " t=" << std::fixed << std::setprecision(1) << dt << "s\n";

        auto print_board = [&](const char* tag, const SelfplayEngine<Gomoku>::SelfplayResult& r) {
            const char* opening = r.balanced_opening ? "balanced" : "empty";
            std::cout << "[SelfPlay] " << tag
                      << " opening=" << opening
                      << " game_len=" << r.game_len
                      << " winner=" << r.winner
                      << " size=" << r.board_size
                      << " rule=" << rule_to_string(r.rule) << "\n";
            const int N = r.board_size;
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
