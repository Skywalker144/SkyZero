// Parallel C++ self-play for 2048: N worker threads run single-game Gumbel
// afterstate MCTS, all feeding one central batched inference server (GPU). This
// is the throughput path — the GPU sees big batches and the cores stay busy.
//
//   ./selfplay2048_par --model data2048/model_ts.pt --games 400 --sims 64 \
//       --threads 48 --batch 256 --out data2048/selfplay --noise 1
//
// Optional --out writes npz shards (state (N,16) int8 raw exponents,
// policy_target (N,4) f32, value_target (N,1) f32) — the schema az2048/train.py
// consumes directly.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>     // getpid (distinct daemon prefixes across processes)

#include <zip.h>

#include "envs/game2048.h"
#include "infer_server_2048.h"
#include "npy.h"          // make_npy_buffer
#include "skyzero_2048.h"

using namespace skyzero;

// Set by SIGINT in daemon mode -> workers stop starting new games and drain.
static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop.store(true); }

// --- V7.1-schema log appends (replaces the Phase-B selfplay_log.py shim) ----
// logs/selfplay.tsv : producer<TAB>iter_or_version<TAB>games<TAB>rows<TAB>seconds
//   (bucket/schedule read col4=rows; run.sh cumulative awk reads col3=games)
static void append_selfplay_tsv(const std::string& log_dir, const char* producer,
                                long iter_or_ver, long games, long rows, double secs) {
    if (log_dir.empty()) return;
    std::filesystem::create_directories(log_dir);
    const std::string path = log_dir + "/selfplay.tsv";
    bool fresh = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (fresh) f << "producer\titer\tgames\trows\tseconds\n";
    f << producer << '\t' << iter_or_ver << '\t' << games << '\t' << rows
      << '\t' << (long)(secs + 0.5) << '\n';
}

// logs/selfplay_stats.tsv : iter ts games rows sp_seconds avg_score best_score e1..e17
static void append_selfplay_stats(const std::string& log_dir, long iter, long games,
                                  long rows, double secs, long avg, long best,
                                  const std::vector<int>& tile_hist) {
    if (log_dir.empty()) return;
    const int MAX_EXP = 17;
    std::filesystem::create_directories(log_dir);
    const std::string path = log_dir + "/selfplay_stats.tsv";
    bool fresh = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (fresh) {
        f << "iter\ttimestamp\tgames\tnew_rows\tsp_seconds\tavg_score\tbest_score";
        for (int e = 1; e <= MAX_EXP; ++e) f << "\te" << e;
        f << '\n';
    }
    f << iter << '\t' << (long)std::time(nullptr) << '\t' << games << '\t' << rows
      << '\t' << (long)(secs + 0.5) << '\t' << avg << '\t' << best;
    for (int e = 1; e <= MAX_EXP; ++e)
        f << '\t' << (e < (int)tile_hist.size() ? tile_hist[e] : 0);
    f << '\n';
}

// Read "iter" from models/latest.meta.json (daemon version tag). -1 if absent.
static long read_meta_iter(const std::string& meta_path) {
    std::ifstream f(meta_path);
    if (!f) return -1;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto p = s.find("\"iter\"");
    if (p == std::string::npos) return -1;
    p = s.find(':', p);
    if (p == std::string::npos) return -1;
    return std::strtol(s.c_str() + p + 1, nullptr, 10);
}

// NB: file_clock's epoch is in the FUTURE on libstdc++, so a valid count is
// NEGATIVE — never use "< 0" as the error sentinel. Errors return MTIME_ERR.
static constexpr long long MTIME_ERR = std::numeric_limits<long long>::min();
static long long file_mtime_ns(const std::string& path) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) return MTIME_ERR;
    return (long long)t.time_since_epoch().count();
}

// Run sims-warmup-cmd, parse trailing int (NUM_SIMULATIONS). <=0 -> keep current.
static int poll_sims(const std::string& cmd) {
    if (cmd.empty()) return 0;
    FILE* pp = popen(cmd.c_str(), "r");
    if (!pp) return 0;
    char buf[256]; std::string out;
    while (fgets(buf, sizeof(buf), pp)) out += buf;
    pclose(pp);
    // last whitespace-delimited token that parses as int
    int val = 0;
    size_t i = out.size();
    while (i > 0) {
        size_t e = i; while (e > 0 && std::isspace((unsigned char)out[e-1])) --e;
        size_t b = e; while (b > 0 && !std::isspace((unsigned char)out[b-1])) --b;
        if (e > b) { val = std::atoi(out.substr(b, e - b).c_str()); if (val > 0) break; }
        i = b;
    }
    return val;
}

// ---- minimal thread-safe npz writer for the 2048 schema ----
class Npz2048Writer {
public:
    Npz2048Writer(std::string dir, int max_rows, std::string prefix)
        : dir_(std::move(dir)), max_rows_(max_rows), prefix_(std::move(prefix)) {
        std::filesystem::create_directories(dir_);
    }
    void append(const std::vector<int8_t>& state16, const std::array<float, 4>& policy, float value) {
        std::lock_guard<std::mutex> lk(m_);
        states_.insert(states_.end(), state16.begin(), state16.end());
        for (float p : policy) policies_.push_back(p);
        values_.push_back(value);
        weights_.push_back(1.0f);   // uniform sample weight (V7.1 schema slot)
        if (++rows_ >= max_rows_) flush_locked();
    }
    void flush() { std::lock_guard<std::mutex> lk(m_); flush_locked(); }
    // Daemon: flush the current version's buffer and start a new prefix/part run.
    void rotate(const std::string& new_prefix) {
        std::lock_guard<std::mutex> lk(m_);
        flush_locked();
        prefix_ = new_prefix;
        part_ = 0;
    }
    int64_t total() const { return total_; }

private:
    void flush_locked() {
        if (rows_ == 0) return;
        const std::string path = dir_ + "/" + prefix_ + "_part_" + std::to_string(part_++) + ".npz";
        const std::string tmp = path + ".tmp";
        int err = 0;
        zip_t* z = zip_open(tmp.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err);
        add(z, "state.npy", "|i1", {rows_, 16}, states_.data(), states_.size());
        add(z, "policy_target.npy", "<f4", {rows_, 4}, policies_.data(), policies_.size() * 4);
        add(z, "value_target.npy", "<f4", {rows_, 1}, values_.data(), values_.size() * 4);
        add(z, "sample_weight.npy", "<f4", {rows_}, weights_.data(), weights_.size() * 4);
        zip_close(z);
        std::filesystem::rename(tmp, path);
        total_ += rows_;
        states_.clear(); policies_.clear(); values_.clear(); weights_.clear(); rows_ = 0;
    }
    void add(zip_t* z, const char* name, const char* descr,
             std::vector<int64_t> shape, const void* data, size_t nbytes) {
        auto buf = make_npy_buffer(descr, shape, data, nbytes);
        uint8_t* raw = static_cast<uint8_t*>(std::malloc(buf.size()));
        std::memcpy(raw, buf.data(), buf.size());
        zip_source_t* src = zip_source_buffer(z, raw, buf.size(), 1);
        zip_int64_t idx = zip_file_add(z, name, src, ZIP_FL_OVERWRITE);
        zip_set_file_compression(z, idx, ZIP_CM_STORE, 0);
    }

    std::string dir_;
    int max_rows_;
    std::string prefix_;
    std::mutex m_;
    std::vector<int8_t> states_;
    std::vector<float> policies_;
    std::vector<float> values_;
    std::vector<float> weights_;
    int rows_ = 0, part_ = 0;
    int64_t total_ = 0;
};

int main(int argc, char** argv) {
    std::string model_path = "data2048/model_ts.pt", out_dir, prefix = "cpp";
    int num_games = 400, sims = 64, threads = 48, batch = 256, wait_us = 300, max_rows = 50000;
    int games_per_worker = 64, server_threads = 1;
    int td_steps = 0;              // 0 = full MC return; >0 = n-step TD bootstrap
    int max_moves = 0;             // >0: cap moves/game (eval; long deterministic games else never end)
    float value_scale = 4000.0f;
    bool value_transform = false;  // wrap value in MuZero h() (see infer_server)
    float gamma = 0.999f;          // discount on future reward (was hardcoded)
    bool noise = true;
    uint64_t seed = 1;
    std::string device_str = "cuda";
    std::string log_dir, sims_warmup_cmd;
    long iter_tag = -1;            // selfplay.tsv/stats iter column (bounded mode)
    std::string eval_log, eval_network;  // bounded-eval: write a python/evaluate.py-style eval.tsv row
    bool daemon = false;
    int model_watch_poll_ms = 2000;
    int progress_secs = 15;        // periodic [selfplay] games/s line (0 = off)
    int stats_games = 1000;        // daemon: emit a selfplay_stats row every N completed games

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--games") num_games = std::stoi(next());
        else if (a == "--sims") sims = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--slot-games") games_per_worker = std::stoi(next());
        else if (a == "--server-threads") server_threads = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--wait-us") wait_us = std::stoi(next());
        else if (a == "--value-scale") value_scale = std::stof(next());
        else if (a == "--value-transform") value_transform = (std::stoi(next()) != 0);
        else if (a == "--gamma") gamma = std::stof(next());
        else if (a == "--td-steps") td_steps = std::stoi(next());
        else if (a == "--out" || a == "--output-dir") out_dir = next();
        else if (a == "--prefix") prefix = next();
        else if (a == "--noise") noise = std::stoi(next()) != 0;
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device_str = next();
        else if (a == "--log-dir") log_dir = next();
        else if (a == "--iter") iter_tag = std::stol(next());
        else if (a == "--daemon") daemon = true;
        else if (a == "--model-watch-poll-ms") model_watch_poll_ms = std::stoi(next());
        else if (a == "--sims-warmup-cmd") sims_warmup_cmd = next();
        else if (a == "--progress-secs") progress_secs = std::stoi(next());
        else if (a == "--stats-games") stats_games = std::stoi(next());
        else if (a == "--eval-log") eval_log = next();
        else if (a == "--eval-network") eval_network = next();
        else if (a == "--max-moves") max_moves = std::stoi(next());
    }

    // models/latest.meta.json sits beside the model (daemon version tag).
    std::string meta_path = model_path;
    if (auto d = meta_path.rfind(".pt"); d != std::string::npos)
        meta_path = meta_path.substr(0, d) + ".meta.json";

    long daemon_version = 0;
    if (daemon) {
        std::signal(SIGINT, on_sigint);
        std::signal(SIGTERM, on_sigint);
        if (!sims_warmup_cmd.empty()) { int s = poll_sims(sims_warmup_cmd); if (s > 0) sims = s; }
        long v = read_meta_iter(meta_path);
        daemon_version = (v < 0) ? 0 : v;
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "daemon_v%06ld_p%d", daemon_version, (int)getpid());
        prefix = pfx;
    }

    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    InferenceServer2048 server(model_path, device, value_scale, batch, wait_us, server_threads,
                               value_transform);
    auto infer = [&server](const std::vector<int8_t>& enc) {
        return server.submit(enc).get();
    };

    {
        bool probe = false;
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--probe") probe = true;
        if (probe) {
            Game2048 gp;
            std::vector<std::vector<int8_t>> boards = {
                {1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0},
                {1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,0},
                {5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0},
            };
            for (auto& b : boards) {
                auto [lg, v] = infer(gp.encode_state(b));
                std::printf("server: logits=[%.4f %.4f %.4f %.4f] value=%.1f\n", lg[0], lg[1], lg[2], lg[3], v);
            }
            return 0;
        }
    }

    std::unique_ptr<Npz2048Writer> writer;
    if (!out_dir.empty()) writer = std::make_unique<Npz2048Writer>(out_dir, max_rows, prefix);

    SkyZero2048Config cfg;
    cfg.num_simulations = sims;
    cfg.gumbel_noise = noise;
    cfg.td_steps = td_steps;
    cfg.gamma = gamma;

    Game2048 game;
    std::atomic<int> next_game{0};
    std::atomic<long> sum_score{0}, sum_moves{0}, games_done{0};
    std::atomic<long> moves_done{0};   // incremented per move PLAYED (smooth samples/s)
    std::mutex agg_m;
    std::vector<int> tile_hist(20, 0);
    std::vector<long> eval_scores;   // per-game scores for the eval.tsv median (eval mode only)
    long best_score = 0;
    long ver_best = 0;   // best score within the current daemon version (reset each reload)
    const int G = std::max(1, games_per_worker);

    auto t0 = std::chrono::steady_clock::now();

    // Each worker runs G games in lockstep, batching the NN evals of all its
    // live slots per simulation tick. Slots refill from the global counter so
    // the batch stays ~G regardless of differing game lengths.
    auto worker = [&](int tid) {
        struct Slot {
            std::unique_ptr<SkyZero2048MCTS> mcts;
            std::vector<int8_t> state;
            long score = 0;
            std::vector<std::vector<int8_t>> ts;
            std::vector<std::array<float, 4>> tp;
            std::vector<int> tr;
            std::vector<float> tv;   // per-step MCTS search value (TD bootstrap)
            int moves = 0;           // moves played this game (for --max-moves cap)
            bool active = false;
        };
        std::mt19937 rng(seed + 7919 * tid + 1);
        std::vector<Slot> slots(G);
        for (auto& s : slots) s.mcts = std::make_unique<SkyZero2048MCTS>(game, cfg, infer, rng());

        auto finalize = [&](Slot& s) {
            if (writer && !s.tr.empty()) {
                auto vals = skyzero::compute_value_targets(s.tr, s.tv, cfg.gamma, cfg.td_steps);
                for (size_t t = 0; t < s.ts.size(); ++t) writer->append(s.ts[t], s.tp[t], vals[t]);
            }
            int me = game.max_tile_exp(s.state);
            sum_score.fetch_add(s.score);
            sum_moves.fetch_add(static_cast<long>(s.tr.size()));
            games_done.fetch_add(1);
            std::lock_guard<std::mutex> lk(agg_m);
            if (s.score > best_score) best_score = s.score;
            if (s.score > ver_best) ver_best = s.score;
            if (me < static_cast<int>(tile_hist.size())) tile_hist[me]++;
            if (!eval_log.empty()) eval_scores.push_back(s.score);
        };

        auto eval_batch = [&](std::vector<std::vector<int8_t>>& encs) {
            std::vector<std::future<InferenceServer2048::Result>> futs;
            futs.reserve(encs.size());
            for (auto& e : encs) futs.push_back(server.submit(std::move(e)));
            std::vector<InferenceServer2048::Result> res(futs.size());
            for (size_t i = 0; i < futs.size(); ++i) res[i] = futs[i].get();
            return res;
        };

        while (true) {
            // Daemon shutdown: abandon in-progress games immediately rather
            // than playing them to terminal (2048 games are long).
            if (daemon && g_stop.load()) break;
            // Refill empty slots with fresh games.
            bool any = false;
            for (auto& s : slots) {
                if (!s.active) {
                    bool start;
                    if (daemon) {
                        start = !g_stop.load();           // run until SIGINT
                    } else {
                        int g = next_game.fetch_add(1);
                        start = (g < num_games);
                    }
                    if (start) {
                        s.state = game.get_initial_state(rng);
                        s.score = 0; s.moves = 0; s.ts.clear(); s.tp.clear(); s.tr.clear(); s.tv.clear();
                        s.active = true;
                    }
                }
                any |= s.active;
            }
            if (!any) break;

            // One move for every active slot: fresh search, batched over slots.
            for (auto& s : slots) if (s.active) s.mcts->begin(s.state);

            // Root evals.
            {
                std::vector<std::vector<int8_t>> encs;
                std::vector<int> idx;
                for (int i = 0; i < G; ++i)
                    if (slots[i].active && !slots[i].mcts->root_terminal()) {
                        encs.push_back(game.encode_state(slots[i].mcts->root_state()));
                        idx.push_back(i);
                    }
                if (!encs.empty()) {
                    auto res = eval_batch(encs);
                    for (size_t k = 0; k < idx.size(); ++k)
                        slots[idx[k]].mcts->apply_root_eval(res[k].first, res[k].second);
                }
            }

            // Simulations, batched across slots until every slot's search is done.
            while (true) {
                std::vector<std::vector<int8_t>> encs;
                std::vector<int> idx;
                bool active_any = false;
                for (int i = 0; i < G; ++i) {
                    if (!slots[i].active || slots[i].mcts->done()) continue;
                    active_any = true;
                    auto e = slots[i].mcts->select_leaf();
                    if (!e.empty()) { encs.push_back(std::move(e)); idx.push_back(i); }
                }
                if (!active_any) break;
                if (!encs.empty()) {
                    auto res = eval_batch(encs);
                    for (size_t k = 0; k < idx.size(); ++k)
                        slots[idx[k]].mcts->apply_leaf(res[k].first, res[k].second);
                }
            }

            // Play the chosen move in each slot; finalize finished games.
            for (auto& s : slots) {
                if (!s.active) continue;
                auto out = s.mcts->result();
                if (out.best_action < 0) { finalize(s); s.active = false; continue; }
                if (writer) { s.ts.push_back(s.state); s.tp.push_back(out.improved_policy);
                              s.tv.push_back(out.root_value); }
                moves_done.fetch_add(1, std::memory_order_relaxed);  // one move/sample produced
                auto mr = game.apply_move(s.state, out.best_action);
                if (writer) s.tr.push_back(mr.reward);
                s.score += mr.reward;
                s.state = game.spawn_random(mr.afterstate, rng);
                s.moves++;
                if (game.is_terminal(s.state) || (max_moves > 0 && s.moves >= max_moves)) {
                    finalize(s); s.active = false;
                }
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);

    // Periodic progress (V7.1-style): games/s + evals/s while self-play runs.
    std::atomic<bool> mon_stop{false};
    std::thread monitor;
    if (progress_secs > 0) {
        monitor = std::thread([&] {
            long last_g = 0, last_s = 0; int64_t last_e = 0;
            auto t_prev = std::chrono::steady_clock::now();
            while (!mon_stop.load()) {
                for (int k = 0; k < progress_secs * 10 && !mon_stop.load(); ++k)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (mon_stop.load()) break;
                auto now = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(now - t_prev).count();
                double total = std::chrono::duration<double>(now - t0).count();
                long g = games_done.load(), s = moves_done.load();
                int64_t e = server.total_evals();
                double gps = dt > 0 ? (g - last_g) / dt : 0.0;
                double sps = dt > 0 ? (s - last_s) / dt : 0.0;     // moves PLAYED/s (smooth)
                double eps = dt > 0 ? (e - last_e) / dt : 0.0;
                double avg = total > 0 ? g / total : 0.0;
                double avglen = g > 0 ? (double)sum_moves.load() / g : 0.0;  // avg over finished games
                if (daemon)
                    std::printf("[selfplay] %ld games  %.1f games/s (avg %.1f)  %.0f samples/s  avglen %.0f  %.0f evals/s\n",
                                g, gps, avg, sps, avglen, eps);
                else
                    std::printf("[selfplay] %ld/%d games (%.0f%%)  %.1f games/s (avg %.1f)  %.0f samples/s  avglen %.0f  %.0f evals/s\n",
                                g, num_games, num_games > 0 ? 100.0 * g / num_games : 0.0, gps, avg, sps, avglen, eps);
                std::fflush(stdout);
                last_g = g; last_s = s; last_e = e; t_prev = now;
            }
        });
    }
    auto stop_monitor = [&] { mon_stop.store(true); if (monitor.joinable()) monitor.join(); };

    if (daemon) {
        // Poll the model mtime; on change, close out the version (flush + tsv
        // row), hot-reload the served weights, rotate to the new prefix.
        std::printf("[daemon] v%06ld started: %d threads sims=%d poll=%dms model=%s\n",
                    daemon_version, threads, sims, model_watch_poll_ms, model_path.c_str());
        std::fflush(stdout);
        long long last_mtime = file_mtime_ns(model_path);
        int64_t rows_base = 0; long games_base = 0;        // selfplay.tsv (per-version cadence)
        auto ver_t0 = std::chrono::steady_clock::now();

        // selfplay_stats.tsv (avg/best score + tile reach) for view_loss's selfplay
        // plot. Emitted every ~stats_games COMPLETED games — DECOUPLED from model
        // reloads — so each row averages a run.sh-comparable sample. A per-reload
        // window is often only tens of games and 2048 scores are heavy-tailed (mean
        // is carried by rare 8192+ games), so small windows read as wild avg swings
        // / phantom regressions. Tagged with the live daemon_version as the iter col.
        long stats_games_base = 0, stats_score_base = 0;
        int64_t stats_rows_base = 0;
        std::vector<int> stats_tile_base(tile_hist.size(), 0);
        auto stats_t0 = ver_t0;
        auto emit_stats = [&]() {
            long games_now = games_done.load();
            long sgames = games_now - stats_games_base;
            if (sgames <= 0) return;
            long score_now = sum_score.load();
            int64_t rows_now = writer ? writer->total() : 0;
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - stats_t0).count();
            long avg = (score_now - stats_score_base) / sgames;
            std::vector<int> tiles(tile_hist.size());
            long sbest;
            {
                std::lock_guard<std::mutex> lk(agg_m);
                for (size_t i = 0; i < tile_hist.size(); ++i) tiles[i] = tile_hist[i] - stats_tile_base[i];
                stats_tile_base = tile_hist;
                sbest = ver_best; ver_best = 0;
            }
            append_selfplay_stats(log_dir, daemon_version, sgames, rows_now - stats_rows_base,
                                  secs, avg, sbest, tiles);
            stats_games_base = games_now; stats_score_base = score_now;
            stats_rows_base = rows_now; stats_t0 = std::chrono::steady_clock::now();
        };
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(model_watch_poll_ms));
            if (g_stop.load()) break;
            // Periodic stats every ~stats_games games (independent of reloads below).
            if (games_done.load() - stats_games_base >= stats_games) emit_stats();
            long long m = file_mtime_ns(model_path);
            if (m == last_mtime || m == MTIME_ERR) continue;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));  // settle (atomic export)
            long long m2 = file_mtime_ns(model_path);
            if (m2 != m) { last_mtime = m2; continue; }                  // still changing

            long new_ver = read_meta_iter(meta_path);
            if (new_ver < 0) new_ver = daemon_version + 1;
            char pfx[64];
            std::snprintf(pfx, sizeof(pfx), "daemon_v%06ld_p%d", new_ver, (int)getpid());

            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ver_t0).count();
            if (writer) writer->rotate(pfx);                             // flush old version's rows
            int64_t rows_now = writer ? writer->total() : 0;
            long games_now = games_done.load();
            append_selfplay_tsv(log_dir, "daemon", daemon_version,
                                games_now - games_base, rows_now - rows_base, secs);

            server.reload(model_path);
            if (!sims_warmup_cmd.empty()) { int s = poll_sims(sims_warmup_cmd); if (s > 0) cfg.num_simulations = s; }
            std::printf("[daemon] reload v%06ld -> v%06ld (sims=%d, +%ld games +%lld rows in %.0fs)\n",
                        daemon_version, new_ver, cfg.num_simulations,
                        games_now - games_base, (long long)(rows_now - rows_base), secs);
            std::fflush(stdout);

            daemon_version = new_ver; last_mtime = m;
            rows_base = rows_now; games_base = games_now;
            ver_t0 = std::chrono::steady_clock::now();
        }
        for (auto& th : pool) th.join();
        stop_monitor();
        if (writer) writer->flush();
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ver_t0).count();
        int64_t rows_now = writer ? writer->total() : 0;
        append_selfplay_tsv(log_dir, "daemon", daemon_version,
                            games_done.load() - games_base, rows_now - rows_base, secs);
        emit_stats();   // flush the trailing partial-window stats row
        std::printf("[daemon] stopped at v%06ld (total %ld games, %lld rows)\n",
                    daemon_version, games_done.load(), (long long)rows_now);
        return 0;
    }

    for (auto& th : pool) th.join();
    stop_monitor();
    if (writer) writer->flush();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::printf("\n=== %d games, %d threads, sims=%d, batch<=%d ===\n", num_games, threads, sims, batch);
    std::printf("avg_score=%ld best=%ld avg_len=%.0f  (%.1fs, %.1f games/s)\n",
                sum_score.load() / std::max(1, num_games), best_score,
                (double)sum_moves.load() / std::max(1, num_games), secs, num_games / secs);
    std::printf("server forwards=%ld evals=%ld  avg_batch=%.1f  evals/s=%.0f\n",
                server.total_forwards(), server.total_evals(),
                (double)server.total_evals() / std::max<int64_t>(1, server.total_forwards()),
                server.total_evals() / secs);
    std::printf("tiles:");
    for (int e = 1; e < (int)tile_hist.size(); ++e) if (tile_hist[e]) std::printf(" %d:%d", 1 << e, tile_hist[e]);
    std::printf("\n");
    if (writer) std::printf("wrote %ld rows to %s\n", writer->total(), out_dir.c_str());

    // V7.1-schema log rows (replaces the Phase-B selfplay_log.py shim).
    long games_total = games_done.load();
    append_selfplay_tsv(log_dir, "main", iter_tag, games_total, writer ? writer->total() : 0, secs);
    append_selfplay_stats(log_dir, iter_tag, games_total, writer ? writer->total() : 0, secs,
                          sum_score.load() / std::max(1, num_games), best_score, tile_hist);

    // Bounded-eval (--eval-log): one python/evaluate.py-schema row to eval.tsv —
    // avg/median/max score, avg max-tile, tile reach-rates. view_loss.py reads it
    // unchanged. Only fires in eval mode; self-play/daemon leaves eval_log empty.
    if (!eval_log.empty()) {
        std::sort(eval_scores.begin(), eval_scores.end());
        const int ng = std::max(1, num_games);
        const size_t n = eval_scores.size();
        const double avg = (double)sum_score.load() / ng;
        const double median = n == 0 ? 0.0
            : (n % 2 ? (double)eval_scores[n / 2]
                     : 0.5 * (eval_scores[n / 2 - 1] + eval_scores[n / 2]));
        double sum_tile = 0.0;
        for (int e = 0; e < (int)tile_hist.size(); ++e) sum_tile += (double)tile_hist[e] * (1L << e);
        const double avg_max_tile = sum_tile / ng;
        const int milestones[] = {256, 512, 1024, 2048, 4096, 8192};
        double reach[6];
        for (int k = 0; k < 6; ++k) {
            long c = 0;
            for (int e = 0; e < (int)tile_hist.size(); ++e)
                if ((1L << e) >= milestones[k]) c += tile_hist[e];
            reach[k] = (double)c / ng;
        }
        std::filesystem::path lp(eval_log);
        if (lp.has_parent_path()) std::filesystem::create_directories(lp.parent_path());
        bool fresh = !std::filesystem::exists(lp);
        std::ofstream f(lp, std::ios::app);
        if (fresh)
            f << "iter\ttimestamp\tnetwork\tgames\tavg_score\tmedian_score\tmax_score"
                 "\tavg_max_tile\tr256\tr512\tr1024\tr2048\tr4096\tr8192\n";
        f << iter_tag << '\t' << (long)std::time(nullptr) << '\t' << eval_network << '\t' << num_games
          << '\t' << (long)(avg + 0.5) << '\t' << (long)(median + 0.5)
          << '\t' << best_score << '\t' << (long)(avg_max_tile + 0.5);
        for (int k = 0; k < 6; ++k) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.4f", reach[k]);
            f << '\t' << buf;
        }
        f << '\n';
        std::printf("[eval] iter=%ld games=%d avg=%.0f median=%.0f best=%ld "
                    "reach2048=%.2f reach4096=%.2f -> %s\n",
                    iter_tag, num_games, avg, median, best_score, reach[3], reach[4], eval_log.c_str());
    }
    return 0;
}
