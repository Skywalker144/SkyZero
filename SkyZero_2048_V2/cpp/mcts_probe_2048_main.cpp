// Empty-board-style MCTS root-value diagnostic for 2048 (V7.1 mcts_probe
// analogue). Loads a TorchScript model, runs Gumbel afterstate MCTS on a FIXED
// canonical board, and appends a row to probe.tsv so value drift is trackable
// across iters. 2048's value head is SCALAR (not WDL), so we log root_value
// (raw points) and the Gumbel-selected action, plus raw NN values on a few
// fixed boards for a search-free sanity trend.
//
//   ./mcts_probe_2048 --model data/models/latest.pt --config configs/baseline/run.cfg \
//       --iter N --log data/logs/probe.tsv
#include <array>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "envs/game2048.h"
#include "infer_server_2048.h"
#include "skyzero_2048.h"

using namespace skyzero;

// Minimal run.cfg (bash KEY=VALUE) reader: returns "" if key absent.
static std::string cfg_get(const std::string& path, const std::string& key) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        size_t s = line.find_first_not_of(" \t");
        if (s == std::string::npos || line[s] == '#') continue;
        size_t eq = line.find('=', s);
        if (eq == std::string::npos) continue;
        std::string k = line.substr(s, eq - s);
        while (!k.empty() && std::isspace((unsigned char)k.back())) k.pop_back();
        if (k != key) continue;
        std::string v = line.substr(eq + 1);
        if (auto h = v.find('#'); h != std::string::npos) v = v.substr(0, h);  // strip comment
        size_t b = v.find_first_not_of(" \t\"");
        size_t e = v.find_last_not_of(" \t\"\r\n");
        return (b == std::string::npos) ? "" : v.substr(b, e - b + 1);
    }
    return "";
}

int main(int argc, char** argv) {
    std::string model_path, config_path, log_path;
    long iter = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--config") config_path = next();
        else if (a == "--iter") iter = std::stol(next());
        else if (a == "--log") log_path = next();
    }
    if (model_path.empty()) { std::fprintf(stderr, "[probe] --model required\n"); return 1; }

    float value_scale = 30.0f;     // h-SPACE scale (MuZero h() value, hardcoded on)
    float gamma = 0.999f;
    int sims = 400;
    std::string device_str = "cuda";
    if (!config_path.empty()) {
        if (auto s = cfg_get(config_path, "VALUE_SCALE"); !s.empty()) value_scale = std::stof(s);
        if (auto s = cfg_get(config_path, "GAMMA"); !s.empty()) gamma = std::stof(s);
        if (auto s = cfg_get(config_path, "PROBE_NUM_SIMULATIONS"); !s.empty()) sims = std::stoi(s);
        else if (auto s2 = cfg_get(config_path, "SIMS"); !s2.empty()) sims = std::stoi(s2);
        if (auto s = cfg_get(config_path, "DEVICE"); !s.empty()) device_str = s;
    }

    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    InferenceServer2048 server(model_path, device, value_scale, /*max_batch*/8, /*wait_us*/0, 1);
    auto infer = [&server](const std::vector<int8_t>& enc) { return server.submit(enc).get(); };

    Game2048 game;
    // Fixed canonical boards (deterministic -> comparable across iters).
    const std::vector<int8_t> probe_board = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<std::vector<int8_t>> fixed = {
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 0},
        {5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    SkyZero2048Config cfg;
    cfg.num_simulations = sims;
    cfg.gamma = gamma;
    cfg.gumbel_noise = false;             // deterministic probe
    SkyZero2048MCTS mcts(game, cfg, infer, /*seed*/12345);
    auto out = mcts.search(probe_board);

    std::array<float, 3> raw{};
    for (int i = 0; i < 3; ++i) raw[i] = infer(game.encode_state(fixed[i])).second;

    std::printf("[probe] iter=%ld sims=%d root_value=%.1f best_action=%d "
                "raw=[%.1f %.1f %.1f]\n",
                iter, sims, out.root_value, out.best_action, raw[0], raw[1], raw[2]);

    if (!log_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(log_path).parent_path());
        bool fresh = !std::filesystem::exists(log_path);
        std::ofstream f(log_path, std::ios::app);
        if (fresh) f << "iter\ttimestamp\troot_value\tbest_action\traw0\traw1\traw2\n";
        f << iter << '\t' << (long)std::time(nullptr) << '\t' << out.root_value << '\t'
          << out.best_action << '\t' << raw[0] << '\t' << raw[1] << '\t' << raw[2] << '\n';
    }
    return 0;
}
