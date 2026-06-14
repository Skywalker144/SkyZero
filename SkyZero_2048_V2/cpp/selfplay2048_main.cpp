// Minimal C++ self-play driver for 2048 — bridge test.
//
// Loads a TorchScript Net2048 (exported by az2048/export.py), runs the
// single-threaded afterstate MCTS (skyzero_2048.h) over a batch of games, and
// reports scores + wall time. This proves the C++<->model path end to end; the
// parallel / batched-inference version comes next.
//
//   ./selfplay2048 --model data2048/model_ts.pt --games 50 --sims 64

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <torch/script.h>

#include "envs/game2048.h"
#include "skyzero_2048.h"

using namespace skyzero;

// Mirror of value_transform.py / infer_server_2048.h h^-1 (h-space -> raw points).
static inline float inv_value_h(float y) {
    const float e = 1e-3f;
    const float z = (std::sqrt(1.f + 4.f * e * (std::fabs(y) + 1.f + e)) - 1.f) / (2.f * e);
    return (y < 0.f ? -1.f : 1.f) * (z * z - 1.f);
}

int main(int argc, char** argv) {
    std::string model_path = "data2048/model_ts.pt";
    int num_games = 50;
    int sims = 64;
    float value_scale = 30.0f;     // h-SPACE scale (MuZero h() value, hardcoded on)
    uint64_t seed = 1;
    std::string device_str = "cpu";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--games") num_games = std::stoi(next());
        else if (a == "--sims") sims = std::stoi(next());
        else if (a == "--value-scale") value_scale = std::stof(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device_str = next();
    }

    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
        module.eval();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "failed to load %s: %s\n", model_path.c_str(), e.what());
        return 1;
    }
    torch::NoGradGuard no_grad;

    Game2048 game;
    SkyZero2048Config cfg;
    cfg.num_simulations = sims;
    cfg.gumbel_noise = false;   // deterministic Gumbel selection for eval play

    const int C = Game2048::NUM_PLANES, A = Game2048::AREA;

    // Batch-1 inference: encode state -> (policy_logits[4], value in raw points).
    auto infer = [&](const std::vector<int8_t>& state) -> std::pair<std::array<float, 4>, float> {
        auto enc = game.encode_state(state);              // int8, C*A
        auto x = torch::empty({1, C, A}, torch::kFloat32);
        float* xp = x.data_ptr<float>();
        for (int i = 0; i < C * A; ++i) xp[i] = static_cast<float>(enc[i]);
        x = x.view({1, C, Game2048::SIZE, Game2048::SIZE}).to(device);

        auto out = module.forward({x}).toTuple();
        auto pol = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
        auto val = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
        // Return raw policy LOGITS (the MCTS legal-masks + softmaxes internally;
        // Gumbel needs the logits) and value in RAW points.
        const float* pp = pol.data_ptr<float>();
        std::array<float, 4> logits{};
        for (int a = 0; a < 4; ++a) logits[a] = pp[a];
        const float scaled = val.data_ptr<float>()[0] * value_scale;
        const float value = inv_value_h(scaled);   // MuZero h^-1 -> raw points
        return {logits, value};
    };

    // Probe mode: print the net's policy/value for a couple of fixed boards and
    // exit. Used to cross-check the C++ bridge against the Python net.
    {
        bool probe = false;
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--probe") probe = true;
        if (probe) {
            std::vector<std::vector<int8_t>> boards = {
                {1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0},
                {1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,0},
                {5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0},
            };
            for (auto& b : boards) {
                auto [p, v] = infer(b);
                std::printf("probe: policy=[%.4f %.4f %.4f %.4f] value=%.1f\n",
                            p[0], p[1], p[2], p[3], v);
            }
            return 0;
        }
    }

    long total_score = 0, best_score = 0;
    int best_exp = 0;
    std::vector<int> tile_hist(20, 0);
    auto t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < num_games; ++g) {
        std::mt19937 rng(seed + g);
        SkyZero2048MCTS mcts(game, cfg, infer, seed + 100000 + g);
        auto state = game.get_initial_state(rng);
        long score = 0;
        while (!game.is_terminal(state)) {
            auto out = mcts.search(state);
            if (out.best_action < 0) break;
            auto mr = game.apply_move(state, out.best_action);
            score += mr.reward;
            state = game.spawn_random(mr.afterstate, rng);
        }
        int me = game.max_tile_exp(state);
        total_score += score;
        if (score > best_score) best_score = score;
        if (me > best_exp) best_exp = me;
        if (me < (int)tile_hist.size()) tile_hist[me]++;
        std::printf("game %3d: score=%6ld maxtile=%d\n", g, score, 1 << me);
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("\n%d games @%d sims: avg=%ld best=%ld besttile=%d  (%.1fs, %.0f ms/game)\n",
                num_games, sims, total_score / num_games, best_score, 1 << best_exp,
                secs, 1000.0 * secs / num_games);
    return 0;
}
