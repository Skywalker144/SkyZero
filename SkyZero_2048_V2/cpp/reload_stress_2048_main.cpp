// Stress test for InferenceServer2048::reload() concurrency (Phase C gate).
//
// libtorch isn't TSan-instrumented (and spawns its own threadpools), so a clean
// -fsanitize=thread run is impractical. Instead we hammer the reader/writer race
// directly: K threads call submit().get() in a tight loop while the main thread
// repeatedly reload()s the (same) model. Asserts: no crash/deadlock and every
// result stays finite. Exercises the exact window guarded by model_mu_.
//
//   ./reload_stress_2048 --model M [--threads 8 --reloads 80 --device cpu]
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "envs/game2048.h"
#include "infer_server_2048.h"

using namespace skyzero;

int main(int argc, char** argv) {
    std::string model_path, device_str = "cpu";
    int threads = 8, reloads = 80;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--reloads") reloads = std::stoi(next());
        else if (a == "--device") device_str = next();
    }
    if (model_path.empty()) { std::fprintf(stderr, "[stress] --model required\n"); return 2; }

    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    InferenceServer2048 server(model_path, device, 4000.0f, /*max_batch*/16, /*wait_us*/0, /*srv*/2);

    Game2048 game;
    const std::vector<int8_t> board = {1, 2, 1, 0, 0, 3, 0, 2, 1, 0, 4, 0, 0, 1, 0, 1};
    const auto enc = game.encode_state(board);

    std::atomic<bool> stop{false};
    std::atomic<long> ok{0}, bad{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            while (!stop.load()) {
                auto r = server.submit(enc).get();
                bool finite = std::isfinite(r.second);
                for (float x : r.first) finite = finite && std::isfinite(x);
                (finite ? ok : bad).fetch_add(1);
            }
        });
    }

    for (int n = 0; n < reloads; ++n) {
        server.reload(model_path);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    stop.store(true);
    for (auto& th : pool) th.join();

    std::printf("[stress] reloads=%d threads=%d ok=%ld bad=%ld\n", reloads, threads, ok.load(), bad.load());
    if (bad.load() != 0) { std::printf("[stress] FAIL: non-finite results\n"); return 1; }
    std::printf("[stress] PASS\n");
    return 0;
}
