#ifndef INFER_SERVER_2048_H
#define INFER_SERVER_2048_H

// Central batched inference server for 2048 self-play.
//
// Many worker threads each run a single-game Gumbel afterstate MCTS and submit
// their leaf states here (blocking on a future). One server thread accumulates
// concurrent requests into a batch, runs a single TorchScript forward on the
// GPU, and fulfills every promise. With enough workers the batch stays full, so
// the GPU is actually utilized — the whole point of the C++ path. Workers never
// touch LibTorch; only this server thread does.

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <torch/script.h>

#include "envs/game2048.h"

namespace skyzero {

// Inverse of the MuZero value-scaling h() (mirror of python/value_transform.py;
// keep EPS and the formula in sync). Maps an h-space value back to raw points.
// Always applied to the net's value output (the head learns h-space).
inline float inv_value_h(float y, float eps = 1e-3f) {
    const float s = static_cast<float>((y > 0.f) - (y < 0.f));
    const float z = (std::sqrt(1.f + 4.f * eps * (std::fabs(y) + 1.f + eps)) - 1.f)
                    / (2.f * eps);
    return s * (z * z - 1.f);
}

class InferenceServer2048 {
public:
    using Result = std::pair<std::array<float, 4>, float>;  // (policy logits, raw value)

    InferenceServer2048(const std::string& model_path, torch::Device device,
                        float value_scale, int max_batch, int wait_us, int num_threads = 1)
        : device_(device), value_scale_(value_scale),
          max_batch_(std::max(1, max_batch)), wait_us_(wait_us) {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        for (int i = 0; i < std::max(1, num_threads); ++i)
            threads_.emplace_back([this] { loop(); });
    }

    ~InferenceServer2048() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : threads_) if (t.joinable()) t.join();
    }

    // enc must be NUM_PLANES*AREA int8 (the encoded leaf state).
    std::future<Result> submit(std::vector<int8_t> enc) {
        auto req = std::make_shared<Request>();
        req->enc = std::move(enc);
        auto fut = req->prom.get_future();
        { std::lock_guard<std::mutex> lk(m_); q_.push(std::move(req)); }
        cv_.notify_one();
        return fut;
    }

    int64_t total_forwards() const { return forwards_.load(); }
    int64_t total_evals() const { return evals_.load(); }

    // Hot-swap the served weights (daemon mode, on latest.pt update). The slow
    // jit::load runs OUTSIDE the lock; only the pointer swap is exclusive, so
    // in-flight forwards drain and the next forward picks up the new module.
    void reload(const std::string& model_path) {
        auto m = torch::jit::load(model_path, device_);
        m.eval();
        std::unique_lock<std::shared_mutex> ml(model_mu_);
        module_ = std::move(m);
    }

private:
    struct Request {
        std::vector<int8_t> enc;
        std::promise<Result> prom;
    };

    void loop() {
        torch::NoGradGuard no_grad;
        const int C = Game2048::NUM_PLANES, A = Game2048::AREA;
        while (true) {
            std::vector<std::shared_ptr<Request>> batch;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                // Brief wait to let more requests accumulate into a fuller batch.
                if (static_cast<int>(q_.size()) < max_batch_ && wait_us_ > 0) {
                    lk.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(wait_us_));
                    lk.lock();
                }
                while (!q_.empty() && static_cast<int>(batch.size()) < max_batch_) {
                    batch.push_back(std::move(q_.front()));
                    q_.pop();
                }
            }
            if (batch.empty()) continue;

            const int B = static_cast<int>(batch.size());
            auto x = torch::empty({B, C, A}, torch::kFloat32);
            float* xp = x.data_ptr<float>();
            for (int b = 0; b < B; ++b) {
                const auto& e = batch[b]->enc;
                for (int i = 0; i < C * A; ++i) xp[b * C * A + i] = static_cast<float>(e[i]);
            }
            x = x.view({B, C, Game2048::SIZE, Game2048::SIZE}).to(device_);

            c10::intrusive_ptr<c10::ivalue::Tuple> out;
            {
                // Shared lock: many forwards run concurrently; reload() waits
                // for them to drain before swapping module_.
                std::shared_lock<std::shared_mutex> ml(model_mu_);
                out = module_.forward({x}).toTuple();
            }
            auto pol = out->elements()[0].toTensor().to(torch::kCPU).contiguous();
            auto val = out->elements()[1].toTensor().to(torch::kCPU).contiguous();
            const float* pp = pol.data_ptr<float>();
            const float* vp = val.data_ptr<float>();
            for (int b = 0; b < B; ++b) {
                std::array<float, 4> lg{pp[b * 4], pp[b * 4 + 1], pp[b * 4 + 2], pp[b * 4 + 3]};
                const float scaled = vp[b] * value_scale_;
                const float raw_v = inv_value_h(scaled);   // MuZero h^-1 -> raw points
                batch[b]->prom.set_value({lg, raw_v});
            }
            forwards_.fetch_add(1);
            evals_.fetch_add(B);
        }
    }

    torch::jit::script::Module module_;
    std::shared_mutex model_mu_;            // guards module_ across reload()
    torch::Device device_;
    float value_scale_;
    int max_batch_;
    int wait_us_;

    std::mutex m_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<Request>> q_;
    bool stop_ = false;
    std::vector<std::thread> threads_;
    std::atomic<int64_t> forwards_{0};
    std::atomic<int64_t> evals_{0};
};

}  // namespace skyzero

#endif
