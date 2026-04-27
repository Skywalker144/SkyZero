#ifndef SKYZERO_NPZ_WRITER_H
#define SKYZERO_NPZ_WRITER_H

// Minimal NPZ writer: each .npz is a zip archive of .npy v1.0 files.
// Supports int8 / float32 arrays. Enough for the selfplay schema.
//
// Schema written per flush (V5):
//   state                   (N, num_planes, H, W)        int8     (V5: 5*15*15)
//   global_features         (N, num_global_features)     float32  (V5: 12-dim)
//   policy_target           (N, H*W)                     float32
//   opponent_policy_target  (N, H*W)                     float32
//   opponent_policy_mask    (N,)                         float32
//   value_target            (N, 3)                       float32
//   sample_weight           (N,)                         float32
//
// Threading: append() is called by the main selfplay collection thread.
// When a chunk fills up, its six buffers are moved into a FlushJob and
// handed off to a background writer thread (one), so the main thread
// never blocks on zip/disk I/O. The writer-job queue has a soft cap; if
// exceeded, append() blocks to provide backpressure.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <zip.h>

#include "policy_surprise_weighting.h"

namespace skyzero {

// ---------------------------------------------------------------------------
// .npy buffer builder (NumPy format 1.0, little-endian hosts only).
// ---------------------------------------------------------------------------
inline std::vector<uint8_t> make_npy_buffer(
    const std::string& descr,                 // e.g. "<i1" or "<f4"
    const std::vector<int64_t>& shape,
    const void* data,
    size_t nbytes
) {
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) {
            header << ", ";
        }
    }
    header << "), }";
    std::string header_str = header.str();

    // Pad with spaces so that (10 + header_len) is a multiple of 64, and
    // ends with '\n'. (10 = 6 magic + 2 version + 2 header-len.)
    const size_t unpadded = 10 + header_str.size() + 1;   // +1 for '\n'
    const size_t padded = ((unpadded + 63) / 64) * 64;
    header_str.append(padded - unpadded, ' ');
    header_str.push_back('\n');

    std::vector<uint8_t> buf;
    buf.reserve(10 + header_str.size() + nbytes);
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    buf.insert(buf.end(), std::begin(magic), std::end(magic));
    buf.push_back(0x01);   // major version
    buf.push_back(0x00);   // minor version
    const uint16_t hlen = static_cast<uint16_t>(header_str.size());
    buf.push_back(static_cast<uint8_t>(hlen & 0xFF));
    buf.push_back(static_cast<uint8_t>((hlen >> 8) & 0xFF));
    buf.insert(buf.end(), header_str.begin(), header_str.end());
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    buf.insert(buf.end(), bytes, bytes + nbytes);
    return buf;
}

// ---------------------------------------------------------------------------
// NpzWriter: accumulates TrainSamples in RAM, flushes a .npz per chunk.
// Thread-safe for append() / flush().
// ---------------------------------------------------------------------------
class NpzWriter {
public:
    NpzWriter(
        std::filesystem::path output_dir,
        std::string file_prefix,
        int board_size,
        int num_planes,
        int max_rows_per_file,
        int max_pending_jobs = 4,
        int num_global_features = 12,
        int state_row_override = -1   // V5: overrides num_planes*board_size² when state is padded to MAX
    )
        : output_dir_(std::move(output_dir)),
          file_prefix_(std::move(file_prefix)),
          board_size_(board_size),
          num_planes_(num_planes),
          num_global_features_(num_global_features),
          area_(board_size * board_size),
          state_row_(state_row_override > 0 ? state_row_override : num_planes * board_size * board_size),
          max_rows_(max_rows_per_file),
          max_pending_jobs_(std::max(1, max_pending_jobs)) {
        std::filesystem::create_directories(output_dir_);
        writer_thread_ = std::thread([this]() { writer_loop(); });
    }

    ~NpzWriter() {
        try {
            flush();
        } catch (...) {
            // swallow: destructor must not throw
        }
        {
            std::lock_guard<std::mutex> lk(job_mutex_);
            stop_.store(true);
        }
        job_cv_.notify_all();
        drain_cv_.notify_all();
        if (writer_thread_.joinable()) writer_thread_.join();
    }

    void append(const TrainSample& s) {
        std::unique_lock<std::mutex> lock(m_);
        if (static_cast<int>(s.state.size()) != state_row_) {
            throw std::runtime_error("NpzWriter: state size mismatch");
        }
        if (static_cast<int>(s.policy_target.size()) != area_) {
            throw std::runtime_error("NpzWriter: policy_target size mismatch");
        }
        if (static_cast<int>(s.opponent_policy_target.size()) != area_) {
            throw std::runtime_error("NpzWriter: opponent_policy_target size mismatch");
        }

        state_buf_.insert(state_buf_.end(), s.state.begin(), s.state.end());
        global_buf_.insert(global_buf_.end(), s.global_features.begin(), s.global_features.end());
        policy_buf_.insert(policy_buf_.end(), s.policy_target.begin(), s.policy_target.end());
        opp_policy_buf_.insert(opp_policy_buf_.end(), s.opponent_policy_target.begin(), s.opponent_policy_target.end());
        opp_mask_buf_.push_back(s.has_opponent_policy ? 1.0f : 0.0f);
        value_buf_.insert(value_buf_.end(), s.value_target.begin(), s.value_target.end());
        weight_buf_.push_back(s.sample_weight);
        ++rows_;
        total_rows_written_ += 1;

        if (rows_ >= max_rows_) {
            enqueue_current_chunk_locked(lock);
        }
    }

    // Flush the current in-memory chunk AND wait for all pending writer jobs
    // to finish. Used at engine shutdown.
    void flush() {
        std::unique_lock<std::mutex> lock(m_);
        enqueue_current_chunk_locked(lock);
        // Wait until writer drains.
        std::unique_lock<std::mutex> jlk(job_mutex_);
        drain_cv_.wait(jlk, [this]() { return jobs_.empty() && !job_in_flight_; });
        if (writer_error_) {
            auto e = writer_error_;
            writer_error_ = nullptr;
            std::rethrow_exception(e);
        }
    }

    int64_t total_rows_written() const {
        std::lock_guard<std::mutex> lock(m_);
        return total_rows_written_;
    }

    // Daemon mode: drain the current chunk + writer queue, then switch to a
    // new file_prefix and reset the part counter. After this returns, the
    // next append() lands in <new_prefix>_part_0000.npz.
    void rotate(std::string new_prefix) {
        flush();
        std::lock_guard<std::mutex> lock(m_);
        file_prefix_ = std::move(new_prefix);
        part_counter_ = 0;
    }

private:
    struct FlushJob {
        std::filesystem::path path;
        int64_t rows = 0;
        std::vector<int8_t> state_buf;
        std::vector<float> global_buf;
        std::vector<float> policy_buf;
        std::vector<float> opp_policy_buf;
        std::vector<float> opp_mask_buf;
        std::vector<float> value_buf;
        std::vector<float> weight_buf;
    };

    // Must be called with m_ held. `lock` is released while we block on
    // job queue capacity, then re-acquired before returning.
    void enqueue_current_chunk_locked(std::unique_lock<std::mutex>& lock) {
        if (rows_ == 0) return;

        auto job = std::make_unique<FlushJob>();
        job->path = output_dir_ / (file_prefix_ + "_part_" +
                                   format4(part_counter_++) + ".npz");
        job->rows = rows_;
        job->state_buf = std::move(state_buf_);
        job->global_buf = std::move(global_buf_);
        job->policy_buf = std::move(policy_buf_);
        job->opp_policy_buf = std::move(opp_policy_buf_);
        job->opp_mask_buf = std::move(opp_mask_buf_);
        job->value_buf = std::move(value_buf_);
        job->weight_buf = std::move(weight_buf_);

        state_buf_.clear();
        global_buf_.clear();
        policy_buf_.clear();
        opp_policy_buf_.clear();
        opp_mask_buf_.clear();
        value_buf_.clear();
        weight_buf_.clear();
        rows_ = 0;

        // Hand off to writer thread, with backpressure.
        lock.unlock();
        {
            std::unique_lock<std::mutex> jlk(job_mutex_);
            job_space_cv_.wait(jlk, [this]() {
                return stop_.load() || jobs_.size() < static_cast<size_t>(max_pending_jobs_);
            });
            if (!stop_.load()) {
                jobs_.push_back(std::move(job));
                job_cv_.notify_one();
            }
            if (writer_error_) {
                auto e = writer_error_;
                writer_error_ = nullptr;
                lock.lock();
                std::rethrow_exception(e);
            }
        }
        lock.lock();
    }

    void writer_loop() {
        while (true) {
            std::unique_ptr<FlushJob> job;
            {
                std::unique_lock<std::mutex> jlk(job_mutex_);
                job_cv_.wait(jlk, [this]() { return stop_.load() || !jobs_.empty(); });
                if (stop_.load() && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
                job_in_flight_ = true;
            }
            job_space_cv_.notify_one();

            try {
                write_job_to_disk(*job);
            } catch (...) {
                std::lock_guard<std::mutex> jlk(job_mutex_);
                writer_error_ = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> jlk(job_mutex_);
                job_in_flight_ = false;
            }
            drain_cv_.notify_all();
        }
    }

    void write_job_to_disk(const FlushJob& job) {
        const auto tmp = job.path.string() + ".tmp";
        int err = 0;
        zip_t* archive = zip_open(tmp.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err);
        if (archive == nullptr) {
            throw std::runtime_error("zip_open failed for " + tmp);
        }

        // V5: state padded to MAX_BOARD_SIZE regardless of game's board_size.
        // state_dim derived from state_row_/num_planes_ (=225 → 15 for V5).
        const int64_t state_dim = static_cast<int64_t>(std::sqrt(static_cast<double>(state_row_ / num_planes_)));
        add_int8_entry(archive, "state.npy",
                       {job.rows, num_planes_, state_dim, state_dim},
                       job.state_buf);
        add_float_entry(archive, "global_features.npy",
                        {job.rows, num_global_features_},
                        job.global_buf);
        add_float_entry(archive, "policy_target.npy",
                        {job.rows, area_},
                        job.policy_buf);
        add_float_entry(archive, "opponent_policy_target.npy",
                        {job.rows, area_},
                        job.opp_policy_buf);
        add_float_entry(archive, "opponent_policy_mask.npy",
                        {job.rows},
                        job.opp_mask_buf);
        add_float_entry(archive, "value_target.npy",
                        {job.rows, 3},
                        job.value_buf);
        add_float_entry(archive, "sample_weight.npy",
                        {job.rows},
                        job.weight_buf);

        if (zip_close(archive) < 0) {
            zip_discard(archive);
            throw std::runtime_error("zip_close failed for " + tmp);
        }
        std::filesystem::rename(tmp, job.path);
    }

    static std::string format4(int n) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%04d", n);
        return buf;
    }

    void add_int8_entry(zip_t* archive, const char* name,
                        std::vector<int64_t> shape,
                        const std::vector<int8_t>& data) {
        auto buf = make_npy_buffer("|i1", shape, data.data(), data.size());
        add_buffer(archive, name, std::move(buf));
    }

    void add_float_entry(zip_t* archive, const char* name,
                         std::vector<int64_t> shape,
                         const std::vector<float>& data) {
        auto buf = make_npy_buffer("<f4", shape, data.data(), data.size() * sizeof(float));
        add_buffer(archive, name, std::move(buf));
    }

    void add_buffer(zip_t* archive, const char* name, std::vector<uint8_t> buf) {
        // libzip takes ownership of the buffer when freep=1 is used with
        // zip_source_buffer. We allocate with malloc so free() is safe.
        uint8_t* raw = static_cast<uint8_t*>(std::malloc(buf.size()));
        if (raw == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(raw, buf.data(), buf.size());
        zip_source_t* src = zip_source_buffer(archive, raw, buf.size(), 1);
        if (src == nullptr) {
            std::free(raw);
            throw std::runtime_error("zip_source_buffer failed");
        }
        const zip_int64_t idx = zip_file_add(archive, name, src, ZIP_FL_OVERWRITE);
        if (idx < 0) {
            zip_source_free(src);
            throw std::runtime_error(std::string("zip_file_add failed for ") + name);
        }
        // Default compression is DEFLATE; use STORE for speed since data is
        // already sizeable and compressibility of float tensors is low.
        zip_set_file_compression(archive, idx, ZIP_CM_STORE, 0);
    }

    std::filesystem::path output_dir_;
    std::string file_prefix_;
    int board_size_;
    int num_planes_;
    int num_global_features_;
    int area_;
    int state_row_;
    int max_rows_;
    int max_pending_jobs_;

    mutable std::mutex m_;
    std::vector<int8_t> state_buf_;
    std::vector<float> global_buf_;
    std::vector<float> policy_buf_;
    std::vector<float> opp_policy_buf_;
    std::vector<float> opp_mask_buf_;
    std::vector<float> value_buf_;
    std::vector<float> weight_buf_;
    int64_t rows_ = 0;
    int64_t total_rows_written_ = 0;
    int part_counter_ = 0;

    // Writer thread plumbing.
    std::thread writer_thread_;
    std::mutex job_mutex_;
    std::condition_variable job_cv_;         // signalled when job arrives
    std::condition_variable job_space_cv_;   // signalled when queue shrinks
    std::condition_variable drain_cv_;       // signalled when writer idle
    std::deque<std::unique_ptr<FlushJob>> jobs_;
    bool job_in_flight_ = false;
    std::atomic<bool> stop_{false};
    std::exception_ptr writer_error_;
};

}  // namespace skyzero

#endif
