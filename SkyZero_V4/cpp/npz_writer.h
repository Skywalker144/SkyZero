#ifndef SKYZERO_NPZ_WRITER_H
#define SKYZERO_NPZ_WRITER_H

// Minimal NPZ writer: each .npz is a zip archive of .npy v1.0 files.
// Supports int8 / float32 arrays. Enough for the selfplay schema.
//
// Schema written per flush:
//   state                   (N, num_planes, H, W)  int8
//   policy_target           (N, H*W)               float32
//   opponent_policy_target  (N, H*W)               float32
//   opponent_policy_mask    (N,)                   float32
//   value_target            (N, 3)                 float32
//   sample_weight           (N,)                   float32

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
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
        int max_rows_per_file
    )
        : output_dir_(std::move(output_dir)),
          file_prefix_(std::move(file_prefix)),
          board_size_(board_size),
          num_planes_(num_planes),
          area_(board_size * board_size),
          state_row_(num_planes * board_size * board_size),
          max_rows_(max_rows_per_file) {
        std::filesystem::create_directories(output_dir_);
    }

    ~NpzWriter() {
        try {
            flush();
        } catch (...) {
            // swallow: destructor must not throw
        }
    }

    void append(const TrainSample& s) {
        std::lock_guard<std::mutex> lock(m_);
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
        policy_buf_.insert(policy_buf_.end(), s.policy_target.begin(), s.policy_target.end());
        opp_policy_buf_.insert(opp_policy_buf_.end(), s.opponent_policy_target.begin(), s.opponent_policy_target.end());
        opp_mask_buf_.push_back(s.has_opponent_policy ? 1.0f : 0.0f);
        value_buf_.insert(value_buf_.end(), s.value_target.begin(), s.value_target.end());
        weight_buf_.push_back(s.sample_weight);
        ++rows_;
        total_rows_written_ += 1;

        if (rows_ >= max_rows_) {
            flush_locked();
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(m_);
        flush_locked();
    }

    int64_t total_rows_written() const {
        std::lock_guard<std::mutex> lock(m_);
        return total_rows_written_;
    }

private:
    void flush_locked() {
        if (rows_ == 0) {
            return;
        }

        const auto path = output_dir_ / (file_prefix_ + "_part_" +
                                         format4(part_counter_++) + ".npz");
        // Atomic: write to .tmp then rename
        const auto tmp = path.string() + ".tmp";

        int err = 0;
        zip_t* archive = zip_open(tmp.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err);
        if (archive == nullptr) {
            throw std::runtime_error("zip_open failed for " + tmp);
        }

        add_int8_entry(archive, "state.npy",
                       {rows_, num_planes_, board_size_, board_size_},
                       state_buf_);
        add_float_entry(archive, "policy_target.npy",
                        {rows_, area_},
                        policy_buf_);
        add_float_entry(archive, "opponent_policy_target.npy",
                        {rows_, area_},
                        opp_policy_buf_);
        add_float_entry(archive, "opponent_policy_mask.npy",
                        {rows_},
                        opp_mask_buf_);
        add_float_entry(archive, "value_target.npy",
                        {rows_, 3},
                        value_buf_);
        add_float_entry(archive, "sample_weight.npy",
                        {rows_},
                        weight_buf_);

        if (zip_close(archive) < 0) {
            zip_discard(archive);
            throw std::runtime_error("zip_close failed for " + tmp);
        }
        std::filesystem::rename(tmp, path);

        // Reset buffers
        state_buf_.clear();
        policy_buf_.clear();
        opp_policy_buf_.clear();
        opp_mask_buf_.clear();
        value_buf_.clear();
        weight_buf_.clear();
        rows_ = 0;
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
    int area_;
    int state_row_;
    int max_rows_;

    mutable std::mutex m_;
    std::vector<int8_t> state_buf_;
    std::vector<float> policy_buf_;
    std::vector<float> opp_policy_buf_;
    std::vector<float> opp_mask_buf_;
    std::vector<float> value_buf_;
    std::vector<float> weight_buf_;
    int64_t rows_ = 0;
    int64_t total_rows_written_ = 0;
    int part_counter_ = 0;
};

}  // namespace skyzero

#endif
