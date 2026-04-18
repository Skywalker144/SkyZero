#ifndef SKYZERO_REPLAYBUFFER_H
#define SKYZERO_REPLAYBUFFER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace skyzero {

struct TrainSample {
    std::vector<int8_t> state;
    int8_t to_play = 1;
    std::vector<float> policy_target;
    std::vector<float> opponent_policy_target;
    std::array<float, 3> value_target{0.0f, 0.0f, 0.0f};
    float sample_weight = 1.0f;
    uint8_t is_full_search = 1;
};

struct ReplayBufferState {
    int board_size = 0;
    int action_size = 0;
    int min_buffer_size = 0;
    int linear_threshold = 0;
    float alpha = 0.75f;
    int max_buffer_size = 0;
    int ptr = 0;
    int size = 0;
    int total_samples_added = 0;
    int games_count = 0;

    std::vector<int8_t> states;
    std::vector<int8_t> to_play;
    std::vector<float> policy_targets;
    std::vector<float> opponent_policy_targets;
    std::vector<float> value_targets;
    std::vector<float> sample_weights;
    std::vector<uint8_t> is_full_search;
};

class ReplayBuffer {
public:
    ReplayBuffer(
        int board_size,
        int min_buffer_size = 10000,
        int linear_threshold = 10000,
        float alpha = 0.75f,
        int max_buffer_size = 3000000
    )
        : board_size_(board_size),
          action_size_(board_size * board_size),
          min_buffer_size_(min_buffer_size),
          linear_threshold_(linear_threshold),
          alpha_(alpha),
          max_buffer_size_(max_buffer_size),
          ptr_(0),
          size_(0),
          total_samples_added_(0),
          games_count_(0),
          data_(max_buffer_size) {}

    int size() const { return size_; }
    int min_buffer_size() const { return min_buffer_size_; }
    int games_count() const { return games_count_; }

    void override_params(int min_buf, int linear_thresh, float alpha, int max_buf) {
        min_buffer_size_ = min_buf;
        linear_threshold_ = linear_thresh;
        alpha_ = alpha;
        // Only shrink max_buffer_size (cannot grow pre-allocated storage)
        if (max_buf <= max_buffer_size_) {
            max_buffer_size_ = max_buf;
        }
    }

    int get_window_size() const {
        if (total_samples_added_ < linear_threshold_) {
            return total_samples_added_;
        }
        const double ratio = static_cast<double>(total_samples_added_) / static_cast<double>(linear_threshold_);
        const double window = static_cast<double>(linear_threshold_) * std::pow(ratio, alpha_);
        return std::min(static_cast<int>(window), max_buffer_size_);
    }

    int add_game(const std::vector<TrainSample>& game_memory) {
        if (game_memory.empty()) {
            return 0;
        }
        for (const auto& s : game_memory) {
            validate_sample_shape(s);
            data_[ptr_] = s;
            ptr_ = (ptr_ + 1) % max_buffer_size_;
            size_ = std::min(size_ + 1, max_buffer_size_);
            total_samples_added_ += 1;
        }
        games_count_ += 1;
        return static_cast<int>(game_memory.size());
    }

    std::vector<TrainSample> sample(int batch_size, std::mt19937& rng) const {
        if (size_ < batch_size || batch_size <= 0) {
            return {};
        }

        const int window_size = std::min(size_, get_window_size());
        if (window_size <= 0) {
            return {};
        }
        const int start_index = (ptr_ - window_size + max_buffer_size_) % max_buffer_size_;

        std::vector<TrainSample> batch;
        batch.reserve(batch_size);

        if (start_index < ptr_) {
            std::uniform_int_distribution<int> dist(start_index, ptr_ - 1);
            for (int i = 0; i < batch_size; ++i) {
                batch.push_back(data_[dist(rng)]);
            }
        } else {
            const int seg0 = max_buffer_size_ - start_index;
            const int seg1 = ptr_;
            const int total = seg0 + seg1;
            std::uniform_int_distribution<int> dist(0, total - 1);
            for (int i = 0; i < batch_size; ++i) {
                const int pick = dist(rng);
                const int idx = (pick < seg0) ? (start_index + pick) : (pick - seg0);
                batch.push_back(data_[idx]);
            }
        }
        return batch;
    }

    void clear() {
        ptr_ = 0;
        size_ = 0;
        total_samples_added_ = 0;
        games_count_ = 0;
        data_.assign(max_buffer_size_, TrainSample{});
    }

    ReplayBufferState get_state() const {
        ReplayBufferState st;
        st.board_size = board_size_;
        st.action_size = action_size_;
        st.min_buffer_size = min_buffer_size_;
        st.linear_threshold = linear_threshold_;
        st.alpha = alpha_;
        st.max_buffer_size = max_buffer_size_;
        st.ptr = ptr_;
        st.size = size_;
        st.total_samples_added = total_samples_added_;
        st.games_count = games_count_;

        const size_t n = static_cast<size_t>(size_);
        const size_t board_cells = static_cast<size_t>(board_size_) * static_cast<size_t>(board_size_);
        const size_t action_cells = static_cast<size_t>(action_size_);

        st.states.reserve(n * board_cells);
        st.to_play.reserve(n);
        st.policy_targets.reserve(n * action_cells);
        st.opponent_policy_targets.reserve(n * action_cells);
        st.value_targets.reserve(n * 3);
        st.sample_weights.reserve(n);
        st.is_full_search.reserve(n);

        const int oldest = (ptr_ - size_ + max_buffer_size_) % max_buffer_size_;
        for (int i = 0; i < size_; ++i) {
            const int idx = (oldest + i) % max_buffer_size_;
            const auto& s = data_[idx];
            st.states.insert(st.states.end(), s.state.begin(), s.state.end());
            st.to_play.push_back(s.to_play);
            st.policy_targets.insert(st.policy_targets.end(), s.policy_target.begin(), s.policy_target.end());
            st.opponent_policy_targets.insert(st.opponent_policy_targets.end(), s.opponent_policy_target.begin(), s.opponent_policy_target.end());
            st.value_targets.insert(st.value_targets.end(), s.value_target.begin(), s.value_target.end());
            st.sample_weights.push_back(s.sample_weight);
            st.is_full_search.push_back(s.is_full_search);
        }
        return st;
    }

    void load_state(const ReplayBufferState& st) {
        if (st.board_size <= 0 || st.action_size <= 0 || st.max_buffer_size <= 0 || st.size < 0) {
            throw std::runtime_error("ReplayBuffer: invalid state metadata");
        }

        board_size_ = st.board_size;
        action_size_ = st.action_size;
        min_buffer_size_ = st.min_buffer_size;
        linear_threshold_ = st.linear_threshold;
        alpha_ = st.alpha;
        max_buffer_size_ = st.max_buffer_size;

        const size_t n = static_cast<size_t>(st.size);
        const size_t board_cells = static_cast<size_t>(board_size_) * static_cast<size_t>(board_size_);
        const size_t action_cells = static_cast<size_t>(action_size_);

        if (st.states.size() != n * board_cells ||
            st.to_play.size() != n ||
            st.policy_targets.size() != n * action_cells ||
            st.opponent_policy_targets.size() != n * action_cells ||
            st.value_targets.size() != n * 3 ||
            st.sample_weights.size() != n ||
            st.is_full_search.size() != n) {
            throw std::runtime_error("ReplayBuffer: invalid state tensor sizes");
        }

        data_.assign(max_buffer_size_, TrainSample{});
        const int kept_size = std::min(st.size, max_buffer_size_);
        const int src_offset = st.size - kept_size;

        for (int i = 0; i < kept_size; ++i) {
            const int src_i = src_offset + i;
            TrainSample s;
            s.state.assign(
                st.states.begin() + static_cast<size_t>(src_i) * board_cells,
                st.states.begin() + static_cast<size_t>(src_i + 1) * board_cells
            );
            s.to_play = st.to_play[src_i];
            s.policy_target.assign(
                st.policy_targets.begin() + static_cast<size_t>(src_i) * action_cells,
                st.policy_targets.begin() + static_cast<size_t>(src_i + 1) * action_cells
            );
            s.opponent_policy_target.assign(
                st.opponent_policy_targets.begin() + static_cast<size_t>(src_i) * action_cells,
                st.opponent_policy_targets.begin() + static_cast<size_t>(src_i + 1) * action_cells
            );
            s.value_target = {
                st.value_targets[static_cast<size_t>(src_i) * 3 + 0],
                st.value_targets[static_cast<size_t>(src_i) * 3 + 1],
                st.value_targets[static_cast<size_t>(src_i) * 3 + 2],
            };
            s.sample_weight = st.sample_weights[src_i];
            s.is_full_search = st.is_full_search[src_i];
            data_[i] = std::move(s);
        }

        size_ = kept_size;
        ptr_ = kept_size % max_buffer_size_;
        total_samples_added_ = std::max(kept_size, st.total_samples_added);
        games_count_ = st.games_count;
    }

    // ── Streaming binary I/O (avoids full-buffer copy) ──────────────

    static constexpr uint32_t kRBFileMagic   = 0x534B5242;  // "SKRB"
    static constexpr uint32_t kRBFileVersion  = 1;
    static constexpr int      kSaveChunkSize  = 50000;

    // Header layout (fixed 48 bytes)
    struct alignas(4) RBFileHeader {
        uint32_t magic;
        uint32_t version;
        int32_t  board_size;
        int32_t  action_size;
        int32_t  min_buffer_size;
        int32_t  linear_threshold;
        float    alpha;
        int32_t  max_buffer_size;
        int32_t  ptr;
        int32_t  size;
        int32_t  total_samples_added;
        int32_t  games_count;
    };
    static_assert(sizeof(RBFileHeader) == 48, "RBFileHeader must be 48 bytes");

    bool save_to_file(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("ReplayBuffer::save_to_file: cannot open " + path);
        }

        // ── Write header ──
        RBFileHeader hdr{};
        hdr.magic              = kRBFileMagic;
        hdr.version            = kRBFileVersion;
        hdr.board_size         = board_size_;
        hdr.action_size        = action_size_;
        hdr.min_buffer_size    = min_buffer_size_;
        hdr.linear_threshold   = linear_threshold_;
        hdr.alpha              = alpha_;
        hdr.max_buffer_size    = max_buffer_size_;
        hdr.ptr                = ptr_;
        hdr.size               = size_;
        hdr.total_samples_added = total_samples_added_;
        hdr.games_count        = games_count_;
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        if (size_ == 0) {
            return out.good();
        }

        const int board_cells  = board_size_ * board_size_;
        const int action_cells = action_size_;
        const int oldest = (ptr_ - size_ + max_buffer_size_) % max_buffer_size_;

        // Helper: iterate the circular buffer in chunks and write one field
        // per sample via a caller-provided lambda that copies bytes into buf.
        auto write_field_chunked = [&](int bytes_per_sample, auto copy_fn) {
            const int chunk = kSaveChunkSize;
            std::vector<char> buf(static_cast<size_t>(chunk) * bytes_per_sample);
            for (int start = 0; start < size_; start += chunk) {
                const int count = std::min(chunk, size_ - start);
                for (int i = 0; i < count; ++i) {
                    const int idx = (oldest + start + i) % max_buffer_size_;
                    copy_fn(buf.data() + static_cast<size_t>(i) * bytes_per_sample, data_[idx]);
                }
                out.write(buf.data(), static_cast<std::streamsize>(count) * bytes_per_sample);
            }
        };

        // 1) states  (int8 × board_cells per sample)
        write_field_chunked(board_cells, [&](char* dst, const TrainSample& s) {
            std::memcpy(dst, s.state.data(), board_cells);
        });

        // 2) to_play (int8 × 1)
        write_field_chunked(1, [](char* dst, const TrainSample& s) {
            *reinterpret_cast<int8_t*>(dst) = s.to_play;
        });

        // 3) policy_targets (float × action_cells)
        const int policy_bytes = action_cells * static_cast<int>(sizeof(float));
        write_field_chunked(policy_bytes, [&](char* dst, const TrainSample& s) {
            std::memcpy(dst, s.policy_target.data(), policy_bytes);
        });

        // 4) opponent_policy_targets (float × action_cells)
        write_field_chunked(policy_bytes, [&](char* dst, const TrainSample& s) {
            std::memcpy(dst, s.opponent_policy_target.data(), policy_bytes);
        });

        // 5) value_targets (float × 3)
        constexpr int value_bytes = 3 * static_cast<int>(sizeof(float));
        write_field_chunked(value_bytes, [](char* dst, const TrainSample& s) {
            std::memcpy(dst, s.value_target.data(), value_bytes);
        });

        // 6) sample_weights (float × 1)
        write_field_chunked(static_cast<int>(sizeof(float)), [](char* dst, const TrainSample& s) {
            std::memcpy(dst, &s.sample_weight, sizeof(float));
        });

        // 7) is_full_search (uint8 × 1)
        write_field_chunked(1, [](char* dst, const TrainSample& s) {
            *reinterpret_cast<uint8_t*>(dst) = s.is_full_search;
        });

        if (!out.good()) {
            throw std::runtime_error("ReplayBuffer::save_to_file: write error on " + path);
        }
        return true;
    }

    bool load_from_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("ReplayBuffer::load_from_file: cannot open " + path);
        }

        // ── Read & validate header ──
        RBFileHeader hdr{};
        in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!in || hdr.magic != kRBFileMagic) {
            throw std::runtime_error("ReplayBuffer::load_from_file: invalid magic");
        }
        if (hdr.version != kRBFileVersion) {
            throw std::runtime_error("ReplayBuffer::load_from_file: unsupported version");
        }
        if (hdr.board_size <= 0 || hdr.action_size <= 0 || hdr.max_buffer_size <= 0 || hdr.size < 0) {
            throw std::runtime_error("ReplayBuffer::load_from_file: invalid header values");
        }

        board_size_          = hdr.board_size;
        action_size_         = hdr.action_size;
        min_buffer_size_     = hdr.min_buffer_size;
        linear_threshold_    = hdr.linear_threshold;
        alpha_               = hdr.alpha;
        max_buffer_size_     = hdr.max_buffer_size;
        total_samples_added_ = hdr.total_samples_added;
        games_count_         = hdr.games_count;

        const int file_size  = hdr.size;
        const int kept_size  = std::min(file_size, max_buffer_size_);
        const int skip_count = file_size - kept_size;

        const int board_cells  = board_size_ * board_size_;
        const int action_cells = action_size_;

        data_.assign(max_buffer_size_, TrainSample{});

        // Pre-allocate vectors in each kept sample
        for (int i = 0; i < kept_size; ++i) {
            data_[i].state.resize(board_cells);
            data_[i].policy_target.resize(action_cells);
            data_[i].opponent_policy_target.resize(action_cells);
        }

        // Helper: read one field for all samples, skipping the first skip_count
        auto read_field_chunked = [&](int bytes_per_sample, auto load_fn) {
            const int chunk = kSaveChunkSize;
            std::vector<char> buf(static_cast<size_t>(chunk) * bytes_per_sample);

            // Skip samples that won't fit
            int remaining_skip = skip_count;
            while (remaining_skip > 0) {
                const int n = std::min(chunk, remaining_skip);
                in.read(buf.data(), static_cast<std::streamsize>(n) * bytes_per_sample);
                remaining_skip -= n;
            }

            // Read kept samples
            for (int start = 0; start < kept_size; start += chunk) {
                const int count = std::min(chunk, kept_size - start);
                in.read(buf.data(), static_cast<std::streamsize>(count) * bytes_per_sample);
                for (int i = 0; i < count; ++i) {
                    load_fn(buf.data() + static_cast<size_t>(i) * bytes_per_sample, data_[start + i]);
                }
            }
        };

        // 1) states
        read_field_chunked(board_cells, [&](const char* src, TrainSample& s) {
            std::memcpy(s.state.data(), src, board_cells);
        });

        // 2) to_play
        read_field_chunked(1, [](const char* src, TrainSample& s) {
            s.to_play = *reinterpret_cast<const int8_t*>(src);
        });

        // 3) policy_targets
        const int policy_bytes = action_cells * static_cast<int>(sizeof(float));
        read_field_chunked(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.policy_target.data(), src, policy_bytes);
        });

        // 4) opponent_policy_targets
        read_field_chunked(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.opponent_policy_target.data(), src, policy_bytes);
        });

        // 5) value_targets
        constexpr int value_bytes = 3 * static_cast<int>(sizeof(float));
        read_field_chunked(value_bytes, [](const char* src, TrainSample& s) {
            std::memcpy(s.value_target.data(), src, value_bytes);
        });

        // 6) sample_weights
        read_field_chunked(static_cast<int>(sizeof(float)), [](const char* src, TrainSample& s) {
            std::memcpy(&s.sample_weight, src, sizeof(float));
        });

        // 7) is_full_search
        read_field_chunked(1, [](const char* src, TrainSample& s) {
            s.is_full_search = *reinterpret_cast<const uint8_t*>(src);
        });

        size_ = kept_size;
        ptr_  = kept_size % max_buffer_size_;
        total_samples_added_ = std::max(kept_size, hdr.total_samples_added);
        games_count_         = hdr.games_count;

        if (!in) {
            throw std::runtime_error("ReplayBuffer::load_from_file: read error on " + path);
        }
        return true;
    }

private:
    void validate_sample_shape(const TrainSample& s) const {
        const size_t board_cells = static_cast<size_t>(board_size_) * static_cast<size_t>(board_size_);
        if (s.state.size() != board_cells) {
            throw std::runtime_error("ReplayBuffer: sample.state has wrong size");
        }
        if (s.policy_target.size() != static_cast<size_t>(action_size_)) {
            throw std::runtime_error("ReplayBuffer: sample.policy_target has wrong size");
        }
        if (s.opponent_policy_target.size() != static_cast<size_t>(action_size_)) {
            throw std::runtime_error("ReplayBuffer: sample.opponent_policy_target has wrong size");
        }
    }

    int board_size_;
    int action_size_;

    int min_buffer_size_;
    int linear_threshold_;
    float alpha_;
    int max_buffer_size_;

    int ptr_;
    int size_;
    int total_samples_added_;
    int games_count_;

    std::vector<TrainSample> data_;
};

}  // namespace skyzero

#endif
