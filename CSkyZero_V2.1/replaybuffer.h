#ifndef SKYZERO_REPLAYBUFFER_H
#define SKYZERO_REPLAYBUFFER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
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
