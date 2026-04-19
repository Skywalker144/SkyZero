#ifndef SKYZERO_NPZ_WRITER_H
#define SKYZERO_NPZ_WRITER_H

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "alphazero.h"
#include "numpywrite.h"

namespace skyzero {

// Accumulates TrainSamples and writes them as NPZ files for the Python pipeline.
// Thread-safe: multiple selfplay workers can call add_game concurrently.
template <typename Game>
class NpzDataWriter {
public:
    NpzDataWriter(
        Game& game,
        const std::string& output_dir,
        int max_rows_per_file
    )
        : game_(game),
          output_dir_(output_dir),
          max_rows_per_file_(max_rows_per_file),
          board_size_(game.board_size),
          num_planes_(game.num_planes),
          board_area_(game.board_size * game.board_size),
          num_rows_(0),
          file_counter_(0)
    {
        namespace fs = std::filesystem;
        fs::create_directories(output_dir);
    }

    void add_game(const std::vector<TrainSample>& samples) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& s : samples) {
            // Encode the state for NN input
            auto encoded = game_.encode_state(s.state, s.to_play);

            encoded_buf_.insert(encoded_buf_.end(), encoded.begin(), encoded.end());
            policy_buf_.insert(policy_buf_.end(), s.policy_target.begin(), s.policy_target.end());
            opp_policy_buf_.insert(opp_policy_buf_.end(), s.opponent_policy_target.begin(), s.opponent_policy_target.end());
            value_buf_.push_back(s.value_target[0]);
            value_buf_.push_back(s.value_target[1]);
            value_buf_.push_back(s.value_target[2]);
            weight_buf_.push_back(s.sample_weight);
            opp_policy_weight_buf_.push_back(s.opp_policy_weight);
            num_rows_++;

            if (num_rows_ >= max_rows_per_file_) {
                write_npz_file();
            }
        }
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (num_rows_ > 0) {
            write_npz_file();
        }
    }

    int total_rows_written() const { return total_rows_written_; }

private:
    void write_npz_file() {
        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::string filename = output_dir_ + "/data_" + std::to_string(ms) + "_" + std::to_string(file_counter_++) + ".npz";

        const int N = num_rows_;
        const int C = num_planes_;
        const int H = board_size_;
        const int W = board_size_;

        // Create NumpyBuffers and copy data
        NumpyBuffer<int8_t> encodedBuf({(int64_t)N, (int64_t)C, (int64_t)H, (int64_t)W}, "|i1");
        std::memcpy(encodedBuf.data, encoded_buf_.data(), encoded_buf_.size() * sizeof(int8_t));

        NumpyBuffer<float> policyBuf({(int64_t)N, (int64_t)board_area_}, "<f4");
        std::memcpy(policyBuf.data, policy_buf_.data(), policy_buf_.size() * sizeof(float));

        NumpyBuffer<float> oppPolicyBuf({(int64_t)N, (int64_t)board_area_}, "<f4");
        std::memcpy(oppPolicyBuf.data, opp_policy_buf_.data(), opp_policy_buf_.size() * sizeof(float));

        NumpyBuffer<float> valueBuf({(int64_t)N, 3}, "<f4");
        std::memcpy(valueBuf.data, value_buf_.data(), value_buf_.size() * sizeof(float));

        NumpyBuffer<float> weightBuf({(int64_t)N}, "<f4");
        std::memcpy(weightBuf.data, weight_buf_.data(), weight_buf_.size() * sizeof(float));

        NumpyBuffer<float> oppPolicyWeightBuf({(int64_t)N}, "<f4");
        std::memcpy(oppPolicyWeightBuf.data, opp_policy_weight_buf_.data(), opp_policy_weight_buf_.size() * sizeof(float));

        uint64_t encodedBytes = encodedBuf.prepareHeaderWithNumRows(N);
        uint64_t policyBytes = policyBuf.prepareHeaderWithNumRows(N);
        uint64_t oppPolicyBytes = oppPolicyBuf.prepareHeaderWithNumRows(N);
        uint64_t valueBytes = valueBuf.prepareHeaderWithNumRows(N);
        uint64_t weightBytes = weightBuf.prepareHeaderWithNumRows(N);
        uint64_t oppPolicyWeightBytes = oppPolicyWeightBuf.prepareHeaderWithNumRows(N);

        ZipFile zip(filename);
        zip.writeBuffer("encodedInputNCHW.npy", encodedBuf.dataIncludingHeader, encodedBytes);
        zip.writeBuffer("policyTargetsN.npy", policyBuf.dataIncludingHeader, policyBytes);
        zip.writeBuffer("opponentPolicyTargetsN.npy", oppPolicyBuf.dataIncludingHeader, oppPolicyBytes);
        zip.writeBuffer("valueTargetsN.npy", valueBuf.dataIncludingHeader, valueBytes);
        zip.writeBuffer("sampleWeightsN.npy", weightBuf.dataIncludingHeader, weightBytes);
        zip.writeBuffer("oppPolicyWeightsN.npy", oppPolicyWeightBuf.dataIncludingHeader, oppPolicyWeightBytes);
        zip.close();

        std::cout << "Wrote " << N << " rows to " << filename << std::endl;

        total_rows_written_ += N;

        // Clear buffers
        encoded_buf_.clear();
        policy_buf_.clear();
        opp_policy_buf_.clear();
        value_buf_.clear();
        weight_buf_.clear();
        opp_policy_weight_buf_.clear();
        num_rows_ = 0;
    }

    Game& game_;
    std::string output_dir_;
    int max_rows_per_file_;
    int board_size_;
    int num_planes_;
    int board_area_;
    int num_rows_;
    int file_counter_;
    int total_rows_written_ = 0;
    std::mutex mutex_;

    std::vector<int8_t> encoded_buf_;
    std::vector<float> policy_buf_;
    std::vector<float> opp_policy_buf_;
    std::vector<float> value_buf_;
    std::vector<float> weight_buf_;
    std::vector<float> opp_policy_weight_buf_;
};

}  // namespace skyzero

#endif  // SKYZERO_NPZ_WRITER_H
