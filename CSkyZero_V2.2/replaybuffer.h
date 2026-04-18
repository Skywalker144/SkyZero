#ifndef SKYZERO_REPLAYBUFFER_H
#define SKYZERO_REPLAYBUFFER_H

// ============================================================================
// File-based Replay Buffer (KataGomo-style)
//
// Instead of keeping all samples in memory, samples are written to chunk files
// on disk. Training reads batches by randomly sampling from chunk files within
// a power-law growing window.
//
// Data flow:
//   selfplay worker -> SelfPlayResult -> main thread -> add_game() writes to
//   current chunk file -> when chunk is full, finalize and start a new one ->
//   sample() picks random samples from chunks within the window -> returns
//   a batch of TrainSample for training.
//
// Chunk binary format (per file):
//   [ChunkHeader: 64 bytes]
//   [field-major arrays]
//     1) states:             int8_t  [num_samples * board_size^2]
//     2) to_play:            int8_t  [num_samples]
//     3) policy_targets:     float   [num_samples * action_size]
//     4) opp_policy_targets: float   [num_samples * action_size]
//     5) value_targets:      float   [num_samples * 3]
//     6) sample_weights:     float   [num_samples]
//     7) is_full_search:     uint8_t [num_samples]
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace skyzero {

// ── TrainSample (unchanged from V2.1) ──────────────────────────────────────

struct TrainSample {
    std::vector<int8_t> state;
    int8_t to_play = 1;
    std::vector<float> policy_target;
    std::vector<float> opponent_policy_target;
    std::array<float, 3> value_target{0.0f, 0.0f, 0.0f};
    float sample_weight = 1.0f;
    uint8_t is_full_search = 1;
};

// ── Chunk file format ──────────────────────────────────────────────────────

struct ChunkHeader {
    uint32_t magic;            // "SKCH" = 0x534B4348
    uint32_t version;          // 1
    int32_t  board_size;
    int32_t  action_size;
    int32_t  num_samples;      // number of samples in this chunk
    int32_t  chunk_id;         // sequential chunk id
    int64_t  first_sample_idx; // global sample index of the first sample
    uint8_t  reserved[28];     // pad to 64 bytes
};
static_assert(sizeof(ChunkHeader) == 64, "ChunkHeader must be 64 bytes");

static constexpr uint32_t kChunkMagic   = 0x534B4348;  // "SKCH"
static constexpr uint32_t kChunkVersion = 1;

// ── Chunk metadata (in-memory, lightweight) ────────────────────────────────

struct ChunkInfo {
    std::string path;              // full path to the chunk file
    int chunk_id = 0;              // sequential chunk id
    int num_samples = 0;           // samples in this chunk
    int64_t first_sample_idx = 0;  // global index of first sample
};

// ── ReplayBuffer: file-based, KataGomo-style ───────────────────────────────

class ReplayBuffer {
public:
    ReplayBuffer(
        int board_size,
        int min_buffer_size = 10000,
        int linear_threshold = 10000,
        float alpha = 0.75f,
        int max_buffer_size = 3000000,
        int samples_per_chunk = 50000,
        const std::string& data_dir = "data"
    )
        : board_size_(board_size),
          action_size_(board_size * board_size),
          min_buffer_size_(min_buffer_size),
          linear_threshold_(linear_threshold),
          alpha_(alpha),
          max_buffer_size_(max_buffer_size),
          samples_per_chunk_(std::max(1, samples_per_chunk)),
          next_chunk_id_(0),
          total_samples_added_(0),
          games_count_(0)
    {
        namespace fs = std::filesystem;
        chunk_dir_ = (fs::path(data_dir) / "chunks").string();
        fs::create_directories(chunk_dir_);
    }

    int size() const {
        int total = 0;
        for (const auto& ci : chunks_) {
            total += ci.num_samples;
        }
        return total;
    }

    int min_buffer_size() const { return min_buffer_size_; }
    int games_count() const { return games_count_; }

    void override_params(int min_buf, int linear_thresh, float alpha, int max_buf) {
        min_buffer_size_ = min_buf;
        linear_threshold_ = linear_thresh;
        alpha_ = alpha;
        max_buffer_size_ = max_buf;
    }

    int get_window_size() const {
        if (total_samples_added_ < linear_threshold_) {
            return static_cast<int>(total_samples_added_);
        }
        const double ratio = static_cast<double>(total_samples_added_) / static_cast<double>(linear_threshold_);
        const double window = static_cast<double>(linear_threshold_) * std::pow(ratio, alpha_);
        return std::min(static_cast<int>(window), max_buffer_size_);
    }

    // ── Write path ─────────────────────────────────────────────────────────

    int add_game(const std::vector<TrainSample>& game_memory) {
        if (game_memory.empty()) return 0;

        for (const auto& s : game_memory) {
            validate_sample_shape(s);
            write_buffer_.push_back(s);
            total_samples_added_ += 1;

            if (static_cast<int>(write_buffer_.size()) >= samples_per_chunk_) {
                flush_write_buffer();
            }
        }
        games_count_ += 1;
        return static_cast<int>(game_memory.size());
    }

    /// Flush any remaining samples in the write buffer to a chunk file.
    /// Called before training and before checkpoint save.
    void flush() {
        if (!write_buffer_.empty()) {
            flush_write_buffer();
        }
    }

    // ── Read path (sampling) ───────────────────────────────────────────────

    std::vector<TrainSample> sample(int batch_size, std::mt19937& rng) {
        flush();  // ensure all pending samples are on disk

        if (chunks_.empty()) return {};

        // Compute window
        const int total_on_disk = size();
        if (total_on_disk < batch_size || batch_size <= 0) return {};

        const int window_size = std::min(total_on_disk, get_window_size());
        if (window_size <= 0) return {};

        // Find which chunks are in the window.
        // Chunks are ordered oldest-to-newest. We want the most recent
        // `window_size` samples.
        const int64_t global_end = total_samples_added_;
        const int64_t global_start = global_end - window_size;

        // Collect (chunk_index, samples_in_window) for chunks that overlap
        struct WindowChunk {
            int chunk_idx;
            int offset_in_chunk;   // first sample index within chunk that is in window
            int count_in_window;   // how many samples from this chunk are in window
        };
        std::vector<WindowChunk> window_chunks;
        int window_sample_count = 0;

        for (int i = 0; i < static_cast<int>(chunks_.size()); ++i) {
            const auto& ci = chunks_[i];
            const int64_t chunk_start = ci.first_sample_idx;
            const int64_t chunk_end = chunk_start + ci.num_samples;

            // Overlap with [global_start, global_end)?
            const int64_t overlap_start = std::max(chunk_start, global_start);
            const int64_t overlap_end = std::min(chunk_end, global_end);
            if (overlap_start >= overlap_end) continue;

            WindowChunk wc;
            wc.chunk_idx = i;
            wc.offset_in_chunk = static_cast<int>(overlap_start - chunk_start);
            wc.count_in_window = static_cast<int>(overlap_end - overlap_start);
            window_sample_count += wc.count_in_window;
            window_chunks.push_back(wc);
        }

        if (window_sample_count < batch_size) return {};

        // Randomly assign each of the batch_size draws to a window chunk,
        // weighted by count_in_window
        std::vector<int> chunk_draw_counts(window_chunks.size(), 0);
        {
            std::vector<int> cdf;
            cdf.reserve(window_chunks.size());
            int cumsum = 0;
            for (const auto& wc : window_chunks) {
                cumsum += wc.count_in_window;
                cdf.push_back(cumsum);
            }
            std::uniform_int_distribution<int> dist(0, cumsum - 1);
            for (int b = 0; b < batch_size; ++b) {
                int r = dist(rng);
                int lo = 0, hi = static_cast<int>(cdf.size()) - 1;
                while (lo < hi) {
                    int mid = (lo + hi) / 2;
                    if (cdf[mid] <= r) lo = mid + 1;
                    else hi = mid;
                }
                chunk_draw_counts[lo]++;
            }
        }

        // For each chunk that needs samples, load and randomly pick
        std::vector<TrainSample> batch;
        batch.reserve(batch_size);

        for (int wi = 0; wi < static_cast<int>(window_chunks.size()); ++wi) {
            if (chunk_draw_counts[wi] == 0) continue;

            const auto& wc = window_chunks[wi];
            const auto& ci = chunks_[wc.chunk_idx];

            auto chunk_samples = load_chunk_range(
                ci.path, ci.num_samples, wc.offset_in_chunk, wc.count_in_window
            );

            const int n_draws = chunk_draw_counts[wi];
            const int n_available = static_cast<int>(chunk_samples.size());
            std::uniform_int_distribution<int> pick_dist(0, n_available - 1);
            for (int d = 0; d < n_draws; ++d) {
                batch.push_back(chunk_samples[pick_dist(rng)]);
            }
        }

        return batch;
    }

    // ── Garbage collection: remove chunks entirely outside the window ──────

    void gc() {
        if (chunks_.empty()) return;
        const int64_t global_end = total_samples_added_;
        const int window_size = get_window_size();
        const int64_t global_start = global_end - window_size;

        namespace fs = std::filesystem;
        int freed = 0;
        auto it = chunks_.begin();
        while (it != chunks_.end()) {
            const int64_t chunk_end = it->first_sample_idx + it->num_samples;
            if (chunk_end <= global_start) {
                std::error_code ec;
                fs::remove(it->path, ec);
                it = chunks_.erase(it);
                freed++;
            } else {
                break;  // chunks are ordered, so remaining ones are in window
            }
        }
        if (freed > 0) {
            std::cout << "[ReplayBuffer GC] Removed " << freed << " old chunk files\n";
        }
    }

    void clear() {
        namespace fs = std::filesystem;
        for (const auto& ci : chunks_) {
            std::error_code ec;
            fs::remove(ci.path, ec);
        }
        chunks_.clear();
        write_buffer_.clear();
        next_chunk_id_ = 0;
        total_samples_added_ = 0;
        games_count_ = 0;
    }

    // ── Checkpoint save/load (metadata only — chunk files persist on disk) ─

    bool save_manifest(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("ReplayBuffer::save_manifest: cannot open " + path);
        }

        const uint32_t magic = 0x534B524D;  // "SKRM" (SkyZero Replay Manifest)
        const uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));

        auto write_i32 = [&](int32_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
        auto write_i64 = [&](int64_t v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
        auto write_f32 = [&](float v)   { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };

        write_i32(board_size_);
        write_i32(action_size_);
        write_i32(min_buffer_size_);
        write_i32(linear_threshold_);
        write_f32(alpha_);
        write_i32(max_buffer_size_);
        write_i32(samples_per_chunk_);
        write_i32(next_chunk_id_);
        write_i64(total_samples_added_);
        write_i32(games_count_);

        // Chunk directory path
        {
            auto len = static_cast<int32_t>(chunk_dir_.size());
            write_i32(len);
            out.write(chunk_dir_.data(), len);
        }

        // Number of chunks
        write_i32(static_cast<int32_t>(chunks_.size()));

        // Each chunk: path_len, path, chunk_id, num_samples, first_sample_idx
        for (const auto& ci : chunks_) {
            auto plen = static_cast<int32_t>(ci.path.size());
            write_i32(plen);
            out.write(ci.path.data(), plen);
            write_i32(ci.chunk_id);
            write_i32(ci.num_samples);
            write_i64(ci.first_sample_idx);
        }

        return out.good();
    }

    bool load_manifest(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("ReplayBuffer::load_manifest: cannot open " + path);
        }

        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!in || magic != 0x534B524D || version != 1) {
            throw std::runtime_error("ReplayBuffer::load_manifest: invalid file");
        }

        auto read_i32 = [&]() -> int32_t { int32_t v; in.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };
        auto read_i64 = [&]() -> int64_t { int64_t v; in.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };
        auto read_f32 = [&]() -> float   { float v;   in.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };

        board_size_ = read_i32();
        action_size_ = read_i32();
        min_buffer_size_ = read_i32();
        linear_threshold_ = read_i32();
        alpha_ = read_f32();
        max_buffer_size_ = read_i32();
        samples_per_chunk_ = read_i32();
        next_chunk_id_ = read_i32();
        total_samples_added_ = read_i64();
        games_count_ = read_i32();

        // Chunk directory path
        {
            int32_t len = read_i32();
            chunk_dir_.resize(len);
            in.read(chunk_dir_.data(), len);
        }

        // Chunks
        int32_t num_chunks = read_i32();
        chunks_.clear();
        chunks_.reserve(num_chunks);
        for (int32_t i = 0; i < num_chunks; ++i) {
            ChunkInfo ci;
            int32_t plen = read_i32();
            ci.path.resize(plen);
            in.read(ci.path.data(), plen);
            ci.chunk_id = read_i32();
            ci.num_samples = read_i32();
            ci.first_sample_idx = read_i64();
            chunks_.push_back(std::move(ci));
        }

        if (!in) {
            throw std::runtime_error("ReplayBuffer::load_manifest: read error on " + path);
        }

        // Verify that chunk files still exist; remove entries for missing files
        namespace fs = std::filesystem;
        auto it = chunks_.begin();
        while (it != chunks_.end()) {
            if (!fs::exists(it->path)) {
                std::cout << "[ReplayBuffer] Warning: chunk file missing: " << it->path << "\n";
                it = chunks_.erase(it);
            } else {
                ++it;
            }
        }

        // Ensure chunk directory exists
        fs::create_directories(chunk_dir_);

        return true;
    }

    // ── Legacy compatibility: load from old .rb file into chunk files ──────

    bool load_from_legacy_rb(const std::string& rb_path) {
        std::ifstream in(rb_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("ReplayBuffer::load_from_legacy_rb: cannot open " + rb_path);
        }

        // Read old RBFileHeader (48 bytes)
        struct alignas(4) LegacyHeader {
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
        static_assert(sizeof(LegacyHeader) == 48);

        LegacyHeader hdr{};
        in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!in || hdr.magic != 0x534B5242) {
            throw std::runtime_error("ReplayBuffer::load_from_legacy_rb: invalid magic");
        }

        board_size_ = hdr.board_size;
        action_size_ = hdr.action_size;
        total_samples_added_ = hdr.total_samples_added;
        games_count_ = hdr.games_count;

        const int file_size = hdr.size;
        if (file_size == 0) return true;

        const int board_cells = board_size_ * board_size_;
        const int action_cells = action_size_;

        // Read all samples into memory temporarily
        std::vector<TrainSample> all_samples(file_size);
        for (auto& s : all_samples) {
            s.state.resize(board_cells);
            s.policy_target.resize(action_cells);
            s.opponent_policy_target.resize(action_cells);
        }

        auto read_field = [&](int bytes_per_sample, auto load_fn) {
            const int chunk = 50000;
            std::vector<char> buf(static_cast<size_t>(chunk) * bytes_per_sample);
            for (int start = 0; start < file_size; start += chunk) {
                const int count = std::min(chunk, file_size - start);
                in.read(buf.data(), static_cast<std::streamsize>(count) * bytes_per_sample);
                for (int i = 0; i < count; ++i) {
                    load_fn(buf.data() + static_cast<size_t>(i) * bytes_per_sample, all_samples[start + i]);
                }
            }
        };

        read_field(board_cells, [&](const char* src, TrainSample& s) {
            std::memcpy(s.state.data(), src, board_cells);
        });
        read_field(1, [](const char* src, TrainSample& s) {
            s.to_play = *reinterpret_cast<const int8_t*>(src);
        });
        const int policy_bytes = action_cells * static_cast<int>(sizeof(float));
        read_field(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.policy_target.data(), src, policy_bytes);
        });
        read_field(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.opponent_policy_target.data(), src, policy_bytes);
        });
        constexpr int value_bytes = 3 * static_cast<int>(sizeof(float));
        read_field(value_bytes, [](const char* src, TrainSample& s) {
            std::memcpy(s.value_target.data(), src, value_bytes);
        });
        read_field(static_cast<int>(sizeof(float)), [](const char* src, TrainSample& s) {
            std::memcpy(&s.sample_weight, src, sizeof(float));
        });
        read_field(1, [](const char* src, TrainSample& s) {
            s.is_full_search = *reinterpret_cast<const uint8_t*>(src);
        });

        // Write these samples into chunk files
        chunks_.clear();
        write_buffer_.clear();
        next_chunk_id_ = 0;
        int64_t saved_total = total_samples_added_;
        total_samples_added_ = 0;

        for (int i = 0; i < file_size; ++i) {
            write_buffer_.push_back(std::move(all_samples[i]));
            total_samples_added_ += 1;
            if (static_cast<int>(write_buffer_.size()) >= samples_per_chunk_) {
                flush_write_buffer();
            }
        }
        if (!write_buffer_.empty()) {
            flush_write_buffer();
        }

        // Restore the true total_samples_added (it may be larger than file_size
        // due to samples that were overwritten in the old circular buffer)
        total_samples_added_ = std::max(static_cast<int64_t>(file_size), saved_total);

        std::cout << "[ReplayBuffer] Migrated " << file_size << " samples from legacy .rb into "
                  << chunks_.size() << " chunk files\n";
        return true;
    }

    // ── Legacy compatibility: load from archive-embedded state ─────────────

    struct LegacyState {
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

    void load_legacy_state(const LegacyState& st) {
        board_size_ = st.board_size;
        action_size_ = st.action_size;
        total_samples_added_ = st.total_samples_added;
        games_count_ = st.games_count;

        const size_t n = static_cast<size_t>(st.size);
        const size_t board_cells = static_cast<size_t>(board_size_) * board_size_;
        const size_t action_cells = static_cast<size_t>(action_size_);

        chunks_.clear();
        write_buffer_.clear();
        next_chunk_id_ = 0;
        int64_t saved_total = total_samples_added_;
        total_samples_added_ = 0;

        for (size_t i = 0; i < n; ++i) {
            TrainSample s;
            s.state.assign(
                st.states.begin() + i * board_cells,
                st.states.begin() + (i + 1) * board_cells
            );
            s.to_play = st.to_play[i];
            s.policy_target.assign(
                st.policy_targets.begin() + i * action_cells,
                st.policy_targets.begin() + (i + 1) * action_cells
            );
            s.opponent_policy_target.assign(
                st.opponent_policy_targets.begin() + i * action_cells,
                st.opponent_policy_targets.begin() + (i + 1) * action_cells
            );
            s.value_target = {
                st.value_targets[i * 3 + 0],
                st.value_targets[i * 3 + 1],
                st.value_targets[i * 3 + 2],
            };
            s.sample_weight = st.sample_weights[i];
            s.is_full_search = (i < st.is_full_search.size()) ? st.is_full_search[i] : 1;

            write_buffer_.push_back(std::move(s));
            total_samples_added_ += 1;
            if (static_cast<int>(write_buffer_.size()) >= samples_per_chunk_) {
                flush_write_buffer();
            }
        }
        if (!write_buffer_.empty()) {
            flush_write_buffer();
        }

        total_samples_added_ = std::max(static_cast<int64_t>(n), saved_total);
    }

    // ── Accessors for checkpoint integration ───────────────────────────────

    int64_t total_samples_added() const { return total_samples_added_; }
    const std::string& chunk_dir() const { return chunk_dir_; }

private:
    // ── Write a full chunk to disk ─────────────────────────────────────────

    void flush_write_buffer() {
        if (write_buffer_.empty()) return;

        const int n = static_cast<int>(write_buffer_.size());
        const int64_t first_idx = total_samples_added_ - n;

        // Build chunk file path
        std::ostringstream fname;
        fname << "chunk_" << std::setw(8) << std::setfill('0') << next_chunk_id_ << ".bin";
        namespace fs = std::filesystem;
        const std::string path = (fs::path(chunk_dir_) / fname.str()).string();

        write_chunk_file(path, write_buffer_, next_chunk_id_, first_idx);

        ChunkInfo ci;
        ci.path = path;
        ci.chunk_id = next_chunk_id_;
        ci.num_samples = n;
        ci.first_sample_idx = first_idx;
        chunks_.push_back(std::move(ci));

        next_chunk_id_++;
        write_buffer_.clear();

        // Run GC to remove old chunks outside the window
        gc();
    }

    void write_chunk_file(
        const std::string& path,
        const std::vector<TrainSample>& samples,
        int chunk_id,
        int64_t first_sample_idx
    ) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("ReplayBuffer: cannot write chunk file " + path);
        }

        const int n = static_cast<int>(samples.size());
        const int board_cells = board_size_ * board_size_;
        const int action_cells = action_size_;

        ChunkHeader hdr{};
        hdr.magic = kChunkMagic;
        hdr.version = kChunkVersion;
        hdr.board_size = board_size_;
        hdr.action_size = action_size_;
        hdr.num_samples = n;
        hdr.chunk_id = chunk_id;
        hdr.first_sample_idx = first_sample_idx;
        std::memset(hdr.reserved, 0, sizeof(hdr.reserved));
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // 1) states
        for (const auto& s : samples) {
            out.write(reinterpret_cast<const char*>(s.state.data()), board_cells);
        }
        // 2) to_play
        for (const auto& s : samples) {
            out.write(reinterpret_cast<const char*>(&s.to_play), 1);
        }
        // 3) policy_targets
        {
            const int bytes = action_cells * static_cast<int>(sizeof(float));
            for (const auto& s : samples) {
                out.write(reinterpret_cast<const char*>(s.policy_target.data()), bytes);
            }
        }
        // 4) opponent_policy_targets
        {
            const int bytes = action_cells * static_cast<int>(sizeof(float));
            for (const auto& s : samples) {
                out.write(reinterpret_cast<const char*>(s.opponent_policy_target.data()), bytes);
            }
        }
        // 5) value_targets
        for (const auto& s : samples) {
            out.write(reinterpret_cast<const char*>(s.value_target.data()), 3 * sizeof(float));
        }
        // 6) sample_weights
        for (const auto& s : samples) {
            out.write(reinterpret_cast<const char*>(&s.sample_weight), sizeof(float));
        }
        // 7) is_full_search
        for (const auto& s : samples) {
            out.write(reinterpret_cast<const char*>(&s.is_full_search), 1);
        }

        if (!out.good()) {
            throw std::runtime_error("ReplayBuffer: write error on chunk file " + path);
        }
    }

    // ── Load a range of samples from a chunk file ──────────────────────────

    std::vector<TrainSample> load_chunk_range(
        const std::string& path,
        int total_in_chunk,
        int offset,
        int count
    ) const {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("ReplayBuffer: cannot read chunk file " + path);
        }

        const int board_cells = board_size_ * board_size_;
        const int action_cells = action_size_;

        // Skip header
        in.seekg(sizeof(ChunkHeader));

        std::vector<TrainSample> result(count);
        for (auto& s : result) {
            s.state.resize(board_cells);
            s.policy_target.resize(action_cells);
            s.opponent_policy_target.resize(action_cells);
        }

        // Read a field-major array, seeking to offset, reading count entries
        auto read_field = [&](int bytes_per_sample, auto load_fn) {
            const auto field_start = in.tellg();
            in.seekg(field_start + static_cast<std::streamoff>(
                static_cast<int64_t>(offset) * bytes_per_sample
            ));

            std::vector<char> buf(static_cast<size_t>(count) * bytes_per_sample);
            in.read(buf.data(), static_cast<std::streamsize>(count) * bytes_per_sample);

            for (int i = 0; i < count; ++i) {
                load_fn(buf.data() + static_cast<size_t>(i) * bytes_per_sample, result[i]);
            }

            // Seek to end of entire field (start + total_in_chunk * bytes_per_sample)
            in.seekg(field_start + static_cast<std::streamoff>(
                static_cast<int64_t>(total_in_chunk) * bytes_per_sample
            ));
        };

        // 1) states
        read_field(board_cells, [&](const char* src, TrainSample& s) {
            std::memcpy(s.state.data(), src, board_cells);
        });
        // 2) to_play
        read_field(1, [](const char* src, TrainSample& s) {
            s.to_play = *reinterpret_cast<const int8_t*>(src);
        });
        // 3) policy_targets
        const int policy_bytes = action_cells * static_cast<int>(sizeof(float));
        read_field(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.policy_target.data(), src, policy_bytes);
        });
        // 4) opponent_policy_targets
        read_field(policy_bytes, [&](const char* src, TrainSample& s) {
            std::memcpy(s.opponent_policy_target.data(), src, policy_bytes);
        });
        // 5) value_targets
        constexpr int value_bytes = 3 * static_cast<int>(sizeof(float));
        read_field(value_bytes, [](const char* src, TrainSample& s) {
            std::memcpy(s.value_target.data(), src, value_bytes);
        });
        // 6) sample_weights
        read_field(static_cast<int>(sizeof(float)), [](const char* src, TrainSample& s) {
            std::memcpy(&s.sample_weight, src, sizeof(float));
        });
        // 7) is_full_search
        read_field(1, [](const char* src, TrainSample& s) {
            s.is_full_search = *reinterpret_cast<const uint8_t*>(src);
        });

        return result;
    }

    void validate_sample_shape(const TrainSample& s) const {
        const size_t board_cells = static_cast<size_t>(board_size_) * board_size_;
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

    // ── Member variables ───────────────────────────────────────────────────

    int board_size_;
    int action_size_;

    int min_buffer_size_;
    int linear_threshold_;
    float alpha_;
    int max_buffer_size_;
    int samples_per_chunk_;

    std::string chunk_dir_;          // directory where chunk files are stored
    std::vector<ChunkInfo> chunks_;  // ordered oldest-to-newest
    std::vector<TrainSample> write_buffer_;  // pending samples not yet flushed
    int next_chunk_id_;
    int64_t total_samples_added_;
    int games_count_;
};

}  // namespace skyzero

#endif
