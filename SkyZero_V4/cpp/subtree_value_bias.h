#ifndef SKYZERO_SUBTREE_VALUE_BIAS_H
#define SKYZERO_SUBTREE_VALUE_BIAS_H

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

namespace skyzero {

// ---------------------------------------------------------------------------
// SubtreeValueBiasEntry — one bucket accumulating observed NN-vs-subtree error
// ---------------------------------------------------------------------------
struct SubtreeValueBiasEntry {
    float delta_utility_sum = 0.0f;
    float weight_sum = 0.0f;
    mutable std::atomic_flag lock = ATOMIC_FLAG_INIT;

    void add(float delta, float weight) {
        while (lock.test_and_set(std::memory_order_acquire));
        delta_utility_sum += delta;
        weight_sum += weight;
        lock.clear(std::memory_order_release);
    }

    void subtract(float delta, float weight) {
        while (lock.test_and_set(std::memory_order_acquire));
        delta_utility_sum -= delta;
        weight_sum -= weight;
        lock.clear(std::memory_order_release);
    }

    // Read current bias (caller does NOT hold lock — snapshot read)
    float get_bias() const {
        while (lock.test_and_set(std::memory_order_acquire));
        float d = delta_utility_sum;
        float w = weight_sum;
        lock.clear(std::memory_order_release);
        if (w > 0.001f) {
            return d / w;
        }
        return 0.0f;
    }
};

// ---------------------------------------------------------------------------
// LocalPatternHasher — zobrist hash over a (2*radius+1)^2 window around a move
//
// For gomoku on a flat vector<int8_t> state:
//   state[r * board_size + c] in {-1, 0, +1}
//   color index: 0=empty, 1=black(+1), 2=white(-1), 3=off-board
// ---------------------------------------------------------------------------
struct LocalPatternHasher {
    int board_size = 0;
    int radius = 0;
    int window_size = 0; // 2*radius+1

    // zobrist[color_idx * window_area + dy * window_size + dx]
    std::vector<uint64_t> zobrist_pattern;
    uint64_t zobrist_player[3] = {}; // index by (player+1): 0=-1, 1=0(unused), 2=+1

    // zobrist for prev_action and action locations
    std::vector<uint64_t> zobrist_prev_action;
    std::vector<uint64_t> zobrist_action;

    LocalPatternHasher() = default;

    void init(int bs, int r, std::mt19937_64& rng) {
        board_size = bs;
        radius = r;
        window_size = 2 * r + 1;
        const int num_colors = 4; // empty, black, white, off-board
        const int window_area = window_size * window_size;

        zobrist_pattern.resize(num_colors * window_area);
        for (auto& z : zobrist_pattern) {
            z = rng();
        }
        for (auto& z : zobrist_player) {
            z = rng();
        }

        const int max_actions = bs * bs;
        zobrist_prev_action.resize(max_actions + 1); // +1 for "no prev action" sentinel
        zobrist_action.resize(max_actions);
        for (auto& z : zobrist_prev_action) {
            z = rng();
        }
        for (auto& z : zobrist_action) {
            z = rng();
        }
    }

    // Get the full bucket key for (player, prev_action, action, local pattern)
    // state is the board BEFORE the action is played.
    uint64_t get_key(
        int player,       // +1 or -1
        int prev_action,  // -1 if none
        int action,
        const std::vector<int8_t>& state
    ) const {
        // Start with player + move location hashes
        uint64_t hash = zobrist_player[player + 1];
        const int prev_idx = (prev_action < 0) ? (board_size * board_size) : prev_action;
        hash ^= zobrist_prev_action[prev_idx];
        hash ^= zobrist_action[action];

        // Add local pattern around 'action'
        if (radius > 0) {
            const int ar = action / board_size;
            const int ac = action % board_size;
            const int window_area = window_size * window_size;
            (void)window_area;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int r = ar + dy;
                    const int c = ac + dx;
                    int color_idx;
                    if (r < 0 || r >= board_size || c < 0 || c >= board_size) {
                        color_idx = 3; // off-board
                    } else {
                        const int8_t v = state[r * board_size + c];
                        if (v == 1) color_idx = 1;       // black
                        else if (v == -1) color_idx = 2;  // white
                        else color_idx = 0;               // empty
                    }
                    const int wy = dy + radius;
                    const int wx = dx + radius;
                    hash ^= zobrist_pattern[color_idx * window_size * window_size + wy * window_size + wx];
                }
            }
        }

        return hash;
    }
};

// ---------------------------------------------------------------------------
// SubtreeValueBiasTable — sharded hash table of bias entries
// ---------------------------------------------------------------------------
struct SubtreeValueBiasTable {
    int num_shards;
    std::vector<std::unordered_map<uint64_t, std::shared_ptr<SubtreeValueBiasEntry>>> shards;
    std::vector<std::mutex> shard_mutexes;
    LocalPatternHasher pattern_hasher;

    SubtreeValueBiasTable(int board_size, int num_shards_, int pattern_radius)
        : num_shards(num_shards_),
          shards(num_shards_),
          shard_mutexes(num_shards_)
    {
        // Use a fixed seed so the zobrist tables are deterministic across runs
        std::mt19937_64 rng(0xABCD1234ULL ^ static_cast<uint64_t>(board_size));
        pattern_hasher.init(board_size, pattern_radius, rng);
    }

    // Look up or create the entry for the given bucket key.
    // state is the board BEFORE the move 'action' is played.
    std::shared_ptr<SubtreeValueBiasEntry> get(
        int player,
        int prev_action,
        int action,
        const std::vector<int8_t>& state
    ) {
        const uint64_t key = pattern_hasher.get_key(player, prev_action, action, state);
        const int shard_idx = static_cast<int>(key % static_cast<uint64_t>(num_shards));

        std::lock_guard<std::mutex> lock(shard_mutexes[shard_idx]);
        auto& slot = shards[shard_idx][key];
        if (!slot) {
            slot = std::make_shared<SubtreeValueBiasEntry>();
        }
        return slot;
    }

    // Remove entries that are not referenced by any node (use_count == 1 means
    // only the table itself holds it). Call between games or during tree reuse.
    void clear_unused() {
        for (int i = 0; i < num_shards; ++i) {
            std::lock_guard<std::mutex> lock(shard_mutexes[i]);
            for (auto it = shards[i].begin(); it != shards[i].end(); ) {
                if (it->second.use_count() <= 1) {
                    it = shards[i].erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
};

}  // namespace skyzero

#endif // SKYZERO_SUBTREE_VALUE_BIAS_H
