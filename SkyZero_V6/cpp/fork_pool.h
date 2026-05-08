#ifndef SKYZERO_FORK_POOL_H
#define SKYZERO_FORK_POOL_H

// KataGomo-style cross-game position pool. Workers save mid-game positions
// at game end; new games occasionally start from a randomly drawn pool entry
// instead of running balanced opening. Implements docs/v6_fork_pool_design.md
// (Section "Architecture / new module").
//
// Lock granularity: single mutex. Workers hit the pool once per game start
// (load) and once per game end (save), not per move — contention is
// negligible for the 32-worker selfplay setup.
//
// Filter on (board_size, rule): linear scan from a random start index for
// matching entries, then swap-and-pop. Acceptable up to capacity ~100k.

#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <utility>
#include <vector>

#include "envs/gomoku.h"  // for RuleType (used as one of the filter keys)

namespace skyzero {

template <typename Game>
class ForkPool {
public:
    explicit ForkPool(size_t capacity) : capacity_(capacity) {}

    // Save a position. board is in game-stride (NOT canvas-padded, NOT V5
    // encoded). to_play is the side to move at this position.
    void save(std::vector<int8_t> board, int to_play, int board_size, RuleType rule) {
        std::lock_guard<std::mutex> lk(mu_);
        if (capacity_ == 0) return;
        if (entries_.size() >= capacity_) {
            entries_.pop_front();  // FIFO eviction
        }
        entries_.push_back(Entry{std::move(board), to_play, board_size, rule});
    }

    // Try to load a position matching (board_size, rule). On hit, writes
    // (board, to_play) to out args, removes the entry, returns true.
    bool try_load(int board_size, RuleType rule, std::mt19937& rng,
                  std::vector<int8_t>& out_board, int& out_to_play) {
        std::lock_guard<std::mutex> lk(mu_);
        if (entries_.empty()) return false;
        const size_t n = entries_.size();
        std::uniform_int_distribution<size_t> pick(0, n - 1);
        const size_t start = pick(rng);
        for (size_t k = 0; k < n; ++k) {
            const size_t idx = (start + k) % n;
            if (entries_[idx].board_size == board_size && entries_[idx].rule == rule) {
                out_board = std::move(entries_[idx].board);
                out_to_play = entries_[idx].to_play;
                // Swap-and-pop using deque's erase. O(n) worst case but the
                // pool is small relative to the per-game work.
                entries_.erase(entries_.begin() + idx);
                return true;
            }
        }
        return false;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return entries_.size();
    }

private:
    struct Entry {
        std::vector<int8_t> board;
        int to_play;
        int board_size;
        RuleType rule;
    };

    mutable std::mutex mu_;
    std::deque<Entry> entries_;
    size_t capacity_;
};

}  // namespace skyzero

#endif
