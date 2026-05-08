// fork_pool_test — sanity tests for ForkPool<Game>.
//
// Tests:
//   1. save + try_load round-trip with single thread
//   2. capacity FIFO eviction
//   3. (board_size, rule) filter correctness
//   4. concurrent push/load: N threads × M ops each, no segfaults, all
//      pushed entries either remain in pool or get loaded out
//
// Run: cpp/build/fork_pool_test

#include <atomic>
#include <cassert>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "envs/gomoku.h"
#include "fork_pool.h"

using skyzero::ForkPool;
using skyzero::Gomoku;
using skyzero::RuleType;

static int failures = 0;

#define EXPECT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { \
        std::fprintf(stderr, "FAIL %s:%d  %s != %s  (%lld vs %lld)\n", \
                     __FILE__, __LINE__, #a, #b, (long long)_a, (long long)_b); \
        ++failures; \
    } \
} while (0)

#define EXPECT_TRUE(c) do { \
    if (!(c)) { \
        std::fprintf(stderr, "FAIL %s:%d  !(%s)\n", __FILE__, __LINE__, #c); \
        ++failures; \
    } \
} while (0)

static std::vector<int8_t> mk_board(int N, int salt) {
    std::vector<int8_t> b(N * N, 0);
    b[salt % (N * N)] = 1;
    return b;
}

// Test 1: save → try_load → board content preserved.
static void test_round_trip() {
    ForkPool<Gomoku> pool(/*capacity=*/100);
    auto b_in = mk_board(15, 42);
    pool.save(b_in, /*to_play=*/+1, /*board_size=*/15, RuleType::RENJU);
    EXPECT_EQ(pool.size(), (size_t)1);

    std::mt19937 rng(0);
    std::vector<int8_t> b_out;
    int to_play_out = 0;
    EXPECT_TRUE(pool.try_load(15, RuleType::RENJU, rng, b_out, to_play_out));
    EXPECT_EQ(pool.size(), (size_t)0);
    EXPECT_EQ(to_play_out, +1);
    EXPECT_EQ(b_out.size(), b_in.size());
    EXPECT_TRUE(b_out == b_in);
    std::printf("test_round_trip: OK\n");
}

// Test 2: capacity → FIFO eviction.
static void test_capacity_fifo() {
    ForkPool<Gomoku> pool(/*capacity=*/100);
    for (int i = 0; i < 200; ++i) {
        pool.save(mk_board(15, i), +1, 15, RuleType::RENJU);
    }
    EXPECT_EQ(pool.size(), (size_t)100);
    // FIFO: the first 100 should be evicted; we should be able to load 100
    // entries, all with salt >= 100 (board[salt % 225] == 1).
    std::mt19937 rng(0);
    int loads = 0;
    for (int i = 0; i < 200; ++i) {
        std::vector<int8_t> b;
        int t = 0;
        if (pool.try_load(15, RuleType::RENJU, rng, b, t)) ++loads;
    }
    EXPECT_EQ(loads, 100);
    EXPECT_EQ(pool.size(), (size_t)0);
    std::printf("test_capacity_fifo: loaded %d (expected 100)\n", loads);
}

// Test 3: filter on (board_size, rule).
static void test_filter() {
    ForkPool<Gomoku> pool(/*capacity=*/1000);
    // Push three buckets:
    //   (15, RENJU) × 50, (17, RENJU) × 50, (15, STANDARD) × 50
    for (int i = 0; i < 50; ++i) pool.save(mk_board(15, i), +1, 15, RuleType::RENJU);
    for (int i = 0; i < 50; ++i) pool.save(mk_board(17, i), +1, 17, RuleType::RENJU);
    for (int i = 0; i < 50; ++i) pool.save(mk_board(15, i), +1, 15, RuleType::STANDARD);
    EXPECT_EQ(pool.size(), (size_t)150);

    // try_load 50 times for (15, RENJU). All hits, all pop one.
    std::mt19937 rng(12345);
    int hits_15_renju = 0;
    for (int i = 0; i < 50; ++i) {
        std::vector<int8_t> b;
        int t = 0;
        if (pool.try_load(15, RuleType::RENJU, rng, b, t)) ++hits_15_renju;
    }
    EXPECT_EQ(hits_15_renju, 50);
    EXPECT_EQ(pool.size(), (size_t)100);

    // 51st try_load for (15, RENJU) should miss (bucket empty).
    std::vector<int8_t> b;
    int t = 0;
    EXPECT_TRUE(!pool.try_load(15, RuleType::RENJU, rng, b, t));
    // But (17, RENJU) and (15, STANDARD) still hit.
    EXPECT_TRUE(pool.try_load(17, RuleType::RENJU, rng, b, t));
    EXPECT_TRUE(pool.try_load(15, RuleType::STANDARD, rng, b, t));
    std::printf("test_filter: hits=%d (expected 50), final size=%zu\n",
                hits_15_renju, pool.size());
}

// Test 4: concurrent save / load across N threads.
static void test_concurrent() {
    ForkPool<Gomoku> pool(/*capacity=*/10000);
    constexpr int N_THREADS = 8;
    constexpr int N_OPS = 1000;
    std::atomic<int> total_saves{0};
    std::atomic<int> total_hits{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < N_THREADS; ++t) {
        ts.emplace_back([&, t]() {
            std::mt19937 rng(0xdeadbeef + t);
            for (int i = 0; i < N_OPS; ++i) {
                if (i % 2 == 0) {
                    pool.save(mk_board(15, t * N_OPS + i), +1, 15, RuleType::RENJU);
                    ++total_saves;
                } else {
                    std::vector<int8_t> b;
                    int pla = 0;
                    if (pool.try_load(15, RuleType::RENJU, rng, b, pla)) ++total_hits;
                }
            }
        });
    }
    for (auto& th : ts) th.join();
    const int saved = total_saves.load();
    const int hits = total_hits.load();
    const size_t remaining = pool.size();
    // Invariant: saved == hits + remaining (no entries lost or duplicated).
    EXPECT_EQ((size_t)saved, (size_t)hits + remaining);
    std::printf("test_concurrent: saved=%d hits=%d remaining=%zu (saved == hits + remaining: OK)\n",
                saved, hits, remaining);
}

int main() {
    test_round_trip();
    test_capacity_fifo();
    test_filter();
    test_concurrent();

    if (failures == 0) {
        std::printf("ALL FORK_POOL TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURES\n", failures);
    return 1;
}
