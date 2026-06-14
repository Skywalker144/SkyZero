// Standalone unit test for cpp/envs/game2048.h. No LibTorch dependency:
//   g++ -std=c++17 -I cpp game2048_test.cpp -o /tmp/g2048_test && /tmp/g2048_test
//
// Board printed/entered row-major as exponents (0 = empty, e = tile 2^e).

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "envs/game2048.h"

using skyzero::Game2048;

static int g_checks = 0;
static int g_fails = 0;

static void check(bool cond, const char* msg) {
    ++g_checks;
    if (!cond) {
        ++g_fails;
        std::printf("  FAIL: %s\n", msg);
    }
}

static std::vector<int8_t> board(std::initializer_list<int> v) {
    std::vector<int8_t> b;
    for (int x : v) b.push_back(static_cast<int8_t>(x));
    return b;
}

static void expect_board(const std::vector<int8_t>& got,
                         const std::vector<int8_t>& want, const char* msg) {
    bool eq = got.size() == want.size();
    for (size_t i = 0; eq && i < got.size(); ++i) eq = (got[i] == want[i]);
    check(eq, msg);
    if (!eq) {
        std::printf("    got : ");
        for (auto v : got) std::printf("%d ", v);
        std::printf("\n    want: ");
        for (auto v : want) std::printf("%d ", v);
        std::printf("\n");
    }
}

int main() {
    Game2048 g;

    // --- slide LEFT: compress + single merge ---
    // row0: [2,2,.,.] (exp 1,1) -> [4,.,.,.] (exp 2), reward 4
    // row1: [2,.,2,4] (1,0,1,2) -> [4,4,.,.] (2,2), reward 4
    // row2: [4,4,4,.] (2,2,2,0) -> [8,4,.,.] (3,2,0,0), reward 8  (leftmost pair merges)
    // row3: [2,2,2,2] (1,1,1,1) -> [4,4,.,.] (2,2,0,0), reward 8  (two merges)
    {
        auto s = board({1,1,0,0,
                        1,0,1,2,
                        2,2,2,0,
                        1,1,1,1});
        auto r = g.apply_move(s, 3);  // left
        check(r.changed, "left changed");
        expect_board(r.afterstate, board({2,0,0,0,
                                          2,2,0,0,
                                          3,2,0,0,
                                          2,2,0,0}), "left afterstate");
        check(r.reward == 4 + 4 + 8 + 8, "left reward");
    }

    // --- slide RIGHT mirrors LEFT ---
    // row3: [2,2,2,2] -> [.,.,4,4] (0,0,2,2), reward 8
    {
        auto s = board({0,0,0,0,
                        0,0,0,0,
                        0,0,0,0,
                        1,1,1,1});
        auto r = g.apply_move(s, 1);  // right
        check(r.changed, "right changed");
        expect_board(r.afterstate, board({0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0,
                                          0,0,2,2}), "right afterstate");
        check(r.reward == 8, "right reward");
    }

    // --- slide UP: column-wise ---
    // col0: rows [2,2,.,.] -> [4,.,.,.], reward 4
    {
        auto s = board({1,0,0,0,
                        1,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto r = g.apply_move(s, 0);  // up
        check(r.changed, "up changed");
        expect_board(r.afterstate, board({2,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0}), "up afterstate");
        check(r.reward == 4, "up reward");
    }

    // --- slide DOWN ---
    {
        auto s = board({1,0,0,0,
                        1,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto r = g.apply_move(s, 2);  // down
        check(r.changed, "down changed");
        expect_board(r.afterstate, board({0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0,
                                          2,0,0,0}), "down afterstate");
        check(r.reward == 4, "down reward");
    }

    // --- no-double-merge: [4,4,4,4]=exp(2,2,2,2) left -> [8,8,.,.]=(3,3,0,0), reward 16 ---
    {
        auto s = board({2,2,2,2,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto r = g.apply_move(s, 3);
        expect_board(r.afterstate, board({3,3,0,0,
                                          0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0}), "no-double-merge afterstate");
        check(r.reward == 8 + 8, "no-double-merge reward");
    }

    // --- illegal move: already left-packed, no merges possible in that dir ---
    {
        auto s = board({2,1,0,0,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto r = g.apply_move(s, 3);  // left: nothing moves or merges
        check(!r.changed, "illegal left unchanged");
        check(r.reward == 0, "illegal left zero reward");
        expect_board(r.afterstate, s, "illegal left identity");
    }

    // --- legal actions on the same board: left illegal, right legal, up/down legal ---
    {
        auto s = board({2,1,0,0,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto legal = g.get_legal_actions(s);
        check(legal.size() == 4, "legal size 4");
        check(legal[3] == 0, "left illegal");
        check(legal[1] == 1, "right legal");
        // Both tiles already in row 0: up moves nothing (illegal), down does.
        check(legal[0] == 0, "up illegal");
        check(legal[2] == 1, "down legal");
        check(!g.is_terminal(s), "not terminal");
    }

    // --- terminal: full checkerboard with no equal neighbors ---
    {
        auto s = board({1,2,1,2,
                        2,1,2,1,
                        1,2,1,2,
                        2,1,2,1});
        check(g.is_terminal(s), "checkerboard terminal");
        auto legal = g.get_legal_actions(s);
        check(legal[0]==0 && legal[1]==0 && legal[2]==0 && legal[3]==0, "no legal moves");
    }

    // --- spawn distribution: probabilities sum to 1, 2 entries per empty cell ---
    {
        auto s = board({1,1,1,1,
                        1,1,1,1,
                        1,1,1,0,    // one empty cell at loc 11
                        1,1,1,1});
        auto dist = g.spawn_distribution(s);
        check(dist.size() == 2, "one empty -> 2 spawn outcomes");
        double sum = 0.0; for (auto& o : dist) sum += o.prob;
        check(std::abs(sum - 1.0) < 1e-9, "spawn probs sum to 1");
        check(dist[0].cell == 11 && dist[1].cell == 11, "spawn cell correct");
        check(std::abs(dist[0].prob - 0.9) < 1e-9, "spawn p(2)=0.9");
        check(std::abs(dist[1].prob - 0.1) < 1e-9, "spawn p(4)=0.1");
    }

    // --- spawn distribution over many empties sums to 1 ---
    {
        std::vector<int8_t> s(Game2048::AREA, 0);
        s[0] = 5;  // one occupied -> 15 empties
        auto dist = g.spawn_distribution(s);
        check(dist.size() == 15 * 2, "15 empties -> 30 outcomes");
        double sum = 0.0; for (auto& o : dist) sum += o.prob;
        check(std::abs(sum - 1.0) < 1e-9, "many-empty probs sum to 1");
    }

    // --- initial state has exactly two non-empty tiles, each exp 1 or 2 ---
    {
        std::mt19937 rng(12345);
        for (int t = 0; t < 200; ++t) {
            auto s = g.get_initial_state(rng);
            int n = 0;
            for (int8_t v : s) {
                if (v != 0) { ++n; check(v == 1 || v == 2, "initial tile is 2 or 4"); }
            }
            check(n == 2 || n == 1, "initial has 1-2 tiles");  // 1 if both landed same cell? no: two distinct draws
            check(n == 2, "initial has exactly 2 tiles");
        }
    }

    // --- spawn_random places exactly one new tile ---
    {
        std::mt19937 rng(777);
        auto after = board({1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto next = g.spawn_random(after, rng);
        int n = 0; for (int8_t v : next) if (v != 0) ++n;
        check(n == 2, "spawn_random adds exactly one tile");
    }

    // --- encode_state: each cell sets exactly one plane; empty -> plane 0 ---
    {
        auto s = board({0,1,2,11,
                        0,0,0,0,
                        0,0,0,0,
                        0,0,0,0});
        auto enc = g.encode_state(s);
        check(enc.size() == static_cast<size_t>(Game2048::NUM_PLANES) * Game2048::AREA,
              "encode size");
        for (int loc = 0; loc < Game2048::AREA; ++loc) {
            int set = 0, which = -1;
            for (int e = 0; e < Game2048::NUM_PLANES; ++e) {
                if (enc[e * Game2048::AREA + loc]) { ++set; which = e; }
            }
            check(set == 1, "exactly one plane per cell");
            int want = s[loc];
            if (want >= Game2048::NUM_PLANES) want = Game2048::NUM_PLANES - 1;
            check(which == want, "plane index == exponent");
        }
    }

    // --- exponent above cap clamps into top plane ---
    {
        auto s = std::vector<int8_t>(Game2048::AREA, 0);
        s[0] = 17;  // above NUM_PLANES-1
        auto enc = g.encode_state(s);
        check(enc[(Game2048::NUM_PLANES - 1) * Game2048::AREA + 0] == 1, "over-cap clamps to top plane");
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
