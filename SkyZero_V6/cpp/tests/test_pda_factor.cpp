// Unit tests for KataGomo PDA factor math + sign-flip per to_play.
// Tests both the per-move 2f/(f+1) redistribute formula and the
// pda_signed_active helper that drives global feature dim 12-13.
//
// Run: cpp/build/test_pda_factor

#include <cassert>
#include <cmath>
#include <cstdio>

#include "alphazero.h"

using skyzero::PdaState;
using skyzero::pda_signed_active;

static int failures = 0;

#define EXPECT_NEAR(a, b, tol) do { \
    auto _a = static_cast<double>(a); auto _b = static_cast<double>(b); \
    if (!(std::abs(_a - _b) <= (tol))) { \
        std::fprintf(stderr, "FAIL %s:%d  %s ≈ %s  (%.6f vs %.6f, tol=%g)\n", \
                     __FILE__, __LINE__, #a, #b, _a, _b, (double)(tol)); \
        ++failures; \
    } \
} while (0)

#define EXPECT_TRUE(c) do { \
    if (!(c)) { \
        std::fprintf(stderr, "FAIL %s:%d  !(%s)\n", __FILE__, __LINE__, #c); \
        ++failures; \
    } \
} while (0)

// 1. f=2 (one doubling) → favored 4/3, disfavored 2/3.
//    Sum = 4/3 + 2/3 = 2 — total budget conserved on average per (favored,
//    disfavored) move pair (KataGomo program/play.cpp:1027-1051).
static void test_factor_f2() {
    const double f = 2.0;
    const double favored = 2.0 * f / (f + 1.0);
    const double disfavored = 2.0 / (f + 1.0);
    EXPECT_NEAR(favored, 4.0 / 3.0, 1e-12);
    EXPECT_NEAR(disfavored, 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(favored + disfavored, 2.0, 1e-12);
    std::printf("test_factor_f2: ok\n");
}

// 2. f=8 (max default ratio) → favored 16/9, disfavored 2/9.
static void test_factor_f8() {
    const double f = 8.0;
    const double favored = 2.0 * f / (f + 1.0);
    const double disfavored = 2.0 / (f + 1.0);
    EXPECT_NEAR(favored, 16.0 / 9.0, 1e-12);
    EXPECT_NEAR(disfavored, 2.0 / 9.0, 1e-12);
    EXPECT_NEAR(favored + disfavored, 2.0, 1e-12);
    std::printf("test_factor_f8: ok\n");
}

// 3. PdaState sign-flip from to_play perspective.
//    KataGomo searchnnhelpers.cpp:20-25: signed_pda_doublings is positive when
//    `to_play == pda.side` (favored) and negative when `to_play == -pda.side`
//    (disfavored).
static void test_signed_active_per_to_play() {
    PdaState p;
    p.abs_doublings = 2.0;   // f = 4
    p.side = +1;             // black is favored

    auto bk = pda_signed_active(&p, +1);
    EXPECT_TRUE(bk.second);                       // pda_active
    EXPECT_NEAR(bk.first, +2.0, 1e-12);          // black sees +2

    auto wh = pda_signed_active(&p, -1);
    EXPECT_TRUE(wh.second);
    EXPECT_NEAR(wh.first, -2.0, 1e-12);          // white sees -2
    std::printf("test_signed_active_per_to_play: ok\n");
}

// 4. Disabled PDA → both pda_active=false, signed=0 regardless of to_play.
static void test_signed_active_disabled() {
    PdaState p;   // side=0 by default
    auto a = pda_signed_active(&p, +1);
    EXPECT_TRUE(!a.second);
    EXPECT_NEAR(a.first, 0.0, 1e-12);
    auto b = pda_signed_active(&p, -1);
    EXPECT_TRUE(!b.second);
    EXPECT_NEAR(b.first, 0.0, 1e-12);

    auto n = pda_signed_active(nullptr, +1);
    EXPECT_TRUE(!n.second);
    EXPECT_NEAR(n.first, 0.0, 1e-12);
    std::printf("test_signed_active_disabled: ok\n");
}

// 5. White-favored: same magnitude but flipped semantics. White sees +2,
//    black sees -2.
static void test_signed_active_white_side() {
    PdaState p;
    p.abs_doublings = 2.0;
    p.side = -1;
    auto wh = pda_signed_active(&p, -1);
    EXPECT_NEAR(wh.first, +2.0, 1e-12);
    auto bk = pda_signed_active(&p, +1);
    EXPECT_NEAR(bk.first, -2.0, 1e-12);
    std::printf("test_signed_active_white_side: ok\n");
}

int main() {
    test_factor_f2();
    test_factor_f8();
    test_signed_active_per_to_play();
    test_signed_active_disabled();
    test_signed_active_white_side();
    if (failures == 0) {
        std::printf("ALL PDA TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURES\n", failures);
    return 1;
}
