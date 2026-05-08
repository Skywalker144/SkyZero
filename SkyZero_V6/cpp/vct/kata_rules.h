#ifndef SKYZERO_VCT_KATA_RULES_H
#define SKYZERO_VCT_KATA_RULES_H

// Minimal Rules subset for KataGomo VCFsolver. We only need basicRule
// (FREESTYLE/STANDARD/RENJU) and the 3 ZOBRIST_BASIC_RULE_HASH constants
// (lifted verbatim from KataGomo cpp/game/rules.cpp:246-250).
//
// VCNRule / firstPassWin / maxMoves / passnum hashes are NOT used by
// VCFsolver, so dropped.

#include "hash128.h"

namespace skyzero {
namespace vct {

struct Rules {
    static constexpr int BASICRULE_FREESTYLE = 0;
    static constexpr int BASICRULE_STANDARD = 1;
    static constexpr int BASICRULE_RENJU = 2;
    int basicRule;

    Rules() : basicRule(BASICRULE_FREESTYLE) {}
    explicit Rules(int br) : basicRule(br) {}

    // KataGomo cpp/game/rules.cpp:246-250 — sha256-derived constants kept
    // verbatim so the VCF transposition table hashes match KataGomo's table
    // bit-for-bit (helpful when cross-checking results).
    static constexpr Hash128 ZOBRIST_BASIC_RULE_HASH[3] = {
        Hash128(0x72eeccc72c82a5e7ULL, 0x0d1265e413623e2bULL),  // FREESTYLE
        Hash128(0x125bfe48a41042d5ULL, 0x061866b5f2b98a79ULL),  // STANDARD
        Hash128(0xa384ece9d8ee713cULL, 0xfdc9f3b5d1f3732bULL),  // RENJU
    };
};

}  // namespace vct
}  // namespace skyzero

#endif
