#ifndef SKYZERO_VCT_VCF_SOLVER_H
#define SKYZERO_VCT_VCF_SOLVER_H

// Lifted from KataGomo cpp/vcfsolver/VCFsolver.h. Only changes:
//  * include paths repointed to V6 vct/
//  * #pragma once → header guard
//  * wrapped in namespace skyzero::vct
//  * dropped `using namespace std;` (use std:: explicitly inside the cpp)
//  * dropped non-ASCII Chinese comments (originals lived in GBK and rendered
//    as garbage on UTF-8 systems)
//
// Algorithm and field semantics unchanged. Renju forbidden-move handling
// (3-3, 4-4, overline) is preserved as-is.

#include <cstdint>
#include <vector>

#include "hash128.h"
#include "kata_board.h"
#include "kata_rules.h"
#include "vcf_hash_table.h"

namespace skyzero {
namespace vct {

class VCFsolver {
public:
    static const int sz = Board::MAX_LEN;
    static const uint8_t C_EM = 0;    // empty
    static const uint8_t C_MY = 1;    // self stone (color is rotated to "my=1, opp=2" perspective inside the solver)
    static const uint8_t C_OPP = 2;   // opponent stone

    static uint64_t MAXNODE;

    static Hash128 zob_board[2][sz][sz];  // (pla-1, y, x)
    static const Hash128 zob_plaWhite;
    static const Hash128 zob_plaBlack;

    // hashTable
    static VCFHashTable hashtable;

    // RULE
    Rules rules;
    uint8_t forbiddenSide;  // self color id when self is forbidden (renju black) / 0 if not

    // board
    int xsize, ysize;
    uint8_t rootboard[sz][sz];   // board[y][x]
    uint8_t board[sz][sz];       // board[y][x]
    int32_t movenum;
    Hash128 boardhash;
    int64_t oppFourPos;          // opponent's four-in-a-row threat point, -1 if none

    uint8_t mystonecount[4][sz][sz];   // 4 directions × board cells: count of "my" stones in 5-cell window starting at (y, x)
    uint8_t oppstonecount[4][sz][sz];  // ditto for opponent
    std::vector<int64_t> threes;
    uint64_t threeCount;

    uint64_t nodenum;

    // result
    int32_t rootresultpos;
    int32_t bestmovenum;

    static uint64_t totalAborted;
    static uint64_t totalSolved;
    static uint64_t totalnodenum;

    static void init();
    VCFsolver(const Rules rules) : rules(rules) { threes.resize(4 * sz * sz); }
    void solve(const Board& kataboard, uint8_t pla, uint8_t& res, uint16_t& loc);
    void print();
    void printRoot();
    static void run(const Board& board, const Rules& rules, uint8_t pla, uint8_t& res, uint16_t& loc) {
        VCFsolver solver(rules);
        solver.solve(board, pla, res, loc);
    }

public:
    int32_t setBoard(const Board& board, uint8_t pla);  // returns >0 immediate win / 0 unknown

    uint32_t findEmptyPos(int t, int y, int x);
    uint32_t findDefendPosOfFive(int y, int x);

    // For renju and standard
    void addNeighborSix(int y, int x, uint8_t pla, int factor);

    int32_t solveIter(bool isRoot);
    int32_t play(int x, int y, uint8_t pla, bool updateHash);
    void undo(int x, int y, int64_t oppFourPos, uint64_t threeCount, bool updateHash);

    // For renju
    bool isForbiddenMove(int y, int x, bool fiveForbidden = false);
    bool checkLife3(int y, int x, int t);
    void printForbiddenMap();
};

}  // namespace vct
}  // namespace skyzero

#endif
