#ifndef SKYZERO_VCT_SKYZERO_ADAPTER_H
#define SKYZERO_VCT_SKYZERO_ADAPTER_H

// Boundary between V6 SkyZero (board-stride int8 layout, V6 RuleType enum,
// canvas-stride MCTS outputs) and the lifted KataGomo VCFsolver. Convert
// SkyZero board → temporary KataBoard → VCFsolver::run → decode loc back to
// canvas-stride.

#include <cstdint>
#include <vector>

#include "../envs/gomoku.h"  // for RuleType

namespace skyzero {
namespace vct {

struct AdapterResult {
    // Solver result codes (KataGomo VCFsolver convention):
    //   1 = win for `to_play`
    //   2 = no win for `to_play` (provably or within budget — see KataGomo
    //       VCFsolver.cpp:226-233)
    //   3 = budget exhausted, undetermined
    int result = 2;

    // First move of the winning sequence in canvas-stride coordinates
    // (r * MAX_BOARD_SIZE + c). Only meaningful when result == 1. Else -1.
    int first_move_canvas = -1;
};

// One-time process startup. Calls VCFsolver::init() to seed its zobrist
// table from a fixed mt19937_64. Idempotent (std::call_once-guarded). Must
// be called before any solve_vcf() invocation.
void global_init();

// state: V6 board-stride, length board_size * board_size, values
// {-1=white, 0=empty, +1=black}.
// to_play: +1 black, -1 white.
// rule: V6 RuleType enum.
// max_nodes: solver node budget; defaults to KataGomo standard = 50000.
AdapterResult solve_vcf(
    const std::vector<int8_t>& state,
    int board_size,
    int to_play,
    skyzero::RuleType rule,
    int max_nodes = 50000
);

}  // namespace vct
}  // namespace skyzero

#endif
