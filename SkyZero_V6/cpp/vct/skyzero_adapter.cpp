#include "skyzero_adapter.h"

#include <mutex>

#include "kata_board.h"
#include "kata_rules.h"
#include "vcf_solver.h"

namespace skyzero {
namespace vct {

namespace {
std::once_flag g_init_flag;
}

void global_init() {
    std::call_once(g_init_flag, []() {
        VCFsolver::init();
    });
}

AdapterResult solve_vcf(
    const std::vector<int8_t>& state,
    int board_size,
    int to_play,
    skyzero::RuleType rule,
    int max_nodes
) {
    AdapterResult out;
    out.result = 2;
    out.first_move_canvas = -1;

    if (board_size <= 0 || board_size > Board::MAX_LEN) return out;

    // Override solver node budget for this call. MAXNODE is a static — this
    // is process-global, not per-call. Acceptable because every selfplay
    // worker shares the same budget config.
    VCFsolver::MAXNODE = static_cast<uint64_t>(max_nodes);

    Board kbd(board_size, board_size);
    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            const int8_t v = state[y * board_size + x];
            if (v == 0) continue;
            const Color c = (v == 1) ? C_BLACK : C_WHITE;
            kbd.playMoveAssumeLegal(Board::getLoc(x, y, board_size), c);
        }
    }

    Rules rules;
    switch (rule) {
        case skyzero::RuleType::FREESTYLE: rules.basicRule = Rules::BASICRULE_FREESTYLE; break;
        case skyzero::RuleType::STANDARD:  rules.basicRule = Rules::BASICRULE_STANDARD;  break;
        case skyzero::RuleType::RENJU:     rules.basicRule = Rules::BASICRULE_RENJU;     break;
    }

    const uint8_t pla = (to_play == 1) ? C_BLACK : C_WHITE;
    uint8_t res = 2;
    uint16_t loc = Board::NULL_LOC;
    VCFsolver::run(kbd, rules, pla, res, loc);

    out.result = static_cast<int>(res);
    if (res == 1 && loc != Board::NULL_LOC) {
        // KataGomo bordered loc:  loc = (x+1) + (y+1) * (xsize + 1)
        const int xsize_plus_1 = board_size + 1;
        const int y_plus_1 = static_cast<int>(loc) / xsize_plus_1;
        const int x_plus_1 = static_cast<int>(loc) - y_plus_1 * xsize_plus_1;
        const int x = x_plus_1 - 1;
        const int y = y_plus_1 - 1;
        if (x >= 0 && x < board_size && y >= 0 && y < board_size) {
            // Canvas-stride uses Gomoku::MAX_BOARD_SIZE (compile-time const,
            // typically 17). Read it through the Game class for consistency.
            const int M = skyzero::Gomoku::MAX_BOARD_SIZE;
            out.first_move_canvas = y * M + x;
        }
    }
    return out;
}

}  // namespace vct
}  // namespace skyzero
