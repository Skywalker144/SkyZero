#ifndef SKYZERO_ENVS_RESULTS_BEFORE_NN_H
#define SKYZERO_ENVS_RESULTS_BEFORE_NN_H

// Pre-NN tactical analysis. 1:1 port of KataGomo
// gamelogic.cpp:80-115 (getMovePriority + connectionLengthOneDirection) +
// gamelogic.cpp:319-412 (ResultsBeforeNN::init), with KataGomo's
// VCN/maxMoves/pass branches stripped (V6 doesn't support them).
//
// Output feeds Gomoku::compute_global_features() dim 6-11 (KataGomo
// nninputs.cpp:809-864 V101 rowGlobal[7..12]) and the V6 spatial plane 5
// (myOnlyLoc one-hot; KataGomo nninputs spatial[5]).
//
// Header-only because the data flows through hot inference call sites
// (alphazero_parallel.h, alphazero_tree_parallel.h, selfplay_manager.h)
// and the per-call cost is cheap enough that translation-unit boundary
// overhead would dominate.

#include <cstdint>
#include <vector>

#include "../envs/gomoku.h"
#include "../vct/skyzero_adapter.h"

namespace skyzero {

// Move priority for a given empty cell on a board, KataGomo-aligned.
// Lower = more urgent. Numbering follows V6 doc (different literal values
// than KataGomo core/global.h MP_FIVE=1/etc., but the same semantic order
// and the same `<` comparator).
//
// (KataGomo source: core/global.h MP_FIVE/MP_OPPOFOUR/MP_MYLIFEFOUR +
// gamelogic.cpp:55-115.)
enum class MovePriority : int8_t {
    INVALID    = -1,
    FIVE       =  0,   // Placing here gives ME an immediate 5-in-a-row.
    OPPOFOUR   =  1,   // Opp would 5-in-a-row at this cell next move; I must block here.
    MYLIFEFOUR =  2,   // Placing here gives ME a life-four (open 4-in-a-row).
    NORMAL     = 126,
};

// Walk one direction from (r, c) along (dr, dc), counting consecutive `color`
// stones until blocked by an empty/wall/opp. Returns count + sets `is_life`.
// Mirrors KataGomo gamelogic.cpp:22-53. is_six_win controls the "long-row
// counts as life" suppression: when overline does NOT win for `color` (e.g.
// RENJU+Black, STANDARD), an empty cell isn't a "life boundary" if the cell
// just past it is also `color` — because extending here would land in an
// overline, not a 5.
inline int connection_length_one_direction(
    const std::vector<int8_t>& state,
    int board_size,
    int r, int c,
    int dr, int dc,
    int color,
    bool is_six_win,
    bool& is_life
) {
    is_life = false;
    int len = 0;
    int rr = r;
    int cc = c;
    while (true) {
        rr += dr;
        cc += dc;
        if (rr < 0 || rr >= board_size || cc < 0 || cc >= board_size) break;
        const int8_t s = state[rr * board_size + cc];
        if (s == color) {
            ++len;
        } else if (s == 0) {
            is_life = true;
            if (!is_six_win) {
                const int rrr = rr + dr;
                const int ccc = cc + dc;
                if (rrr >= 0 && rrr < board_size && ccc >= 0 && ccc < board_size
                    && state[rrr * board_size + ccc] == color) {
                    is_life = false;
                }
            }
            break;
        } else {
            break;
        }
    }
    return len;
}

// MovePriority for one direction, given placing `to_play` at (r, c).
// KataGomo gamelogic.cpp:55-78 (getMovePriorityOneDirectionAssumeLegal).
inline MovePriority move_priority_one_direction(
    const std::vector<int8_t>& state,
    int board_size,
    int r, int c,
    int dr, int dc,
    int to_play,
    bool is_six_win_me,
    bool is_six_win_opp
) {
    const int opp = -to_play;
    bool my_life_a, my_life_b, opp_life_a, opp_life_b;
    const int my_con =
        connection_length_one_direction(state, board_size, r, c, dr, dc, to_play, is_six_win_me, my_life_a) +
        connection_length_one_direction(state, board_size, r, c, -dr, -dc, to_play, is_six_win_me, my_life_b) +
        1;
    const int opp_con =
        connection_length_one_direction(state, board_size, r, c, dr, dc, opp, is_six_win_opp, opp_life_a) +
        connection_length_one_direction(state, board_size, r, c, -dr, -dc, opp, is_six_win_opp, opp_life_b) +
        1;

    if (my_con == 5 || (my_con > 5 && is_six_win_me)) return MovePriority::FIVE;
    if (opp_con == 5 || (opp_con > 5 && is_six_win_opp)) return MovePriority::OPPOFOUR;
    if (my_con == 4 && my_life_a && my_life_b) return MovePriority::MYLIFEFOUR;
    return MovePriority::NORMAL;
}

// Aggregate MovePriority across all 4 directions, then apply Renju+Black
// forbidden-move filter (KataGomo gamelogic.cpp:112).
//   FREESTYLE: black overline wins, white overline wins.
//   STANDARD:  no overline wins for either.
//   RENJU:     black exact-5; white overline wins.
inline MovePriority get_move_priority(
    const Gomoku& g,
    const std::vector<int8_t>& state,
    int to_play,
    int r, int c
) {
    if (state[r * g.board_size + c] != 0) return MovePriority::INVALID;

    const bool is_six_win_me =
        (g.rule == RuleType::FREESTYLE) ? true :
        (g.rule == RuleType::STANDARD)  ? false :
        /* RENJU */                        (to_play == -1);   // white can win on overline
    const bool is_six_win_opp =
        (g.rule == RuleType::FREESTYLE) ? true :
        (g.rule == RuleType::STANDARD)  ? false :
        /* RENJU */                        (to_play == 1);    // when I'm black, opp(white) overlines win

    constexpr int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    MovePriority best = MovePriority::NORMAL;
    for (auto& d : dirs) {
        const MovePriority mp = move_priority_one_direction(
            state, g.board_size, r, c, d[0], d[1], to_play, is_six_win_me, is_six_win_opp);
        if (static_cast<int>(mp) < static_cast<int>(best)) best = mp;
    }

    // KataGomo gamelogic.cpp:112: Renju+Black life-four at a forbidden point
    // downgrades to NORMAL. Otherwise we'd treat a forbidden cell as a forced
    // win, which it isn't — Black playing here loses immediately.
    if (best == MovePriority::MYLIFEFOUR && g.rule == RuleType::RENJU && to_play == 1) {
        if (g.is_renju_forbidden_at(state, r, c)) return MovePriority::NORMAL;
    }
    return best;
}

// Pre-NN tactical state for one (state, to_play) pair. Populated by init().
// Feeds:
//   - global feature dims 6-11 (winner / VCF results)
//   - spatial plane 5 (my_only_loc one-hot)
//   - inference-time short-circuits: my-side immediate-win sees winner ==
//     to_play && my_only_canvas >= 0 and skips the NN.
//
// Convention:
//   winner: +1=black wins, -1=white wins, 0=undecided. Filled when this
//           position has a forced win for to_play (immediate 5, life-four
//           with no opp-four, or solver-found VCF).
//   my_only_canvas: canvas-stride index of the unique forced move
//           (immediate-five, must-block opp-four, or VCF first move). -1
//           if no forced move was found.
//   myVCF / oppVCF: VCFsolver result codes (KataGomo convention):
//           0 = solver was not called for this side
//           1 = side can VCF-win
//           2 = side cannot VCF-win
//           3 = solver budget exhausted (undetermined)
//   calculatedVCF: true iff init() actually ran the solver (gated on has_vcf
//           AND the position not already being decided by MP_FIVE / opp-four /
//           own life-four).
struct ResultsBeforeNN {
    bool inited = false;
    bool calculatedVCF = false;
    int  winner = 0;
    int  my_only_canvas = -1;
    uint8_t myVCF = 0;
    uint8_t oppVCF = 0;

    // KataGomo gamelogic.cpp:319-412.
    void init(const Gomoku& g,
              const std::vector<int8_t>& state,
              int to_play,
              bool has_vcf,
              int vct_max_nodes = 50000) {
        if (inited) return;
        inited = true;

        const int N = g.board_size;
        const int M = Gomoku::MAX_BOARD_SIZE;

        // Pass 1: scan all empty cells for MP_FIVE / OPPOFOUR / MYLIFEFOUR.
        // KataGomo aborts early on MP_FIVE; we mirror that.
        bool opp_has_four = false;
        bool my_life_four = false;
        int  my_life_four_canvas = -1;
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                if (state[r * N + c] != 0) continue;
                const MovePriority mp = get_move_priority(g, state, to_play, r, c);
                const int canvas = r * M + c;
                if (mp == MovePriority::FIVE) {
                    winner = to_play;
                    my_only_canvas = canvas;
                    return;
                }
                if (mp == MovePriority::OPPOFOUR) {
                    opp_has_four = true;
                    my_only_canvas = canvas;   // KataGomo also overwrites here.
                } else if (mp == MovePriority::MYLIFEFOUR) {
                    my_life_four = true;
                    my_life_four_canvas = canvas;
                }
            }
        }

        // Opp has a four-threat → I must block; winner stays undecided.
        if (opp_has_four) return;

        // I have life-four, opp doesn't → I win in ≤2 moves.
        if (my_life_four) {
            winner = to_play;
            my_only_canvas = my_life_four_canvas;
            return;
        }

        // No has_vcf → leave dim 7-11 as zero (KataGomo skips solver).
        if (!has_vcf) return;

        // Run VCF solver for both sides (KataGomo gamelogic.cpp:401-411).
        calculatedVCF = true;

        // Opp's VCF — note has-vcf encodes "opp can force win regardless of
        // my move", critical signal even when I'm not threatened.
        const auto opp_r = vct::solve_vcf(state, N, -to_play, g.rule, vct_max_nodes);
        oppVCF = static_cast<uint8_t>(opp_r.result);

        const auto my_r = vct::solve_vcf(state, N, to_play, g.rule, vct_max_nodes);
        myVCF = static_cast<uint8_t>(my_r.result);
        if (myVCF == 1 && my_r.first_move_canvas >= 0) {
            winner = to_play;
            my_only_canvas = my_r.first_move_canvas;
        }
    }
};

// Definition of Gomoku::compute_global_features (declared in gomoku.h). Lives
// here because it depends on the full ResultsBeforeNN definition. KataGomo
// nninputs.cpp:809-864 V101 alignment for dims 6-13.
inline GlobalFeatures Gomoku::compute_global_features(
    int ply, int to_play,
    const ResultsBeforeNN& r,
    double pda_signed_for_to_play,
    bool pda_active
) const {
    GlobalFeatures g;   // zero-init
    g.data[0] = (rule == RuleType::FREESTYLE) ? 1.0f : 0.0f;
    g.data[1] = (rule == RuleType::STANDARD)  ? 1.0f : 0.0f;
    g.data[2] = (rule == RuleType::RENJU)     ? 1.0f : 0.0f;
    // dim 3: renju_color_sign (only under Renju, "black has tighter constraints").
    g.data[3] = (rule == RuleType::RENJU) ? (to_play == 1 ? -1.0f : +1.0f) : 0.0f;
    // dim 4: has_forbidden — only RENJU has a forbidden-move concept.
    g.data[4] = (rule == RuleType::RENJU) ? 1.0f : 0.0f;
    // dim 5: ply normalized by actual board area (size-invariant progress signal).
    g.data[5] = float(ply) / float(board_size * board_size);
    // dims 6-11: ResultsBeforeNN-derived (KataGomo rowGlobal[7..12]).
    g.data[6]  = (r.winner == to_play) ? 1.0f : 0.0f;
    g.data[7]  = (r.calculatedVCF && r.myVCF  == 2) ? 1.0f : 0.0f;
    g.data[8]  = (r.calculatedVCF && r.myVCF  == 3) ? 1.0f : 0.0f;
    g.data[9]  = (r.calculatedVCF && r.oppVCF == 1) ? 1.0f : 0.0f;
    g.data[10] = (r.calculatedVCF && r.oppVCF == 2) ? 1.0f : 0.0f;
    g.data[11] = (r.calculatedVCF && r.oppVCF == 3) ? 1.0f : 0.0f;
    // dims 12-13: KataGomo PDA (rowGlobal[15..16]). Default zero outside selfplay.
    g.data[12] = pda_active ? 1.0f : 0.0f;
    g.data[13] = static_cast<float>(0.5 * pda_signed_for_to_play);
    return g;
}

inline GlobalFeatures Gomoku::compute_global_features(int ply, int to_play) const {
    ResultsBeforeNN empty;   // winner=0, my_only=-1, no VCF — dims 6-11 zero.
    return compute_global_features(ply, to_play, empty);
}

// Bundled NN-forward input for one (state, to_play). Built by
// prepare_inference_input(); consumed by every NN-forward call site
// (alphazero_parallel.h / alphazero_tree_parallel.h / random_opening.h /
// policy_init.h / selfplay_manager.h post-game replay).
//
// vct_winning_canvas captures KataGomo nneval.cpp:706-723 myOnlyLoc fast
// path: when ResultsBeforeNN proves the move-to-play is a forced win
// (winner == to_play && my_only_canvas >= 0 — covers MP_FIVE / my_life_four /
// myVCF==1), the caller short-circuits the NN entirely and returns a one-hot
// policy + (1,0,0) value. -1 means "no short-circuit; run the NN".
struct InferenceInput {
    std::vector<int8_t> encoded;                                   // V5: NUM_SPATIAL_PLANES_V5 * MAX_AREA
    std::array<float, GlobalFeatures::DIM> globals{};
    int vct_winning_canvas = -1;
};

// Build InferenceInput for one (state, to_play). Runs ResultsBeforeNN once,
// reuses it for plane 5 + dims 6-11, and exposes the VCT-win canvas for
// optional NN short-circuit. PDA params default off (only selfplay opts in).
inline InferenceInput prepare_inference_input(
    const Gomoku& g,
    const std::vector<int8_t>& state,
    int to_play,
    bool has_vcf,
    int vct_max_nodes = 50000,
    double pda_signed_for_to_play = 0.0,
    bool pda_active = false
) {
    ResultsBeforeNN r;
    r.init(g, state, to_play, has_vcf, vct_max_nodes);

    InferenceInput in;
    in.encoded = g.encode_state_v5(state, to_play, r.my_only_canvas);

    int ply = 0;
    for (int8_t v : state) if (v != 0) ++ply;
    auto gf = g.compute_global_features(ply, to_play, r, pda_signed_for_to_play, pda_active);
    for (int i = 0; i < GlobalFeatures::DIM; ++i) in.globals[i] = gf.data[i];

    in.vct_winning_canvas = (r.winner == to_play && r.my_only_canvas >= 0)
        ? r.my_only_canvas : -1;
    return in;
}

}  // namespace skyzero

#endif  // SKYZERO_ENVS_RESULTS_BEFORE_NN_H
