#ifndef SKYZERO_VCT_KATA_BOARD_H
#define SKYZERO_VCT_KATA_BOARD_H

// Minimal Board subset needed by KataGomo VCFsolver — only the read-only
// queries the solver makes: x_size, y_size, colors[loc], NULL_LOC, and the
// MAX_LEN constant. No zobrist, no capture / suicide / ko logic, no JSON.
//
// Bordered loc encoding (matches KataGomo):
//   loc = (x+1) + (y+1) * (x_size + 1)
//   0 = NULL_LOC (sentinel for off-board / null)
//
// MAX_LEN matches KataGomo COMPILE_MAX_BOARD_LEN. We use 15 (Gomoku
// classical) but VCFsolver gates internally on this — bumping requires
// re-checking VCFsolver's stack-allocated arrays of size [sz][sz].

#include <cstdint>
#include <cstring>

namespace skyzero {
namespace vct {

// Color domain (matches KataGomo cpp/core/global.h):
//   0 = empty, 1 = black, 2 = white, anything else = wall sentinel.
using Color = int8_t;
using Loc = short;

static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;

inline Color getOpp(Color c) { return c ^ 3; }

struct Board {
    static constexpr int MAX_LEN = 15;
    static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;
    static constexpr int MAX_ARR_SIZE = (MAX_LEN + 1) * (MAX_LEN + 2) + 1;

    static constexpr Loc NULL_LOC = 0;
    // Real KataGomo PASS_LOC = 1; VCFsolver doesn't use pass moves but the
    // sentinel is referenced in some legality checks.
    static constexpr Loc PASS_LOC = 1;

    int x_size;
    int y_size;
    Color colors[MAX_ARR_SIZE];

    Board() : x_size(MAX_LEN), y_size(MAX_LEN) {
        init(MAX_LEN, MAX_LEN);
    }
    Board(int x, int y) : x_size(x), y_size(y) {
        init(x, y);
    }

    static Loc getLoc(int x, int y, int x_size_) {
        return (Loc)((x + 1) + (y + 1) * (x_size_ + 1));
    }

    // VCFsolver doesn't actually mutate this Board after setBoard reads from
    // it, but the public-API method is referenced by the adapter when
    // populating the board from V6 state. Mirrors KataGomo's signature; no
    // capture / suicide handling needed (gomoku has none).
    void playMoveAssumeLegal(Loc loc, Color color) {
        colors[loc] = color;
    }

private:
    void init(int xS, int yS) {
        // Fill border with wall sentinels, interior with empty.
        std::memset(colors, C_WALL, sizeof(colors));
        for (int y = 0; y < yS; ++y) {
            for (int x = 0; x < xS; ++x) {
                colors[getLoc(x, y, xS)] = C_EMPTY;
            }
        }
    }
};

}  // namespace vct
}  // namespace skyzero

#endif
