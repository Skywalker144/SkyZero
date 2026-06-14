#ifndef SKYZERO_ENVS_GAME2048_H
#define SKYZERO_ENVS_GAME2048_H

// 2048 environment for Stochastic AlphaZero (afterstate formulation).
//
// Unlike the two-player, deterministic Gomoku env, 2048 is a single-agent
// stochastic MDP:
//   * one action = one of 4 slide directions (up / right / down / left),
//   * applying a slide is DETERMINISTIC and yields an "afterstate" plus a
//     reward (the sum of the values of tiles created by merges this move),
//   * the environment then spawns one random tile (value 2 w.p. 0.9, value 4
//     w.p. 0.1) in a uniformly-random empty cell, producing the next state.
//
// The MCTS that consumes this env (cpp/skyzero_2048.h, written in M2) must
// therefore distinguish DECISION nodes (player picks a direction) from CHANCE
// nodes (env spawns a tile). This header only provides the game rules; it does
// not depend on LibTorch so it can be unit-tested with plain g++.
//
// State encoding: a length-16 vector of int8_t EXPONENTS, row-major
// (loc = r*SIZE + c). 0 means empty; e>0 means a tile of value 2^e. So tile
// "2" is exponent 1, "2048" is exponent 11, "32768" is exponent 15.

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace skyzero {

class Game2048 {
public:
    static constexpr int SIZE = 4;
    static constexpr int AREA = SIZE * SIZE;   // 16 cells
    static constexpr int NUM_ACTIONS = 4;      // 0=up, 1=right, 2=down, 3=left

    // One-hot exponent planes 0..NUM_PLANES-1 (plane 0 == empty cell). Caps the
    // representable tile at 2^(NUM_PLANES-1) = 32768; exponents above the cap
    // are clamped into the top plane in encode_state.
    static constexpr int NUM_PLANES = 16;

    // Tile-spawn probabilities (standard 2048).
    static constexpr double PROB_2 = 0.9;   // spawns exponent 1 (tile 2)
    static constexpr double PROB_4 = 0.1;   // spawns exponent 2 (tile 4)

    struct MoveResult {
        std::vector<int8_t> afterstate;  // board after the deterministic slide+merge
        int reward = 0;                  // sum of values of tiles created by merges
        bool changed = false;            // false => illegal move (board unchanged)
    };

    // One possible random spawn from an afterstate.
    struct SpawnOutcome {
        int cell;       // loc of the empty cell that gets filled
        int exp;        // 1 (tile 2) or 2 (tile 4)
        double prob;    // joint probability over (cell, value)
    };

    // ---- Initial state: two random tiles. ----
    std::vector<int8_t> get_initial_state(std::mt19937& rng) const {
        std::vector<int8_t> state(AREA, 0);
        spawn_tile_inplace(state, rng);
        spawn_tile_inplace(state, rng);
        return state;
    }

    // ---- Legal actions: a direction is legal iff sliding changes the board. ----
    std::vector<uint8_t> get_legal_actions(const std::vector<int8_t>& state) const {
        std::vector<uint8_t> legal(NUM_ACTIONS, 0);
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            legal[a] = apply_move(state, a).changed ? 1 : 0;
        }
        return legal;
    }

    // Terminal iff no direction produces a change (board full and no merges).
    bool is_terminal(const std::vector<int8_t>& state) const {
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            if (apply_move(state, a).changed) return false;
        }
        return true;
    }

    // ---- Deterministic slide + merge (the afterstate transition). ----
    // Each of the 4 lines orthogonal to the move is compressed toward the
    // moving edge; equal adjacent tiles merge once (exponent +1), scoring the
    // value of the newly created tile (2^(e+1)).
    MoveResult apply_move(const std::vector<int8_t>& state, int action) const {
        MoveResult res;
        res.afterstate = state;
        for (int line = 0; line < SIZE; ++line) {
            std::array<int, SIZE> idx = line_indices(action, line);
            std::array<int8_t, SIZE> in{};
            for (int i = 0; i < SIZE; ++i) in[i] = state[idx[i]];

            std::array<int8_t, SIZE> out{};
            const int line_reward = slide_line(in, out);
            res.reward += line_reward;

            for (int i = 0; i < SIZE; ++i) {
                if (res.afterstate[idx[i]] != out[i]) res.changed = true;
                res.afterstate[idx[i]] = out[i];
            }
        }
        if (!res.changed) {
            res.afterstate = state;  // keep identity on illegal move
            res.reward = 0;
        }
        return res;
    }

    // ---- All (cell, value, prob) spawns from an afterstate. ----
    std::vector<SpawnOutcome> spawn_distribution(const std::vector<int8_t>& afterstate) const {
        std::vector<int> empties;
        for (int i = 0; i < AREA; ++i) if (afterstate[i] == 0) empties.push_back(i);
        std::vector<SpawnOutcome> out;
        if (empties.empty()) return out;
        const double per_cell = 1.0 / static_cast<double>(empties.size());
        out.reserve(empties.size() * 2);
        for (int cell : empties) {
            out.push_back({cell, 1, per_cell * PROB_2});
            out.push_back({cell, 2, per_cell * PROB_4});
        }
        return out;
    }

    // Sample one spawn (used to step the real game during self-play).
    std::vector<int8_t> spawn_random(const std::vector<int8_t>& afterstate, std::mt19937& rng) const {
        std::vector<int8_t> next = afterstate;
        spawn_tile_inplace(next, rng);
        return next;
    }

    // ---- NN input: NUM_PLANES one-hot exponent planes, each SIZE×SIZE. ----
    // encoded[e*AREA + loc] == 1 where cell `loc` has exponent e (clamped to
    // the top plane). Every cell sets exactly one plane (plane 0 = empty).
    std::vector<int8_t> encode_state(const std::vector<int8_t>& state) const {
        std::vector<int8_t> encoded(static_cast<size_t>(NUM_PLANES) * AREA, 0);
        for (int loc = 0; loc < AREA; ++loc) {
            int e = state[loc];
            if (e >= NUM_PLANES) e = NUM_PLANES - 1;
            encoded[static_cast<size_t>(e) * AREA + loc] = 1;
        }
        return encoded;
    }

    // Highest exponent on the board (0 if empty). For logging / metrics.
    int max_tile_exp(const std::vector<int8_t>& state) const {
        int m = 0;
        for (int8_t v : state) m = std::max(m, static_cast<int>(v));
        return m;
    }

    // ---- Dihedral D4 transforms (inference-time stochastic symmetry). ----
    // A transform_type in [0,7] is (k = type%4 quarter-rotations, flip = type>=4).
    // One rotation maps cell (r,c) -> (N-1-c, r) (90° CCW); a flip then maps
    // c -> N-1-c. transform_encoded applies this per one-hot plane (the NN input).
    //
    // ACTION_PERM is the matching relabel of the 4 slide directions (0=up,1=right,
    // 2=down,3=left): action `a` on a board equals action ACTION_PERM[type][a] on
    // that board transformed by `type`. So to undo the transform on NN logits:
    //   orig_logits[a] = nn_logits[ACTION_PERM[type][a]].
    // Rotating the planes WITHOUT relabeling actions is the classic 2048 trap;
    // this table is VERIFIED equivariant against apply_move in the unit test.
    static constexpr int ACTION_PERM[8][4] = {
        {0, 1, 2, 3}, {3, 0, 1, 2}, {2, 3, 0, 1}, {1, 2, 3, 0},
        {0, 3, 2, 1}, {1, 0, 3, 2}, {2, 1, 0, 3}, {3, 2, 1, 0},
    };

    // Single-plane raw-board dihedral transform (k rotations then optional flip).
    static std::vector<int8_t> transform_board(const std::vector<int8_t>& b, int k, bool flip) {
        std::vector<int8_t> out(AREA, 0);
        for (int r = 0; r < SIZE; ++r) {
            for (int c = 0; c < SIZE; ++c) {
                int rr = r, cc = c;
                for (int t = 0; t < k; ++t) { int nr = SIZE - 1 - cc, nc = rr; rr = nr; cc = nc; }
                if (flip) cc = SIZE - 1 - cc;
                out[rr * SIZE + cc] = b[r * SIZE + c];
            }
        }
        return out;
    }

    // Same dihedral transform applied to an encoded state (NUM_PLANES planes).
    static std::vector<int8_t> transform_encoded(const std::vector<int8_t>& enc, int k, bool flip) {
        std::vector<int8_t> out(enc.size(), 0);
        for (int p = 0; p < NUM_PLANES; ++p) {
            const size_t base = static_cast<size_t>(p) * AREA;
            for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                    int rr = r, cc = c;
                    for (int t = 0; t < k; ++t) { int nr = SIZE - 1 - cc, nc = rr; rr = nr; cc = nc; }
                    if (flip) cc = SIZE - 1 - cc;
                    out[base + static_cast<size_t>(rr) * SIZE + cc] =
                        enc[base + static_cast<size_t>(r) * SIZE + c];
                }
            }
        }
        return out;
    }

private:
    // Map (action, line) to the 4 board locs ordered from the moving edge
    // inward, so sliding is always "toward index 0 of the returned array".
    //   action 0=up:    column `line`, rows 0..3
    //   action 1=right: row `line`, cols 3..0
    //   action 2=down:  column `line`, rows 3..0
    //   action 3=left:  row `line`, cols 0..3
    static std::array<int, SIZE> line_indices(int action, int line) {
        std::array<int, SIZE> idx{};
        for (int i = 0; i < SIZE; ++i) {
            int r = 0, c = 0;
            switch (action) {
                case 0: r = i;            c = line;         break;  // up
                case 1: r = line;         c = SIZE - 1 - i; break;  // right
                case 2: r = SIZE - 1 - i; c = line;         break;  // down
                case 3: r = line;         c = i;            break;  // left
            }
            idx[i] = r * SIZE + c;
        }
        return idx;
    }

    // Compress + merge a single line toward index 0. Returns reward (sum of
    // values of created tiles). Each tile participates in at most one merge.
    static int slide_line(const std::array<int8_t, SIZE>& in, std::array<int8_t, SIZE>& out) {
        std::array<int8_t, SIZE> packed{};
        int n = 0;
        for (int i = 0; i < SIZE; ++i) if (in[i] != 0) packed[n++] = in[i];

        out = {};
        int w = 0;
        int reward = 0;
        for (int i = 0; i < n; ++i) {
            if (i + 1 < n && packed[i] == packed[i + 1]) {
                const int8_t merged = static_cast<int8_t>(packed[i] + 1);
                out[w++] = merged;
                reward += (1 << merged);  // value of the newly created tile
                ++i;                      // consume the partner; no double-merge
            } else {
                out[w++] = packed[i];
            }
        }
        return reward;
    }

    // Place a single tile (exponent 1 w.p. 0.9, exponent 2 w.p. 0.1) in a
    // uniformly-random empty cell. No-op if the board is full.
    void spawn_tile_inplace(std::vector<int8_t>& state, std::mt19937& rng) const {
        std::vector<int> empties;
        for (int i = 0; i < AREA; ++i) if (state[i] == 0) empties.push_back(i);
        if (empties.empty()) return;
        std::uniform_int_distribution<int> cell_dist(0, static_cast<int>(empties.size()) - 1);
        std::uniform_real_distribution<double> val_dist(0.0, 1.0);
        const int cell = empties[cell_dist(rng)];
        state[cell] = (val_dist(rng) < PROB_4) ? 2 : 1;
    }
};

}  // namespace skyzero

#endif
