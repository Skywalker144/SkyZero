#ifndef SKYZERO_ENVS_GOMOKU_H
#define SKYZERO_ENVS_GOMOKU_H

// Ported from CSkyZero_V3/envs/gomoku.h. Opening-library logic removed;
// self-play starts either from an empty board or from a KataGomo-style
// balanced opening generated in random_opening.h.

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace skyzero {

// Rule variants for multi-rule training (V5).
//   FREESTYLE: no forbidden moves; any 5+ stones in a row wins for both colors.
//   STANDARD:  black wins only with exactly 5 (overline ≥6 forbidden, black loses);
//              white can win with any 5+; no three-three/four-four restriction.
//   RENJU:     black: exactly 5 wins; three-three / four-four / long-row forbidden;
//              white: any 5+ wins, no restriction.
enum class RuleType : int8_t {
    FREESTYLE = 0,
    STANDARD = 1,
    RENJU = 2,
};

inline RuleType rule_from_string(const std::string& s) {
    if (s == "freestyle") return RuleType::FREESTYLE;
    if (s == "standard")  return RuleType::STANDARD;
    if (s == "renju")     return RuleType::RENJU;
    throw std::runtime_error("unknown rule: " + s);
}

inline const char* rule_to_string(RuleType r) {
    switch (r) {
        case RuleType::FREESTYLE: return "freestyle";
        case RuleType::STANDARD:  return "standard";
        case RuleType::RENJU:     return "renju";
    }
    return "?";
}

// 12-dim global feature vector fed to KataGoNet.linear_global.
// dims 0-2: rule one-hot (freestyle, standard, renju)
// dim 3:    renju_color_sign (Renju+Black=-1, Renju+White=+1, else 0)
// dim 4:    has_forbidden (rule != FREESTYLE)
// dim 5:    ply / (board_size²)
// dims 6-11: VCF placeholder, zero (Phase C / Sprint 4 will fill via VCF solver)
struct GlobalFeatures {
    static constexpr int DIM = 12;
    float data[DIM];
    GlobalFeatures() { for (int i = 0; i < DIM; ++i) data[i] = 0.0f; }
};

struct GameInitialState {
    std::vector<int8_t> board;
    int to_play;
};

// Compile-time canvas size. Default 15; overridden by CMake via -D when
// scripts/run.cfg sets MAX_BOARD_SIZE (see cpp/CMakeLists.txt). Pattern
// borrowed from KataGomo's COMPILE_MAX_BOARD_LEN.
#ifndef SKYZERO_MAX_BOARD_SIZE
#define SKYZERO_MAX_BOARD_SIZE 15
#endif

class Gomoku {
public:
    static constexpr int MAX_BOARD_SIZE = SKYZERO_MAX_BOARD_SIZE;
    static constexpr int MAX_AREA = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
    static constexpr int NUM_SPATIAL_PLANES_V5 = 5;   // mask + own + opp + fb_b + fb_w

    int board_size;
    int num_planes;
    bool use_renju;                          // legacy: derived from rule
    bool enable_forbidden_point_plane;
    RuleType rule;

    // V5 constructor (preferred).
    Gomoku(int size, RuleType r, bool forbidden_plane)
        : board_size(size),
          num_planes(NUM_SPATIAL_PLANES_V5),
          use_renju(r == RuleType::RENJU),
          enable_forbidden_point_plane(forbidden_plane && r != RuleType::FREESTYLE),
          rule(r) {}

    // Legacy constructor (V4-compatible; bool renju → RuleType::RENJU else FREESTYLE).
    Gomoku(int size = 15, bool renju = true, bool forbidden_plane = true)
        : Gomoku(size, renju ? RuleType::RENJU : RuleType::FREESTYLE, forbidden_plane) {}

    GameInitialState get_initial_state(std::mt19937& /*rng*/) const {
        return {std::vector<int8_t>(board_size * board_size, 0), 1};
    }

    std::vector<uint8_t> get_is_legal_actions(const std::vector<int8_t>& state, int /*to_play*/) const {
        std::vector<uint8_t> legal(state.size(), 0);
        for (size_t i = 0; i < state.size(); ++i) {
            legal[i] = (state[i] == 0) ? 1 : 0;
        }
        // Forbidden points are legal moves for Black, but playing on one
        // results in an immediate loss (checked in get_winner).
        return legal;
    }

    // Transform a single board location under D4 symmetry.
    // sym ∈ [0,8): low 2 bits = number of 90° rotations (k), bit 2 = horizontal flip.
    // Mapping matches utils.h::transform_encoded_state: (r,c) source → (rr,cc) dest.
    int transform_loc(int loc, int sym) const {
        int r = loc / board_size;
        int c = loc % board_size;
        const int k = sym & 3;
        const bool do_flip = (sym & 4) != 0;
        int rr = r;
        int cc = c;
        for (int t = 0; t < k; ++t) {
            const int nr = board_size - 1 - cc;
            const int nc = rr;
            rr = nr;
            cc = nc;
        }
        if (do_flip) {
            cc = board_size - 1 - cc;
        }
        return rr * board_size + cc;
    }

    // Returns the D4 subgroup that leaves `state` invariant. Always contains 0 (identity).
    std::vector<int> get_board_symmetries(const std::vector<int8_t>& state) const {
        std::vector<int> out;
        out.push_back(0);
        const int n = board_size * board_size;
        for (int sym = 1; sym < 8; ++sym) {
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                if (state[i] != state[transform_loc(i, sym)]) {
                    ok = false;
                    break;
                }
            }
            if (ok) out.push_back(sym);
        }
        return out;
    }

    // Like get_is_legal_actions, but among each symmetry orbit only the
    // representative with the smallest loc index stays legal. When the board
    // has only the trivial symmetry, returns the original mask unchanged.
    std::vector<uint8_t> get_canonical_legal_actions(
        const std::vector<int8_t>& state, int to_play
    ) const {
        auto legal = get_is_legal_actions(state, to_play);
        auto syms = get_board_symmetries(state);
        if (syms.size() <= 1) {
            return legal;
        }
        const int n = board_size * board_size;
        for (int loc = 0; loc < n; ++loc) {
            if (!legal[loc]) continue;
            for (int sym : syms) {
                if (sym == 0) continue;
                const int mapped = transform_loc(loc, sym);
                if (mapped < loc) {
                    legal[loc] = 0;
                    break;
                }
            }
        }
        return legal;
    }

    std::vector<int8_t> get_next_state(const std::vector<int8_t>& state, int action, int to_play) const {
        auto next = state;
        next[action] = static_cast<int8_t>(to_play);
        return next;
    }

    int get_winner(const std::vector<int8_t>& state, int last_action = -1, int last_player = 0) const {
        if (use_renju && last_action >= 0 && last_player == 1) {
            const int row = last_action / board_size;
            const int col = last_action % board_size;
            ForbiddenPointFinder fpf(board_size);
            for (int i = 0; i < static_cast<int>(state.size()); ++i) {
                if (i == last_action || state[i] == 0) {
                    continue;
                }
                const int r = i / board_size;
                const int c = i % board_size;
                fpf.set_stone(r, c, state[i] == 1 ? C_BLACK : C_WHITE);
            }
            if (fpf.is_forbidden(row, col)) {
                return -1;
            }
        }

        const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                const int stone = state[r * board_size + c];
                if (stone == 0) {
                    continue;
                }
                for (const auto& d : dirs) {
                    const int pr = r - d[0];
                    const int pc = c - d[1];
                    if (on_board(pr, pc) && state[pr * board_size + pc] == stone) {
                        continue;
                    }
                    int len = 1;
                    int nr = r + d[0];
                    int nc = c + d[1];
                    while (on_board(nr, nc) && state[nr * board_size + nc] == stone) {
                        ++len;
                        nr += d[0];
                        nc += d[1];
                    }
                    if (stone == 1 && len == 5) {
                        return stone;  // Black wins with exactly 5 (overline is forbidden)
                    }
                    if (stone == -1 && len >= 5) {
                        return stone;  // White wins with 5 or more
                    }
                }
            }
        }

        if (std::all_of(state.begin(), state.end(), [](int8_t v) { return v != 0; })) {
            return 0;
        }
        return 2;
    }

    bool is_terminal(const std::vector<int8_t>& state, int last_action = -1, int last_player = 0) const {
        return get_winner(state, last_action, last_player) != 2;
    }

    // ============================================================
    // Canvas-stride API (KataGomo NNPos::locToPos style).
    // ============================================================
    // MCTS / NN code works in canvas-stride pos = r*MAX_BOARD_SIZE + c
    // (so policy logits / visit counts / legal masks have fixed size MAX_AREA),
    // while game-internal state stays board-stride loc = r*board_size + c.
    // The overloads below translate at the boundary, mirroring KataGomo's
    // NNPos::locToPos / posToLoc. off-board canvas positions:
    //   - in legal mask: false (never sampled by MCTS)
    //   - in canvas_pos_to_loc: returns -1 (caller must pre-mask to skip)
    static constexpr int loc_to_canvas_pos(int loc, int board_size_) {
        const int r = loc / board_size_;
        const int c = loc % board_size_;
        return r * MAX_BOARD_SIZE + c;
    }
    // Returns -1 if (r,c) is outside [0, board_size_). Caller is expected to
    // only feed canvas positions that legal mask marks true.
    static constexpr int canvas_pos_to_loc(int canvas_pos, int board_size_) {
        const int r = canvas_pos / MAX_BOARD_SIZE;
        const int c = canvas_pos % MAX_BOARD_SIZE;
        return (r < board_size_ && c < board_size_) ? r * board_size_ + c : -1;
    }

    // Canvas-stride legal mask, size = MAX_AREA. off-board cells = 0.
    std::vector<uint8_t> get_is_legal_actions_canvas(
        const std::vector<int8_t>& state, int to_play
    ) const {
        const auto legal_local = get_is_legal_actions(state, to_play);  // size N²
        std::vector<uint8_t> out(MAX_AREA, 0);
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                out[r * MAX_BOARD_SIZE + c] = legal_local[r * board_size + c];
            }
        }
        return out;
    }

    // Canvas-stride canonical legal mask (used by tree-parallel MCTS to dedupe
    // D4-symmetric root moves). Re-projects board-stride canonical mask onto
    // canvas. off-board cells = 0.
    std::vector<uint8_t> get_canonical_legal_actions_canvas(
        const std::vector<int8_t>& state, int to_play
    ) const {
        const auto legal_local = get_canonical_legal_actions(state, to_play);  // size N²
        std::vector<uint8_t> out(MAX_AREA, 0);
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                out[r * MAX_BOARD_SIZE + c] = legal_local[r * board_size + c];
            }
        }
        return out;
    }

    std::vector<int8_t> get_next_state_canvas(
        const std::vector<int8_t>& state, int canvas_pos, int to_play
    ) const {
        const int loc = canvas_pos_to_loc(canvas_pos, board_size);
        // off-board canvas positions should have been masked out by legal mask.
        // -1 here is a programming bug; return state unchanged so the caller's
        // game loop can detect via subsequent is_terminal_canvas check.
        if (loc < 0) return state;
        return get_next_state(state, loc, to_play);
    }

    int get_winner_canvas(
        const std::vector<int8_t>& state, int last_canvas_pos = -1, int last_player = 0
    ) const {
        const int last_loc = (last_canvas_pos >= 0)
            ? canvas_pos_to_loc(last_canvas_pos, board_size) : -1;
        return get_winner(state, last_loc, last_player);
    }

    bool is_terminal_canvas(
        const std::vector<int8_t>& state, int last_canvas_pos = -1, int last_player = 0
    ) const {
        return get_winner_canvas(state, last_canvas_pos, last_player) != 2;
    }

    std::vector<int8_t> encode_state(const std::vector<int8_t>& state, int to_play) const {
        const int area = board_size * board_size;
        std::vector<int8_t> encoded(num_planes * area, 0);
        // Plane 0: current player's stones
        // Plane 1: opponent's stones
        for (int i = 0; i < area; ++i) {
            encoded[i] = (state[i] == to_play) ? 1 : 0;
            encoded[area + i] = (state[i] == -to_play) ? 1 : 0;
        }

        // Plane 2: forbidden points when current player is Black (to_play == 1)
        // Plane 3: forbidden points when current player is White (to_play == -1)
        // The populated plane implicitly indicates whose turn it is.
        if (enable_forbidden_point_plane && use_renju) {
            ForbiddenPointFinder fpf(board_size);
            for (int i = 0; i < area; ++i) {
                if (state[i] == 0) {
                    continue;
                }
                fpf.set_stone(i / board_size, i % board_size, state[i] == 1 ? C_BLACK : C_WHITE);
            }
            const int forbidden_plane = (to_play == 1) ? 2 : 3;
            for (int i = 0; i < area; ++i) {
                if (state[i] != 0) {
                    continue;
                }
                const int r = i / board_size;
                const int c = i % board_size;
                encoded[forbidden_plane * area + i] = fpf.is_forbidden(r, c) ? 1 : 0;
            }
        }

        return encoded;
    }

    std::vector<int8_t> encode_state_batch(
        const std::vector<std::vector<int8_t>>& states,
        const std::vector<int8_t>& to_plays
    ) const {
        const int batch = static_cast<int>(states.size());
        const int area = board_size * board_size;
        std::vector<int8_t> out(batch * num_planes * area, 0);
        for (int b = 0; b < batch; ++b) {
            const int8_t tp = to_plays[b];
            const size_t base = static_cast<size_t>(b) * num_planes * area;
            for (int i = 0; i < area; ++i) {
                out[base + i] = (states[b][i] == tp) ? 1 : 0;
                out[base + area + i] = (states[b][i] == -tp) ? 1 : 0;
            }

            if (enable_forbidden_point_plane && use_renju) {
                ForbiddenPointFinder fpf(board_size);
                for (int i = 0; i < area; ++i) {
                    if (states[b][i] == 0) continue;
                    fpf.set_stone(i / board_size, i % board_size,
                                  states[b][i] == 1 ? C_BLACK : C_WHITE);
                }
                const int forbidden_plane = (tp == 1) ? 2 : 3;
                for (int i = 0; i < area; ++i) {
                    if (states[b][i] != 0) continue;
                    if (fpf.is_forbidden(i / board_size, i % board_size)) {
                        out[base + forbidden_plane * area + i] = 1;
                    }
                }
            }
        }
        return out;
    }

    // ============================================================
    // V5 API: 5-plane padded encode + global features + multi-rule winner
    // ============================================================

    // V5 encode: 5 planes, padded to MAX_BOARD_SIZE × MAX_BOARD_SIZE = 15×15.
    // Plane layout:
    //   0: on-board mask  (1 inside [0, board_size), 0 in padding)
    //   1: own stones     (1 where state[i] == to_play)
    //   2: opp stones     (1 where state[i] == -to_play)
    //   3: forbidden_black (1 where forbidden when current player is black)
    //   4: forbidden_white (1 where forbidden when current player is white)
    // Output stride is MAX_BOARD_SIZE = 15 regardless of board_size, so smaller
    // boards (e.g., 13×13) have the right/bottom rows/cols filled with 0
    // (mask=0 outside).
    std::vector<int8_t> encode_state_v5(const std::vector<int8_t>& state, int to_play) const {
        constexpr int M = MAX_BOARD_SIZE;
        constexpr int A = MAX_AREA;
        const int N = board_size;
        std::vector<int8_t> encoded(NUM_SPATIAL_PLANES_V5 * A, 0);

        // Plane 0: on-board mask
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                encoded[0 * A + r * M + c] = 1;
            }
        }

        // Planes 1-2: own / opp
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                const int8_t s = state[r * N + c];   // input stride = N (board_size)
                const int dst = r * M + c;           // output stride = M (15)
                if (s == to_play) encoded[1 * A + dst] = 1;
                else if (s == -to_play) encoded[2 * A + dst] = 1;
            }
        }

        // Planes 3-4: forbidden (only fill when rule has forbidden semantics)
        if (enable_forbidden_point_plane && rule != RuleType::FREESTYLE) {
            ForbiddenPointFinder fpf(N);
            for (int i = 0; i < N * N; ++i) {
                if (state[i] == 0) continue;
                fpf.set_stone(i / N, i % N, state[i] == 1 ? C_BLACK : C_WHITE);
            }
            const int fb_plane = (to_play == 1) ? 3 : 4;
            // STANDARD only forbids long-row for black; we still write that into
            // the same plane via FPF.is_forbidden, which checks all renju
            // patterns. For STANDARD, callers should ignore this plane content
            // for non-long-row patterns OR (simpler) train the network to
            // interpret the plane as a generic "forbidden hint" — the rule
            // one-hot in global features tells it which subset to weigh.
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    if (state[r * N + c] != 0) continue;
                    if (fpf.is_forbidden(r, c)) {
                        encoded[fb_plane * A + r * M + c] = 1;
                    }
                }
            }
        }

        return encoded;
    }

    // Batched V5 encode: shape (batch, 5, M, M) flattened.
    std::vector<int8_t> encode_state_v5_batch(
        const std::vector<std::vector<int8_t>>& states,
        const std::vector<int8_t>& to_plays
    ) const {
        const int batch = static_cast<int>(states.size());
        constexpr int per_sample = NUM_SPATIAL_PLANES_V5 * MAX_AREA;
        std::vector<int8_t> out(static_cast<size_t>(batch) * per_sample, 0);
        for (int b = 0; b < batch; ++b) {
            auto enc = encode_state_v5(states[b], to_plays[b]);
            std::copy(enc.begin(), enc.end(), out.begin() + b * per_sample);
        }
        return out;
    }

    // 12-dim global features (KataGoNet.linear_global input).
    GlobalFeatures compute_global_features(int ply, int to_play) const {
        GlobalFeatures g;   // zero-init
        g.data[0] = (rule == RuleType::FREESTYLE) ? 1.0f : 0.0f;
        g.data[1] = (rule == RuleType::STANDARD)  ? 1.0f : 0.0f;
        g.data[2] = (rule == RuleType::RENJU)     ? 1.0f : 0.0f;
        // dim 3: renju_color_sign (only fires under Renju, captures "black has tighter constraints")
        g.data[3] = (rule == RuleType::RENJU) ? (to_play == 1 ? -1.0f : +1.0f) : 0.0f;
        // dim 4: has_forbidden (RENJU or STANDARD both have forbidden semantics for black)
        g.data[4] = (rule != RuleType::FREESTYLE) ? 1.0f : 0.0f;
        // dim 5: ply normalized by actual board area (not MAX_AREA — different boards
        // have different game lengths, this gives a size-invariant progress signal)
        g.data[5] = float(ply) / float(board_size * board_size);
        // dims 6-11: VCF placeholder, all zero
        return g;
    }

    // V5 multi-rule winner check. Returns 1 (black wins), -1 (white wins),
    // 0 (draw, board full), 2 (game ongoing).
    // last_action / last_player describe the move just played; used for
    // forbidden-move detection (RENJU full check, STANDARD long-row only).
    int get_winner_v5(const std::vector<int8_t>& state, int last_action = -1, int last_player = 0) const {
        // Forbidden-move detection: if the last player was Black and the rule
        // forbids the move, Black loses immediately.
        if (last_action >= 0 && last_player == 1 && rule != RuleType::FREESTYLE) {
            const int row = last_action / board_size;
            const int col = last_action % board_size;
            if (rule == RuleType::RENJU) {
                ForbiddenPointFinder fpf(board_size);
                for (int i = 0; i < static_cast<int>(state.size()); ++i) {
                    if (i == last_action || state[i] == 0) continue;
                    fpf.set_stone(i / board_size, i % board_size,
                                  state[i] == 1 ? C_BLACK : C_WHITE);
                }
                if (fpf.is_forbidden(row, col)) return -1;   // black loses
            } else {  // STANDARD
                if (is_overline_at(state, row, col, /*color=*/1)) return -1;
            }
        }

        // Standard win check: scan for runs.
        const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                const int stone = state[r * board_size + c];
                if (stone == 0) continue;
                for (const auto& d : dirs) {
                    const int pr = r - d[0];
                    const int pc = c - d[1];
                    if (on_board(pr, pc) && state[pr * board_size + pc] == stone) continue;
                    int len = 1;
                    int nr = r + d[0];
                    int nc = c + d[1];
                    while (on_board(nr, nc) && state[nr * board_size + nc] == stone) {
                        ++len;
                        nr += d[0];
                        nc += d[1];
                    }
                    // Black: FREESTYLE wins on 5+, STANDARD/RENJU only on exactly 5.
                    // (overline already handled above; if we see len>=6 here for black under
                    // STANDARD/RENJU it means the overline check missed — should not happen)
                    if (stone == 1) {
                        if (rule == RuleType::FREESTYLE) {
                            if (len >= 5) return 1;
                        } else {
                            if (len == 5) return 1;
                        }
                    }
                    // White: always wins on 5+ (all rules).
                    if (stone == -1 && len >= 5) return -1;
                }
            }
        }

        if (std::all_of(state.begin(), state.end(), [](int8_t v) { return v != 0; })) {
            return 0;
        }
        return 2;
    }

private:
    // Check whether the stone at (r, c) of given color forms a run of length ≥ 6
    // through any of the 4 directions. Used for STANDARD overline detection.
    bool is_overline_at(const std::vector<int8_t>& state, int r, int c, int color) const {
        const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (const auto& d : dirs) {
            int len = 1;
            int nr = r + d[0], nc = c + d[1];
            while (on_board(nr, nc) && state[nr * board_size + nc] == color) {
                ++len; nr += d[0]; nc += d[1];
            }
            nr = r - d[0]; nc = c - d[1];
            while (on_board(nr, nc) && state[nr * board_size + nc] == color) {
                ++len; nr -= d[0]; nc -= d[1];
            }
            if (len >= 6) return true;
        }
        return false;
    }

    static constexpr int C_EMPTY = 0;
    static constexpr int C_BLACK = 1;
    static constexpr int C_WHITE = 2;
    static constexpr int C_WALL = 3;

    bool on_board(int r, int c) const {
        return r >= 0 && r < board_size && c >= 0 && c < board_size;
    }

    struct ForbiddenPointFinder {
        int size;
        std::vector<int> board;

        explicit ForbiddenPointFinder(int n) : size(n), board((n + 2) * (n + 2), C_WALL) {
            clear();
        }

        void clear() {
            for (int r = 1; r <= size; ++r) {
                for (int c = 1; c <= size; ++c) {
                    board[r * (size + 2) + c] = C_EMPTY;
                }
            }
        }

        void set_stone(int r, int c, int stone) {
            board[(r + 1) * (size + 2) + (c + 1)] = stone;
        }

        int get_stone(int r, int c) const {
            return board[(r + 1) * (size + 2) + (c + 1)];
        }

        std::array<int, 2> get_dir(int d) const {
            if (d == 1) return {1, 0};
            if (d == 2) return {0, 1};
            if (d == 3) return {1, 1};
            return {1, -1};
        }

        int check_line_length(int x, int y, int color, int d) const {
            const auto dir = get_dir(d);
            const int dx = dir[0];
            const int dy = dir[1];
            int len = 1;

            int i = x + dx;
            int j = y + dy;
            while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                ++len;
                i += dx;
                j += dy;
            }

            i = x - dx;
            j = y - dy;
            while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                ++len;
                i -= dx;
                j -= dy;
            }
            return len;
        }

        bool is_five(int x, int y, int color, int d = 0) {
            if (get_stone(x, y) != C_EMPTY) {
                return false;
            }
            set_stone(x, y, color);
            bool found = false;
            if (d == 0) {
                for (int k = 1; k <= 4; ++k) {
                    const int len = check_line_length(x, y, color, k);
                    if ((color == C_BLACK && len == 5) || (color == C_WHITE && len >= 5)) {
                        found = true;
                        break;
                    }
                }
            } else {
                const int len = check_line_length(x, y, color, d);
                found = (color == C_BLACK) ? (len == 5) : (len >= 5);
            }
            set_stone(x, y, C_EMPTY);
            return found;
        }

        bool is_overline(int x, int y) {
            if (get_stone(x, y) != C_EMPTY) {
                return false;
            }
            set_stone(x, y, C_BLACK);
            bool overline = false;
            for (int d = 1; d <= 4; ++d) {
                const int len = check_line_length(x, y, C_BLACK, d);
                if (len == 5) {
                    set_stone(x, y, C_EMPTY);
                    return false;
                }
                if (len >= 6) {
                    overline = true;
                }
            }
            set_stone(x, y, C_EMPTY);
            return overline;
        }

        int is_open_four(int x, int y, int color, int d) {
            if (get_stone(x, y) != C_EMPTY || is_five(x, y, color) || (color == C_BLACK && is_overline(x, y))) {
                return 0;
            }
            set_stone(x, y, color);
            const auto dir = get_dir(d);
            const int dx = dir[0];
            const int dy = dir[1];

            int nline = 1;
            int i = x - dx;
            int j = y - dy;
            while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                ++nline;
                i -= dx;
                j -= dy;
            }
            if (!(i >= 0 && i < size && j >= 0 && j < size) || get_stone(i, j) != C_EMPTY || !is_five(i, j, color, d)) {
                set_stone(x, y, C_EMPTY);
                return 0;
            }

            i = x + dx;
            j = y + dy;
            while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                ++nline;
                i += dx;
                j += dy;
            }
            if (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == C_EMPTY && is_five(i, j, color, d)) {
                set_stone(x, y, C_EMPTY);
                return (nline == 4) ? 1 : 2;
            }
            set_stone(x, y, C_EMPTY);
            return 0;
        }

        bool is_four(int x, int y, int color, int d) {
            if (get_stone(x, y) != C_EMPTY || is_five(x, y, color) || (color == C_BLACK && is_overline(x, y))) {
                return false;
            }
            set_stone(x, y, color);
            const auto dir = get_dir(d);
            const int dx = dir[0];
            const int dy = dir[1];
            bool found = false;
            for (int sign : {1, -1}) {
                int i = x + dx * sign;
                int j = y + dy * sign;
                while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                    i += dx * sign;
                    j += dy * sign;
                }
                if (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == C_EMPTY && is_five(i, j, color, d)) {
                    found = true;
                    break;
                }
            }
            set_stone(x, y, C_EMPTY);
            return found;
        }

        bool is_open_three(int x, int y, int color, int d) {
            if (is_five(x, y, color) || (color == C_BLACK && is_overline(x, y))) {
                return false;
            }
            set_stone(x, y, color);
            const auto dir = get_dir(d);
            const int dx = dir[0];
            const int dy = dir[1];
            bool found = false;
            for (int sign : {1, -1}) {
                int i = x + dx * sign;
                int j = y + dy * sign;
                while (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == color) {
                    i += dx * sign;
                    j += dy * sign;
                }
                if (i >= 0 && i < size && j >= 0 && j < size && get_stone(i, j) == C_EMPTY && is_open_four(i, j, color, d) == 1) {
                    if (color == C_BLACK) {
                        if (!is_double_four(i, j) && !is_double_three(i, j) && !is_overline(i, j)) {
                            found = true;
                            break;
                        }
                    } else {
                        found = true;
                        break;
                    }
                }
            }
            set_stone(x, y, C_EMPTY);
            return found;
        }

        bool is_double_four(int x, int y) {
            if (get_stone(x, y) != C_EMPTY || is_five(x, y, C_BLACK)) {
                return false;
            }
            int nfour = 0;
            for (int d = 1; d <= 4; ++d) {
                const int ret = is_open_four(x, y, C_BLACK, d);
                if (ret == 2) {
                    nfour += 2;
                } else if (ret == 1 || is_four(x, y, C_BLACK, d)) {
                    nfour += 1;
                }
            }
            return nfour >= 2;
        }

        bool is_double_three(int x, int y) {
            if (get_stone(x, y) != C_EMPTY || is_five(x, y, C_BLACK)) {
                return false;
            }
            int nthree = 0;
            for (int d = 1; d <= 4; ++d) {
                if (is_open_three(x, y, C_BLACK, d)) {
                    ++nthree;
                }
            }
            return nthree >= 2;
        }

        bool is_forbidden(int x, int y) {
            if (get_stone(x, y) != C_EMPTY) {
                return false;
            }
            int nearby_black = 0;
            for (int i = std::max(0, x - 2); i <= std::min(size - 1, x + 2); ++i) {
                for (int j = std::max(0, y - 2); j <= std::min(size - 1, y + 2); ++j) {
                    if (i == x && j == y) {
                        continue;
                    }
                    if (get_stone(i, j) == C_BLACK) {
                        const int xd = std::abs(i - x);
                        const int yd = std::abs(j - y);
                        if ((xd + yd) != 3) {
                            ++nearby_black;
                        }
                    }
                }
            }
            if (nearby_black < 2) {
                return false;
            }
            return is_double_three(x, y) || is_double_four(x, y) || is_overline(x, y);
        }
    };
};

}  // namespace skyzero

#endif
