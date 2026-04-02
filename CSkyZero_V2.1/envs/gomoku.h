#ifndef SKYZERO_ENVS_GOMOKU_H
#define SKYZERO_ENVS_GOMOKU_H

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace skyzero {

struct GameInitialState {
    std::vector<int8_t> board;
    int to_play;
};

class Gomoku {
public:
    int board_size;
    int num_planes;
    bool use_renju;
    bool enable_forbidden_point_plane;

    struct Opening {
        std::vector<int8_t> board;
        int to_play;
    };
    std::vector<Opening> openings_;
    std::vector<float> opening_weights_;
    float empty_board_prob_ = 1.0f;

    Gomoku(int size = 15, bool renju = true, bool forbidden_plane = true)
        : board_size(size),
          num_planes(3 + (forbidden_plane ? 1 : 0)),
          use_renju(renju),
          enable_forbidden_point_plane(forbidden_plane) {}

    void load_openings(const std::string& path, float empty_board_prob = 0.0f) {
        empty_board_prob_ = empty_board_prob;
        std::ifstream fs(path);
        if (!fs.is_open()) {
            std::cerr << "Warning: Could not open opening file: " << path << ". Using empty board only.\n";
            empty_board_prob_ = 1.0f;
            return;
        }

        struct RawOpening {
            std::string weight_str;
            std::string black_str;
            std::string white_str;
        };
        std::vector<RawOpening> raw_list;
        std::string line;
        while (std::getline(fs, line)) {
            RawOpening ro;
            ro.weight_str = line;
            if (!std::getline(fs, ro.black_str)) break;
            if (!std::getline(fs, ro.white_str)) break;
            raw_list.push_back(ro);
        }

        float sum_explicit = 0.0f;
        int num_implicit = 0;
        for (const auto& ro : raw_list) {
            if (ro.weight_str.empty() || std::all_of(ro.weight_str.begin(), ro.weight_str.end(), [](unsigned char ch){ return std::isspace(ch); })) {
                num_implicit++;
            } else {
                sum_explicit += std::stof(ro.weight_str);
            }
        }

        float implicit_weight = 0.0f;
        if (num_implicit > 0) {
            implicit_weight = std::max(0.0f, (1.0f - sum_explicit) / num_implicit);
        }

        const int center = board_size / 2;
        for (const auto& ro : raw_list) {
            Opening op;
            op.board.assign(board_size * board_size, 0);
            int stones = 0;

            auto place = [&](const std::string& s, int color) {
                std::stringstream ss(s);
                std::string coord;
                while (ss >> coord) {
                    size_t comma = coord.find(',');
                    if (comma != std::string::npos) {
                        int dx = std::stoi(coord.substr(0, comma));
                        int dy = std::stoi(coord.substr(comma + 1));
                        int r = center + dx;
                        int c = center + dy;
                        if (r >= 0 && r < board_size && c >= 0 && c < board_size) {
                            op.board[r * board_size + c] = static_cast<int8_t>(color);
                            stones++;
                        }
                    }
                }
            };

            place(ro.black_str, 1);
            place(ro.white_str, -1);
            op.to_play = (stones % 2 == 0) ? 1 : -1;
            openings_.push_back(op);

            float w = 0.0f;
            if (ro.weight_str.empty() || std::all_of(ro.weight_str.begin(), ro.weight_str.end(), [](unsigned char ch){ return std::isspace(ch); })) {
                w = implicit_weight;
            } else {
                w = std::stof(ro.weight_str);
            }
            opening_weights_.push_back(w);
        }

        if (openings_.empty()) {
            empty_board_prob_ = 1.0f;
        }
    }

    GameInitialState get_initial_state(std::mt19937& rng) const {
        if (openings_.empty()) {
            return {std::vector<int8_t>(board_size * board_size, 0), 1};
        }

        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        if (uni(rng) < empty_board_prob_) {
            return {std::vector<int8_t>(board_size * board_size, 0), 1};
        }

        std::discrete_distribution<int> dist(opening_weights_.begin(), opening_weights_.end());
        const auto& op = openings_[dist(rng)];
        return {op.board, op.to_play};
    }

    std::vector<uint8_t> get_is_legal_actions(const std::vector<int8_t>& state, int to_play) const {
        std::vector<uint8_t> legal(state.size(), 0);
        const bool empty = std::all_of(state.begin(), state.end(), [](int8_t v) { return v == 0; });
        if (empty) {
            // legal.assign(state.size(), 1);
            int center_r = board_size / 2;
            int center_c = board_size / 2;
            int center_loc = center_r * board_size + center_c;
            legal[center_loc] = 1;
            return legal;
        }

        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                const int loc = r * board_size + c;
                if (state[loc] != 0) {
                    continue;
                }
                legal[loc] = is_near_occupied(state, r, c, 3) ? 1 : 0;
            }
        }

        // Forbidden points are legal moves for Black, but playing on one
        // results in an immediate loss (checked in get_winner).

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
            // Plane 0: current player's stones
            // Plane 1: opponent's stones
            for (int i = 0; i < area; ++i) {
                out[base + i] = (states[b][i] == tp) ? 1 : 0;
                out[base + area + i] = (states[b][i] == -tp) ? 1 : 0;
            }

            // Plane 2: forbidden points when current player is Black (tp == 1)
            // Plane 3: forbidden points when current player is White (tp == -1)
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

private:
    static constexpr int C_EMPTY = 0;
    static constexpr int C_BLACK = 1;
    static constexpr int C_WHITE = 2;
    static constexpr int C_WALL = 3;

    bool on_board(int r, int c) const {
        return r >= 0 && r < board_size && c >= 0 && c < board_size;
    }

    bool is_near_occupied(const std::vector<int8_t>& state, int r, int c, int dist) const {
        for (int dr = -dist; dr <= dist; ++dr) {
            for (int dc = -dist; dc <= dist; ++dc) {
                const int nr = r + dr;
                const int nc = c + dc;
                if (!on_board(nr, nc)) {
                    continue;
                }
                if (state[nr * board_size + nc] != 0) {
                    return true;
                }
            }
        }
        return false;
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
