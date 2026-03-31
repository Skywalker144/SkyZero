#ifndef SKYZERO_ENVS_TICTACTOE_H
#define SKYZERO_ENVS_TICTACTOE_H

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace skyzero {

struct GameInitialState {
    std::vector<int8_t> board;
    int to_play;
};

class TicTacToe {
public:
    int board_size = 3;
    int num_planes = 3;

    GameInitialState get_initial_state(std::mt19937& /*rng*/) const {
        return {std::vector<int8_t>(board_size * board_size, 0), 1};
    }

    std::vector<uint8_t> get_is_legal_actions(const std::vector<int8_t>& state, int /*to_play*/) const {
        std::vector<uint8_t> legal(state.size(), 0);
        for (size_t i = 0; i < state.size(); ++i) {
            legal[i] = (state[i] == 0) ? 1 : 0;
        }
        return legal;
    }

    std::vector<int8_t> get_next_state(const std::vector<int8_t>& state, int action, int to_play) const {
        auto next = state;
        next[action] = static_cast<int8_t>(to_play);
        return next;
    }

    int get_winner(const std::vector<int8_t>& state, int /*last_action*/ = -1, int /*last_player*/ = 0) const {
        for (int r = 0; r < 3; ++r) {
            const int a = state[r * 3 + 0];
            if (a != 0 && a == state[r * 3 + 1] && a == state[r * 3 + 2]) {
                return a;
            }
        }
        for (int c = 0; c < 3; ++c) {
            const int a = state[c];
            if (a != 0 && a == state[c + 3] && a == state[c + 6]) {
                return a;
            }
        }
        if (state[0] != 0 && state[0] == state[4] && state[0] == state[8]) {
            return state[0];
        }
        if (state[2] != 0 && state[2] == state[4] && state[2] == state[6]) {
            return state[2];
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

        for (int i = 0; i < area; ++i) {
            encoded[i] = (state[i] == to_play) ? 1 : 0;
            encoded[area + i] = (state[i] == -to_play) ? 1 : 0;
            encoded[2 * area + i] = (to_play > 0) ? 1 : 0;
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
            for (int i = 0; i < area; ++i) {
                const size_t base = static_cast<size_t>(b) * num_planes * area;
                out[base + i] = (states[b][i] == tp) ? 1 : 0;
                out[base + area + i] = (states[b][i] == -tp) ? 1 : 0;
                out[base + 2 * area + i] = (tp > 0) ? 1 : 0;
            }
        }
        return out;
    }
};

}  // namespace skyzero

#endif
