#ifndef SKYZERO_RANDOM_OPENING_H
#define SKYZERO_RANDOM_OPENING_H

// KataGomo-style balanced opening generation.
//
// Ported from KataGomo-Gom2024/cpp/game/randomopening.cpp.
// See /home/sky/.claude/plans/katagomo-bright-teapot.md for the algorithm.
//
// Usage: construct with a Game reference, a single-state inference callback
// (same signature as ParallelMCTS::InferenceFn), an AlphaZeroConfig, and an
// RNG seed. Call initialize(state, to_play) to overwrite the starting
// position with a NN-fair random opening. initialize() always returns; on
// repeated failure it drops rejectProb and retries.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "alphazero.h"

namespace skyzero {

template <typename Game>
class RandomOpening {
public:
    using InferenceFn = std::function<
        std::pair<std::vector<float>, std::array<float, 3>>(const std::vector<int8_t>&)>;

    RandomOpening(const Game& game, InferenceFn infer_fn, const AlphaZeroConfig& cfg, uint64_t seed)
        : game_(game), infer_fn_(std::move(infer_fn)), cfg_(cfg), rng_(seed) {}

    // Overwrites `state` and `to_play` with a balanced-opening start position.
    void initialize(std::vector<int8_t>& state, int& to_play) {
        double reject_prob = cfg_.balanced_opening_reject_prob;
        int tries = 0;
        while (!try_once(state, to_play, reject_prob)) {
            ++tries;
            if (tries > cfg_.balanced_opening_max_tries) {
                tries = 0;
                reject_prob = cfg_.balanced_opening_reject_prob_fallback;
            }
        }
    }

private:
    // WDL-utility from `next_player`'s perspective: positive → next_player
    // is winning, |value| near 0 → balanced.
    double board_value(const std::vector<int8_t>& state, int next_player) const {
        const auto encoded = game_.encode_state(state, next_player);
        auto res = infer_fn_(encoded);
        const auto& v = res.second;
        return static_cast<double>(v[0]) - static_cast<double>(v[2]);
    }

    double rand_uniform() {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        return d(rng_);
    }

    bool rand_bool(double p) {
        if (p <= 0.0) return false;
        if (p >= 1.0) return true;
        return rand_uniform() < p;
    }

    double rand_exponential() {
        std::exponential_distribution<double> d(1.0);
        return d(rng_);
    }

    // Truncated normal: sample in (-1, 1). We use a simple rejection loop on
    // a Gaussian with stddev `stddev` (analogue of KataGomo's nextGaussianTruncated).
    double rand_truncated_gaussian(double stddev) {
        std::normal_distribution<double> d(0.0, stddev);
        for (int i = 0; i < 64; ++i) {
            const double x = d(rng_);
            if (x > -1.0 && x < 1.0) return x;
        }
        return 0.0;
    }

    // Center-biased random move on empty board; distance-weighted near
    // existing stones otherwise.
    int random_nearby_move(const std::vector<int8_t>& state, double avg_dist) {
        const int N = game_.board_size;
        const int area = N * N;
        const bool empty = std::all_of(state.begin(), state.end(), [](int8_t v) { return v == 0; });
        if (empty) {
            const double middle_bonus = 1.5;
            const double xd = rand_truncated_gaussian(middle_bonus * 0.999) / (2.0 * middle_bonus);
            const double yd = rand_truncated_gaussian(middle_bonus * 0.999) / (2.0 * middle_bonus);
            int x = static_cast<int>(std::lround(xd * N + 0.5 * (N - 1)));
            int y = static_cast<int>(std::lround(yd * N + 0.5 * (N - 1)));
            x = std::clamp(x, 0, N - 1);
            y = std::clamp(y, 0, N - 1);
            return y * N + x;
        }

        std::vector<double> prob(area, 0.0);
        const double middle_bonus_factor = 1.5;
        const double half_board_len = std::max(0.5 * (N - 1), 0.5 * (N - 1));
        for (int i1 = 0; i1 < area; ++i1) {
            if (state[i1] == 0) continue;
            const int x1 = i1 % N;
            const int y1 = i1 / N;
            for (int i2 = 0; i2 < area; ++i2) {
                if (state[i2] != 0) continue;
                const int x2 = i2 % N;
                const int y2 = i2 / N;
                const double dist_from_center = std::max(
                    std::abs(x2 - 0.5 * (N - 1)),
                    std::abs(y2 - 0.5 * (N - 1)));
                const double middle_bonus =
                    middle_bonus_factor * (half_board_len - dist_from_center) / half_board_len;
                const double d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
                    + avg_dist * avg_dist;
                prob[i2] += (1.0 + middle_bonus) * std::pow(d2, -2.0);
            }
        }

        double total = 0.0;
        for (double p : prob) total += p;
        if (total <= 0.0) {
            // Fallback: pick any empty square.
            for (int i = 0; i < area; ++i) {
                if (state[i] == 0) return i;
            }
            return 0;
        }
        const double r = (rand_uniform() - 1e-8) * total;
        double acc = 0.0;
        for (int i = 0; i < area; ++i) {
            acc += prob[i];
            if (acc >= r) return i;
        }
        for (int i = area - 1; i >= 0; --i) {
            if (prob[i] > 0.0) return i;
        }
        return 0;
    }

    // Returns a balance move, or -1 if the opening should be rejected.
    int balance_move(const std::vector<int8_t>& state, int next_player, double reject_prob) {
        const int area = game_.board_size * game_.board_size;
        const auto legal = game_.get_is_legal_actions(state, next_player);

        const double root_value_pla = board_value(state, next_player);
        if (root_value_pla < 0.0) {
            const double reject_factor = 1.0 - std::exp(-3.0 * root_value_pla * root_value_pla);
            if (rand_bool(reject_factor) && rand_bool(reject_prob)) return -1;
        }
        const double root_value_opp = board_value(state, -next_player);
        if (root_value_opp < 0.0) {
            const double reject_factor = 1.0 - std::exp(-3.0 * root_value_opp * root_value_opp);
            if (rand_bool(reject_factor) && rand_bool(reject_prob)) return -1;
        }

        std::vector<double> prob(area, 0.0);
        double max_prob = 0.0;
        for (int a = 0; a < area; ++a) {
            if (a >= static_cast<int>(legal.size()) || !legal[a]) continue;
            auto next_state = game_.get_next_state(state, a, next_player);
            if (game_.is_terminal(next_state, a, next_player)) continue;
            // Value of the resulting position, from the *opponent's* POV
            // (they are to move next). `val_opp` > 0 means next_player handed
            // a winning position to the opponent, i.e. next_player lost tempo.
            // `val_opp` near 0 means the position is balanced either way.
            const double val_opp = board_value(next_state, -next_player);
            const double p = std::pow(std::max(0.0, 1.0 - val_opp * val_opp), 4.0);
            prob[a] = p;
            if (p > max_prob) max_prob = p;
        }
        if (rand_bool(1.0 - max_prob) && rand_bool(reject_prob)) return -1;

        double total = 0.0;
        for (double p : prob) total += p;
        if (total <= 0.0) return -1;
        const double r = (rand_uniform() - 1e-8) * total;
        double acc = 0.0;
        for (int i = 0; i < area; ++i) {
            acc += prob[i];
            if (acc >= r) return i;
        }
        return -1;
    }

    bool try_once(std::vector<int8_t>& state, int& to_play, double reject_prob) {
        // KataGomo VCNRULE_NOVC weights for randomMoveNum in [0..11].
        static const std::array<float, 12> kMoveNumWeights =
            {10.0f, 30.0f, 50.0f, 80.0f, 60.0f, 40.0f, 20.0f, 10.0f, 5.0f, 1.0f, 0.0f, 0.0f};

        auto board = state;
        int player = to_play;

        // Sample randomMoveNum.
        double total = 0.0;
        for (float w : kMoveNumWeights) total += w;
        const double pick = rand_uniform() * total;
        double acc = 0.0;
        int random_move_num = 0;
        for (size_t i = 0; i < kMoveNumWeights.size(); ++i) {
            acc += kMoveNumWeights[i];
            if (acc >= pick) { random_move_num = static_cast<int>(i); break; }
        }

        const double avg_dist = rand_exponential() * cfg_.balanced_opening_avg_dist_factor;

        for (int i = 0; i < random_move_num; ++i) {
            const auto legal = game_.get_is_legal_actions(board, player);
            int a = random_nearby_move(board, avg_dist);
            if (a < 0 || a >= static_cast<int>(legal.size()) || !legal[a]) {
                // Pick any legal square as a fallback.
                a = -1;
                for (int j = 0; j < static_cast<int>(legal.size()); ++j) {
                    if (legal[j]) { a = j; break; }
                }
                if (a < 0) return false;
            }
            board = game_.get_next_state(board, a, player);
            if (game_.is_terminal(board, a, player)) return false;
            player = -player;
        }

        const int bmove = balance_move(board, player, reject_prob);
        if (bmove < 0) return false;
        board = game_.get_next_state(board, bmove, player);
        if (game_.is_terminal(board, bmove, player)) return false;
        player = -player;

        state = std::move(board);
        to_play = player;
        return true;
    }

    const Game& game_;
    InferenceFn infer_fn_;
    const AlphaZeroConfig& cfg_;
    std::mt19937 rng_;
};

}  // namespace skyzero

#endif
