#ifndef SKYZERO_POLICY_INIT_H
#define SKYZERO_POLICY_INIT_H

// KataGomo-style "initGamesWithPolicy" opening augmentation.
//
// Ported from KataGomo-Gom2024/cpp/program/playutils.cpp
// (PlayUtils::initializeGameUsingPolicy + getGameInitializationMove).
//
// After the balanced opening is generated (see random_opening.h), this class
// plays an extra ~Exp(1)*avg_move_num moves sampled from the NN policy
// raised to 1/temperature. The effect is to push the starting position off
// the narrow "NN sees both sides as balanced" plateau that balance_move
// selects for, which is what keeps value_target from collapsing into draw.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "alphazero.h"
#include "utils.h"

namespace skyzero {

template <typename Game>
class PolicyInit {
public:
    using InferenceFn = std::function<
        std::pair<std::vector<float>, std::array<float, 3>>(const std::vector<int8_t>&)>;

    PolicyInit(const Game& game, InferenceFn infer_fn, const AlphaZeroConfig& cfg, uint64_t seed)
        : game_(game), infer_fn_(std::move(infer_fn)), cfg_(cfg), rng_(seed) {}

    // Plays ~Exp(1)*avg_move_num policy-sampled moves on `state`/`to_play`,
    // minus 2x the number of stones already on the board (mirrors KataGomo's
    // `randomInitMovenumEquToPolicyInit = 2.0` credit for balance-opening moves).
    // Returns true if the loop completed without the game ending; false if a
    // sampled move terminated the game (caller should retry or drop).
    bool initialize(std::vector<int8_t>& state, int& to_play) {
        const double avg_move_num = static_cast<double>(cfg_.policy_init_avg_move_num);
        if (avg_move_num <= 0.0) return true;
        const double temperature = std::max(0.1, static_cast<double>(cfg_.policy_init_temperature));

        int board_movenum = 0;
        for (int8_t v : state) if (v != 0) ++board_movenum;

        const double random_init_movenum_equ_to_policy_init = 2.0;
        std::exponential_distribution<double> exp_dist(1.0);
        const double raw = exp_dist(rng_) * avg_move_num
            - random_init_movenum_equ_to_policy_init * static_cast<double>(board_movenum);
        int num_moves = static_cast<int>(std::floor(raw));
        if (num_moves <= 0) return true;

        for (int i = 0; i < num_moves; ++i) {
            const int action = sample_policy_move(state, to_play, temperature);
            if (action < 0) return true;  // no legal candidate with positive policy
            state = game_.get_next_state(state, action, to_play);
            if (game_.is_terminal(state, action, to_play)) return false;
            to_play = -to_play;
        }
        return true;
    }

private:
    int sample_policy_move(const std::vector<int8_t>& state, int to_play, double temperature) {
        const auto encoded = game_.encode_state(state, to_play);
        auto res = infer_fn_(encoded);
        std::vector<float> logits = std::move(res.first);

        const int area = game_.board_size * game_.board_size;
        const auto legal = game_.get_is_legal_actions(state, to_play);
        if (static_cast<int>(logits.size()) != area) return -1;
        for (int a = 0; a < area; ++a) {
            if (a >= static_cast<int>(legal.size()) || !legal[a]) {
                logits[a] = -std::numeric_limits<float>::infinity();
            }
        }
        const auto probs = softmax(logits);

        // Build selection values: probs^(1/T). Legal-only, skip zero-prob.
        std::vector<double> select(area, 0.0);
        std::vector<int> actions;
        actions.reserve(static_cast<size_t>(area));
        double total = 0.0;
        for (int a = 0; a < area; ++a) {
            const float p = probs[a];
            if (!(p > 0.0f)) continue;
            const double sv = std::pow(static_cast<double>(p), 1.0 / temperature);
            select[a] = sv;
            actions.push_back(a);
            total += sv;
        }
        if (actions.empty() || total <= 0.0) return -1;

        // KataGomo playutils.cpp:113 — with tiny probability, pick uniformly
        // from legal moves to add a bit of outlier variety.
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        if (u01(rng_) < 0.0002) {
            std::uniform_int_distribution<size_t> pick(0, actions.size() - 1);
            return actions[pick(rng_)];
        }

        const double r = u01(rng_) * total;
        double acc = 0.0;
        for (int a : actions) {
            acc += select[a];
            if (acc >= r) return a;
        }
        return actions.back();
    }

    const Game& game_;
    InferenceFn infer_fn_;
    const AlphaZeroConfig& cfg_;
    std::mt19937 rng_;
};

}  // namespace skyzero

#endif
