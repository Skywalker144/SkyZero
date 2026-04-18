#ifndef SKYZERO_RANDOM_OPENING_H
#define SKYZERO_RANDOM_OPENING_H

// Online balanced opening generator for Gomoku (AlphaZero self-play).
//
// Adapted from KataGomo-Gom2024's randomopening.cpp.
//
// Algorithm per game:
//   1. Sample move count N from weighted distribution.
//   2. Place N random stones using distance-weighted scatter.
//   3. Pick one balanced move via NN evaluation: score = (1 - V^2)^k.
//   4. If too unbalanced, retry (up to max_retries).

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "envs/gomoku.h"

namespace skyzero {

struct RandomOpeningConfig {
    bool enabled = false;
    int min_moves = 0;                   // KataGomo NOVC: distribution starts at 0
    int max_moves = 11;                  // KataGomo NOVC: distribution length = 12
    float balance_power = 4.0f;          // exponent k in (1 - V^2)^k (selfplay)
    float reject_prob = 0.995f;          // initial probabilistic rejection strength
    float reject_prob_fallback = 0.8f;   // fallback after max_retries (loop continues)
    int max_retries = 20;                // retries before reject_prob decays to fallback
    int nearby_dist = 3;                 // Chebyshev distance for "near existing stones"
};

// Global statistics (thread-safe)
namespace opening_stats {
    inline std::atomic<int64_t> tried_count{0};
    inline std::atomic<int64_t> succeed_count{0};
    inline std::atomic<int64_t> eval_count{0};
}

class RandomOpeningGenerator {
public:
    using InferenceFn = std::function<
        std::pair<std::vector<float>, std::array<float, 4>>(const std::vector<int8_t>&)>;
    using BatchInferenceFn = std::function<
        std::vector<std::pair<std::vector<float>, std::array<float, 4>>>(
            const std::vector<std::vector<int8_t>>&)>;

    RandomOpeningGenerator(const Gomoku& game, const RandomOpeningConfig& cfg)
        : game_(game), cfg_(cfg), area_(game.board_size * game.board_size) {}

    GameInitialState generate(
        std::mt19937& rng,
        const InferenceFn& infer_fn,
        const BatchInferenceFn& batch_infer_fn
    ) {
        // KataGomo-style retry with rejectProb decay, plus hard bail-out.
        double reject_prob = cfg_.reject_prob;
        int tries = 0;
        int total_tries = 0;
        const int absolute_max = std::max(1, cfg_.max_retries) * 10;
        while (true) {
            opening_stats::tried_count.fetch_add(1, std::memory_order_relaxed);
            ++total_tries;

            // Phase 1: sample move count
            int num_moves = sample_move_count(rng);

            // Phase 2: scatter random nearby moves
            std::vector<int8_t> board(area_, 0);
            int to_play = 1;  // Black first
            int last_action = -1;
            int last_player = 0;

            bool ok = scatter_moves(board, to_play, last_action, last_player,
                                    num_moves, rng);
            if (ok) {
                // Hard bail: if even the relaxed filter can't pass, skip the
                // balance check and return the scatter result. Prevents the
                // infinite retry loop when NN is heavily offense-biased.
                if (total_tries >= absolute_max) {
                    opening_stats::succeed_count.fetch_add(1, std::memory_order_relaxed);
                    return {std::move(board), to_play};
                }

                // Phase 3: pick one balanced move via NN evaluation
                int balanced_loc = pick_balanced_move(
                    board, to_play, last_action, last_player,
                    reject_prob, rng, infer_fn, batch_infer_fn);
                if (balanced_loc >= 0) {
                    board[balanced_loc] = static_cast<int8_t>(to_play);
                    int next_player = -to_play;
                    if (!game_.is_terminal(board, balanced_loc, to_play)) {
                        opening_stats::succeed_count.fetch_add(1, std::memory_order_relaxed);
                        return {std::move(board), next_player};
                    }
                }
            }

            if (++tries > cfg_.max_retries) {
                tries = 0;
                reject_prob = cfg_.reject_prob_fallback;
            }
        }
    }

private:
    const Gomoku& game_;
    RandomOpeningConfig cfg_;
    int area_;

    // Weighted distribution biased toward mid-range move counts (for Renju)
    int sample_move_count(std::mt19937& rng) {
        // KataGomo NOVC distribution (peak at 3 moves), indices 0..11
        static const std::vector<float> weights = {
            10.0f, 30.0f, 50.0f, 80.0f, 60.0f, 40.0f,
            20.0f, 10.0f,  5.0f,  1.0f,  0.0f,  0.0f
        };

        int max_idx = std::min(static_cast<int>(weights.size()),
                               cfg_.max_moves + 1);
        int min_idx = std::min(cfg_.min_moves, max_idx);

        // Build sub-distribution [min_moves, max_moves]
        std::vector<float> sub_weights(weights.begin() + min_idx,
                                       weights.begin() + max_idx);
        if (sub_weights.empty()) return cfg_.min_moves;

        std::discrete_distribution<int> dist(sub_weights.begin(), sub_weights.end());
        return min_idx + dist(rng);
    }

    // Place num_moves stones using distance-weighted random scatter
    bool scatter_moves(
        std::vector<int8_t>& board, int& to_play,
        int& last_action, int& last_player,
        int num_moves, std::mt19937& rng
    ) {
        const int bs = game_.board_size;
        std::exponential_distribution<double> exp_dist(1.0);
        double avg_dist = exp_dist(rng) * 0.8;

        for (int m = 0; m < num_moves; ++m) {
            int loc = sample_nearby_move(board, rng, avg_dist);
            if (loc < 0) return false;

            // Check forbidden: for Black under Renju, placing on a forbidden
            // point means immediate loss. We skip such positions in scatter.
            if (game_.use_renju && to_play == 1) {
                auto test_board = board;
                test_board[loc] = 1;
                int winner = game_.get_winner(test_board, loc, 1);
                if (winner == -1) {
                    // Forbidden point — retry this move with different location
                    // Try a few times, then give up on this opening
                    bool found = false;
                    for (int retry = 0; retry < 10; ++retry) {
                        loc = sample_nearby_move(board, rng, avg_dist);
                        if (loc < 0) break;
                        test_board = board;
                        test_board[loc] = 1;
                        winner = game_.get_winner(test_board, loc, 1);
                        if (winner != -1) { found = true; break; }
                    }
                    if (!found) return false;
                }
            }

            board[loc] = static_cast<int8_t>(to_play);
            last_action = loc;
            last_player = to_play;

            // Check terminal
            if (game_.is_terminal(board, last_action, last_player)) {
                return false;
            }

            to_play = -to_play;
        }
        return true;
    }

    // Sample a random empty position, weighted by proximity to existing stones
    // and center bias. For empty board, uses Gaussian around center.
    int sample_nearby_move(
        const std::vector<int8_t>& board,
        std::mt19937& rng,
        double avg_dist
    ) {
        const int bs = game_.board_size;
        const bool is_empty = std::all_of(board.begin(), board.end(),
                                          [](int8_t v) { return v == 0; });

        if (is_empty) {
            // Gaussian around center. σ=3 matches KataGomo's effective σ on 15x15.
            std::normal_distribution<double> gauss(0.0, 3.0);
            for (int tries = 0; tries < 50; ++tries) {
                double xd = gauss(rng);
                double yd = gauss(rng);
                int r = static_cast<int>(std::round(xd + 0.5 * (bs - 1)));
                int c = static_cast<int>(std::round(yd + 0.5 * (bs - 1)));
                if (r >= 0 && r < bs && c >= 0 && c < bs) {
                    return r * bs + c;
                }
            }
            // Fallback: center
            return (bs / 2) * bs + (bs / 2);
        }

        // Distance-weighted probability: for each empty cell, sum over all
        // occupied cells: (1 + middleBonus) / (dist^2 + avgDist^2)^2
        std::vector<double> prob(area_, 0.0);
        const double half_board = std::max(0.5 * (bs - 1), 0.5);
        const double avg_dist_sq = avg_dist * avg_dist;

        for (int r2 = 0; r2 < bs; ++r2) {
            for (int c2 = 0; c2 < bs; ++c2) {
                const int loc2 = r2 * bs + c2;
                if (board[loc2] != 0) continue;

                double dist_from_center = std::max(
                    std::abs(r2 - 0.5 * (bs - 1)),
                    std::abs(c2 - 0.5 * (bs - 1)));
                double middle_bonus = 1.5 * (half_board - dist_from_center) / half_board;

                for (int r1 = 0; r1 < bs; ++r1) {
                    for (int c1 = 0; c1 < bs; ++c1) {
                        if (board[r1 * bs + c1] == 0) continue;
                        double dx = r2 - r1;
                        double dy = c2 - c1;
                        double dist_sq = dx * dx + dy * dy;
                        prob[loc2] += (1.0 + middle_bonus) /
                                      ((dist_sq + avg_dist_sq) * (dist_sq + avg_dist_sq));
                    }
                }
            }
        }

        double total = 0.0;
        for (double p : prob) total += p;
        if (total < 1e-30) return -1;

        std::uniform_real_distribution<double> uni(0.0, total);
        double r = uni(rng);
        double cumsum = 0.0;
        for (int i = 0; i < area_; ++i) {
            cumsum += prob[i];
            if (cumsum >= r) return i;
        }
        return area_ - 1;
    }

    // Evaluate all candidate moves and pick one with balanced value.
    // Uses batch NN inference for efficiency.
    int pick_balanced_move(
        const std::vector<int8_t>& board, int to_play,
        int last_action, int last_player,
        double reject_prob,
        std::mt19937& rng,
        const InferenceFn& infer_fn,
        const BatchInferenceFn& batch_infer_fn
    ) {
        const int bs = game_.board_size;
        const int next_player = -to_play;

        // --- Phase A: evaluate root value from both perspectives (2 inferences)
        std::vector<std::vector<int8_t>> root_batch;
        root_batch.push_back(game_.encode_state(board, to_play));
        root_batch.push_back(game_.encode_state(board, next_player));
        opening_stats::eval_count.fetch_add(2, std::memory_order_relaxed);
        auto root_results = batch_infer_fn(root_batch);
        const auto& wdl_pla = root_results[0].second;
        const auto& wdl_opp = root_results[1].second;
        double root_V_pla = static_cast<double>(wdl_pla[0]) - wdl_pla[2];
        double root_V_opp = static_cast<double>(wdl_opp[0]) - wdl_opp[2];

        std::bernoulli_distribution bern_reject(reject_prob);

        // Pre-check: probably all moves are losing from my perspective
        if (root_V_pla < 0) {
            double f = 1.0 - std::exp(-3.0 * root_V_pla * root_V_pla);
            std::bernoulli_distribution bern_f(f);
            if (bern_f(rng) && bern_reject(rng)) return -1;
        }
        // Pre-check: probably all moves are winning (opp's perspective losing)
        if (root_V_opp < 0) {
            double f = 1.0 - std::exp(-3.0 * root_V_opp * root_V_opp);
            std::bernoulli_distribution bern_f(f);
            if (bern_f(rng) && bern_reject(rng)) return -1;
        }

        // KataGomo: nearby filter only activates when position isn't decisively
        // winning for the player-to-move and the board isn't empty.
        int stone_num = 0;
        for (int8_t v : board) if (v != 0) ++stone_num;
        const bool should_check_nearby = (root_V_opp > 0) && (stone_num > 0);

        // --- Phase B: collect candidate positions
        std::vector<int> candidates;
        for (int r = 0; r < bs; ++r) {
            for (int c = 0; c < bs; ++c) {
                const int loc = r * bs + c;
                if (board[loc] != 0) continue;
                if (should_check_nearby &&
                    !game_.is_near_occupied(board, r, c, cfg_.nearby_dist)) continue;
                candidates.push_back(loc);
            }
        }
        if (candidates.empty()) return -1;

        // --- Phase C: place stone, filter terminals/forbidden, encode
        std::vector<std::vector<int8_t>> encoded_batch;
        std::vector<int> valid_candidates;
        encoded_batch.reserve(candidates.size());
        valid_candidates.reserve(candidates.size());

        for (int loc : candidates) {
            auto next_board = board;
            next_board[loc] = static_cast<int8_t>(to_play);

            if (game_.is_terminal(next_board, loc, to_play)) continue;
            if (game_.use_renju && to_play == 1) {
                int winner = game_.get_winner(next_board, loc, to_play);
                if (winner == -1) continue;  // forbidden for Black
            }

            encoded_batch.push_back(game_.encode_state(next_board, next_player));
            valid_candidates.push_back(loc);
        }
        if (valid_candidates.empty()) return -1;

        // --- Phase D: batch NN evaluation of candidates
        opening_stats::eval_count.fetch_add(
            static_cast<int64_t>(encoded_batch.size()), std::memory_order_relaxed);
        auto results = batch_infer_fn(encoded_batch);

        // --- Phase E: compute balance scores (1 - V^2)^k
        std::vector<double> scores(results.size(), 0.0);
        double max_prob = 0.0;
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& wdl = results[i].second;
            double V = static_cast<double>(wdl[0]) - wdl[2];
            double score = std::pow(std::max(0.0, 1.0 - V * V),
                                    static_cast<double>(cfg_.balance_power));
            scores[i] = score;
            max_prob = std::max(max_prob, score);
        }

        // Probabilistic reject based on max balance score
        {
            double p = std::clamp(1.0 - max_prob, 0.0, 1.0);
            std::bernoulli_distribution bern_max(p);
            if (bern_max(rng) && bern_reject(rng)) return -1;
        }

        // Sample proportional to balance scores
        double total = 0.0;
        for (double s : scores) total += s;
        if (total < 1e-30) return -1;

        std::uniform_real_distribution<double> uni(0.0, total);
        double r = uni(rng);
        double cumsum = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            cumsum += scores[i];
            if (cumsum >= r) return valid_candidates[i];
        }
        return valid_candidates.back();
    }
};

}  // namespace skyzero

#endif  // SKYZERO_RANDOM_OPENING_H
