#ifndef SKYZERO_GAME_INITIALIZER_H
#define SKYZERO_GAME_INITIALIZER_H

// Per-game (size, rule) sampler — KataGomo `GameInitializer` style.
// Single mutex protects rng_ and the two discrete distributions; lock is held
// for ~microseconds per game (just sample + Gomoku ctor with RVO), so 96+
// workers asking once per ~100ms-game show no measurable contention.
//
// References (KataGomo /home/sky/RL/SkyZero/KataGomo/cpp/program/play.cpp):
//   - lines 32-42, 253: lock_guard around createGame
//   - lines 320-330: createRulesUnsynchronized
//   - lines 353-356: weighted bSize sampling via rand.nextUInt(probs.data(),n)
// We swap KataGomo's `rand.nextUInt(probs,n)` for std::discrete_distribution
// (functionally equivalent, no external dep on Rand).

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envs/gomoku.h"

namespace skyzero {

class GameInitializer {
public:
    GameInitializer(
        const std::vector<int>& sizes,
        const std::vector<float>& size_relprobs,
        const std::vector<RuleType>& rules,
        const std::vector<float>& rule_relprobs,
        uint64_t seed
    )
        : rng_(seed),
          sizes_(sizes),
          rules_(rules),
          size_dist_(size_relprobs.begin(), size_relprobs.end()),
          rule_dist_(rule_relprobs.begin(), rule_relprobs.end()) {
        // ---- validation ----
        if (sizes_.empty()) {
            throw std::runtime_error("GameInitializer: BOARD_SIZES must have ≥ 1 entry");
        }
        if (sizes_.size() != size_relprobs.size()) {
            throw std::runtime_error(
                "GameInitializer: BOARD_SIZES (" + std::to_string(sizes_.size())
                + ") and BOARD_SIZE_RELPROBS (" + std::to_string(size_relprobs.size())
                + ") must have equal length");
        }
        if (rules_.empty()) {
            throw std::runtime_error("GameInitializer: RULES must have ≥ 1 entry");
        }
        if (rules_.size() != rule_relprobs.size()) {
            throw std::runtime_error(
                "GameInitializer: RULES (" + std::to_string(rules_.size())
                + ") and RULE_RELPROBS (" + std::to_string(rule_relprobs.size())
                + ") must have equal length");
        }
        for (int s : sizes_) {
            if (s < 5 || s > Gomoku::MAX_BOARD_SIZE) {
                throw std::runtime_error(
                    "GameInitializer: board size " + std::to_string(s)
                    + " out of range [5, " + std::to_string(Gomoku::MAX_BOARD_SIZE)
                    + "]. To raise the upper bound, edit cpp/envs/gomoku.h:65, "
                      "re-trace the model, and rebuild C++.");
            }
        }
        auto check_relprobs = [](const std::vector<float>& p, const char* name) {
            float total = 0.0f;
            for (float v : p) {
                if (!(v >= 0.0f) || !std::isfinite(v)) {
                    throw std::runtime_error(
                        std::string("GameInitializer: ") + name
                        + " contains negative / non-finite value");
                }
                total += v;
            }
            if (total <= 0.0f) {
                throw std::runtime_error(
                    std::string("GameInitializer: ") + name
                    + " sums to 0; need at least one positive weight");
            }
        };
        check_relprobs(size_relprobs, "BOARD_SIZE_RELPROBS");
        check_relprobs(rule_relprobs, "RULE_RELPROBS");
    }

    // Sample a fresh (size, rule) pair under the configured distribution and
    // return a Gomoku instance. Mutex protects the rng + dists; Gomoku ctor
    // is cheap enough to keep inside the lock.
    Gomoku create_game() {
        std::lock_guard<std::mutex> lock(mtx_);
        const int size = sizes_[size_dist_(rng_)];
        const RuleType rule = rules_[rule_dist_(rng_)];
        const bool forbidden_plane = (rule != RuleType::FREESTYLE);
        return Gomoku(size, rule, forbidden_plane);
    }

    // Print resolved (normalized) distribution. Called once at selfplay startup
    // so the user can eyeball that BOARD_SIZE_RELPROBS / RULE_RELPROBS map onto
    // the percentages they intended.
    void log_distribution(std::ostream& os) const {
        const auto size_probs = size_dist_.probabilities();
        const auto rule_probs = rule_dist_.probabilities();
        os << "[GameInit] board_mix={";
        for (size_t i = 0; i < sizes_.size(); ++i) {
            if (i) os << ", ";
            os << sizes_[i] << ":" << std::fixed << std::setprecision(1)
               << (size_probs[i] * 100.0) << "%";
        }
        os << "} rule_mix={";
        for (size_t i = 0; i < rules_.size(); ++i) {
            if (i) os << ", ";
            os << rule_to_string(rules_[i]) << ":"
               << std::fixed << std::setprecision(1)
               << (rule_probs[i] * 100.0) << "%";
        }
        os << "} (per-game)" << std::endl;
    }

private:
    mutable std::mutex mtx_;
    std::mt19937_64 rng_;
    std::vector<int> sizes_;
    std::vector<RuleType> rules_;
    std::discrete_distribution<int> size_dist_;
    std::discrete_distribution<int> rule_dist_;
};

}  // namespace skyzero

#endif
