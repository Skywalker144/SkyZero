#ifndef SKYZERO_POLICY_SURPRISE_WEIGHTING_H
#define SKYZERO_POLICY_SURPRISE_WEIGHTING_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace skyzero {

// TrainSample — the data written to NPZ files
struct TrainSample {
    std::vector<int8_t> state;
    int8_t to_play = 1;
    std::vector<float> policy_target;
    std::vector<float> opponent_policy_target;
    std::array<float, 3> value_target{0.0f, 0.0f, 0.0f};
    float sample_weight = 1.0f;
    float opp_policy_weight = 1.0f;   // 0 if the row has no next move
};

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

// ---------------------------------------------------------------------------
// PolicySurpriseSample — intermediate struct during selfplay before weighting
// Aligned to Python V3 return_memory sample dict
// ---------------------------------------------------------------------------
struct PolicySurpriseSample {
    std::vector<int8_t> state;
    int8_t to_play = 1;

    std::vector<float> policy_target;
    std::vector<float> opponent_policy_target;
    std::array<float, 3> outcome{0.0f, 0.0f, 0.0f};
    std::vector<float> nn_policy;
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};
    std::array<float, 3> v_mix{0.0f, 0.0f, 0.0f};         // WDL from search root
    std::array<float, 3> value_target{0.0f, 0.0f, 0.0f};
    float sample_weight = 1.0f;
    float opp_policy_weight = 1.0f;
};

// ---------------------------------------------------------------------------
// KL divergence (vector version)
// ---------------------------------------------------------------------------
inline float compute_kl_divergence(
    const std::vector<float>& policy_target,
    const std::vector<float>& policy_prior,
    float epsilon = 1e-10f
) {
    if (policy_target.empty() || policy_prior.empty() || policy_target.size() != policy_prior.size()) {
        return 0.0f;
    }

    float sum_p = 0.0f;
    float sum_q = 0.0f;
    for (size_t i = 0; i < policy_target.size(); ++i) {
        sum_p += clampf(policy_target[i], epsilon, 1.0f);
        sum_q += clampf(policy_prior[i], epsilon, 1.0f);
    }
    if (sum_p <= epsilon || sum_q <= epsilon) {
        return 0.0f;
    }

    float kl = 0.0f;
    for (size_t i = 0; i < policy_target.size(); ++i) {
        if (policy_target[i] <= 0.0f) {
            continue;
        }
        const float p = clampf(policy_target[i], epsilon, 1.0f) / sum_p;
        const float q = clampf(policy_prior[i], epsilon, 1.0f) / sum_q;
        kl += p * (std::log(p) - std::log(q));
    }
    return std::max(0.0f, kl);
}

// ---------------------------------------------------------------------------
// KL divergence (array<3> version — for value WDL)
// ---------------------------------------------------------------------------
inline float compute_kl_divergence(
    const std::array<float, 3>& target,
    const std::array<float, 3>& prior,
    float epsilon = 1e-10f
) {
    float sum_p = 0.0f;
    float sum_q = 0.0f;
    for (int i = 0; i < 3; ++i) {
        sum_p += clampf(target[i], epsilon, 1.0f);
        sum_q += clampf(prior[i], epsilon, 1.0f);
    }
    if (sum_p <= epsilon || sum_q <= epsilon) {
        return 0.0f;
    }
    float kl = 0.0f;
    for (int i = 0; i < 3; ++i) {
        if (target[i] <= 0.0f) {
            continue;
        }
        const float p = clampf(target[i], epsilon, 1.0f) / sum_p;
        const float q = clampf(prior[i], epsilon, 1.0f) / sum_q;
        kl += p * (std::log(p) - std::log(q));
    }
    return std::max(0.0f, kl);
}

// ---------------------------------------------------------------------------
// compute_policy_surprise_weights
// Aligned to Python V3 policy_surprise_weighting.py
// target_weights[i] = sample.sample_weight
// ---------------------------------------------------------------------------
inline std::vector<float> compute_policy_surprise_weights(
    const std::vector<PolicySurpriseSample>& game_data,
    float policy_surprise_data_weight = 0.5f,
    float value_surprise_data_weight = 0.1f
) {
    const int n = static_cast<int>(game_data.size());
    if (n == 0) {
        return {};
    }

    std::vector<float> target_weights(n, 0.0f);
    std::vector<float> policy_surprises(n, 0.0f);
    std::vector<float> value_surprises(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        const auto& s = game_data[i];
        // Policy Surprise (KL between search policy and NN prior)
        policy_surprises[i] = compute_kl_divergence(s.policy_target, s.nn_policy);
        // Value Surprise (KL between value_target and NN value probs)
        value_surprises[i] = std::min(compute_kl_divergence(s.value_target, s.nn_value_probs), 1.0f);
        // Weight = sample_weight (aligned to Python V3)
        target_weights[i] = s.sample_weight;
    }

    const float sum_weights = std::accumulate(target_weights.begin(), target_weights.end(), 0.0f);
    if (sum_weights <= 1e-8f) {
        return std::vector<float>(n, 0.0f);
    }

    float avg_p_surprise = 0.0f;
    float avg_v_surprise = 0.0f;
    for (int i = 0; i < n; ++i) {
        avg_p_surprise += policy_surprises[i] * target_weights[i];
        avg_v_surprise += value_surprises[i] * target_weights[i];
    }
    avg_p_surprise /= sum_weights;
    avg_v_surprise /= sum_weights;

    // Dynamic scaling of value surprise weight if average surprise is very low
    float actual_v_weight = value_surprise_data_weight;
    if (avg_v_surprise < 0.01f) {
        actual_v_weight *= (avg_v_surprise / 0.01f);
    }
    const float baseline_weight_ratio = std::max(0.0f, 1.0f - policy_surprise_data_weight - actual_v_weight);
    const float p_threshold = avg_p_surprise * 1.5f;

    std::vector<float> p_prob_values(n, 0.0f);
    std::vector<float> v_prob_values(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float w = target_weights[i];
        const float ps = policy_surprises[i];
        const float vs = value_surprises[i];
        // Surprise weighting logic (aligned to Python V3)
        p_prob_values[i] = w * ps + (1.0f - w) * std::max(0.0f, ps - p_threshold);
        v_prob_values[i] = w * vs;
    }

    const float sum_p_prob = std::max(std::accumulate(p_prob_values.begin(), p_prob_values.end(), 0.0f), 1e-10f);
    const float sum_v_prob = std::max(std::accumulate(v_prob_values.begin(), v_prob_values.end(), 0.0f), 1e-10f);

    std::vector<float> final_weights(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float w = target_weights[i];
        const float term_base = baseline_weight_ratio * w;
        const float term_p = policy_surprise_data_weight * p_prob_values[i] * sum_weights / sum_p_prob;
        const float term_v = actual_v_weight * v_prob_values[i] * sum_weights / sum_v_prob;
        final_weights[i] = term_base + term_p + term_v;
    }
    return final_weights;
}

// ---------------------------------------------------------------------------
// apply_surprise_weighting_to_game
// Aligned to Python V3: delete outcome/nn_policy/nn_value_probs/v_mix
// then stochastic rounding by weight
// ---------------------------------------------------------------------------
inline std::vector<TrainSample> apply_surprise_weighting_to_game(
    const std::vector<PolicySurpriseSample>& game_data,
    const std::vector<float>& weights,
    std::mt19937& rng
) {
    std::vector<TrainSample> weighted_data;
    if (game_data.empty() || game_data.size() != weights.size()) {
        return weighted_data;
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (size_t i = 0; i < game_data.size(); ++i) {
        const float w = weights[i];
        if (w <= 0.0f) {
            continue;
        }
        const int floor_w = static_cast<int>(std::floor(w));
        const float frac = w - static_cast<float>(floor_w);

        auto make_train = [&]() {
            TrainSample ts;
            ts.state = game_data[i].state;
            ts.to_play = game_data[i].to_play;
            ts.policy_target = game_data[i].policy_target;
            ts.opponent_policy_target = game_data[i].opponent_policy_target;
            ts.value_target = game_data[i].value_target;
            // Relative importance is already encoded by the floor+bernoulli
            // insertion count derived from final_weights[i]; each written row
            // contributes at full weight.
            ts.sample_weight = 1.0f;
            ts.opp_policy_weight = game_data[i].opp_policy_weight;
            return ts;
        };

        for (int k = 0; k < floor_w; ++k) {
            weighted_data.push_back(make_train());
        }
        if (uni(rng) < frac) {
            weighted_data.push_back(make_train());
        }
    }
    return weighted_data;
}

}  // namespace skyzero

#endif
