#ifndef SKYZERO_UTILS_H
#define SKYZERO_UTILS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <atomic>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <chrono>

namespace skyzero {

inline std::atomic<bool> stop_requested{false};

inline void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n[Signal] Ctrl+C detected. Requesting graceful shutdown...\n";
        stop_requested = true;
    }
}

inline std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float v = std::exp(logits[i] - max_logit);
        probs[i] = v;
        sum += v;
    }
    if (sum <= 1e-20f) {
        const float p = 1.0f / static_cast<float>(probs.size());
        std::fill(probs.begin(), probs.end(), p);
        return probs;
    }
    for (float& p : probs) {
        p /= sum;
    }
    return probs;
}

inline std::vector<float> temperature_transform(const std::vector<float>& probs, float temp) {
    std::vector<float> out(probs.size(), 0.0f);
    if (probs.empty()) {
        return out;
    }
    if (temp <= 1e-10f) {
        const float max_v = *std::max_element(probs.begin(), probs.end());
        size_t count = 0;
        for (float v : probs) {
            if (v == max_v) {
                ++count;
            }
        }
        if (count == 0) {
            return out;
        }
        const float tie_p = 1.0f / static_cast<float>(count);
        for (size_t i = 0; i < probs.size(); ++i) {
            out[i] = (probs[i] == max_v) ? tie_p : 0.0f;
        }
        return out;
    }
    if (std::fabs(temp - 1.0f) < 1e-10f) {
        return probs;
    }

    std::vector<float> logits;
    std::vector<size_t> idx;
    logits.reserve(probs.size());
    idx.reserve(probs.size());

    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > 0.0f) {
            idx.push_back(i);
            logits.push_back(std::log(probs[i]) / temp);
        }
    }
    if (logits.empty()) {
        return probs;
    }
    const auto scaled = softmax(logits);
    for (size_t k = 0; k < idx.size(); ++k) {
        out[idx[k]] = scaled[k];
    }
    return out;
}

inline std::vector<float> root_temperature_transform(
    const std::vector<float>& policy,
    int current_step,
    float root_temperature_init,
    float root_temperature_final,
    int board_size
) {
    const float decay = std::pow(0.5f, static_cast<float>(current_step) / static_cast<float>(board_size));
    const float current_temp = root_temperature_final + (root_temperature_init - root_temperature_final) * decay;
    return temperature_transform(policy, current_temp);
}

inline std::vector<float> sample_dirichlet(const std::vector<float>& alphas, std::mt19937& rng) {
    std::vector<float> out(alphas.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < alphas.size(); ++i) {
        const float a = std::max(1e-6f, alphas[i]);
        std::gamma_distribution<float> gamma(a, 1.0f);
        out[i] = gamma(rng);
        sum += out[i];
    }
    if (sum <= 1e-20f) {
        const float u = 1.0f / static_cast<float>(out.size());
        std::fill(out.begin(), out.end(), u);
        return out;
    }
    for (float& v : out) {
        v /= sum;
    }
    return out;
}

inline std::vector<float> add_shaped_dirichlet_noise(
    const std::vector<float>& policy,
    float total_dirichlet_alpha,
    float epsilon,
    std::mt19937& rng
) {
    std::vector<size_t> legal_idx;
    for (size_t i = 0; i < policy.size(); ++i) {
        if (policy[i] > 0.0f) {
            legal_idx.push_back(i);
        }
    }
    if (legal_idx.empty()) {
        return policy;
    }

    std::vector<float> log_probs(legal_idx.size(), 0.0f);
    for (size_t i = 0; i < legal_idx.size(); ++i) {
        log_probs[i] = std::log(std::min(policy[legal_idx[i]], 0.01f) + 1e-20f);
    }
    const float log_mean = std::accumulate(log_probs.begin(), log_probs.end(), 0.0f) / static_cast<float>(log_probs.size());

    std::vector<float> alpha_shape(log_probs.size(), 0.0f);
    float alpha_shape_sum = 0.0f;
    for (size_t i = 0; i < log_probs.size(); ++i) {
        alpha_shape[i] = std::max(log_probs[i] - log_mean, 0.0f);
        alpha_shape_sum += alpha_shape[i];
    }

    const float uniform = 1.0f / static_cast<float>(legal_idx.size());
    std::vector<float> alpha_weights(legal_idx.size(), uniform);
    if (alpha_shape_sum > 1e-10f) {
        for (size_t i = 0; i < legal_idx.size(); ++i) {
            alpha_weights[i] = 0.5f * (alpha_shape[i] / alpha_shape_sum) + 0.5f * uniform;
        }
    }

    std::vector<float> alphas(legal_idx.size(), 0.0f);
    for (size_t i = 0; i < legal_idx.size(); ++i) {
        alphas[i] = alpha_weights[i] * total_dirichlet_alpha;
    }
    const auto noise = sample_dirichlet(alphas, rng);

    std::vector<float> out = policy;
    for (size_t i = 0; i < legal_idx.size(); ++i) {
        const size_t idx = legal_idx[i];
        out[idx] = policy[idx] * (1.0f - epsilon) + noise[i] * epsilon;
    }
    return out;
}

inline std::vector<float> reshape_rotate_flip_flat(
    const std::vector<float>& flat,
    int board_size,
    int k,
    bool do_flip
) {
    std::vector<float> out(flat.size(), 0.0f);
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
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
            out[rr * board_size + cc] = flat[r * board_size + c];
        }
    }
    return out;
}

inline std::vector<int8_t> transform_encoded_state(
    const std::vector<int8_t>& encoded,
    int channels,
    int board_size,
    int k,
    bool do_flip
) {
    const size_t plane_size = static_cast<size_t>(board_size) * static_cast<size_t>(board_size);
    if (encoded.size() != static_cast<size_t>(channels) * plane_size) {
        throw std::runtime_error("transform_encoded_state: bad input shape");
    }
    std::vector<int8_t> out(encoded.size(), 0);
    for (int ch = 0; ch < channels; ++ch) {
        const size_t base = static_cast<size_t>(ch) * plane_size;
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
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
                out[base + static_cast<size_t>(rr) * board_size + cc] = encoded[base + static_cast<size_t>(r) * board_size + c];
            }
        }
    }
    return out;
}

}  // namespace skyzero

#endif
