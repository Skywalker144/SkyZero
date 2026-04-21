#ifndef SKYZERO_UTILS_H
#define SKYZERO_UTILS_H

// Ported verbatim from CSkyZero_V3/utils.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

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
