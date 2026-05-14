import numpy as np


def compute_kl_divergence(p, q, epsilon=1e-10):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    if p.size == 0 or q.size == 0 or p.shape != q.shape:
        return 0.0

    p_clip = np.clip(p, epsilon, 1.0)
    q_clip = np.clip(q, epsilon, 1.0)

    sum_p = np.sum(p_clip)
    sum_q = np.sum(q_clip)
    if sum_p <= epsilon or sum_q <= epsilon:
        return 0.0

    p_norm = p_clip / sum_p
    q_norm = q_clip / sum_q

    mask = p > 0
    if not np.any(mask):
        return 0.0

    kl = np.sum(p_norm[mask] * (np.log(p_norm[mask]) - np.log(q_norm[mask])))
    return max(0.0, float(kl))


def compute_policy_surprise_weights(
    game_data,
    policy_surprise_data_weight=0.5,
    value_surprise_data_weight=0.1,
):
    n = len(game_data)
    if n == 0:
        return np.array([], dtype=np.float32)

    # Extract base weights and compute surprises
    target_weights = np.array([d.get("sample_weight", 1.0) for d in game_data], dtype=np.float32)
    policy_surprises = np.zeros(n, dtype=np.float32)
    value_surprises = np.zeros(n, dtype=np.float32)

    for i, d in enumerate(game_data):
        policy_surprises[i] = compute_kl_divergence(
            d["policy_target"], d["nn_policy"]
        )
        # Value surprise: how much does the raw NN value diverge from the final outcome?
        value_surprises[i] = min(
            compute_kl_divergence(
                d["value_target"], d["nn_value_probs"]
            ), 1.0
        )

    sum_weights = float(np.sum(target_weights))
    if sum_weights <= 1e-8:
        return np.zeros(n, dtype=np.float32)

    # Weighted average surprises
    avg_p_surprise = float(np.sum(policy_surprises * target_weights) / sum_weights)
    avg_v_surprise = float(np.sum(value_surprises * target_weights) / sum_weights)

    # Safety: reduce value weight when avg value surprise is very low
    actual_v_weight = value_surprise_data_weight
    if avg_v_surprise < 0.01:
        actual_v_weight *= avg_v_surprise / 0.01

    baseline_weight_ratio = max(0.0, 1.0 - policy_surprise_data_weight - actual_v_weight)

    # Policy bonus: two-stage weighting with threshold at 1.5x average
    p_threshold = avg_p_surprise * 1.5
    p_prob = target_weights * policy_surprises + (1.0 - target_weights) * np.maximum(
        0.0, policy_surprises - p_threshold
    )
    v_prob = target_weights * value_surprises

    sum_p_prob = max(float(np.sum(p_prob)), 1e-10)
    sum_v_prob = max(float(np.sum(v_prob)), 1e-10)

    # Combine three terms: baseline + policy bonus + value bonus
    weights = (
        baseline_weight_ratio * target_weights
        + policy_surprise_data_weight * p_prob * sum_weights / sum_p_prob
        + actual_v_weight * v_prob * sum_weights / sum_v_prob
    )

    return weights.astype(np.float32)


def apply_surprise_weighting(game_memory, weights):

    weighted = []
    for i, sample in enumerate(game_memory):
        w = float(weights[i])
        if w <= 0.0:
            continue
        for _ in range(int(np.floor(w))):
            weighted.append(dict(sample))  # shallow copy
        if np.random.randn() < (w - np.floor(w)):
            weighted.append(dict(sample))

    return weighted
