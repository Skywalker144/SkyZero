import numpy as np


def compute_kl_divergence(policy_target, policy_prior, epsilon=1e-10):
    mask = policy_target > 0 
    
    p = np.clip(policy_target, epsilon, 1.0)
    q = np.clip(policy_prior, epsilon, 1.0)
    p /= np.sum(p)
    q /= np.sum(q)
    
    kl = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
    return max(0.0, kl)

def compute_policy_surprise_weights(game_data, board_size, policy_surprise_data_weight=0.5, value_surprise_data_weight=0.1):
    
    n_positions = len(game_data)
    if n_positions == 0:
        return []

    now_factor = 1 / (1 + (board_size ** 2) * 0.016)

    last_outcome = game_data[-1]["outcome"]
    if last_outcome == 1:
        current_target = np.array([1.0, 0.0, 0.0])
    elif last_outcome == 0:
        current_target = np.array([0.0, 1.0, 0.0])
    else:
        current_target = np.array([0.0, 0.0, 1.0])
    
    smoothed_targets = [None] * n_positions
    for i in range(n_positions - 1, -1, -1):
        search_value = game_data[i]["root_value"]
        current_target = current_target + now_factor * (search_value - current_target)
        smoothed_targets[i] = current_target.copy()
    
    target_weigts = []
    policy_surprises = []
    value_surprises = []

    for i in range(n_positions):
        sample = game_data[i]
        p_kl = compute_kl_divergence(sample["policy_target"], sample["nn_policy"])
        policy_surprises.append(p_kl)

        v_kl = compute_kl_divergence(smoothed_targets[i], sample["value_probs"])
        value_surprises.append(min(v_kl, 1))

        w = 1 if sample["is_full_search"] else 0
        target_weigts.append(w)
    
    sum_weights = sum(target_weigts)
    if sum_weights <= 1e-8:
        return [0.0] * n_positions
    
    avg_p_surprise = sum(s * w for s, w in zip(policy_surprises, target_weigts)) / sum_weights
    avg_v_surprise = sum(s * w for s, w in zip(value_surprises, target_weigts)) / sum_weights

    actual_v_weight = value_surprise_data_weight
    if avg_v_surprise < 0.01:
        actual_v_weight *= (avg_v_surprise / 0.01)
    
    baseline_weight_ratio = 1 - policy_surprise_data_weight - actual_v_weight

    p_threshold = avg_p_surprise * 1.5
    p_prob_values = []
    v_prob_values = []
    for i in range(n_positions):
        w = target_weigts[i]
        ps = policy_surprises[i]
        vs = value_surprises[i]

        p_prob = w * ps + (1 - w) * max(0, ps - p_threshold)
        p_prob_values.append(p_prob)

        v_prob = w * vs
        v_prob_values.append(v_prob)
    
    sum_p_prob = max(sum(p_prob_values), 1e-10)
    sum_v_prob = max(sum(v_prob_values), 1e-10)

    fianl_weights = []
    for i in range(n_positions):
        w = target_weigts[i]
        p_p = p_prob_values[i]
        v_p = v_prob_values[i]

        term_base = baseline_weight_ratio * w
        term_p = policy_surprise_data_weight * p_p * sum_weights / sum_p_prob
        term_v = actual_v_weight * v_p * sum_weights / sum_v_prob

        fianl_weights.append(term_base + term_p + term_v)

    return fianl_weights


def compute_policy_surprise_weights_(game_data, baseline_weight_ratio=0.5, value_surprise_data_weight=0.1):

    n_positions = len(game_data)
    if n_positions == 0:
        return []

    target_weights = []
    policy_surprises = []
    value_surprises = []

    full_search_count = 0

    for sample in game_data:
        # (final_state, encoded_state, policy_target, nn_policy, value_probs, root_value, outcome, is_full_search)
        policy_target = sample["policy_target"]
        policy_prior = sample["nn_policy"]
        nn_value_probs = sample["value_probs"]
        outcome = sample["outcome"]
        is_full_search = sample["is_full_search"]

        w = 1.0 if is_full_search else 0.0
        target_weights.append(w)

        if is_full_search:
            full_search_count += 1

        p_kl = compute_kl_divergence(policy_target, policy_prior)
        policy_surprises.append(p_kl)

        if outcome == 1:
            target_v = np.array([1.0, 0.0, 0.0])
        elif outcome == 0:
            target_v = np.array([0.0, 1.0, 0.0])
        else:
            target_v = np.array([0.0, 0.0, 1.0])

        v_kl = compute_kl_divergence(target_v, nn_value_probs)
        value_surprises.append(min(v_kl, 1))

    sum_weights = sum(target_weights)

    if sum_weights <= 1e-8:
        return [0.0] * n_positions

    avg_p_surprise = sum(s * w for s, w in zip(policy_surprises, target_weights)) / sum_weights
    avg_v_surprise = sum(s * w for s, w in zip(value_surprises, target_weights)) / sum_weights

    actual_v_weight = value_surprise_data_weight
    if avg_v_surprise < 0.01:
        actual_v_weight *= (avg_v_surprise / 0.01)

    p_surprise_weight = 1.0 - baseline_weight_ratio - actual_v_weight

    p_threshold = avg_p_surprise * 1.5

    p_prop_values = []
    v_prop_values = []
    for i in range(n_positions):
        w = target_weights[i]
        ps = policy_surprises[i]
        vs = value_surprises[i]

        p_prop = w * ps + (1.0 - w) * max(0.0, ps - p_threshold)
        p_prop_values.append(p_prop)

        v_prop = w * vs
        v_prop_values.append(v_prop)

    sum_p_prop = max(sum(p_prop_values), 1e-10)
    sum_v_prop = max(sum(v_prop_values), 1e-10)

    final_weights = []
    for i in range(n_positions):
        w = target_weights[i]
        p_prop = p_prop_values[i]
        v_prop = v_prop_values[i]

        term_base = baseline_weight_ratio * w
        term_p = p_surprise_weight * p_prop * (sum_weights / sum_p_prop)
        term_v = actual_v_weight * v_prop * (sum_weights / sum_v_prop)

        final_weights.append(term_base + term_p + term_v)

    return final_weights


def apply_surprise_weighting_to_game_(game_data, weights):
    weighted_data = []

    for sample, weight in zip(game_data, weights):
        if weight <= 0:
            continue

        floor_weight = int(np.floor(weight))
        fractional = weight - floor_weight

        for _ in range(floor_weight):
            weighted_data.append(sample)
        if np.random.random() < fractional:
            weighted_data.append(sample)

    return weighted_data

# Fast
def apply_surprise_weighting_to_game(game_data, weights):
    weighted_data = []
    rand = np.random.random
    
    for sample, weight in zip(game_data, weights):
        if weight <= 0:
            continue
        floor_weight = int(weight)
        
        if floor_weight > 0:
            weighted_data.extend([sample] * floor_weight)
        
        if (weight - floor_weight) > rand():
            weighted_data.append(sample)
    return weighted_data