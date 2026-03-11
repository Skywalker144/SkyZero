import numpy as np


def compute_kl_divergence(policy_target, policy_prior, epsilon=1e-10):
    mask = policy_target > 0 
    
    p = np.clip(policy_target, epsilon, 1.0)
    q = np.clip(policy_prior, epsilon, 1.0)
    p /= np.sum(p)
    q /= np.sum(q)
    
    kl = np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
    return max(0.0, kl)


def scalar_to_probs(v):
    """Converts a scalar value v in [-1, 1] to a probability distribution [Win, Draw, Loss]."""
    v_win = max(0.0, v)
    v_loss = max(0.0, -v)
    v_draw = max(0.0, 1.0 - (v_win + v_loss))
    probs = np.array([v_win, v_draw, v_loss])
    return probs / (np.sum(probs) + 1e-12)


def compute_policy_surprise_weights(game_data, board_size, policy_surprise_data_weight=0.5, value_surprise_data_weight=0.1):
    n_positions = len(game_data)
    if n_positions == 0:
        return []

    # Smoothing factor based on board size
    now_factor = 1.0 / (1.0 + (board_size ** 2) * 0.016)

    # Initialize current_target from the final outcome
    # alphazero.py: 1=win, 0=draw, -1=loss. Prob index: [0:win, 1:draw, 2:loss]
    last_outcome = game_data[-1]["outcome"]
    if last_outcome == 1:
        current_target = np.array([1.0, 0.0, 0.0])
    elif last_outcome == 0:
        current_target = np.array([0.0, 1.0, 0.0])
    else:  # -1 or other negative
        current_target = np.array([0.0, 0.0, 1.0])
    
    smoothed_targets = [None] * n_positions
    for i in range(n_positions - 1, -1, -1):
        if i < n_positions - 1 and game_data[i]["to_play"] != game_data[i+1]["to_play"]:
            current_target = np.array([current_target[2], current_target[1], current_target[0]])
            
        # Convert scalar v_mix to prob vector
        search_value_vec = scalar_to_probs(game_data[i]["v_mix"])
        # Smoothed target update
        current_target = (1.0 - now_factor) * current_target + now_factor * search_value_vec
        smoothed_targets[i] = current_target.copy()
    
    target_weights = []
    policy_surprises = []
    value_surprises = []

    for i in range(n_positions):
        sample = game_data[i]
        
        # Policy Surprise (KL between search policy and NN prior)
        p_kl = compute_kl_divergence(sample["policy_target"], sample["nn_policy"])
        policy_surprises.append(p_kl)

        # Value Surprise (KL between smoothed value target and NN value probs)
        v_kl = compute_kl_divergence(smoothed_targets[i], sample["nn_value_probs"])
        value_surprises.append(min(v_kl, 1.0))

        target_weights.append(sample.get("sample_weight", 1.0))
    
    sum_weights = sum(target_weights)
    if sum_weights <= 1e-8:
        return [0.0] * n_positions
    
    avg_p_surprise = sum(s * w for s, w in zip(policy_surprises, target_weights)) / sum_weights
    avg_v_surprise = sum(s * w for s, w in zip(value_surprises, target_weights)) / sum_weights

    # Dynamic scaling of value surprise weight if average surprise is very low
    actual_v_weight = value_surprise_data_weight
    if avg_v_surprise < 0.01:
        actual_v_weight *= (avg_v_surprise / 0.01)
    
    baseline_weight_ratio = max(0.0, 1.0 - policy_surprise_data_weight - actual_v_weight)

    p_threshold = avg_p_surprise * 1.5
    p_prob_values = []
    v_prob_values = []
    for i in range(n_positions):
        w = target_weights[i]
        ps = policy_surprises[i]
        vs = value_surprises[i]

        # Surprise weighting logic
        p_prob = w * ps + (1.0 - w) * max(0.0, ps - p_threshold)
        p_prob_values.append(p_prob)

        v_prob = w * vs
        v_prob_values.append(v_prob)
    
    sum_p_prob = max(sum(p_prob_values), 1e-10)
    sum_v_prob = max(sum(v_prob_values), 1e-10)

    final_weights = []
    for i in range(n_positions):
        w = target_weights[i]
        p_p = p_prob_values[i]
        v_p = v_prob_values[i]

        term_base = baseline_weight_ratio * w
        term_p = policy_surprise_data_weight * p_p * (sum_weights / sum_p_prob)
        term_v = actual_v_weight * v_p * (sum_weights / sum_v_prob)

        final_weights.append(term_base + term_p + term_v)

    return final_weights


def apply_surprise_weighting_to_game(game_data, weights):
    """Applies importance sampling using the calculated weights."""
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
