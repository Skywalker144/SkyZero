import math
import numpy as np

def softmax(x):
    max_logit = np.max(x)
    exp_x = np.exp(x - max_logit)
    return exp_x / np.sum(exp_x)

def temperature_transform(probs, temp):
    probs = np.asarray(probs, dtype=np.float64)
    if temp <= 1e-10:
        max_val = np.max(probs)
        max_mask = (probs == max_val)
        return max_mask.astype(np.float64) / np.sum(max_mask)
    if abs(temp - 1.0) < 1e-10:
        return probs
    
    probs = np.maximum(probs, 1e-10)
    logits = np.log(probs) / temp
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def add_dirichlet_noise(policy, alpha=0.03, epsilon=0.25):
    # In the AlphaZero paper, alpha is set to approximately 10 divided by the board area.
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    if nonzero_count <= 1:
        return policy
    noise = np.random.dirichlet([alpha] * nonzero_count)
    new_policy = policy.copy()
    new_policy[nonzero_mask] = (1 - epsilon) * policy[nonzero_mask] + epsilon * noise
    return new_policy

def random_augment_batch(batch, board_size):
    """D4 augmentation: per-sample random rotation (0/90/180/270) + optional flip.

    Transforms `encoded_state` (C, H, W) and any spatially-laid-out policy
    target (H*W) consistently. Value targets are D4-invariant so untouched.
    """
    spatial_policy_keys = ("policy_target", "opp_policy_target")

    augmented_batch = []
    for sample in batch:
        k = np.random.randint(0, 4)
        flip = bool(np.random.choice([True, False]))

        state = sample["encoded_state"]
        aug_state = np.rot90(state, k, axes=(1, 2))
        if flip:
            aug_state = np.flip(aug_state, axis=2)

        new_sample = sample.copy()
        new_sample["encoded_state"] = aug_state.copy()
        for key in spatial_policy_keys:
            if key not in sample:
                continue
            policy = sample[key].reshape(board_size, board_size)
            aug_policy = np.rot90(policy, k)
            if flip:
                aug_policy = np.flip(aug_policy, axis=1)
            new_sample[key] = aug_policy.flatten().copy()
        augmented_batch.append(new_sample)
    return augmented_batch

def print_board(board):
    rows, cols = board.shape
    print("   ", end="")
    for col in range(cols):
        print(f"{col:2d} ", end="")
    print()
    for row in range(rows):
        print(f"{row:2d} ", end="")
        for col in range(cols):
            if board[row, col] == 1:
                print(" × ", end="")
            elif board[row, col] == -1:
                print(" ○ ", end="")
            else:
                print(" · ", end="")
        print()
