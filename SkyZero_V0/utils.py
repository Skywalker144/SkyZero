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

def add_dirichlet_noise(policy, alpha=0.3, epsilon=0.25, is_root=True):
    if not is_root:
        return policy
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    if nonzero_count <= 1:
        return policy
    
    noise = np.random.dirichlet([alpha] * nonzero_count)
    new_policy = policy.copy()
    new_policy[nonzero_mask] = (1 - epsilon) * policy[nonzero_mask] + epsilon * noise
    return new_policy

def random_augment_batch(batch, board_size):
    augmented_batch = []
    for sample in batch:
        k = np.random.randint(0, 4)
        flip = np.random.choice([True, False])
        
        state = sample["encoded_state"]
        policy = sample["policy_target"].reshape(board_size, board_size)
        
        aug_state = np.rot90(state, k, axes=(1, 2))
        aug_policy = np.rot90(policy, k)
        
        if flip:
            aug_state = np.flip(aug_state, axis=2)
            aug_policy = np.flip(aug_policy, axis=1)
            
        new_sample = sample.copy()
        new_sample["encoded_state"] = aug_state.copy()
        new_sample["policy_target"] = aug_policy.flatten().copy()
        augmented_batch.append(new_sample)
    return augmented_batch

def print_board(board):
    current_board = board[-1] if board.ndim == 3 else board
    rows, cols = current_board.shape
    print("   ", end="")
    for col in range(cols):
        print(f"{col:2d} ", end="")
    print()
    for row in range(rows):
        print(f"{row:2d} ", end="")
        for col in range(cols):
            if current_board[row, col] == 1:
                print(" × ", end="")
            elif current_board[row, col] == -1:
                print(" ○ ", end="")
            else:
                print(" · ", end="")
        print()
