import math
import time
import numpy as np
from matplotlib import pyplot as plt


def softmax(policy_logits):
    max_logit = np.max(policy_logits)
    policy = np.exp(policy_logits - max_logit)
    policy_sum = np.sum(policy)
    policy /= policy_sum
    return policy


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


def random_augment_batch(batch, board_size):
    # batch is { "encoded_state": np.ndarray(B, C, H, W), ... }
    if not batch:
        return batch
        
    batch_size = len(batch["encoded_state"])
    
    # In-place modify Numpy arrays for fast augmentation
    for i in range(batch_size):
        transform_type = np.random.randint(0, 8)
        k = transform_type % 4
        do_flip = transform_type >= 4
        
        if k == 0 and not do_flip:
            continue
            
        batch["encoded_state"][i] = np.rot90(batch["encoded_state"][i], k=k, axes=(1, 2))
        
        p_2d = batch["policy_target"][i].reshape(board_size, board_size)
        opp_p_2d = batch["opponent_policy_target"][i].reshape(board_size, board_size)
        
        aug_p_2d = np.rot90(p_2d, k=k)
        aug_opp_p_2d = np.rot90(opp_p_2d, k=k)
        
        if do_flip:
            batch["encoded_state"][i] = np.flip(batch["encoded_state"][i], axis=2)
            aug_p_2d = np.flip(aug_p_2d, axis=1)
            aug_opp_p_2d = np.flip(aug_opp_p_2d, axis=1)
            
        batch["policy_target"][i] = aug_p_2d.flatten()
        batch["opponent_policy_target"][i] = aug_opp_p_2d.flatten()
        
    return batch


def drop_last(memory, batch_size):
    len_memory = len(memory)
    memory = memory[:len_memory - len_memory % batch_size]
    return memory

def temperature_transform(probs, temp):
    probs = np.asarray(probs, dtype=np.float64)
    
    if temp <= 1e-10:
        max_val = np.max(probs)
        max_mask = (probs == max_val)
        return max_mask.astype(np.float64) / np.sum(max_mask)
    if abs(temp - 1.0) < 1e-10:
        return probs
    non_zero_mask = probs > 0
    if not np.any(non_zero_mask):
        return probs
    logits = np.log(probs[non_zero_mask])
    logits /= temp
    
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    
    probs_normalized = exp_logits / np.sum(exp_logits)
    scaled = np.zeros_like(probs)
    scaled[non_zero_mask] = probs_normalized
    return scaled
