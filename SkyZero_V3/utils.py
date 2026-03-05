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


def random_augment_sample(sample, board_size):
    # game_data: dictionary (encoded_state, final_state, policy_target, opponent_policy, value_variance, outcome)
    transform_type = np.random.randint(0, 8)
    k = transform_type % 4
    do_flip = transform_type >= 4

    state = sample["encoded_state"]
    p_target = sample["policy_target"]
    opp_p_target = sample["opponent_policy_target"]

    aug_state = np.rot90(state, k=k, axes=(1, 2))
    p_2d = p_target.reshape(board_size, board_size)
    opp_p_2d = opp_p_target.reshape(board_size, board_size)
    aug_p_2d = np.rot90(p_2d, k=k)
    aug_opp_p_2d = np.rot90(opp_p_2d, k=k)
    if do_flip:
        aug_state = np.flip(aug_state, axis=2)
        aug_p_2d = np.flip(aug_p_2d, axis=1)
        aug_opp_p_2d = np.flip(aug_opp_p_2d, axis=1)
    aug_p_target = aug_p_2d.flatten()
    aug_opp_p_target = aug_opp_p_2d.flatten()

    new_sample = sample.copy()
    new_sample.update({
        "encoded_state": aug_state.copy(),
        "policy_target": aug_p_target.copy(),
        "opponent_policy_target": aug_opp_p_target.copy(),
    })
    return new_sample


def random_augment_batch(batch, board_size):
    augmented_batch = []
    for sample in batch:
        aug_sample = random_augment_sample(sample, board_size)
        augmented_batch.append(aug_sample)
    return augmented_batch


def drop_last(memory, batch_size):
    len_memory = len(memory)
    memory = memory[:len_memory - len_memory % batch_size]
    return memory

