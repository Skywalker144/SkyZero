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

    f_state = sample['final_state']
    state = sample['encoded_state']
    p_target = sample['policy_target']
    opp_p_target = sample['opponent_policy_target']

    aug_f_state = np.rot90(f_state, k=k)
    aug_state = np.rot90(state, k=k, axes=(1, 2))
    p_2d = p_target.reshape(board_size, board_size)
    opp_p_2d = opp_p_target.reshape(board_size, board_size)
    aug_p_2d = np.rot90(p_2d, k=k)
    aug_opp_p_2d = np.rot90(opp_p_2d, k=k)
    if do_flip:
        aug_f_state = np.flip(aug_f_state, axis=1)
        aug_state = np.flip(aug_state, axis=2)
        aug_p_2d = np.flip(aug_p_2d, axis=1)
        aug_opp_p_2d = np.flip(aug_opp_p_2d, axis=1)
    aug_p_target = aug_p_2d.flatten()
    aug_opp_p_target = aug_opp_p_2d.flatten()

    new_sample = sample.copy()
    new_sample.update({
        'final_state': aug_f_state.copy(),
        'encoded_state': aug_state.copy(),
        'policy_target': aug_p_target.copy(),
        'opponent_policy_target': aug_opp_p_target.copy(),
    })
    return new_sample


def random_augment_batch(batch, board_size):
    """
    处理字典列表形式的 batch
    """
    augmented_batch = []
    for sample in batch:
        aug_sample = random_augment_sample(sample, board_size)
        augmented_batch.append(aug_sample)
    return augmented_batch


def drop_last(memory, batch_size):
    len_memory = len(memory)
    memory = memory[:len_memory - len_memory % batch_size]
    return memory


def add_dirichlet_noise(policy, alpha, epsilon=0.25):
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    noise = np.random.dirichlet(np.full(nonzero_count, alpha))
    policy[nonzero_mask] = (policy[nonzero_mask] * (1 - epsilon) + noise * epsilon)
    return policy


def add_shaped_dirichlet_noise(policy_t, total_dirichlet_alpha=10.83, epsilon=0.25):
    nonzero_mask = policy_t > 0
    legal_count = np.sum(nonzero_mask)

    if legal_count == 0:
        return policy_t

    # 1. 计算对数概率，并限制最大值为 0.01 防止极端值
    # 添加 1e-20 防止 log(0)
    log_probs = np.log(np.minimum(policy_t[nonzero_mask], 0.01) + 1e-20)
    log_mean = np.mean(log_probs)

    # 2. 减去均值并截断负值，得到形状
    alpha_shape = np.maximum(log_probs - log_mean, 0.0)
    alpha_shape_sum = np.sum(alpha_shape)

    # 3. 计算混合分布：50% 均匀 + 50% 形状
    uniform = 1.0 / legal_count

    alpha_weights = np.empty(legal_count)
    if alpha_shape_sum > 1e-10:
        # 归一化形状部分并混合
        alpha_weights = 0.5 * (alpha_shape / alpha_shape_sum) + 0.5 * uniform
    else:
        # 如果形状部分全为0（例如所有概率都极小或相等），则退化为均匀分布
        alpha_weights.fill(uniform)

    # 4. 缩放到总 alpha
    alphas = alpha_weights * total_dirichlet_alpha

    noise = np.random.dirichlet(alphas)

    new_policy = policy_t.copy()
    new_policy[nonzero_mask] = (policy_t[nonzero_mask] * (1 - epsilon) + noise * epsilon)
    return new_policy


def root_temperature_transform(policy, current_step, args, board_size):
    decay_factor = math.pow(0.5, current_step / board_size)
    current_temp = args['root_temperature_final'] + (args['root_temperature_init'] - args['root_temperature_final']) * decay_factor
    new_policy = temperature_transform(policy, current_temp)
    return new_policy


def temperature_transform(probs, temp):
    probs = np.asarray(probs, dtype=np.float64)
    
    # 1. 处理 temp = 0 的特殊情况（Argmax 逻辑）
    if temp <= 1e-10:  # 使用极小值判断代替精确等于0
        max_val = np.max(probs)
        max_mask = (probs == max_val)
        return max_mask.astype(np.float64) / np.sum(max_mask)
    # 2. 处理 temp = 1 的情况（直接返回，避免计算）
    if abs(temp - 1.0) < 1e-10:
        return probs
    # 3. 核心计算优化
    # 只有非零元素参与计算，避免 log(0) 警告
    non_zero_mask = probs > 0
    if not np.any(non_zero_mask):
        return probs
    # 直接在原空间计算或在对数空间计算
    # 优化点：如果 temp 不是特别小，直接用幂运算通常比 log-exp 转换快
    # 但为了数值稳定性，我们采用 log 空间处理
    logits = np.log(probs[non_zero_mask])
    logits /= temp
    
    # Softmax 技巧：减去最大值防止溢出
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    
    # 归一化
    probs_normalized = exp_logits / np.sum(exp_logits)
    # 填充回原形状
    scaled = np.zeros_like(probs)
    scaled[non_zero_mask] = probs_normalized
    return scaled
