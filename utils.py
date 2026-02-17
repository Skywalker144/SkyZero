import math

import matplotlib.pyplot as plt
import numpy as np


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
                print(" ● ", end="")
            elif current_board[row, col] == -1:
                print(" ○ ", end="")
            else:
                print(" · ", end="")
        print()


def augment_data(memory, board_size):
    """
    对一组样本生成所有8种对称变换（用于完整数据增强）。
    
    Args:
        memory: 样本列表，每个样本是 (state, action_probs, outcome, for_train) 的元组
        board_size: 棋盘大小
        
    Returns:
        增强后的样本列表（原始数量的8倍）
    """
    augmented_memory = []

    for state, action_probs, outcome, for_train in memory:

        action_probs_2d = action_probs.reshape(board_size, board_size)

        for k in [0, 1, 2, 3]:
            rot_state = np.rot90(state, k=k, axes=(1, 2))
            rot_probs = np.rot90(action_probs_2d, k=k)
            augmented_memory.append((rot_state.copy(), rot_probs.flatten(), outcome, for_train))

            flip_state = np.flip(rot_state, axis=2)
            flip_probs = np.flip(rot_probs, axis=1)
            augmented_memory.append((flip_state.copy(), flip_probs.flatten(), outcome, for_train))

    return augmented_memory


def random_augment_sample(state, action_probs, board_size):
    """
    对单个样本随机应用8种对称变换中的一种。
    
    8种变换包括：
    - 旋转 0°, 90°, 180°, 270°
    - 以及每种旋转后的水平翻转
    
    Args:
        state: 状态数组，形状为 (C, H, W)
        action_probs: 动作概率，形状为 (board_size^2,)
        board_size: 棋盘大小
        
    Returns:
        (transformed_state, transformed_probs): 变换后的状态和动作概率
    """
    # 随机选择变换类型：0-7
    # 0-3: 旋转 k*90° 不翻转
    # 4-7: 旋转 (k-4)*90° 后翻转
    transform_type = np.random.randint(0, 8)

    k = transform_type % 4  # 旋转次数
    do_flip = transform_type >= 4  # 是否翻转

    action_probs_2d = action_probs.reshape(board_size, board_size)

    # 先旋转
    rot_state = np.rot90(state, k=k, axes=(1, 2))
    rot_probs = np.rot90(action_probs_2d, k=k)

    # 再翻转（如果需要）
    if do_flip:
        rot_state = np.flip(rot_state, axis=2)
        rot_probs = np.flip(rot_probs, axis=1)

    return rot_state.copy(), rot_probs.flatten().copy()


def random_augment_batch(batch, board_size):
    """
    对一个batch的样本进行随机数据增强（仅用于正方形棋盘）。
    每个样本独立地随机选择8种变换中的一种。
    
    Args:
        batch: 样本列表，每个样本是 (state, action_probs, *others) 的元组
        board_size: 棋盘大小（正方形）
        
    Returns:
        增强后的batch（数量不变）
    """
    augmented_batch = []
    for item in batch:
        state = item[0]
        action_probs = item[1]
        others = item[2:]
        aug_state, aug_probs = random_augment_sample(state, action_probs, board_size)
        augmented_batch.append((aug_state, aug_probs) + others)
    return augmented_batch


def random_augment_sample_rect(state, action_probs, board_height, board_width):
    """
    对单个样本随机应用水平翻转变换（用于非正方形棋盘如Connect4）。
    
    非正方形棋盘只支持2种变换：
    - 不变换
    - 水平翻转
    
    Args:
        state: 状态数组，形状为 (C, H, W)
        action_probs: 动作概率，形状为 (board_height * board_width,)
        board_height: 棋盘高度
        board_width: 棋盘宽度
        
    Returns:
        (transformed_state, transformed_probs): 变换后的状态和动作概率
    """
    # 随机选择是否翻转
    do_flip = np.random.random() < 0.5

    if do_flip:
        # 水平翻转
        flip_state = np.flip(state, axis=2)
        action_probs_2d = action_probs.reshape(board_height, board_width)
        flip_probs = np.flip(action_probs_2d, axis=1)
        return flip_state.copy(), flip_probs.flatten().copy()
    else:
        return state.copy(), action_probs.copy()


def random_augment_batch_rect(batch, board_height, board_width):
    """
    对一个batch的样本进行随机数据增强（用于非正方形棋盘）。
    每个样本独立地随机选择是否水平翻转。
    
    Args:
        batch: 样本列表，每个样本是 (state, action_probs, *others) 的元组
        board_height: 棋盘高度
        board_width: 棋盘宽度
        
    Returns:
        增强后的batch（数量不变）
    """
    augmented_batch = []
    for item in batch:
        state = item[0]
        action_probs = item[1]
        others = item[2:]
        aug_state, aug_probs = random_augment_sample_rect(state, action_probs, board_height, board_width)
        augmented_batch.append((aug_state, aug_probs) + others)
    return augmented_batch


def random_augment_sample_connect4(state, action_probs):
    """
    对Connect4等列动作游戏的单个样本随机应用水平翻转。
    
    Connect4的动作只有列（7个），翻转时action_probs也需要反转顺序。
    
    Args:
        state: 状态数组，形状为 (C, H, W)
        action_probs: 动作概率，形状为 (num_columns,)
        
    Returns:
        (transformed_state, transformed_probs): 变换后的状态和动作概率
    """
    do_flip = np.random.random() < 0.5

    if do_flip:
        # 水平翻转状态
        flip_state = np.flip(state, axis=2)
        # 反转动作概率（列的顺序反转）
        flip_probs = np.flip(action_probs)
        return flip_state.copy(), flip_probs.copy()
    else:
        return state.copy(), action_probs.copy()


def random_augment_batch_connect4(batch):
    """
    对Connect4等列动作游戏的batch进行随机数据增强。
    
    Args:
        batch: 样本列表，每个样本是 (state, action_probs, *others) 的元组
        
    Returns:
        增强后的batch（数量不变）
    """
    augmented_batch = []
    for item in batch:
        state = item[0]
        action_probs = item[1]
        others = item[2:]
        aug_state, aug_probs = random_augment_sample_connect4(state, action_probs)
        augmented_batch.append((aug_state, aug_probs) + others)
    return augmented_batch


def drop_last(memory, batch_size):
    len_memory = len(memory)
    memory = memory[:len_memory - len_memory % batch_size]
    return memory


def add_dirichlet_noise_origin(policy, alpha, epsilon=0.25):
    noise = np.random.dirichlet([alpha] * len(policy))
    policy = (1 - epsilon) * policy + epsilon * noise
    return policy


def add_dirichlet_noise(policy, alpha, epsilon=0.25):
    policy = policy.copy()  # 避免原地修改传入的数组
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    noise = np.random.dirichlet(np.full(nonzero_count, alpha))
    policy[nonzero_mask] = (policy[nonzero_mask] * (1 - epsilon) + noise * epsilon)
    return policy


def root_temperature_transform(policy, current_step, args, board_size):
    decay_factor = math.pow(0.5, current_step / board_size)
    current_temp = args['root_temperature_final'] + (args['root_temperature_init'] - args['root_temperature_final']) * decay_factor
    new_policy = temperature_transform(policy, current_temp)
    return new_policy


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


def temperature_transform(probs, temp):
    probs = np.asarray(probs, dtype=np.float64)

    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("输入概率分布的和必须为 1。")

    if temp == 0:
        zero_mask = (probs == 0)
        non_zero_prob = probs[~zero_mask]
        if len(non_zero_prob) == 0:
            return np.zeros_like(probs)

        max_val = non_zero_prob.max()
        max_mask = (probs == max_val)
        count = max_mask.sum()
        scaled = np.zeros_like(probs, dtype=np.float64)
        scaled[max_mask] = 1.0 / count
        return scaled

    non_zero_mask = (probs != 0)
    if not non_zero_mask.any():
        return np.zeros_like(probs)

    exponent = 1.0 / temp
    log_probs = np.log(probs[non_zero_mask])
    log_scaled = exponent * log_probs

    log_scaled -= log_scaled.max()

    exp_probs = np.exp(log_scaled)

    log_sum_exp = np.log(exp_probs.sum())
    log_probs_normalized = log_scaled - log_sum_exp
    probs_normalized = np.exp(log_probs_normalized)

    scaled = np.zeros_like(probs, dtype=np.float64)
    scaled[non_zero_mask] = probs_normalized

    return scaled


def sample_truncated_normal(mu, sigma, n_samples=1000):
    """
    从截断正态分布 (x > 0) 中采样。

    参数:
        mu (float): 正态分布的均值
        sigma (float): 正态分布的标准差
        n_samples (int): 采样数量，默认为1000

    返回:
        numpy.ndarray: 服从截断正态分布 (x > 0) 的样本
    """
    # 截断范围：x > 0
    lower_bound = 0
    upper_bound = np.inf

    # 标准化截断界限
    a = (lower_bound - mu) / sigma
    b = (upper_bound - mu) / sigma

    # 生成样本
    samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples)

    if n_samples == 1:
        samples = samples[0]

    return samples


if __name__ == '__main__':
    mu = 1
    sigma = 1
    samples = sample_truncated_normal(mu, sigma, n_samples=1000000)
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='g', label='Sampled Data')
    plt.title('Samples from Truncated Normal Distribution (x > 0)')
    plt.xlabel('Sample Value')
    plt.ylabel('Density')
    plt.grid()
    plt.show()
