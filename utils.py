import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import truncnorm


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
        memory: 样本列表，每个样本是 (state, action_probs, outcome, num_sims) 的元组
        board_size: 棋盘大小
        
    Returns:
        增强后的样本列表（原始数量的8倍）
    """
    augmented_memory = []

    for state, action_probs, outcome, num_sims in memory:

        action_probs_2d = action_probs.reshape(board_size, board_size)

        for k in [0, 1, 2, 3]:
            rot_state = np.rot90(state, k=k, axes=(1, 2))
            rot_probs = np.rot90(action_probs_2d, k=k)
            augmented_memory.append((rot_state.copy(), rot_probs.flatten(), outcome, num_sims))

            flip_state = np.flip(rot_state, axis=2)
            flip_probs = np.flip(rot_probs, axis=1)
            augmented_memory.append((flip_state.copy(), flip_probs.flatten(), outcome, num_sims))

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
    对一个batch的样本进行随机数据增强。
    每个样本独立地随机选择8种变换中的一种。
    
    Args:
        batch: 样本列表，每个样本是 (state, action_probs, outcome, num_sims) 的元组
        board_size: 棋盘大小
        
    Returns:
        增强后的batch（数量不变）
    """
    augmented_batch = []
    for state, action_probs, outcome, num_sims in batch:
        aug_state, aug_probs = random_augment_sample(state, action_probs, board_size)
        augmented_batch.append((aug_state, aug_probs, outcome, num_sims))
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
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    noise = np.random.dirichlet(np.full(nonzero_count, alpha))
    policy[nonzero_mask] = (policy[nonzero_mask] * (1 - epsilon) + noise * epsilon)
    return policy


def add_dirichlet_noise_sm(policy, epsilon=0.25):
    nonzero_mask = policy > 0
    nonzero_count = np.sum(nonzero_mask)
    alpha = 1 / math.sqrt(nonzero_count)
    noise = np.random.dirichlet(np.full(nonzero_count, alpha))
    policy[nonzero_mask] = (policy[nonzero_mask] * (1 - epsilon) + noise * epsilon)
    return policy


#
# def temperature_transform(probs: np.ndarray, temperature: float) -> np.ndarray:
#     """
#     对概率分布应用温度变换（简化版，低检查，高性能）。
#
#     假设:
#         - probs 是一个 NumPy 一维数组。
#         - probs 的元素都是非负的。
#         - temperature 是一个非负浮点数。
#
#     参数:
#         probs (np.ndarray): 一维 NumPy 数组，表示概率分布。可能包含0。
#         temperature (float): 温度参数 T (T >= 0)。
#
#     返回:
#         np.ndarray: 经过温度变换后的新概率分布数组。
#                     原始为0的项将保持为0（在数值精度范围内）。
#     """
#
#     # --- T = 0 (极限情况：one-hot 或均分给最大值) ---
#     if temperature == 0:
#         # 检查是否全零，如果是则直接返回全零
#         if np.all(probs == 0):
#             return np.zeros_like(probs, dtype=float)  # 或者 probs.dtype 如果想保持
#
#         max_prob = np.max(probs)
#         is_max = (probs == max_prob)
#         num_max = np.sum(is_max)  # 计算有多少个最大值
#
#         transformed_probs = np.zeros_like(probs, dtype=float)  # 使用 float 保证能存放 1.0/num_max
#         transformed_probs[is_max] = 1.0 / num_max
#         return transformed_probs
#
#     # --- T = 1 (分布不变) ---
#     # 直接比较浮点数可能不精确，但为了效率这里直接比较
#     # 如果担心 T=1.000000001 的情况，可以用 np.isclose(temperature, 1.0)
#     if temperature == 1.0:
#         return probs  # 直接返回原始数组（或 probs.copy() 如果不想被修改）
#
#     # --- T > 0 且 T != 1 ---
#
#     # 1. 计算 p_i^(1/T)
#     #    不使用 .astype(np.float64)，使用输入数组的原始类型进行计算
#     #    这可能更快，但在极端温度下精度较低或更容易溢出/下溢
#     exponent = 1.0 / temperature
#     # 使用 errstate 避免打印警告，对性能影响很小
#     with np.errstate(invalid='ignore'):  # 0^x 可能触发 invalid 但结果是0
#         powered_probs = np.power(probs, exponent)
#
#     # 2. 处理上溢导致的 inf (下溢会自动变0，后续处理)
#     powered_probs = np.nan_to_num(powered_probs, nan=0.0, posinf=np.inf)  # nan基本不会出现（除非输入有nan）
#
#     # 3. 归一化
#     sum_powered_probs = np.sum(powered_probs)
#
#     # 处理和为0（下溢）或无穷大（上溢）的特殊情况
#     if sum_powered_probs == 0:
#         # 所有项都下溢为0，行为类似于T=0
#         # （代码复用 T=0 的逻辑）
#         if np.all(probs == 0):
#             return np.zeros_like(probs, dtype=float)
#         max_prob = np.max(probs)
#         is_max = (probs == max_prob)
#         num_max = np.sum(is_max)
#         transformed_probs = np.zeros_like(probs, dtype=float)
#         # 防止 num_max 为 0 （虽然理论上如果 probs 不全零，num_max >= 1）
#         if num_max > 0:
#             transformed_probs[is_max] = 1.0 / num_max
#         return transformed_probs
#
#     elif np.isinf(sum_powered_probs):
#         # 存在上溢项
#         is_inf = np.isinf(powered_probs)
#         num_inf = np.sum(is_inf)
#         transformed_probs = np.zeros_like(probs, dtype=float)
#         if num_inf > 0:
#             transformed_probs[is_inf] = 1.0 / num_inf
#         # 如果 sum 是 inf 但没有具体的 inf 项（极不可能），则返回全零
#         return transformed_probs
#     else:
#         # 正常归一化
#         # 移除了强制置零和二次归一化步骤
#         transformed_probs = powered_probs / sum_powered_probs
#         # 注意：极小的浮点误差可能导致原为0的位置出现非常小的非零值
#         # 如果严格要求0必须是精确的0.0，则需要加回 transformed_probs[probs == 0] = 0.0
#
#     return transformed_probs
#

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
