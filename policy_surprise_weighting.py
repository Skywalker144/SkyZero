"""
Policy Surprise Weighting (PSW) - KataGo 方法实现

来源: KataGo Methods (https://github.com/lightvector/KataGo)

核心思想:
    KataGo 通过增加那些"策略训练目标相对于策略先验非常令人惊讶"的训练样本的
    采样频率来改进学习效率。这是 KataGo 训练中最重要的改进之一。

算法:
    1. 在一局游戏的所有 full-searched 位置中，计算每个位置的 KL 散度:
       KL(policy_target || policy_prior)
    
    2. 重新分配频率权重:
       - 约一半的总权重均匀分配（基础权重）
       - 另一半权重按照 KL 散度成比例分配
    
    3. 每个位置被写入训练数据 floor(frequency_weight) 次，
       再以 (frequency_weight - floor(frequency_weight)) 的概率额外写入一次

注意:
    - 这不是重要性采样（不会按比例缩小梯度权重）
    - 使用的是带噪声的 softmax root policy prior
    - 可以允许 "fast" search 如果 KL 散度非常大也获得非零权重
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_kl_divergence(policy_target: np.ndarray, policy_prior: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算从 policy_prior 到 policy_target 的 KL 散度
    
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    
    其中 P = policy_target, Q = policy_prior
    
    Args:
        policy_target: 策略训练目标 (来自 MCTS 搜索)
        policy_prior: 策略先验 (来自神经网络，可能带有 Dirichlet 噪声)
        epsilon: 小常数，用于数值稳定性
    
    Returns:
        KL 散度值 (非负)
    """
    # 确保输入是有效的概率分布
    p = np.asarray(policy_target, dtype=np.float64)
    q = np.asarray(policy_prior, dtype=np.float64)
    
    # 添加 epsilon 防止 log(0) 和除以 0
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # 重新归一化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 只在 p > epsilon 的位置计算 KL（避免 0 * log(0) 问题）
    mask = p > epsilon
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    return max(0.0, kl)  # KL 散度理论上非负


def compute_policy_surprise_weights(
    game_data: List[Tuple],
    baseline_weight_ratio: float = 0.5,
    min_weight: float = 0.0,
) -> List[float]:
    """
    计算一局游戏中每个位置的 Policy Surprise Weight (PSW)
    
    基于 KataGo 原版逻辑 (cpp/program/play.cpp):
    - 动态计算 fast search 的包含阈值
    - 混合"基础权重"和"惊奇权重"
    - 保持总权重和(即样本总数)与原始 full search 样本数一致
    
    Args:
        game_data: 一局游戏的数据 (encoded_state, policy_target, outcome, for_train, policy_prior)
        baseline_weight_ratio: 基础权重(均匀部分)的比例。对应 KataGo 的 (1 - policySurpriseDataWeight)。
                             剩余部分 (1 - baseline_weight_ratio) 将根据 surprise 大小分配。
        min_weight: 最小权重 (KataGo 默认为 0，但可以设置小值防止完全饿死)
    
    Returns:
        每个位置的频率权重列表
    """
    n_positions = len(game_data)
    if n_positions == 0:
        return []
    
    # 1. 解析数据并计算 KL 散度 (Surprise)
    target_weights = []   # 对应 KataGo 的 targetWeight (Full search=1, Fast=0)
    surprises = []        # 对应 KataGo 的 policySurprise
    
    full_search_count = 0
    
    for sample in game_data:
        # 解包数据 (encoded_state, policy_target, outcome, for_train, policy_prior)
        if len(sample) >= 5:
            policy_target = sample[1]
            for_train = sample[3]
            policy_prior = sample[4]
            
            # Full search (for_train=True) 的 target_weight 为 1.0
            # Fast search (for_train=False) 的 target_weight 为 0.0
            w = 1.0 if for_train else 0.0
            target_weights.append(w)
            
            if for_train:
                full_search_count += 1
            
            # 计算 Surprise (KL 散度)
            kl = compute_kl_divergence(policy_target, policy_prior)
            surprises.append(kl)
        else:
            # 数据格式不完整，默认权重
            target_weights.append(1.0)
            surprises.append(0.0)
            full_search_count += 1

    sum_weights = sum(target_weights)  # 实际上等于 full_search_count
    
    # 如果没有 full search 样本，直接返回全 0 (或者抛出异常)
    if sum_weights <= 1e-8:
        return [0.0] * n_positions

    # 2. 计算平均 Surprise 和 Fast Search 的阈值
    # KataGo: double averagePolicySurpriseWeighted = sumPolicySurpriseWeighted / sumWeights;
    sum_surprise_weighted = sum(s * w for s, w in zip(surprises, target_weights))
    avg_surprise = sum_surprise_weighted / sum_weights
    
    # KataGo: double thresholdToIncludeReduced = averagePolicySurpriseWeighted * 1.5;
    threshold_to_include_reduced = avg_surprise * 1.5
    
    # 3. 计算 Surprise Proportional Value
    # 这是每个样本在"惊奇权重"部分的相对贡献份额
    surprise_prop_values = []
    for s, w in zip(surprises, target_weights):
        # KataGo 逻辑:
        # policySurprisePropValue = targetWeight * policySurprise + (1-targetWeight) * max(0.0, policySurprise-threshold);
        # 对于 Full Search (w=1): prop = surprise
        # 对于 Fast Search (w=0): prop = max(0, surprise - threshold)
        prop = w * s + (1.0 - w) * max(0.0, s - threshold_to_include_reduced)
        surprise_prop_values.append(prop)
        
    sum_surprise_prop = sum(surprise_prop_values)
    sum_surprise_prop = max(sum_surprise_prop, 1e-10) # 防止除以 0
    
    # 4. 计算最终权重
    # KataGo 逻辑:
    # newValue = (1.0 - policySurpriseDataWeight) * targetWeight
    #          + policySurpriseDataWeight * policySurprisePropValue * sumWeights / sumPolicySurprisePropValue;
    
    policy_surprise_data_weight = 1.0 - baseline_weight_ratio
    final_weights = []
    
    for i in range(n_positions):
        w = target_weights[i]
        prop = surprise_prop_values[i]
        
        # 基础部分 (Uniform-ish)
        term1 = baseline_weight_ratio * w
        
        # 惊奇部分 (Proportional)
        # 注意: sumWeights / sumProp 是归一化系数，确保惊奇部分的总权重也等于 sumWeights
        term2 = policy_surprise_data_weight * prop * (sum_weights / sum_surprise_prop)
        
        weight = term1 + term2
        
        # 应用最小权重 (仅对非零权重的项，或者对所有 Full Search 项? 
        # KataGo 没有显式 min_weight，但为了稳定性保留此功能，主要针对 Full Search)
        if w > 0 and weight < min_weight:
            weight = min_weight
            
        final_weights.append(weight)

    return final_weights



def apply_surprise_weighting_to_game(game_data: List[Tuple], weights: List[float], ) -> List[Tuple]:
    """
    根据 Policy Surprise Weights 重复/采样游戏数据
    
    KataGo 的方法:
    - 每个位置被写入训练数据 floor(weight) 次
    - 再以 (weight - floor(weight)) 的概率额外写入一次
    
    注意: 输出保留原始 sample 格式不变（包含 policy_prior）。
    调用方应负责在写入 replay buffer 前移除多余字段。
    
    *** 修改: 将所有输出样本的 for_train (index 3) 强制设为 True ***
    因为如果样本被 PSW 选中（即使它原本是 fast search），说明它具有较高的 KL 散度，
    值得用于策略训练。
    
    Args:
        game_data: 原始游戏数据
        weights: 每个位置的频率权重
    
    Returns:
        按权重重复后的游戏数据，且 for_train=True
    """
    weighted_data = []
    
    for sample, weight in zip(game_data, weights):
        if weight <= 0:
            continue
        
        # 强制设置 for_train = True (index 3)
        # sample 是 tuple，不可变，需要重构
        # (encoded_state, policy_target, outcome, for_train, policy_prior)
        if len(sample) >= 4:
            # 保持前3个元素，替换第4个为 True，保持剩下的
            sample_to_add = sample[:3] + (True,) + sample[4:]
        else:
            sample_to_add = sample

        # 计算重复次数
        floor_weight = int(np.floor(weight))
        fractional = weight - floor_weight
        
        # 添加 floor(weight) 次
        for _ in range(floor_weight):
            weighted_data.append(sample_to_add)

        # 以weight - floor(weight) 的概率额外添加一次
        if np.random.random() < fractional:
            weighted_data.append(sample_to_add)
    
    return weighted_data


class PolicySurpriseWeighter:
    """
    Policy Surprise Weighting 管理器
    
    封装 PSW 的配置和应用逻辑，便于集成到训练流程中
    """
    
    def __init__(
        self,
        baseline_weight_ratio: float = 0.5,
        min_weight: float = 0.01,
    ):
        """
        初始化 PSW 管理器
        
        Args:
            baseline_weight_ratio: 均匀分配的权重比例 (1 - policySurpriseDataWeight)
            min_weight: 最小权重
        """
        self.baseline_weight_ratio = baseline_weight_ratio
        self.min_weight = min_weight
        
        # 统计信息
        self.total_samples_before = 0
        self.total_samples_after = 0
        self.total_games = 0
        self.total_kl_sum = 0.0
    
    def process_game(self, game_data: List[Tuple]) -> Tuple[List[Tuple], dict]:
        """
        处理一局游戏的数据，应用 Policy Surprise Weighting
        
        Args:
            game_data: 原始游戏数据，每个元素应包含 policy_prior
        
        Returns:
            (weighted_data, stats): 加权后的数据和统计信息
        """
        if len(game_data) == 0:
            return game_data, {'enabled': False}
        
        # 计算权重
        weights = compute_policy_surprise_weights(
            game_data,
            baseline_weight_ratio=self.baseline_weight_ratio,
            min_weight=self.min_weight,
        )
        
        # 应用权重
        weighted_data = apply_surprise_weighting_to_game(
            game_data, weights
        )
        
        # 更新统计
        self.total_samples_before += len(game_data)
        self.total_samples_after += len(weighted_data)
        self.total_games += 1
        
        # 计算 KL 统计
        kl_values = []
        for sample in game_data:
            if len(sample) >= 5:
                policy_target, policy_prior = sample[1], sample[4]
                kl = compute_kl_divergence(policy_target, policy_prior)
                kl_values.append(kl)
                self.total_kl_sum += kl
        
        stats = {
            'samples_before': len(game_data),
            'samples_after': len(weighted_data),
            'expansion_ratio': len(weighted_data) / max(1, len(game_data)),
            'weights': weights,
            'kl_mean': np.mean(kl_values) if kl_values else 0.0,
            'kl_max': np.max(kl_values) if kl_values else 0.0,
            'kl_std': np.std(kl_values) if kl_values else 0.0,
        }
        
        return weighted_data, stats
    
    def get_global_stats(self) -> dict:
        """获取全局统计信息"""
        return {
            'total_games': self.total_games,
            'total_samples_before': self.total_samples_before,
            'total_samples_after': self.total_samples_after,
            'overall_expansion_ratio': (
                self.total_samples_after / max(1, self.total_samples_before)
            ),
            'avg_kl': (
                self.total_kl_sum / max(1, self.total_samples_before)
            ),
        }


# ============== 用于 MCTS 集成的辅助函数 ==============

def extract_policy_prior_from_root(root_node, action_space_size: int) -> np.ndarray:
    """
    从 MCTS 根节点提取策略先验
    
    Args:
        root_node: MCTS 根节点
        action_space_size: 动作空间大小
    
    Returns:
        policy_prior: 策略先验数组
    """
    policy_prior = np.zeros(action_space_size)
    
    if hasattr(root_node, 'children') and root_node.children:
        for child in root_node.children:
            if hasattr(child, 'action_taken') and hasattr(child, 'prior'):
                policy_prior[child.action_taken] = child.prior
    
    # 归一化
    prior_sum = np.sum(policy_prior)
    if prior_sum > 0:
        policy_prior /= prior_sum
    
    return policy_prior


# ============== 测试代码 ==============

if __name__ == '__main__':
    # 简单测试
    print("Testing Policy Surprise Weighting...")
    
    # 测试 KL 散度计算
    p = np.array([0.8, 0.1, 0.1])
    q = np.array([0.3, 0.3, 0.4])
    kl = compute_kl_divergence(p, q)
    print(f"KL(p||q) = {kl:.4f}")
    
    # 测试相同分布的 KL = 0
    kl_same = compute_kl_divergence(p, p)
    print(f"KL(p||p) = {kl_same:.4f} (should be ~0)")
    
    # 测试 PSW 权重计算
    # 模拟游戏数据: (state, policy_target, outcome, for_train, policy_prior)
    mock_game_data = [
        (None, np.array([0.9, 0.05, 0.05]), 1, True, np.array([0.3, 0.3, 0.4])),   # full search, 高 surprise
        (None, np.array([0.3, 0.3, 0.4]), 1, True, np.array([0.3, 0.3, 0.4])),     # full search, 低 surprise
        (None, np.array([0.5, 0.3, 0.2]), 1, True, np.array([0.4, 0.3, 0.3])),     # full search, 中等 surprise
        (None, np.array([0.7, 0.2, 0.1]), 1, False, np.array([0.2, 0.4, 0.4])),    # fast search, 高 KL
        (None, np.array([0.4, 0.3, 0.3]), 1, False, np.array([0.4, 0.3, 0.3])),    # fast search, 低 KL
    ]
    
    weights = compute_policy_surprise_weights(mock_game_data)
    
    print(f"\nWeights: {weights}")
    print(f"Sum of weights: {sum(weights):.4f}")
    
    # 测试 PSW 管理器
    weighter = PolicySurpriseWeighter(
        baseline_weight_ratio=0.5,
    )
    
    weighted_data, stats = weighter.process_game(mock_game_data)
    print(f"\nPSW Stats: {stats}")
    print(f"Original samples: {len(mock_game_data)}, Weighted samples: {len(weighted_data)}")
    
    print("\nPolicy Surprise Weighting test completed!")

