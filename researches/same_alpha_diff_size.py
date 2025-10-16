import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)


# 定义函数：生成并可视化 Dirichlet 噪声
def visualize_dirichlet_noise(board_size, alpha_value, ax, title):
    # 合法动作数 = board_size * board_size
    num_actions = board_size ** 2
    # 生成 alpha 向量，所有值相同
    alpha = [alpha_value] * num_actions
    # 从 Dirichlet 分布采样
    noise = np.random.dirichlet(alpha)
    # 将一维噪声向量重塑为二维棋盘
    noise_grid = noise.reshape(board_size, board_size)

    # 绘制热图，颜色从黑（0）到白（1）
    im = ax.imshow(noise_grid, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(board_size))
    ax.set_yticks(np.arange(board_size))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 在每个格子上标注数值
    # for i in range(board_size):
    #     for j in range(board_size):
    #         ax.text(j, i, f'{noise_grid[i, j]:.2f}',
    #                 ha='center', va='center', color='red' if noise_grid[i, j] < 0.5 else 'blue')

    return im  # 返回 im 对象以供外部使用


# 设置参数
alpha_value = 0.3  # AlphaZero 常用值
board_sizes = [2, 3, 4]  # 不同棋盘大小：2x2, 3x3, 4x4

# 创建子图
fig, axes = plt.subplots(1, len(board_sizes), figsize=(15, 5))
fig.suptitle(f'Dirichlet Noise Visualization (alpha = {alpha_value})', fontsize=16)

# 为每个棋盘大小生成并可视化噪声，并保存最后一个 im
ims = []
for i, size in enumerate(board_sizes):
    im = visualize_dirichlet_noise(size, alpha_value, axes[i], f'{size}x{size} Board')
    ims.append(im)

# 调整布局并添加颜色条，使用最后一个 im
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.colorbar(ims[-1], ax=axes, orientation='vertical', fraction=0.05, pad=0.05)
plt.show()