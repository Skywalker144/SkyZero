import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet


def plot_dirichlet_heatmap(alpha_values, board_sizes):
    # 创建多图布局，增加高度和宽度以避免重叠
    fig, axes = plt.subplots(
        len(board_sizes), len(alpha_values),
        figsize=(5 * len(alpha_values), 5 * len(board_sizes)),
        constrained_layout=True
    )

    # 如果只有一个board_size或alpha_value，确保axes是二维数组
    if len(board_sizes) == 1 and len(alpha_values) == 1:
        axes = np.array([[axes]])
    elif len(board_sizes) == 1:
        axes = axes.reshape(1, -1)
    elif len(alpha_values) == 1:
        axes = axes.reshape(-1, 1)

    # 对每种组合进行绘图
    for i, board_size in enumerate(board_sizes):
        for j, alpha in enumerate(alpha_values):
            # 创建Dirichlet分布参数
            alpha_vector = np.ones(board_size * board_size) * alpha

            # 生成一个样本并重塑为棋盘形式
            sample = dirichlet.rvs(alpha_vector, size=1)[0]
            board = sample.reshape(board_size, board_size)

            # 绘制灰度热图
            ax = axes[i, j]
            im = ax.imshow(board, cmap='gray', vmin=0, vmax=np.max(board))

            # 设置标题，调整位置和字体大小
            ax.set_title(
                f'Board={board_size}x{board_size}, α={alpha}',
                fontsize=10, pad=10
            )

            # 添加网格线
            ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(-0.5, board_size, 1))
            ax.set_yticks(np.arange(-0.5, board_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # 添加颜色条，调整大小和位置
    plt.colorbar(
        im, ax=axes.ravel().tolist(), orientation='vertical',
        shrink=0.8, pad=0.05
    )

    # 使用constrained_layout替代tight_layout以更好地自动调整间距
    # plt.tight_layout() 已由constrained_layout=True替代

    plt.show()


# 测试不同参数
# alpha_values = [0.03, 0.1, 0.3, 1]
# board_sizes = [9, 19]
alpha_values = [0.03, 0.1, 0.3, 1]
board_sizes = [2, 5, 9, 15, 19]

plot_dirichlet_heatmap(alpha_values, board_sizes)
