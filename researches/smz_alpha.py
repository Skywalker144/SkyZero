import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet


def plot_dirichlet_heatmap(board_sizes):
    # 根据 board_sizes 计算 num_legal_actions 和 alpha_values
    alpha_values = [1 / np.sqrt(board_size * board_size) for board_size in board_sizes]
    num_cols = len(board_sizes)  # 列数等于 board_sizes 数量

    # 创建多图布局
    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(5 * num_cols, 5),  # 单行布局
        constrained_layout=True
    )

    # 如果只有一个 board_size，确保 axes 是一维数组
    if num_cols == 1:
        axes = np.array([axes])

    # 对每种组合进行绘图
    for i, (board_size, alpha) in enumerate(zip(board_sizes, alpha_values)):
        # 创建 Dirichlet 分布参数
        num_legal_actions = board_size * board_size
        alpha_vector = np.ones(num_legal_actions) * alpha

        # 生成一个样本并重塑为棋盘形式
        sample = dirichlet.rvs(alpha_vector, size=1)[0]
        board = sample.reshape(board_size, board_size)

        # 绘制灰度热图（高概率为白色）
        ax = axes[i]
        im = ax.imshow(board, cmap='gray', vmin=0, vmax=np.max(board))  # gray_r 反转灰度

        # 设置标题
        ax.set_title(
            f'Board={board_size}x{board_size}\nα={alpha:.3f}',
            fontsize=10, pad=10
        )

        # 添加网格线
        ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, board_size, 1))
        ax.set_yticks(np.arange(-0.5, board_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # 添加颜色条
    plt.colorbar(
        im, ax=axes.ravel().tolist(), orientation='vertical',
        shrink=0.8, pad=0.05
    )

    plt.show()


# 测试不同棋盘大小
board_sizes = [4, 10, 15]  # 对应 num_legal_actions = 4, 25, 100
plot_dirichlet_heatmap(board_sizes)
