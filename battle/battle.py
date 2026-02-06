"""
两个AlphaZero模型对战脚本
自动读取battle文件夹里的两个checkpoint，进行对战并输出胜率
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from alphazero import AlphaZero, MCTS
from connect4.connect4 import Connect4, print_board
from gomoku_9.gomoku import Gomoku
from nets import ResNet


def load_model_from_checkpoint(checkpoint_path, game, args):
    """从checkpoint加载模型"""
    model = ResNet(game, num_blocks=4, num_channels=256).to(args['device'])
    # model = ResNet(game, num_blocks=2, num_channels=128).to(args['device'])

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def play_game(mcts1, mcts2, game, verbose=False):
    """
    两个MCTS对战一局
    mcts1 执黑（先手，to_play=1）
    mcts2 执白（后手，to_play=-1）
    返回: 获胜者 (1=mcts1胜, -1=mcts2胜, 0=平局)
    """
    state = game.get_initial_state()
    to_play = 1
    
    while not game.is_terminal(state):
        if to_play == 1:
            action_probs = mcts1.search(state, to_play)
        else:
            action_probs = mcts2.search(state, to_play)
        
        # 贪婪选择最优动作
        action = np.argmax(action_probs)
        state = game.get_next_state(state, action, to_play)
        to_play = -to_play
    
    winner = game.get_winner(state)
    
    if verbose:
        print_board(state)
        if winner == 1:
            print("模型1（先手）获胜!")
        elif winner == -1:
            print("模型2（后手）获胜!")
        else:
            print("平局!")
    
    return winner


def battle(num_games=100, num_simulations=200, verbose=False):
    """
    两个AlphaZero对战
    每对checkpoint会进行两轮对战（交换先后手），确保公平
    """
    # 查找battle文件夹中的checkpoint文件
    battle_dir = os.path.dirname(__file__)
    checkpoint_files = glob.glob(os.path.join(battle_dir, "*.pth"))
    
    if len(checkpoint_files) < 2:
        print(f"错误: battle文件夹中需要至少2个checkpoint文件，当前只有 {len(checkpoint_files)} 个")
        return
    
    # 按修改时间排序，选择最新的两个（或者按名字排序）
    checkpoint_files = sorted(checkpoint_files, key=os.path.getmtime)
    
    checkpoint1 = checkpoint_files[0]
    checkpoint2 = checkpoint_files[1]
    
    print(f"模型1 (较早): {os.path.basename(checkpoint1)}")
    print(f"模型2 (较新): {os.path.basename(checkpoint2)}")
    print(f"对战局数: {num_games} (每个模型各执先手 {num_games // 2} 局)")
    print(f"每步MCTS模拟次数: {num_simulations}")
    print("-" * 60)
    
    # 初始化游戏和参数
    game = Gomoku(board_size=9, history_step=4)
    # game = Connect4(history_step=3)
    args = {
        'mode': 'eval',
        'num_simulations': num_simulations,
        'c_puct': 1.4,
        'Q_norm_bounds': [-1, 1],
        'dirichlet_epsilon': 0,  # 评估时不加噪声
        'dirichlet_alpha': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"使用设备: {args['device']}")
    
    # 加载两个模型
    model1 = load_model_from_checkpoint(checkpoint1, game, args)
    model2 = load_model_from_checkpoint(checkpoint2, game, args)
    
    # 创建MCTS
    mcts1 = MCTS(game, args, model1)
    mcts2 = MCTS(game, args, model2)
    
    # 统计结果
    # model1_wins: 模型1赢的局数
    # model2_wins: 模型2赢的局数
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    # 第一轮：模型1执先手
    print("\n第一轮: 模型1(较早)执先手...")
    half_games = num_games // 2
    for i in tqdm(range(half_games), desc="模型1先手"):
        winner = play_game(mcts1, mcts2, game, verbose=verbose)
        if winner == 1:
            model1_wins += 1
        elif winner == -1:
            model2_wins += 1
        else:
            draws += 1
    
    print(f"  模型1先手结果: 模型1胜{model1_wins}, 模型2胜{model2_wins}, 平局{draws}")
    
    # 记录第一轮结果
    round1_m1_wins = model1_wins
    round1_m2_wins = model2_wins
    round1_draws = draws
    
    # 第二轮：模型2执先手（交换位置）
    print("\n第二轮: 模型2(较新)执先手...")
    for i in tqdm(range(half_games), desc="模型2先手"):
        winner = play_game(mcts2, mcts1, game, verbose=verbose)
        if winner == 1:  # mcts2（模型2）赢
            model2_wins += 1
        elif winner == -1:  # mcts1（模型1）赢
            model1_wins += 1
        else:
            draws += 1
    
    round2_m1_wins = model1_wins - round1_m1_wins
    round2_m2_wins = model2_wins - round1_m2_wins
    round2_draws = draws - round1_draws
    print(f"  模型2先手结果: 模型2胜{round2_m2_wins}, 模型1胜{round2_m1_wins}, 平局{round2_draws}")
    
    # 输出总结果
    print("\n" + "=" * 60)
    print("对战结果汇总")
    print("=" * 60)
    print(f"模型1 (较早): {os.path.basename(checkpoint1)}")
    print(f"模型2 (较新): {os.path.basename(checkpoint2)}")
    print("-" * 60)
    print(f"总对战局数: {num_games}")
    print(f"模型1获胜: {model1_wins} 局 ({100 * model1_wins / num_games:.1f}%)")
    print(f"模型2获胜: {model2_wins} 局 ({100 * model2_wins / num_games:.1f}%)")
    print(f"平局: {draws} 局 ({100 * draws / num_games:.1f}%)")
    print("-" * 60)
    
    # 计算胜率（不计平局）
    decided_games = model1_wins + model2_wins
    if decided_games > 0:
        print(f"胜率（不计平局）:")
        print(f"  模型1: {100 * model1_wins / decided_games:.1f}%")
        print(f"  模型2: {100 * model2_wins / decided_games:.1f}%")
    
    print("=" * 60)
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'total_games': num_games
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaZero对战')
    parser.add_argument('--games', type=int, default=10, help='对战局数')
    parser.add_argument('--simulations', type=int, default=600, help='每步MCTS模拟次数')
    parser.add_argument('--verbose', action='store_true', help='是否打印每局棋盘')
    
    args = parser.parse_args()
    
    battle(num_games=args.games, num_simulations=args.simulations, verbose=args.verbose)
