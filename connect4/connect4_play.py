"""
Connect4 人机对战脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.optim as optim
import numpy as np

from alphazero import AlphaZero
from connect4 import Connect4, print_board
from nets import ResNet


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    game = Connect4(history_step=3)
    model = ResNet(game, num_blocks=2, num_channels=128).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    args = {
        'mode': 'eval',
        'num_simulations': 800,  # 对战时可以用更多模拟
        'c_puct': 1.4,
        'temperature': 0.1,
        'dirichlet_epsilon': 0,
        'dirichlet_alpha': 0.5,
        'buffer_size': 10000,
        'file_name': 'connect4',
        'Q_norm_bounds': [-1, 1],
        'device': 'cuda'
    }
    
    print(f"Connect4 AlphaZero vs Human")
    print(f"Board Size: {game.board_height} x {game.board_width}")
    print(f"Action Space: {game.action_space_size} (columns 0-6)")
    print()

    alphazero = AlphaZero(game, model, optimizer, args)
    alphazero.load_checkpoint()
    
    print()
    print("=" * 50)
    print("游戏说明:")
    print("- 输入列号(0-6)来落子")
    print("- 棋子会落到该列的最低空位")
    print("- ● 表示玩家 1 (先手), ○ 表示玩家 -1 (后手)")
    print("=" * 50)
    print()
    
    to_play = int(input(
        '选择先后手:\n'
        '  1 = 你先手 (执 ●)\n'
        ' -1 = AI先手 (你执 ○)\n'
        '请输入: '
    ))
    
    color = 1  # 当前落子方
    state = game.get_initial_state()
    print("\n初始棋盘:")
    print_board(state)
    
    while not game.is_terminal(state):
        legal_actions = game.get_is_legal_actions(state)
        legal_cols = [i for i in range(game.board_width) if legal_actions[i]]
        
        if to_play == 1:
            # 人类玩家
            while True:
                try:
                    move = input(f"你的回合 (可选列: {legal_cols}): ")
                    col = int(move.strip())
                    if col in legal_cols:
                        break
                    else:
                        print(f"无效列号，请选择: {legal_cols}")
                except ValueError:
                    print("请输入有效的列号(0-6)")
            
            action = col
            state = game.get_next_state(state, action, color)
            print(f"\n你落子在第 {col} 列")
            
        elif to_play == -1:
            # AI玩家
            print(f'AI思考中...')
            action, info = alphazero.play(state, color)
            state = game.get_next_state(state, action, color)
            
            action_probs = info['action_probs']
            policy = info['policy']
            value = info['value']
            
            print(f"\nAI落子在第 {action} 列")
            print(f"MCTS访问概率: {action_probs}")
            print(f"神经网络策略: {policy}")
            print(f"AI评估值: {value:.3f} (AI胜率: {info['ai_winrate']*100:.1f}%)")
        
        to_play = -to_play
        color = -color
        print()
        print_board(state)
    
    # 游戏结束
    winner = game.get_winner(state)
    print()
    print("=" * 50)
    if winner == 1:
        print("游戏结束！先手 ● 获胜！")
    elif winner == -1:
        print("游戏结束！后手 ○ 获胜！")
    else:
        print("游戏结束！平局！")
    print("=" * 50)
