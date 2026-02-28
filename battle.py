import os
import sys
import importlib.util
import torch
import numpy as np
from tqdm import tqdm

# Add current directory to path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alphazero_play import TreeReuseAlphaZero
from nets import ResNet
from envs.gomoku import Gomoku
from utils import print_board

def load_args_from_path(args_path):
    # Add the directory containing the script to sys.path so it can find local imports like gomoku_train
    script_dir = os.path.dirname(os.path.abspath(args_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    spec = importlib.util.spec_from_file_location("play_script", args_path)
    if spec is None:
        raise ImportError(f"Could not load module spec from {args_path}")
    if spec.loader is None:
        raise ImportError(f"No loader for module spec from {args_path}")
        
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
        
    # In gomoku_play.py, eval_args is defined and train_args is imported.
    if not hasattr(module, "eval_args"):
        raise AttributeError(f"Module {args_path} does not have 'eval_args'")
    if not hasattr(module, "train_args"):
        raise AttributeError(f"Module {args_path} does not have 'train_args'")
        
    return getattr(module, "eval_args"), getattr(module, "train_args")

def get_game_instance(eval_args, train_args):
    history_step = eval_args.get("history_step", 2)
    board_size = train_args.get("board_size", 15)
    return Gomoku(board_size=board_size, history_step=history_step)

def play_battle(game, model_a, model_b, args, a_starts=True):
    """
    Plays a single game between model_a and model_b.
    Returns: 1 if model_a wins, -1 if model_b wins, 0 if draw.
    """
    state = game.get_initial_state()
    to_play = 1  # Black/First player
    
    # model_a_color determines which "to_play" value belongs to model_a
    model_a_color = 1 if a_starts else -1
    
    mcts_root = None
    while not game.is_terminal(state):
        if to_play == model_a_color:
            action, _, mcts_root = model_a.play(state, to_play, mcts_root, show_progress_bar=False)
            model_a.apply_action(mcts_root, action)
        else:
            action, _, mcts_root = model_b.play(state, to_play, mcts_root, show_progress_bar=False)
            model_b.apply_action(mcts_root, action)
            
        state = game.get_next_state(state, action, to_play)

        to_play = -to_play
        print_board(state)
    winner = game.get_winner(state)
    if winner == 0:
        return 0
    return 1 if winner == model_a_color else -1

def main():
    print("=== AlphaZero Gomoku Battle Arena ===")
    
    # Configuration: Manually set paths and parameters here
    play_script_path = "gomoku/gomoku_play.py"
    checkpoint_a_path = "data/gomoku/gomoku_checkpoint_2026-02-28_11-16-17.pth"  # Path for Model A (New Model)
    checkpoint_b_path = "data/gomoku/gomoku_checkpoint_2026-02-26_10-33-24.pth"   # Path for Model B (Old Model)
    num_games = 10                                    # Number of games to play

    # Load configuration
    try:
        eval_args, train_args = load_args_from_path(play_script_path)
        eval_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        eval_args["mode"] = "eval"
        print(f"Config loaded. Device: {eval_args['device']}")
    except Exception as e:
        print(f"Failed to load configuration from {play_script_path}: {e}")
        return

    # Initialize Game
    try:
        game = get_game_instance(eval_args, train_args)
        print(f"Gomoku game initialized (Size: {game.board_size})")
    except Exception as e:
        print(f"Failed to initialize game: {e}")
        return

    # Initialize Models
    def create_alphazero(checkpoint_path):
        model = ResNet(game, num_blocks=eval_args["num_blocks"], num_channels=eval_args["num_channels"])
        az = TreeReuseAlphaZero(game, model, None, eval_args)
        if not az.load_checkpoint(checkpoint_path):
            raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")
        return az

    print("Loading models...")
    try:
        az_a = create_alphazero(checkpoint_a_path)
        az_b = create_alphazero(checkpoint_b_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Stats tracking
    a_wins = 0
    b_wins = 0
    draws = 0

    print(f"\nStarting Battle: {num_games} games")
    for i in range(num_games):
        # Alternate starting player
        a_starts = (i % 2 == 0)
        p1_name = "Model A" if a_starts else "Model B"
        p2_name = "Model B" if a_starts else "Model A"
        
        print(f"Game {i+1}/{num_games}: {p1_name} (Black) vs {p2_name} (White)...", end="", flush=True)
        
        result = play_battle(game, az_a, az_b, eval_args, a_starts=a_starts)
        
        if result == 1:
            a_wins += 1
            winner_name = "Model A"
        elif result == -1:
            b_wins += 1
            winner_name = "Model B"
        else:
            draws += 1
            winner_name = "Draw"
            
        print(f" Winner: {winner_name}")

    # Final Report
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Model A (New): {a_wins} wins")
    print(f"Model B (Old): {b_wins} wins")
    print(f"Draws:         {draws} draws")
    
    total_played = a_wins + b_wins + draws
    if total_played > 0:
        win_rate = (a_wins / total_played) * 100
        non_loss_rate = ((a_wins + draws) / total_played) * 100
        print(f"Model A Win Rate: {win_rate:.2f}%")
        print(f"Model A Non-loss Rate: {non_loss_rate:.2f}%")
    
    if a_wins > b_wins:
        print("\nConclusion: Model A is stronger!")
    elif b_wins > a_wins:
        print("\nConclusion: Model B is stronger!")
    else:
        print("\nConclusion: The models appear to be of equal strength.")

if __name__ == "__main__":
    main()
