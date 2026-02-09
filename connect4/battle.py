import os
import sys
import glob
import argparse
import time
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphazero import MCTS
from nets import ResNet

# Default configurations for each game type
# These match the settings in *_train.py files
GAME_CONFIGS = {
    'connect4': {
        'module': 'connect4',
        'class': 'Connect4',
        'blocks': 2,
        'channels': 128,
        'kwargs': {'history_step': 3}
    },
    'tictactoe': {
        'module': 'tictactoe',
        'class': 'TicTacToe',
        'blocks': 1,
        'channels': 64,
        'kwargs': {'history_step': 3}
    },
    'gomoku': {
        'module': 'gomoku',
        'class': 'Gomoku',
        'blocks': 8,
        'channels': 256,
        'kwargs': {'board_size': 15, 'history_step': 4}
    }
}

def get_game_context():
    """Identify game based on current directory name."""
    folder_name = os.path.basename(os.getcwd())
    config = GAME_CONFIGS.get(folder_name)
    
    if not config:
        print(f"Warning: Could not auto-detect game config for folder '{folder_name}'.")
        print("Available configs:", list(GAME_CONFIGS.keys()))
        return None, folder_name
    
    return config, folder_name

def load_game_class(config):
    """Dynamically import the game class."""
    try:
        module = __import__(config['module'], fromlist=[config['class']])
        game_cls = getattr(module, config['class'])
        return game_cls
    except ImportError as e:
        print(f"Error importing game class: {e}")
        sys.exit(1)

def parse_timestamp(filename):
    """Extract timestamp from checkpoint filename."""
    # Expected format: name_checkpoint_YYYY-MM-DD_HH-MM-SS.pth
    try:
        base = os.path.basename(filename)
        # Regex to find timestamp
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', base)
        if match:
            dt_str = match.group(1)
            return datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        pass
    return None

def find_checkpoints(game_name, interval_seconds):
    """Find the latest checkpoint and the comparison checkpoint."""
    checkpoint_dir = f"{game_name}_checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        sys.exit(1)

    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        print(f"Error: No .pth files found in '{checkpoint_dir}'.")
        sys.exit(1)

    # Sort files by timestamp
    checkpoints = []
    for f in files:
        dt = parse_timestamp(f)
        if dt:
            checkpoints.append((dt, f))
    
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print("Error: Could not parse timestamps from checkpoint files.")
        sys.exit(1)

    latest_dt, latest_ckpt = checkpoints[-1]
    
    # Find past checkpoint
    target_time = latest_dt.timestamp() - interval_seconds
    past_ckpt = None
    past_dt = None
    
    # Find the latest checkpoint that is <= target_time
    # Iterate backwards
    for dt, f in reversed(checkpoints):
        if dt.timestamp() <= target_time:
            past_ckpt = f
            past_dt = dt
            break
            
    if past_ckpt is None:
        # If no checkpoint is old enough, use the oldest one available
        past_dt, past_ckpt = checkpoints[0]
        print(f"Warning: No checkpoint found older than {interval_seconds}s. Using oldest available.")

    return (latest_ckpt, latest_dt), (past_ckpt, past_dt)

def play_match(game, model1, model2, args):
    """
    Play a single match between two models.
    model1: Player 1 (1)
    model2: Player 2 (-1)
    """
    mcts1 = MCTS(game, args, model1)
    mcts2 = MCTS(game, args, model2)
    
    state = game.get_initial_state()
    to_play = 1
    
    while not game.is_terminal(state):
        if to_play == 1:
            # Player 1 turn (Model 1)
            action_probs = mcts1.search(state, to_play)
        else:
            # Player 2 turn (Model 2)
            # MCTS expects to_play relative to the model perspective?
            # MCTS.search takes to_play.
            # In AlphaZero.selfplay: mcts.search(state, to_play)
            action_probs = mcts2.search(state, to_play)
        
        # Select action (Greedy for evaluation)
        # Can add temperature if needed, but usually battle is deterministic/greedy
        if args['temperature'] == 0:
            action = np.argmax(action_probs)
        else:
            # Add simple temperature sampling if requested
            action = np.random.choice(len(action_probs), p=action_probs)

        state = game.get_next_state(state, action, to_play)
        to_play = -to_play
        
    return game.get_winner(state)

def main():
    parser = argparse.ArgumentParser(description="Battle between latest and past AlphaZero checkpoints.")
    
    # Auto-detect config
    config, folder_name = get_game_context()
    
    # Config arguments
    parser.add_argument('--interval', type=int, default=7200, help='Time interval in seconds to look back for the past model (default: 3600)')
    parser.add_argument('--games', type=int, default=20, help='Total number of games to play (default: 10)')
    parser.add_argument('--sims', type=int, default=600, help='Number of MCTS simulations per move (default: 100)')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for move selection (default: 0.0 for deterministic)')
    
    # Model override arguments
    defaults = config if config else {'blocks': 0, 'channels': 0, 'kwargs': {}}
    parser.add_argument('--blocks', type=int, default=defaults['blocks'], help=f"ResNet blocks (default: {defaults['blocks']})")
    parser.add_argument('--channels', type=int, default=defaults['channels'], help=f"ResNet channels (default: {defaults['channels']})")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running in: {os.getcwd()}")
    print(f"Game: {folder_name}")
    print(f"Device: {device}")
    
    # 1. Load Game Class
    if config:
        GameClass = load_game_class(config)
        game = GameClass(**config['kwargs'])
    else:
        print("Error: Unknown game configuration. Please run this script inside a known game folder (connect4, gomoku, tictactoe).")
        sys.exit(1)
        
    # 2. Find Checkpoints
    (latest_file, latest_dt), (past_file, past_dt) = find_checkpoints(folder_name, args.interval)
    
    print(f"\nModel A (Latest): {os.path.basename(latest_file)}")
    print(f"  Timestamp: {latest_dt}")
    print(f"Model B (Past):   {os.path.basename(past_file)}")
    print(f"  Timestamp: {past_dt}")
    print(f"  Time Diff: {latest_dt - past_dt}")
    
    if latest_file == past_file:
        print("\nWarning: Model A and Model B are the same file!")
        
    # 3. Load Models
    print(f"\nLoading models (Blocks={args.blocks}, Channels={args.channels})...")
    
    model_a = ResNet(game, num_blocks=args.blocks, num_channels=args.channels).to(device)
    checkpoint_a = torch.load(latest_file, map_location=device, weights_only=False)
    model_a.load_state_dict(checkpoint_a['model_state_dict'])
    model_a.eval()
    
    model_b = ResNet(game, num_blocks=args.blocks, num_channels=args.channels).to(device)
    checkpoint_b = torch.load(past_file, map_location=device, weights_only=False)
    model_b.load_state_dict(checkpoint_b['model_state_dict'])
    model_b.eval()
    
    # 4. Battle Loop
    print(f"\nStarting {args.games} games...")
    print(f"Simulations: {args.sims}")
    
    model_a_wins = 0
    model_b_wins = 0
    draws = 0
    
    # args for MCTS
    mcts_args = {
        'num_simulations': args.sims,
        'c_puct': 1.5, # Standard default
        'temperature': 0.1,
        'device': device,
        'Q_norm_bounds': [-1, 1], # Assuming standard bounds
        'mode': 'play'
    }
    
    for i in range(args.games):
        # Swap sides: Even indices -> A is Player 1. Odd indices -> B is Player 1.
        if i % 2 == 0:
            p1_model = model_a
            p2_model = model_b
            p1_name = "Model A"
            p2_name = "Model B"
        else:
            p1_model = model_b
            p2_model = model_a
            p1_name = "Model B"
            p2_name = "Model A"
            
        winner = play_match(game, p1_model, p2_model, mcts_args)
        
        # Winner is 1 (P1), -1 (P2), or 0 (Draw)
        if winner == 1:
            actual_winner = p1_name
        elif winner == -1:
            actual_winner = p2_name
        else:
            actual_winner = "Draw"
            
        print(f"Game {i+1}/{args.games}: {p1_name} (Black) vs {p2_name} (White) -> Winner: {actual_winner}")
        
        if actual_winner == "Model A":
            model_a_wins += 1
        elif actual_winner == "Model B":
            model_b_wins += 1
        else:
            draws += 1

    # 5. Results
    print("\n=== Final Results ===")
    print(f"Model A (Latest): {model_a_wins} wins ({model_a_wins/args.games*100:.1f}%)")
    print(f"Model B (Past):   {model_b_wins} wins ({model_b_wins/args.games*100:.1f}%)")
    print(f"Draws:            {draws} ({draws/args.games*100:.1f}%)")

if __name__ == "__main__":
    main()
