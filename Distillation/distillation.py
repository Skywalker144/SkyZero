import sys
import os
import argparse
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm

# Add the project root to sys.path so envs, nets and other modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gomoku import Gomoku
from nets import ResNet
from alphazero import AlphaZero
from gomoku.gomoku_train import train_args as gomoku_train_args

class DataGenerater:
    def __init__(self, game, model, train_args, dist_args):
        self.game = game
        self.model = model
        self.train_args = train_args
        self.dist_args = dist_args
        
        # Initialize AlphaZero with Gomoku training parameters
        self.alphazero = AlphaZero(self.game, self.model, None, self.train_args)
        
        # Load teacher model's pre-trained checkpoint to generate data
        # self.train_args['data_dir'] points to 'data/gomoku', which is correct for loading.
        if self.alphazero.load_checkpoint():
            print("Successfully loaded teacher model checkpoint from", self.train_args['data_dir'])
        else:
            print("No checkpoint found in", self.train_args['data_dir'], "- using random weights for teacher model.")
        
        self.model.eval()

    def generate_and_save(self, num_samples):
        save_dir = self.dist_args['data_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        all_data = []
        pbar = tqdm(total=num_samples, desc="Generating Data")
        
        with torch.no_grad():
            while len(all_data) < num_samples:
                # selfplay() returns (return_memory, winner, memory_len, final_state)
                memory, _, length, _ = self.alphazero.selfplay()
                
                current_len = len(all_data)
                needed = num_samples - current_len
                
                # Check how much to add to not exceed num_samples
                if length > needed:
                    all_data.extend(memory[:needed])
                    pbar.update(needed)
                else:
                    all_data.extend(memory)
                    pbar.update(length)
                
        pbar.close()
        
        save_path = os.path.join(save_dir, 'distillation_dataset.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"Successfully generated {num_samples} samples and saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['generate', 'train'], default='generate', help='Mode to run: generate data or train distillation model')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    cmd_args = parser.parse_args()
    
    # Distillation specific arguments
    dist_args = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dir': 'Distillation/dataset',
        'batch_size': cmd_args.batch_size,
        'epochs': cmd_args.epochs,
        'num_workers': 0,
        'kl_loss_weight': 0.5,
        'log_interval': 10,
        'lr': 1e-3,
        'weight_decay': 1e-4,
    }
    
    # Initialize Gomoku game
    game = Gomoku(board_size=gomoku_train_args['board_size'], history_step=gomoku_train_args['history_step'])
    
    if cmd_args.mode == 'generate':
        # Instantiate teacher model based on the same config used for training
        teacher_model = ResNet(
            game, 
            num_blocks=gomoku_train_args['num_blocks'], 
            num_channels=gomoku_train_args['num_channels']
        ).to(gomoku_train_args['device'])
        
        # Generator handles running self-play games and collecting samples
        generator = DataGenerater(game, teacher_model, gomoku_train_args, dist_args)
        generator.generate_and_save(cmd_args.num_samples)
        
    elif cmd_args.mode == 'train':
        print("Train mode requires OfflineDistiller which is not fully implemented in this script.")
        # student_model = ResNet(game, num_blocks=2, num_channels=32).to(dist_args['device'])
        # optimizer = optim.Adam(student_model.parameters(), lr=dist_args['lr'], weight_decay=dist_args['weight_decay'])
        # distiller = OfflineDistiller(game, student_model, optimizer, dist_args)
        # distiller.train()