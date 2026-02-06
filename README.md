# SkyZero

A clean, modular implementation of AlphaZero algorithm in PyTorch, designed for training AI agents on board games like Tic-Tac-Toe and Gomoku.

## Features

- **AlphaZero Algorithm**: Full implementation of the AlphaZero algorithm combining Monte Carlo Tree Search (MCTS) with deep neural networks
- **ResNet Architecture**: Residual neural network with KataGo-style global pooling for better board understanding
- **Modular Design**: Easy to add new games by implementing a simple game interface
- **Training Pipeline**: Complete self-play training loop with experience replay
- **Checkpoint System**: Automatic saving and loading of training progress

## Project Structure

```
SkyZero/
├── alphazero.py          # Core AlphaZero implementation (MCTS + training loop)
├── nets.py               # Neural network architectures (ResNet with global pooling)
├── replay_buffer.py      # Experience replay buffer for training
├── utils.py              # Utility functions (data augmentation, etc.)
├── tictactoe/            # Tic-Tac-Toe game implementation
│   ├── tictactoe.py      # Game logic
│   ├── tictactoe_train.py    # Training script
│   ├── tictactoe_selfplay.py # Self-play script
│   └── tictactoe_play.py     # Human vs AI play script
└── gomoku/        # Gomoku (Five in a Row) game implementation
    ├── gomoku.py         # Game logic
    ├── gomoku_train.py   # Training script
    ├── gomoku_selfplay.py# Self-play script
    └── gomoku_play.py    # Human vs AI play script
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm

### Setup

```bash
git clone https://github.com/Skywalker144/SkyZero.git
cd SkyZero
pip install torch numpy matplotlib tqdm
```

## Usage

### Training Tic-Tac-Toe

```bash
cd tictactoe
python tictactoe_train.py
```

### Training Gomoku

```bash
cd gomoku
python gomoku_train.py
```

### Playing Against the AI

After training, you can play against the AI:

```bash
python tictactoe_play.py  # For Tic-Tac-Toe
python gomoku_play.py     # For Gomoku
```

## Algorithm Overview

### Monte Carlo Tree Search (MCTS)

The implementation uses PUCT (Predictor Upper Confidence bounds applied to Trees) for node selection:

```
PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)` - Average value of taking action `a` from state `s`
- `P(s,a)` - Prior probability from neural network
- `N(s)` - Visit count of parent node
- `N(s,a)` - Visit count of current node

### Neural Network

The network consists of:
1. **Input Layer**: Encodes game state with position history
2. **Residual Tower**: Stack of residual blocks for feature extraction
3. **Global Pooling** (optional): KataGo-style global context aggregation
4. **Policy Head**: Outputs action probabilities
5. **Value Head**: Outputs position evaluation (-1 to 1)

### Training Loop

1. Self-play games generate training data
2. Data is stored in replay buffer
3. Network is trained on batches from replay buffer
4. Repeat

## Configuration

Key hyperparameters in training scripts:

| Parameter | Description |
|-----------|-------------|
| `num_simulations` | MCTS simulations per move |
| `c_puct` | Exploration constant |
| `batch_size` | Training batch size |
| `buffer_size` | Replay buffer capacity |
| `num_blocks` | Number of residual blocks |
| `num_channels` | Network channel width |
| `dirichlet_alpha` | Dirichlet noise parameter |
| `temperature` | Action selection temperature |

## Adding New Games

To add a new game, implement the game interface:

```python
class YourGame:
    def __init__(self):
        self.board_size = 9  # Board dimension
        
    def get_initial_state(self):
        """Return initial game state"""
        pass
    
    def get_is_legal_actions(self, state):
        """Return boolean array of legal actions"""
        pass
    
    def get_next_state(self, state, action, to_play):
        """Return new state after action"""
        pass
    
    def get_winner(self, state):
        """Return 1, -1, 0 (draw), or None (ongoing)"""
        pass
    
    def is_terminal(self, state):
        """Return True if game is over"""
        pass
    
    def encode_state(self, state, to_play):
        """Encode state for neural network input"""
        pass
```

## References

- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) - AlphaGo Zero paper
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) - AlphaZero paper
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) - KataGo paper

## License

This project is open source and available under the MIT License.

## Acknowledgments

This implementation is inspired by the original DeepMind AlphaZero paper and various open-source implementations in the community.
