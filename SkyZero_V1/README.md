# SkyZero_V0: Pure AlphaZero

SkyZero_V0 is a concise and efficient implementation of the original AlphaZero algorithm, supporting environments such as Tic-Tac-Toe and Gomoku.

## Project Lineage
- **SkyZero_V0 (Current)**: Pure AlphaZero implementation.
- [SkyZero_V2](../SkyZero_V2/README.md): Added KataGo techniques.
- [SkyZero_V2.1](../SkyZero_V2.1/README.md): Added Auxiliary Tasks.
- [SkyZero_V3](../SkyZero_V3/README.md): Gumbel AlphaZero + KataGo techniques.

## Core Principles
The core idea of AlphaZero is the combination of **Monte Carlo Tree Search (MCTS)** and **Deep Neural Networks (DNN)**.

- **Dual-head Network**: Uses a ResNet-structured neural network to output:
    - **Policy ($p$)**: Predicted probabilities for each legal action.
    - **Value ($v$)**: An estimate of the win rate (-1 to 1).
- **Zero Knowledge**: Starts from random weights and learns solely through self-play based on game rules.

## Algorithm Flow
1. **Self-Play**: Agent selects moves using MCTS guided by the current neural network.
2. **Data Collection**: Stores experience data (state, policy, outcome) into a Replay Buffer.
3. **Optimization**: Trains the network to minimize the loss between predicted and actual outcomes/policies.

## Quick Start
### Training
```bash
python tictactoe/tictactoe_train.py
```
### Play Against AI
```bash
python tictactoe/tictactoe_play.py
```

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- tqdm

## License
Licensed under the [MIT License](LICENSE).
