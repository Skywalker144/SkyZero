# SkyZero_V2: AlphaZero + KataGo Techniques

SkyZero_V2 enhances the base AlphaZero algorithm by integrating advanced training and search techniques inspired by KataGo.

## Project Lineage
- [SkyZero_V0](../SkyZero_V0/README.md): Pure AlphaZero implementation.
- **SkyZero_V2 (Current)**: Added KataGo techniques.
- [SkyZero_V2.1](../SkyZero_V2.1/README.md): Added Auxiliary Tasks.
- [SkyZero_V3](../SkyZero_V3/README.md): Gumbel AlphaZero + KataGo techniques.

## Key Improvements (KataGo Tricks)
- **Global Pooling (KataGPool)**: Incorporates global state information into the convolutional layers.
- **Surprise Weighting**: Prioritizes training on positions where the model's prediction deviates significantly from MCTS results.
- **Shaped Dirichlet Noise**: Adaptive noise for robust exploration without losing strong signals.
- **Forced Playouts**: Ensures minimum simulations for high-priority actions.
- **Temperature Decay**: 
    - **Root Temperature Decay**: Focuses the search as the game progresses.
    - **Move Temperature Decay**: Shifts from exploration to exploitation during a game.
- **Policy Target Pruning**: Reduces noise in MCTS policy targets to improve convergence.

## Quick Start
### Training
```bash
python tictactoe/tictactoe_train.py
```
### Play Against AI
```bash
python tictactoe/tictactoe_play.py
```

## Features
- **Parallel Training**: Multi-process self-play for faster data collection.
- **Model Export**: Supports ONNX export for Web UI deployment.
- **Evaluation**: Automated tournament testing and visualization.

## License
Licensed under the [MIT License](LICENSE).
