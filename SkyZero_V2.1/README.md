# SkyZero_V2.1: AlphaZero + KataGo Tricks + Auxiliary Tasks

SkyZero_V2.1 builds upon V2 by adding auxiliary tasks and stochastic enhancements to further improve training stability and representation learning.

## Project Lineage
- [SkyZero_V0](../SkyZero_V0/README.md): Pure AlphaZero implementation.
- [SkyZero_V2](../SkyZero_V2/README.md): Added KataGo techniques.
- **SkyZero_V2.1 (Current)**: Added Auxiliary Tasks.
- [SkyZero_V3](../SkyZero_V3/README.md): Gumbel AlphaZero + KataGo techniques.

## Key Features
- **Auxiliary Tasks**: Enhances the internal representation of the network by predicting additional game-related metrics.
- **3-Bin Value Head**: Instead of a single scalar, the value head predicts probabilities for (Win, Draw, Loss), allowing for more nuanced state evaluation.
- **Stochastic Transform**: Implements random symmetry transformations (rotations/flips) during self-play and training to improve generalization.
- **Nested Bottleneck Blocks**: Uses advanced ResNet architectures (`NestedBottleneckResBlock`) for deeper and more efficient feature extraction.
- **Global Pooling Residual Blocks**: Further refinement of the KataGPool mechanism integrated into residual connections.

## Quick Start
### Training
```bash
python tictactoe/tictactoe_train.py
```
### Play Against AI
```bash
python tictactoe/tictactoe_play.py
```

## Advanced Training Logic
- **Policy Surprise Weighting**: Integrated training logic to focus on "surprising" states.
- **Root Temperature Transform**: Sophisticated search temperature management.

## License
Licensed under the [MIT License](LICENSE).
