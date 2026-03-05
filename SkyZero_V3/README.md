# SkyZero_V3: Gumbel AlphaZero + KataGo Techniques

SkyZero_V3 is the most advanced version of the series, implementing the **Gumbel AlphaZero** algorithm combined with KataGo's proven techniques.

## Project Lineage
- [SkyZero_V0](../SkyZero_V0/README.md): Pure AlphaZero implementation.
- [SkyZero_V2](../SkyZero_V2/README.md): Added KataGo techniques.
- [SkyZero_V2.1](../SkyZero_V2.1/README.md): Added Auxiliary Tasks.
- **SkyZero_V3 (Current)**: Gumbel AlphaZero + KataGo techniques.

## Gumbel AlphaZero Principles
Gumbel AlphaZero (from ICLR 2022) replaces heuristic exploration with a theoretically sound sampling and search mechanism:
1. **Gumbel-Top-k Sampling**: Uses the Gumbel-Max trick at the root to select candidate actions.
2. **Sequential Halving**: Allocates simulation budgets in stages, pruning less promising actions to focus on the best candidates.
3. **Policy Improvement Guarantee**: Mathematically ensures the target policy is an improvement over the raw network, even with very few simulations.
4. **Unbiased Updates**: Uses a specific loss function that allows the network to learn more efficiently from search results.

## Key Features
- **Parallel Self-Play**: Optimized multi-process data collection.
- **KataGo Techniques**: Integrated Global Pooling and advanced normalization layers.
- **ONNX Export**: Easy deployment to Web environments via ONNX Runtime.
- **Visualization**: Automated training loss and win-rate plotting.

## Quick Start
### Training
```bash
python -m tictactoe.tictactoe_train
```
### Play Against AI
```bash
python -m tictactoe.tictactoe_play
```

## References
- [Policy Improvement by Planning with Gumbel (ICLR 2022)](https://arxiv.org/abs/2111.09794)

## License
Licensed under the [MIT License](LICENSE).
