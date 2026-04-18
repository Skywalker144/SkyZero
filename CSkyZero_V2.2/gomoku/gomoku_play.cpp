#include <iostream>

#include <torch/torch.h>

#include "../envs/gomoku.h"
#include "../playgame.h"

namespace {

skyzero::AlphaZeroConfig build_gomoku_eval_config() {
    skyzero::AlphaZeroConfig cfg;
    cfg.board_size = 15;
    cfg.num_blocks = 4;
    cfg.num_channels = 128;
    cfg.full_search_num_simulations = 600;
    cfg.fast_search_num_simulations = 600;
    cfg.enable_forced_playouts = false;
    cfg.dirichlet_epsilon = 0.0f;

    // Dynamic Variance-Scaled cPUCT
    cfg.cpuct_utility_stdev_prior = 0.40f;
    cfg.cpuct_utility_stdev_prior_weight = 2.0f;
    cfg.cpuct_utility_stdev_scale = 0.85f;

    // Subtree Value Bias (also enabled for play)
    cfg.enable_subtree_value_bias = true;
    cfg.subtree_value_bias_factor = 0.35f;
    cfg.subtree_value_bias_weight_exponent = 0.85f;
    cfg.subtree_value_bias_free_prop = 0.8f;
    cfg.subtree_value_bias_table_shards = 4096;
    cfg.subtree_value_bias_pattern_radius = 2;

    cfg.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return cfg;
}

}  // namespace

int main() {
    auto cfg = build_gomoku_eval_config();
    cfg.lr = 1e-4f;
    cfg.weight_decay = 3e-5f;
    cfg.file_name = "gomoku";
    cfg.data_dir = "data/gomoku";

    skyzero::Gomoku game(cfg.board_size, true, true);
    skyzero::GamePlayer<skyzero::Gomoku> player(game, cfg);
    player.load_checkpoint();
    return player.play();
}
