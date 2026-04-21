#include <iostream>

#include "../alphazero_parallel.h"
#include "../envs/gomoku.h"

namespace {

skyzero::AlphaZeroConfig build_gomoku_config() {
    skyzero::AlphaZeroConfig cfg;
    cfg.board_size = 15;
    cfg.num_blocks = 6;
    cfg.num_channels = 96;
    cfg.lr = 1e-4f;
    cfg.weight_decay = 3e-5f;

    cfg.num_simulations = 64;
    cfg.gumbel_m = 16;
    cfg.gumbel_c_visit = 50.0f;
    cfg.gumbel_c_scale = 1.0f;

    cfg.move_temperature_init = 0.8f;
    cfg.move_temperature_final = 0.2f;

    cfg.batch_size = 256;
    cfg.min_buffer_size = 1e5;
    cfg.linear_threshold = 2e6;
    cfg.replay_alpha = 0.8f;
    cfg.max_buffer_size = 2e7;

    cfg.train_steps_per_generation = 100;
    cfg.target_replay_ratio = 6.0f;
    cfg.savetime_interval = 14400;
    cfg.file_name = "gomoku";
    cfg.data_dir = "data/gomoku";
    cfg.save_on_exit = true;

    cfg.fpu_reduction_max = 0.08f;
    cfg.root_fpu_reduction_max = 0.0f;
    cfg.enable_stochastic_transform_inference_for_root = false;
    cfg.enable_symmetry_inference_for_root = true;
    cfg.enable_stochastic_transform_inference_for_child = true;
    cfg.enable_symmetry_inference_for_child = false;

    // Dynamic Variance-Scaled cPUCT
    cfg.cpuct_utility_stdev_prior = 0.40f;
    cfg.cpuct_utility_stdev_prior_weight = 2.0f;
    cfg.cpuct_utility_stdev_scale = 0.85f;

    // Subtree Value Bias
    cfg.enable_subtree_value_bias = false;
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
    std::signal(SIGINT, skyzero::signal_handler);
    auto cfg = build_gomoku_config();
    skyzero::AlphaZeroParallelConfig pcfg;
    pcfg.num_workers = 32;
    pcfg.num_inference_servers = 2;
    pcfg.inference_batch_size = 128;
    pcfg.inference_batch_wait_us = 100;
    pcfg.leaf_batch_size = 8;
    pcfg.max_games_to_process_per_tick = 200;
    pcfg.idle_sleep_ms = 0;

    skyzero::Gomoku game(cfg.board_size, true, true);
    game.load_openings("envs/gomoku_openings.txt", 0.9f);
    auto model = skyzero::ResNet(game.board_size, game.num_planes, cfg.num_blocks, cfg.num_channels);
    model->to(cfg.device);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay)
    );

    skyzero::AlphaZeroParallel<skyzero::Gomoku> az(game, model, optimizer, cfg, pcfg);
    az.load_checkpoint();

    std::cout << "Start Gomoku training, device=" << (cfg.device.is_cuda() ? "cuda" : "cpu") << "\n";
    az.learn();
    std::cout << "Training finished.\n";
    return 0;
}
