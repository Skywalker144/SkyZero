#include <iostream>

#include "../alphazero_parallel.h"
#include "../envs/tictactoe.h"

namespace {

skyzero::AlphaZeroConfig build_tictactoe_config() {
    skyzero::AlphaZeroConfig cfg;
    cfg.board_size = 3;
    cfg.num_blocks = 2;
    cfg.num_channels = 32;
    cfg.lr = 1e-3f;
    cfg.weight_decay = 3e-5f;

    cfg.full_search_num_simulations = 50;
    cfg.fast_search_num_simulations = 20;
    cfg.full_search_prob = 0.25f;

    cfg.root_temperature_init = 1.25f;
    cfg.root_temperature_final = 1.1f;
    cfg.move_temperature_init = 0.8f;
    cfg.move_temperature_final = 0.2f;

    cfg.total_dirichlet_alpha = 0.27f;
    cfg.dirichlet_epsilon = 0.25f;

    cfg.batch_size = 128;
    cfg.min_buffer_size = 500;
    cfg.linear_threshold = 2048;
    cfg.replay_alpha = 0.75f;
    cfg.max_buffer_size = 100000;
    cfg.train_steps_per_generation = 5;
    cfg.target_replay_ratio = 5.0f;
    cfg.savetime_interval = 120;
    cfg.file_name = "tictactoe";
    cfg.data_dir = "data/tictactoe";
    cfg.save_on_exit = true;

    cfg.enable_forced_playouts = true;
    cfg.forced_playouts_k = 1.0f;
    cfg.fpu_reduction_max = 0.0f;
    cfg.root_fpu_reduction_max = 0.0f;
    cfg.enable_stochastic_transform_inference_for_root = false;
    cfg.enable_symmetry_inference_for_root = true;
    cfg.enable_stochastic_transform_inference_for_child = true;
    cfg.enable_symmetry_inference_for_child = false;

    cfg.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return cfg;
}

}  // namespace

int main() {
    std::signal(SIGINT, skyzero::signal_handler);
    auto cfg = build_tictactoe_config();
    skyzero::AlphaZeroParallelConfig pcfg;
    pcfg.num_workers = 40;
    pcfg.num_inference_servers = 1;
    pcfg.inference_batch_size = 128;
    pcfg.inference_batch_wait_us = 30;
    pcfg.leaf_batch_size = 4;
    pcfg.max_games_to_process_per_tick = 200;
    pcfg.idle_sleep_ms = 0;

    skyzero::TicTacToe game;
    auto model = skyzero::ResNet(game.board_size, game.num_planes, cfg.num_blocks, cfg.num_channels);
    model->to(cfg.device);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr).weight_decay(cfg.weight_decay)
    );

    skyzero::AlphaZeroParallel<skyzero::TicTacToe> az(game, model, optimizer, cfg, pcfg);
    az.load_checkpoint();

    std::cout << "Start TicTacToe training, device=" << (cfg.device.is_cuda() ? "cuda" : "cpu") << "\n";
    az.learn();
    std::cout << "Training finished.\n";
    return 0;
}
