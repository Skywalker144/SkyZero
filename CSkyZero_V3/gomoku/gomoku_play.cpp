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
    cfg.num_simulations = 400;
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
