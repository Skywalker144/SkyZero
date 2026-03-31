#include <iostream>

#include <torch/torch.h>

#include "../envs/tictactoe.h"
#include "../playgame.h"

namespace {

skyzero::AlphaZeroConfig build_tictactoe_eval_config() {
    skyzero::AlphaZeroConfig cfg;
    cfg.board_size = 3;
    cfg.num_blocks = 2;
    cfg.num_channels = 32;
    cfg.full_search_num_simulations = 200;
    cfg.fast_search_num_simulations = 200;
    cfg.enable_forced_playouts = true;
    cfg.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    return cfg;
}

}  // namespace

int main() {
    auto cfg = build_tictactoe_eval_config();
    cfg.lr = 1e-3f;
    cfg.weight_decay = 3e-5f;
    cfg.file_name = "tictactoe";
    cfg.data_dir = "data/tictactoe";

    skyzero::TicTacToe game;
    skyzero::GamePlayer<skyzero::TicTacToe> player(game, cfg);
    player.load_checkpoint();
    return player.play();
}
