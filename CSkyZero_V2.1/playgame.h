#ifndef SKYZERO_PLAYGAME_H
#define SKYZERO_PLAYGAME_H

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "alphazero.h"

namespace skyzero {

namespace playgame_detail {

inline void print_board(const std::vector<int8_t>& board, int size) {
    std::cout << "   ";
    for (int c = 0; c < size; ++c) {
        std::cout << (c < 10 ? " " : "") << c << ' ';
    }
    std::cout << '\n';

    for (int r = 0; r < size; ++r) {
        std::cout << (r < 10 ? " " : "") << r << ' ';
        for (int c = 0; c < size; ++c) {
            const int8_t v = board[r * size + c];
            char ch = '.';
            if (v == 1) ch = 'X';
            if (v == -1) ch = 'O';
            std::cout << ' ' << ch << ' ';
        }
        std::cout << '\n';
    }
}

inline int argmax_index(const std::vector<float>& p) {
    if (p.empty()) {
        return -1;
    }
    return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
}

inline void print_policy_grid(const std::vector<float>& p, int board_size, const std::string& title) {
    std::cout << title << "\n";
    for (int r = 0; r < board_size; ++r) {
        for (int c = 0; c < board_size; ++c) {
            const int idx = r * board_size + c;
            const float v = (idx >= 0 && idx < static_cast<int>(p.size())) ? p[idx] : 0.0f;
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << v << ' ';
        }
        std::cout << '\n';
    }
}

inline bool parse_row_col(const std::string& text, int& row, int& col) {
    std::istringstream iss(text);
    if (!(iss >> row >> col)) {
        return false;
    }
    std::string remain;
    if (iss >> remain) {
        return false;
    }
    return true;
}

inline bool parse_human_side(const std::string& text, int& side) {
    std::istringstream iss(text);
    if (!(iss >> side)) {
        return false;
    }
    std::string remain;
    if (iss >> remain) {
        return false;
    }
    return side == 1 || side == -1;
}

}  // namespace playgame_detail

template <typename Game>
class GamePlayer {
public:
    GamePlayer(Game& game, AlphaZeroConfig cfg)
        : game_(game),
          cfg_(std::move(cfg)),
          model_(ResNet(game_.board_size, game_.num_planes, cfg_.num_blocks, cfg_.num_channels)),
          optimizer_(model_->parameters(), torch::optim::AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay)),
          alphazero_(game_, model_, optimizer_, cfg_),
          mcts_(game_, cfg_, model_) {
        model_->to(cfg_.device);
        model_->eval();
    }

    bool load_checkpoint(const std::string& path = "") {
        return alphazero_.load_checkpoint(path);
    }

    int play() {
        int human_side = 0;
        while (true) {
            std::cout
                << "1 for Human first move and -1 for Human second move\n"
                << "The position of the piece needs to be input in coordinate form.\n"
                << "   (first input the vertical coordinate, then the horizontal coordinate).\n"
                << "Please enter: ";
            std::string line;
            if (!std::getline(std::cin, line)) {
                return 0;
            }
            if (playgame_detail::parse_human_side(line, human_side)) {
                break;
            }
            std::cout << "Invalid input. Please enter 1 or -1.\n";
        }

        std::vector<int8_t> state = game_.get_initial_state();
        int to_play = 1;
        int last_action = -1;
        int last_player = 0;

        root_.reset(new MCTSNode{state, to_play});
        history_.clear();
        playgame_detail::print_board(state, game_.board_size);

        while (true) {
            if (game_.is_terminal(state, last_action, last_player)) {
                const int winner = game_.get_winner(state, last_action, last_player);
                if (winner == 1) {
                    std::cout << "Black wins!\n";
                } else if (winner == -1) {
                    std::cout << "White wins!\n";
                } else {
                    std::cout << "Draw!\n";
                }

                std::cout << "Game Over. 'u' to undo, 'q' to quit: ";
                std::string resp;
                if (!std::getline(std::cin, resp)) {
                    return 0;
                }
                if (resp == "u" || resp == "U") {
                    if (undo_two_moves(state, to_play, last_action, last_player)) {
                        std::cout << "Undo successful.\n";
                        playgame_detail::print_board(state, game_.board_size);
                        continue;
                    }
                    std::cout << "Nothing to undo.\n";
                }
                break;
            }

            if (to_play == human_side) {
                while (true) {
                    std::cout << "Human step (row col / 'u' for undo / 'q' for quit): ";
                    std::string input;
                    if (!std::getline(std::cin, input)) {
                        return 0;
                    }

                    if (input == "u" || input == "U") {
                        if (undo_two_moves(state, to_play, last_action, last_player)) {
                            std::cout << "Undo successful.\n";
                            playgame_detail::print_board(state, game_.board_size);
                        } else {
                            std::cout << "Nothing to undo.\n";
                        }
                        continue;
                    }
                    if (input == "q" || input == "Q") {
                        std::cout << "Exiting game.\n";
                        return 0;
                    }

                    int row = -1;
                    int col = -1;
                    if (!playgame_detail::parse_row_col(input, row, col)) {
                        std::cout << "Invalid input format. Please enter 'row col' (e.g., '7 7').\n";
                        continue;
                    }
                    if (row < 0 || row >= game_.board_size || col < 0 || col >= game_.board_size) {
                        std::cout << "Invalid move: out of board range.\n";
                        continue;
                    }

                    const int action = row * game_.board_size + col;
                    const auto legal = game_.get_is_legal_actions(state, to_play);
                    if (!legal[action]) {
                        std::cout << "Invalid move: (" << row << ", " << col << ") is forbidden or occupied.\n";
                        continue;
                    }

                    push_history(state, to_play, last_action, last_player);
                    state = game_.get_next_state(state, action, to_play);
                    last_action = action;
                    last_player = to_play;
                    to_play = -to_play;
                    advance_root(action, state, to_play);
                    break;
                }
            } else {
                push_history(state, to_play, last_action, last_player);
                std::cout << "AlphaZero step:\n";

                const auto out = mcts_.search(state, to_play, cfg_.full_search_num_simulations, root_);
                const int action = playgame_detail::argmax_index(out.mcts_policy);
                if (action < 0) {
                    std::cout << "No legal action found. Exiting game.\n";
                    return 1;
                }
                const int row = action / game_.board_size;
                const int col = action % game_.board_size;

                state = game_.get_next_state(state, action, to_play);
                last_action = action;
                last_player = to_play;
                to_play = -to_play;
                advance_root(action, state, to_play);

                playgame_detail::print_policy_grid(out.mcts_policy, game_.board_size, "MCTS Strategy:");
                playgame_detail::print_policy_grid(out.nn_policy, game_.board_size, "NN Strategy:");
                std::cout
                    << "Win  Probability: " << std::fixed << std::setprecision(2) << out.nn_value_probs[0] << '\n'
                    << "Draw Probability: " << std::fixed << std::setprecision(2) << out.nn_value_probs[1] << '\n'
                    << "Lose Probability: " << std::fixed << std::setprecision(2) << out.nn_value_probs[2] << '\n';

                const float root_value = out.root_value[0] - out.root_value[2];
                const float nn_value = out.nn_value_probs[0] - out.nn_value_probs[2];
                std::cout
                    << "          "
                    << std::setw(6) << "Win" << "  "
                    << std::setw(6) << "Draw" << "  "
                    << std::setw(6) << "Loss" << "  "
                    << std::setw(6) << "W-L" << '\n'
                    << "  root_value:  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.root_value[0] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.root_value[1] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.root_value[2] * 100.0f) << "%  "
                    << std::showpos << std::fixed << std::setprecision(2) << root_value << std::noshowpos << '\n'
                    << "  nn_value:    "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[0] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[1] * 100.0f) << "%  "
                    << std::setw(6) << std::fixed << std::setprecision(2) << (out.nn_value_probs[2] * 100.0f) << "%  "
                    << std::showpos << std::fixed << std::setprecision(2) << nn_value << std::noshowpos << '\n';
                std::cout << "AI move: (" << row << ", " << col << ")\n";
            }

            playgame_detail::print_board(state, game_.board_size);
        }

        return 0;
    }

private:
    struct Snapshot {
        std::vector<int8_t> state;
        int to_play = 1;
        int last_action = -1;
        int last_player = 0;
    };

    void push_history(const std::vector<int8_t>& state, int to_play, int last_action, int last_player) {
        history_.push_back(Snapshot{state, to_play, last_action, last_player});
    }

    bool undo_two_moves(std::vector<int8_t>& state, int& to_play, int& last_action, int& last_player) {
        if (history_.size() < 2) {
            return false;
        }
        history_.pop_back();
        Snapshot restore = history_.back();
        history_.pop_back();

        state = std::move(restore.state);
        to_play = restore.to_play;
        last_action = restore.last_action;
        last_player = restore.last_player;
        root_.reset(new MCTSNode{state, to_play});
        return true;
    }

    void advance_root(int action, const std::vector<int8_t>& next_state, int next_to_play) {
        std::unique_ptr<MCTSNode> next_root;
        if (root_) {
            for (auto& child : root_->children) {
                if (child && child->action_taken == action) {
                    next_root = std::move(child);
                    break;
                }
            }
        }
        if (next_root) {
            next_root->parent = nullptr;
            root_ = std::move(next_root);
        } else {
            root_.reset(new MCTSNode{next_state, next_to_play});
        }
    }

    Game& game_;
    AlphaZeroConfig cfg_;
    ResNet model_;
    torch::optim::AdamW optimizer_;
    AlphaZero<Game> alphazero_;
    MCTS<Game> mcts_;

    std::vector<Snapshot> history_;
    std::unique_ptr<MCTSNode> root_;
};

}  // namespace skyzero

#endif
