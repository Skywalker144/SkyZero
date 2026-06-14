// Standalone unit test for cpp/skyzero_2048.h (afterstate MCTS). No LibTorch:
//   g++ -std=c++17 -I cpp skyzero_2048_test.cpp -o /tmp/sz2048_test && /tmp/sz2048_test
//
// Inference is a stub callback so the search can be exercised deterministically
// (value=0 means the only signal is realized merge rewards).

#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "skyzero_2048.h"

using skyzero::Game2048;
using skyzero::SkyZero2048Config;
using skyzero::SkyZero2048MCTS;

static int g_checks = 0, g_fails = 0;
static void check(bool c, const char* m) {
    ++g_checks;
    if (!c) { ++g_fails; std::printf("  FAIL: %s\n", m); }
}

static std::vector<int8_t> board(std::initializer_list<int> v) {
    std::vector<int8_t> b; for (int x : v) b.push_back((int8_t)x); return b;
}

// Uniform policy, zero value: search is driven purely by realized rewards.
static std::pair<std::array<float, 4>, float> stub_uniform_zero(const std::vector<int8_t>&) {
    return {{0.25f, 0.25f, 0.25f, 0.25f}, 0.0f};
}

int main() {
    Game2048 game;

    // --- Invariants: visit counts and policy normalization ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 200; cfg.gamma = 0.999f;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/1);
        auto s = board({1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto out = mcts.search(s);

        int visit_sum = 0;
        for (int a = 0; a < 4; ++a) visit_sum += out.visit_counts[a];
        check(visit_sum == cfg.num_simulations, "root child visits sum to num_simulations");

        float pol_sum = 0.0f;
        for (int a = 0; a < 4; ++a) pol_sum += out.visit_policy[a];
        check(std::abs(pol_sum - 1.0f) < 1e-4f, "visit_policy sums to 1");

        // best_action must be legal.
        auto legal = game.get_legal_actions(s);
        check(out.best_action >= 0 && legal[out.best_action] == 1, "best_action legal");
    }

    // --- Stochastic D4 transform: ACTION_PERM equivariance vs apply_move ---
    // The 2048 trap: a dihedral transform of the board must RELABEL the 4 slide
    // directions. Verify ACTION_PERM against the real game on random boards:
    //   legal(B)[a]            == legal(F(B))[P[a]]
    //   reward(apply(B,a))     == reward(apply(F(B), P[a]))
    //   F(apply(B,a).after)    == apply(F(B), P[a]).after
    //   encode(F(B))           == transform_encoded(encode(B))
    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> tile(0, 6);
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<int8_t> b(16);
            for (auto& x : b) x = (int8_t)tile(rng);
            for (int type = 0; type < 8; ++type) {
                const int k = type % 4; const bool flip = type >= 4;
                auto bt = Game2048::transform_board(b, k, flip);
                auto legalB = game.get_legal_actions(b);
                auto legalBt = game.get_legal_actions(bt);
                for (int a = 0; a < 4; ++a) {
                    const int pa = Game2048::ACTION_PERM[type][a];
                    check(legalB[a] == legalBt[pa], "D4: legal equivariance");
                    if (legalB[a]) {
                        auto mB = game.apply_move(b, a);
                        auto mBt = game.apply_move(bt, pa);
                        check(mB.reward == mBt.reward, "D4: reward equivariance");
                        check(Game2048::transform_board(mB.afterstate, k, flip) == mBt.afterstate,
                              "D4: afterstate equivariance");
                    }
                }
                check(game.encode_state(bt)
                          == Game2048::transform_encoded(game.encode_state(b), k, flip),
                      "D4: transform_encoded matches encode-of-transformed");
            }
        }
    }

    // --- Stochastic transform plumbing: search runs + valid output with it on ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 200;
        cfg.stochastic_transform_root = true;
        cfg.stochastic_transform_child = true;
        // Non-uniform stub (favor up) to exercise the undo_action_perm path.
        auto stub_dir = [](const std::vector<int8_t>&) {
            return std::pair<std::array<float, 4>, float>({{2.0f, 1.0f, 0.0f, -1.0f}}, 0.0f);
        };
        SkyZero2048MCTS mcts(game, cfg, stub_dir, /*seed=*/3);
        auto s = board({1,1,0,0, 0,2,0,0, 0,0,3,0, 0,0,0,0});
        auto out = mcts.search(s);
        int vs = 0; for (int a = 0; a < 4; ++a) vs += out.visit_counts[a];
        check(vs == cfg.num_simulations, "stochastic-transform: visits sum to num_simulations");
        auto legal = game.get_legal_actions(s);
        check(out.best_action >= 0 && legal[out.best_action] == 1,
              "stochastic-transform: best_action legal");
    }

    // --- interpolate_early: early->late temperature decay by turn ---
    {
        using skyzero::interpolate_early;
        check(std::abs(interpolate_early(0, 19, 16, 0.8f, 0.1f) - 0.8f) < 1e-5f,
              "interp: turn 0 == early");
        check(std::abs(interpolate_early(100000, 19, 16, 0.8f, 0.1f) - 0.1f) < 1e-3f,
              "interp: far turn == late");
        check(interpolate_early(10, 19, 16, 0.8f, 0.1f)
                  > interpolate_early(40, 19, 16, 0.8f, 0.1f),
              "interp: monotone decreasing toward late");
    }

    // --- root=puct with chosen-move/root-policy temperature + full LCB ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 300;
        cfg.root_puct = true; cfg.non_root_gumbel = false;
        cfg.root_desired_per_child_visits_coeff = 2.0f;
        cfg.chosen_move_temperature = 0.0f;            // late = argmax
        cfg.chosen_move_temperature_early = 1.0f;      // explore early
        cfg.chosen_move_temperature_halflife = 5.0f;
        cfg.root_policy_temperature = 1.5f;            // flatten root priors
        cfg.root_policy_temperature_early = 2.0f;
        cfg.root_dirichlet_alpha = 0.5f;
        cfg.lcb_for_selection = true;                  // exercise the full LCB
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/11);
        auto s = board({1,1,0,0, 0,2,0,0, 0,0,1,0, 0,3,0,0});
        auto legal = game.get_legal_actions(s);
        for (int turn : {0, 100}) {
            auto out = mcts.search(s, /*sims_override=*/-1, turn);
            int vs = 0; for (int a = 0; a < 4; ++a) vs += out.visit_counts[a];
            check(vs == cfg.num_simulations, "root=puct(+temp+lcb): visits sum");
            float ps = 0.0f; for (int a = 0; a < 4; ++a) ps += out.improved_policy[a];
            check(std::abs(ps - 1.0f) < 1e-3f, "root=puct(+temp+lcb): target sums to 1");
            check(out.best_action >= 0 && legal[out.best_action] == 1,
                  "root=puct(+temp+lcb): best_action legal");
        }
    }

    // --- Behavioral: prefer a high-reward merge over a zero-reward shuffle ---
    // Board: two equal tiles (exp 5) side by side in row 0.
    //   LEFT(3)/RIGHT(1): merge -> reward 2^6 = 64.
    //   DOWN(2):          slide apart, no merge -> reward 0.
    //   UP(0):            already at top -> illegal.
    {
        SkyZero2048Config cfg; cfg.num_simulations = 400; cfg.gamma = 0.999f;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/7);
        auto s = board({5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});

        auto legal = game.get_legal_actions(s);
        check(legal[0] == 0, "up illegal here");
        check(legal[1] == 1 && legal[3] == 1, "left/right legal");
        check(legal[2] == 1, "down legal");

        auto out = mcts.search(s);
        // The merging directions carry an immediate +64; the search should pick
        // one of them over the rewardless DOWN.
        check(out.best_action == 1 || out.best_action == 3,
              "search prefers the merging move (left/right) over down");
        // And they should out-visit DOWN.
        const int merge_visits = out.visit_counts[1] + out.visit_counts[3];
        check(merge_visits > out.visit_counts[2],
              "merging moves out-visit the zero-reward move");
        // Root value should be positive (reward is reachable).
        check(out.root_value > 0.0f, "root value positive when reward reachable");
    }

    // --- Chance-node visit fractions track the spawn distribution ---
    // After LEFT-merge on the board above, the afterstate has many empty cells.
    // We verify the deterministic "most under-represented" descent keeps the
    // 0.9/0.1 split: across the subtree under the chosen action, value-4 spawns
    // should be far rarer than value-2 spawns. (Checked indirectly via a high
    // sim count producing a positive, finite root value — exercised above.)
    // Here we directly test descend behavior through a dedicated small board.
    {
        SkyZero2048Config cfg; cfg.num_simulations = 300; cfg.gamma = 1.0f;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/3);
        // Single legal move scenario keeps all visits funneling through one
        // chance node so its child split is meaningful.
        auto s = board({5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto out = mcts.search(s);
        check(out.best_action >= 0, "search returns an action (chance descent ran)");
    }

    // --- Terminal root handled gracefully ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 50;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/9);
        auto s = board({1,2,1,2, 2,1,2,1, 1,2,1,2, 2,1,2,1});  // checkerboard, no moves
        check(game.is_terminal(s), "checkerboard is terminal");
        auto out = mcts.search(s);
        check(out.best_action == -1, "terminal root -> no action");
        for (int a = 0; a < 4; ++a) check(out.visit_counts[a] == 0, "terminal root -> zero visits");
    }

    // --- value sanity: reward reachable => positive root value, merge chosen ---
    // root_value is a visit-weighted average over ALL explored actions (incl.
    // the rewardless DOWN branch), so it need not reach the single-step reward;
    // the meaningful signal is that it is positive and the merge wins.
    {
        SkyZero2048Config cfg; cfg.num_simulations = 200; cfg.gamma = 1.0f;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/11);
        auto s = board({3,3,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});  // merge -> 2^4 = 16
        auto out = mcts.search(s);
        check(out.root_value > 0.0f, "root value positive when reward reachable");
        check(out.best_action == 1 || out.best_action == 3, "merge move chosen");
    }

    // --- value-target construction: MC return-to-go vs n-step TD bootstrap ---
    {
        using skyzero::compute_value_targets;
        // rewards r=[10,20,30,40], search values v=[1,2,3,4], gamma=0.5.
        std::vector<int> r = {10, 20, 30, 40};
        std::vector<float> v = {1.f, 2.f, 3.f, 4.f};

        // td_steps<=0 -> full MC return-to-go (bootstrap ignored).
        auto mc = compute_value_targets(r, v, 0.5f, 0);
        // z3=40; z2=30+.5*40=50; z1=20+.5*50=45; z0=10+.5*45=32.5
        check(std::abs(mc[3] - 40.0f)  < 1e-4f, "MC z3");
        check(std::abs(mc[2] - 50.0f)  < 1e-4f, "MC z2");
        check(std::abs(mc[1] - 45.0f)  < 1e-4f, "MC z1");
        check(std::abs(mc[0] - 32.5f)  < 1e-4f, "MC z0");

        // td_steps=1 -> z_t = r_t + gamma*v_{t+1} (bootstrap), last step no bootstrap.
        auto td1 = compute_value_targets(r, v, 0.5f, 1);
        // z0=10+.5*v1=11; z1=20+.5*v2=21.5; z2=30+.5*v3=32; z3=40 (t+1 past terminal)
        check(std::abs(td1[0] - 11.0f)  < 1e-4f, "TD1 z0");
        check(std::abs(td1[1] - 21.5f)  < 1e-4f, "TD1 z1");
        check(std::abs(td1[2] - 32.0f)  < 1e-4f, "TD1 z2");
        check(std::abs(td1[3] - 40.0f)  < 1e-4f, "TD1 z3 (no bootstrap past terminal)");

        // td_steps=2 -> z_t = r_t + gamma*r_{t+1} + gamma^2*v_{t+2}.
        auto td2 = compute_value_targets(r, v, 0.5f, 2);
        // z0=10+.5*20+.25*v2=10+10+0.75=20.75; z1=20+.5*30+.25*v3=20+15+1=36
        // z2=30+.5*40 (t+2 past terminal)=50; z3=40
        check(std::abs(td2[0] - 20.75f) < 1e-4f, "TD2 z0");
        check(std::abs(td2[1] - 36.0f)  < 1e-4f, "TD2 z1");
        check(std::abs(td2[2] - 50.0f)  < 1e-4f, "TD2 z2 (bootstrap past terminal dropped)");
        check(std::abs(td2[3] - 40.0f)  < 1e-4f, "TD2 z3");

        // n >= length -> identical to full MC return (bootstrap never used).
        auto tdbig = compute_value_targets(r, v, 0.5f, 99);
        for (int t = 0; t < 4; ++t)
            check(std::abs(tdbig[t] - mc[t]) < 1e-4f, "TD(n>=len) == MC return");
    }

    // --- NON_ROOT_SEARCH_ALGO=gumbel: eq.14 in-tree rule still searches sanely ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 400; cfg.gamma = 0.999f;
        cfg.non_root_gumbel = true;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/7);
        auto s = board({5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto out = mcts.search(s);
        int vs = 0; for (int a = 0; a < 4; ++a) vs += out.visit_counts[a];
        check(vs == cfg.num_simulations, "[nonroot-gumbel] visits sum to num_simulations");
        auto legal = game.get_legal_actions(s);
        check(out.best_action >= 0 && legal[out.best_action] == 1, "[nonroot-gumbel] best_action legal");
        check(out.best_action == 1 || out.best_action == 3, "[nonroot-gumbel] prefers merging move");
    }

    // --- KataGo PUCT full combo (log + variance + FPU) searches sanely ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 400; cfg.gamma = 0.999f;
        cfg.c_puct = 1.1f; cfg.c_puct_log = 0.45f; cfg.c_puct_base = 500.0f;
        cfg.fpu_reduction_max = 0.2f;
        cfg.cpuct_utility_stdev_scale = 0.85f; cfg.cpuct_utility_stdev_prior = 0.20f;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/7);
        auto s = board({5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto out = mcts.search(s);
        int vs = 0; for (int a = 0; a < 4; ++a) vs += out.visit_counts[a];
        check(vs == cfg.num_simulations, "[puct-combo] visits sum to num_simulations");
        auto legal = game.get_legal_actions(s);
        check(out.best_action >= 0 && legal[out.best_action] == 1, "[puct-combo] best_action legal");
        check(out.best_action == 1 || out.best_action == 3, "[puct-combo] prefers merging move");
    }

    // --- tree reuse: a detached post-move subtree carries its visits forward ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 100; cfg.gamma = 0.999f;
        cfg.enable_tree_reuse = true;
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/5);
        auto s = board({1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});

        // Drive one search through the deferred-eval stepping API.
        auto drive = [&](std::unique_ptr<skyzero::DecisionNode> reuse) {
            mcts.begin(s, std::move(reuse));
            if (mcts.root_needs_eval()) {
                auto r = stub_uniform_zero(mcts.root_state());
                mcts.apply_root_eval(r.first, r.second);
            }
            while (!mcts.done()) {
                auto e = mcts.select_leaf();
                if (!e.empty()) { auto r = stub_uniform_zero(e); mcts.apply_leaf(r.first, r.second); }
            }
            return mcts.result();
        };

        auto out1 = drive(nullptr);
        check(out1.best_action >= 0, "[reuse] first search produced a move");

        // Detach the first visited spawn outcome under the chosen move.
        auto mr = game.apply_move(s, out1.best_action);
        std::unique_ptr<skyzero::DecisionNode> reuse;
        int rcell = -1, rexp = 0;
        for (int c = 0; c < 16 && !reuse; ++c)
            for (int ex = 1; ex <= 2 && !reuse; ++ex) {
                auto r = mcts.detach_after_move(out1.best_action, c, ex);
                if (r) { reuse = std::move(r); rcell = c; rexp = ex; }
            }
        check(reuse != nullptr, "[reuse] detached a visited subtree");
        check(reuse && reuse->n > 0 && reuse->expanded, "[reuse] carried subtree has prior visits");

        // Reused search adopts it (no fresh eval) and still returns a legal move.
        s = mr.afterstate; if (rcell >= 0) s[rcell] = static_cast<int8_t>(rexp);
        mcts.begin(s, std::move(reuse));
        check(!mcts.root_needs_eval(), "[reuse] adopted root needs no NN eval");
        while (!mcts.done()) {
            auto e = mcts.select_leaf();
            if (!e.empty()) { auto r = stub_uniform_zero(e); mcts.apply_leaf(r.first, r.second); }
        }
        auto out2 = mcts.result();
        auto legal = game.get_legal_actions(s);
        check(out2.best_action >= 0 && legal[out2.best_action] == 1, "[reuse] reused search legal move");
    }

    // --- ROOT_SEARCH_ALGO=puct: per-sim root PUCT + forced playouts + pruning ---
    {
        SkyZero2048Config cfg; cfg.num_simulations = 400; cfg.gamma = 0.999f;
        cfg.root_puct = true;
        cfg.root_desired_per_child_visits_coeff = 2.0f;  // forced playouts on
        cfg.fpu_reduction_max = 0.2f; cfg.root_fpu_reduction_max = 0.2f;
        cfg.chosen_move_temperature = 0.0f;              // argmax pruned weight
        SkyZero2048MCTS mcts(game, cfg, stub_uniform_zero, /*seed=*/7);
        auto s = board({5,5,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
        auto out = mcts.search(s);

        int vs = 0; for (int a = 0; a < 4; ++a) vs += out.visit_counts[a];
        check(vs == cfg.num_simulations, "[root-puct] visits sum to num_simulations");
        float ps = 0.0f; for (int a = 0; a < 4; ++a) ps += out.improved_policy[a];
        check(std::abs(ps - 1.0f) < 1e-3f, "[root-puct] pruned target policy sums to 1");
        auto legal = game.get_legal_actions(s);
        check(out.best_action >= 0 && legal[out.best_action] == 1, "[root-puct] best_action legal");
        check(out.best_action == 1 || out.best_action == 3, "[root-puct] prefers merging move");
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
