#ifndef SKYZERO_ALPHAZERO_H
#define SKYZERO_ALPHAZERO_H

// Config, MCTSNode, and shared helpers for MCTS.
// Ported from CSkyZero_V3/alphazero.h with:
//   * Subtree Value Bias (SVB) removed.
//   * Dynamic variance-scaled cPUCT removed (stdev_factor == 1.0).
//   * Single-threaded `MCTS` class removed (selfplay uses ParallelMCTS only).
//   * Torch save/load `AlphaZero` class removed (Python handles training & checkpoints).
//   * Training / replay-buffer / playout-cap configuration not included here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <torch/torch.h>

namespace skyzero {

// ---------------------------------------------------------------------------
// Search algorithm at the root. GUMBEL is the existing Gumbel + sequential
// halving path; PUCT is KataGomo-aligned vanilla AlphaZero (root Dirichlet,
// PUCT descent every playout, chosenMove temperature for action selection).
// Interior PUCT is identical for both — they share compute_select_params.
// ---------------------------------------------------------------------------
enum class SearchAlgo {
    GUMBEL = 0,
    PUCT = 1,
};

// ---------------------------------------------------------------------------
// Config — only the fields the C++ selfplay path needs.
// Python side owns everything training-related.
// ---------------------------------------------------------------------------
struct AlphaZeroConfig {
    int board_size = 15;

    // Default to Gumbel for backward compat with existing selfplay/elo/play
    // entry points. Switch via SEARCH_ALGO=puct in the .cfg.
    SearchAlgo search_algo = SearchAlgo::GUMBEL;

    // Gumbel MCTS
    int num_simulations = 64;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_noise_enabled = true;

    // Opening exploration window: sample first `half_life` moves ∝ visit counts.
    //   >0 ⇒ global override across all games (window length in plies)
    //   -1 ⇒ per-game game_.board_size auto-adapt (resolved in selfplay_manager.h)
    //    0 ⇒ disabled — always greedy gumbel_action from move 0
    int half_life = 0;
    // Opening sampling temperature: p(a) ∝ N(a)^(1/move_temperature).
    // 1.0 = raw visit counts; <1 sharpens toward SH finalists; →0 = argmax(N).
    float move_temperature = 1.0f;

    // PUCT / FPU
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.08f;
    float fpu_loss_prop = 0.0f;

    // KataGo searchexplorehelpers.cpp:287 — root vs non-root FPU split.
    // Root only sees the descent-time FPU (children that are sometimes
    // unvisited within a single search); the absolute Q at root is never
    // consumed by upstream PUCT, so a smaller reduction wastes fewer sims
    // on long-tail children. KataGo defaults: root 0.1, non-root 0.2.
    // V6 uses root 0.1 / non-root 0.08 (V6 baseline).
    float root_fpu_reduction_max = 0.1f;
    float root_fpu_loss_prop = 0.0f;

    // KataGomo gamelogic.cpp:400-411 + nneval.cpp:706-823. Per NN forward,
    // run a VCT (VCFsolver) for the side-to-move; on a definitive win,
    // collapse the legal mask to {first_move} and force value to (1,0,0)
    // WLD. This makes search converge to the winning move automatically at
    // every expanded node where VCT wins, not just at the search root —
    // mirrors KataGomo's behavior (search isn't aware VCT happened, NN
    // legal-mask handles it).
    bool use_vct = false;
    int  vct_max_nodes = 50000;
    // Cheaper fallback: only run VCT at the search root, override the
    // returned MCTSSearchOutput rather than every leaf NN call. Use when
    // worker CPU is the bottleneck. ~+30 Elo (vs ~+50 for full alignment).
    bool use_vct_at_root_only = false;

    // KataGomo cross-game position pool (docs/v6_fork_pool_design.md). New
    // games occasionally start from a saved mid-game position instead of
    // running balanced opening, so training data covers the natural mid-game
    // manifold. fork_random_top_k > 0 makes the loaded position then play
    // one uniform-random move from the NN's top-K legal moves to add
    // trajectory divergence beyond the loaded entry.
    float fork_load_prob = 0.0f;          // P(start new game from pool); 0 disables
    float fork_save_prob = 0.0f;          // P(save 1 position from this game); 0 disables
    int   fork_pool_cap = 10000;
    int   fork_min_move_to_save = 8;
    int   fork_random_top_k = 0;

    // KataGo nneval.cpp:696 — global NN policy temperature applied to legal
    // logits before softmax everywhere in the tree (root + child). T > 1
    // flattens (more exploration off-policy), T < 1 sharpens. T == 1 is
    // identity. Note: also applies on the Gumbel path via nn_logits, so
    // changing this with SEARCH_ALGO=gumbel may require retuning gumbel_c_visit.
    float nn_policy_temperature = 1.0f;

    // Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297, v1.9+).
    // Multiplies cpuct by `1 + scale * (stdev/prior - 1)` where stdev is
    // Bayesian-shrunk per-visit utility stdev at the parent. High-variance
    // subtrees get more exploration. Applied at non-root only (Gumbel root
    // uses sequential halving, not PUCT). scale=0 disables.
    // KataGo SearchParams() ctor: 0.25 / 1.0 / 0.0; forBot()/forTestsV2(): 0.40 / 2.0 / 0.85 (used here).
    float cpuct_utility_stdev_prior = 0.40f;
    float cpuct_utility_stdev_prior_weight = 2.0f;
    float cpuct_utility_stdev_scale = 0.85f;

    // KataGomo PUCT — root Dirichlet noise (selfplay exploration). Mirrors
    // SearchParams.{rootNoiseEnabled,rootDirichletNoiseTotalConcentration,
    // rootDirichletNoiseWeight}. Only used when search_algo == PUCT and the
    // node passed to search() is a fresh root (n == 0); on tree-reuse we do
    // NOT re-noise — KataGomo's behavior is identical, see Search::search.
    // Total concentration is the sum α₁+…+α_K of the Dirichlet vector;
    // KataGomo's 10.83 ≈ 0.03·361 (AlphaZero Go default × 19² cells). With K
    // legal moves the per-move α is total / K; weight is the convex blend
    //   prior' = (1 − w)·prior + w·dir_sample.
    bool root_noise_enabled = false;
    float root_dirichlet_total_concentration = 10.83f;
    float root_noise_weight = 0.25f;

    // KataGomo PUCT — chosenMove temperature schedule for selfplay action
    // selection over visit counts. Mirrors SearchParams.{chosenMoveTemperature,
    // chosenMoveTemperatureEarly,chosenMoveTemperatureHalflife,chosenMoveSubtract,
    // chosenMovePrune}. KataGomo selfplay8b defaults: 0.15 / 0.75 / 19 / 0 / 1.
    // Effective T at move m: T_final + (T_early − T_final) * 2^(−m / halflife);
    // halflife is in plies and is scaled by board_area/(19*19) internally so
    // shorter games decay faster (KataGomo searchresults.cpp:getChosenMoveLoc).
    // Subtract drops `subtract` from each visit count before tempering, prune
    // zeroes any move with strictly fewer than `prune` visits.
    // T == 0 ⇒ deterministic argmax-visit (Gumbel-style greedy).
    float chosen_move_temperature = 0.15f;
    float chosen_move_temperature_early = 0.75f;
    float chosen_move_temperature_halflife = 19.0f;
    float chosen_move_subtract = 0.0f;
    float chosen_move_prune = 1.0f;

    // Stochastic transform / symmetry at inference time
    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;
    bool enable_symmetry_inference_for_root = false;
    bool enable_symmetry_inference_for_child = false;

    // KataGo-style root symmetry pruning. At root, symmetry-equivalent legal
    // moves are masked to -inf logit so only the orbit's canonical
    // representative (smallest loc) is searched. Leave OFF for training even
    // when the dataloader does D4 augmentation: pruning still prevents the
    // NN from receiving visit-count signal at 7/8 of equivalent positions.
    // Recommended ON for elo / human play.
    bool root_symmetry_pruning = false;

    // Surprise weighting
    float policy_surprise_data_weight = 0.5f;
    float value_surprise_data_weight = 0.1f;

    // Balanced opening (KataGomo-style). Each game samples r ~ U(0,1):
    //   r <  balance_opening_prob → NN-scored random opening;
    //   r >= balance_opening_prob → empty-board start.
    float balance_opening_prob = 0.8f;
    int balanced_opening_max_tries = 20;
    float balanced_opening_avg_dist_factor = 0.8f;
    float balanced_opening_reject_prob = 0.995f;
    float balanced_opening_reject_prob_fallback = 0.8f;
    // Power applied to (1 - v^2) when sampling the final balance move. KataGomo
    // uses 4 for selfplay (looser, more diverse) and 10 for match-mode Elo
    // (sharper concentration on |v|≈0). See randomopening.cpp:152.
    float balanced_opening_value_exponent = 4.0f;

    // Policy-initialization (KataGomo initGamesWithPolicy). After balanced
    // opening, play ~Exp(1)*policy_init_avg_move_num extra moves sampled from
    // the NN policy^(1/temperature) to push the game off the balance plateau.
    // Set avg_move_num <= 0 to disable.
    float policy_init_avg_move_num = 0.0f;
    float policy_init_temperature = 1.0f;

    // KataGomo PCR (Playout Cap Randomization, play.cpp:946-981). Per move,
    // Bernoulli(cheap_search_prob): cheap → reduce sims to cheap_search_visits
    // and multiply step_sample_weight by cheap_search_target_weight; full →
    // run normal sims (and may still be reduced by reduceVisits below). The
    // two paths are MUTUALLY EXCLUSIVE — KataGomo uses if/else if.
    // When cheap with cheap_search_target_weight ≤ 0: also enable
    // KataGomo "removeRootNoise" bundle (play.cpp:1198-1204) — disable root
    // Dirichlet/Gumbel noise AND collapse root_fpu_{reduction_max,loss_prop}
    // to the non-root values, so 100-visit cheap searches don't waste budget
    // on the wide-root exploration meant for full searches.
    // KataGomo selfplay8b defaults: 0.75 / 200 / 0.0; selfplay1: 0.75 / 100 / 0.0.
    // V6 default 0.0 disables PCR (back-compat).
    float cheap_search_prob = 0.0f;
    int   cheap_search_visits = 100;
    float cheap_search_target_weight = 0.0f;

    // Soft resign (KataGomo reduceVisits-aligned: smooth quadratic interpolation,
    // non-sticky, signed-extreme over fixed-frame v_mix; proportional floor
    // adapted for Gumbel-MCTS warmup-stage NUM_SIMULATIONS).
    // eff_min = max(reduced_visits_min_floor, round(num_simulations * reduced_visits_fraction))
    float soft_resign_threshold = 0.9f;        // reduceVisitsThreshold
    int soft_resign_step_threshold = 3;        // reduceVisitsThresholdLookback
    float soft_resign_sample_weight = 0.1f;    // reducedVisitsWeight
    float reduced_visits_fraction = 0.25f;     // 碾压时压到 num_simulations * fraction
    int reduced_visits_min_floor = 16;         // 绝对下限 (= GUMBEL_M)

    // Sprint 2 #3: keep the subtree under the played action as the new
    // root instead of rebuilding from scratch each ply. Gumbel state is
    // search-local (recomputed in gumbel_sequential_halving), so no
    // per-action noise reset is needed.
    bool enable_tree_reuse = true;

    // KataGomo Playout Doubling Advantage (PDA). Per-game coin flip plays
    // one side with up to `pda_max_ratio`× as many simulations as the other,
    // training the network to robustly play asymmetrically advantaged
    // positions. Selfplay-only (evaluation binaries leave it disabled).
    //   pda_normal_prob: P(this game uses PDA). 0 = off (default).
    //   pda_max_ratio:   max factor f for the favored side (KataGomo
    //                    `maxAsymmetricRatio` default = 8.0).
    //   pda_min_visits_floor: minimum visits after redistribution; below this
    //                    a runtime_error is thrown (mirrors KataGomo behavior;
    //                    catches misconfigurations early). KataGomo
    //                    play.cpp:1047 default = 5.
    // KataGomo references: program/play.cpp:427-440 (per-game roll),
    // program/play.cpp:1027-1051 (per-move 2f/(f+1) redistribute).
    float pda_normal_prob = 0.0f;
    float pda_max_ratio = 8.0f;
    int   pda_min_visits_floor = 5;

    torch::Device device = torch::kCPU;
};

// Per-game KataGomo Playout Doubling Advantage state. Rolled once at game
// start (selfplay_manager::selfplay_once); read on each move to redistribute
// `num_simulations` between the two sides via the 2f/(f+1) formula and to
// set global feature dims 12-13 (pda_active gate + signed doublings).
//   side = 0:  PDA disabled this game (no redistribution, dim 12/13 = 0).
//   side = +1: black is the favored side; black gets f×, white gets 1×.
//   side = -1: white is the favored side; white gets f×, black gets 1×.
// where f = 2^abs_doublings is the effective ratio.
struct PdaState {
    double abs_doublings = 0.0;
    int    side = 0;
};

// Resolves PdaState into the per-NN-forward (pda_signed, pda_active) pair
// fed to global feature dims 12-13. KataGomo searchnnhelpers.cpp:20-25:
// pda_signed flips sign per to_play so the favored side always sees a
// positive doublings value.
inline std::pair<double, bool> pda_signed_active(const PdaState* pda, int to_play) {
    if (!pda || pda->side == 0) return {0.0, false};
    const double signed_v = (to_play == pda->side) ? +pda->abs_doublings : -pda->abs_doublings;
    return {signed_v, true};
}

// ---------------------------------------------------------------------------
// Selfplay parallelism config consumed by ParallelMCTS / SelfplayEngine.
// ---------------------------------------------------------------------------
struct SelfplayParallelConfig {
    int num_workers = 32;
    int num_inference_servers = 2;
    int inference_batch_size = 128;
    int inference_batch_wait_us = 100;
    int leaf_batch_size = 8;
    int max_result_queue_size = 0;  // 0 = auto (2 * num_workers); <0 = unbounded
};

// ---------------------------------------------------------------------------
// MCTS Node
// ---------------------------------------------------------------------------
struct MCTSNode {
    std::vector<int8_t> state;
    int to_play = 1;
    float prior = 0.0f;
    MCTSNode* parent = nullptr;
    int action_taken = -1;

    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<float> nn_policy;          // softmax probabilities (legal-masked)
    std::vector<float> nn_logits;          // raw logits (legal-masked, -inf for illegal)
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};  // WDL

    std::array<float, 3> v{0.0f, 0.0f, 0.0f};
    int n = 0;
    float q_sum_sq = 0.0f;   // Σ u_i² where u_i = value[2]-value[0] per backup; feeds variance-scaled cpuct in compute_select_params.
    int vloss = 0;

    bool is_expanded() const { return !children.empty(); }

    void update(const std::array<float, 3>& value) {
        v[0] += value[0];
        v[1] += value[1];
        v[2] += value[2];
        const float u = value[2] - value[0];
        q_sum_sq += u * u;
        n += 1;
    }
};

// ---------------------------------------------------------------------------
// Search output
// ---------------------------------------------------------------------------
struct MCTSSearchOutput {
    std::vector<float> mcts_policy;                         // improved policy (Gumbel)
    std::array<float, 3> v_mix{0.0f, 0.0f, 0.0f};          // WDL v_mix (search root value)
    std::vector<float> nn_policy;                           // raw NN policy
    std::array<float, 3> nn_value_probs{0.0f, 0.0f, 0.0f};  // raw NN value
    int gumbel_action = -1;                                 // selected action by Gumbel (selfplay/eval/play)
    std::vector<float> visit_counts;                        // raw root-child visit counts N(s,a)
    std::vector<std::vector<int>> gumbel_phases;            // surviving actions at each halving phase (16,8,4,2,1)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
inline std::array<float, 3> flip_wdl(const std::array<float, 3>& in) {
    return {in[2], in[1], in[0]};
}

inline float wdl_utility(const std::array<float, 3>& v) {
    return v[0] - v[2];
}

// KataGo nneval.cpp:696. Apply temperature to legal logits in place. Skip
// `-inf` entries (illegal moves) so they remain masked. Caller must have
// already set illegal logits to -inf. T == 1.0 is a no-op.
inline void apply_nn_policy_temperature(std::vector<float>& logits, float T) {
    if (T == 1.0f) return;
    const float inv_t = 1.0f / std::max(T, 1e-6f);
    for (auto& l : logits) {
        if (std::isfinite(l)) l *= inv_t;
    }
}

// ---------------------------------------------------------------------------
// PUCT + FPU helpers.
// (Dynamic variance-scaled cPUCT removed: stdev_factor is fixed at 1.0.)
// ---------------------------------------------------------------------------

struct SelectParams {
    float explore_scaling;
    float fpu_value;
};

// effective_parent_n = node.n (single-thread) or node.n + node.vloss (parallel).
// is_root selects {fpu_reduction_max, fpu_loss_prop} between regular and root_*
// values, mirroring KataGo searchexplorehelpers.cpp:287-288.
// collapse_root_fpu: KataGo "removeRootNoise" bundle (play.cpp:1201-1202).
// When true and is_root, use {fpu_reduction_max, fpu_loss_prop} at root too —
// cheap searches reuse the previous tree's root child stats, so unvisited new
// children should not get the wide-root exploration boost.
inline SelectParams compute_select_params(
    const MCTSNode& node,
    int effective_parent_n,
    float visited_policy_mass,
    const AlphaZeroConfig& cfg,
    bool is_root = false,
    bool collapse_root_fpu = false
) {
    const float total_child_weight = static_cast<float>(std::max(0, effective_parent_n - 1));

    const float c_puct = cfg.c_puct + cfg.c_puct_log
        * std::log((total_child_weight + cfg.c_puct_base) / cfg.c_puct_base);

    std::array<float, 3> parent_q{0.0f, 0.0f, 0.0f};
    if (node.n > 0) {
        parent_q = {node.v[0] / node.n, node.v[1] / node.n, node.v[2] / node.n};
    }
    const float parent_utility = wdl_utility(parent_q);

    // Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297). Sign
    // of u (W-L vs L-W) doesn't matter — variance is invariant under sign
    // flip. node.q_sum_sq accumulates (value[2]-value[0])² per backup, so
    // utility_sq_avg = q_sum_sq / n is the empirical second moment.
    float parent_utility_stdev_factor = 1.0f;
    if (cfg.cpuct_utility_stdev_scale != 0.0f) {
        const float weight_sum = static_cast<float>(node.n);
        float parent_utility_stdev;
        if (node.n <= 0 || weight_sum <= 1.0f) {
            parent_utility_stdev = cfg.cpuct_utility_stdev_prior;
        } else {
            float utility_sq_avg = node.q_sum_sq / weight_sum;
            const float utility_sq = parent_utility * parent_utility;
            // numerical guard (KataGo line 286-287): observed second moment
            // must be ≥ mean² for variance to be non-negative.
            if (utility_sq_avg < utility_sq) utility_sq_avg = utility_sq;
            const float variance_prior = cfg.cpuct_utility_stdev_prior * cfg.cpuct_utility_stdev_prior;
            const float prior_weight = cfg.cpuct_utility_stdev_prior_weight;
            const float numerator = (utility_sq + variance_prior) * prior_weight + utility_sq_avg * weight_sum;
            const float denominator = prior_weight + weight_sum - 1.0f;
            const float shrunk_variance = std::max(0.0f, numerator / denominator - utility_sq);
            parent_utility_stdev = std::sqrt(shrunk_variance);
        }
        parent_utility_stdev_factor = 1.0f + cfg.cpuct_utility_stdev_scale
            * (parent_utility_stdev / cfg.cpuct_utility_stdev_prior - 1.0f);
    }

    const float explore_scaling = c_puct * std::sqrt(total_child_weight + 0.01f) * parent_utility_stdev_factor;

    const float nn_utility = wdl_utility(node.nn_value_probs);
    const float avg_weight = std::min(1.0f, static_cast<float>(std::pow(visited_policy_mass, cfg.fpu_pow)));
    const float parent_utility_for_fpu = avg_weight * parent_utility + (1.0f - avg_weight) * nn_utility;

    const bool use_root_fpu = is_root && !collapse_root_fpu;
    const float fpu_reduction_max = use_root_fpu ? cfg.root_fpu_reduction_max : cfg.fpu_reduction_max;
    const float fpu_loss_prop = use_root_fpu ? cfg.root_fpu_loss_prop : cfg.fpu_loss_prop;
    const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

}  // namespace skyzero

#endif
