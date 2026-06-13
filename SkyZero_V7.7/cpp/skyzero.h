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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace skyzero {

// ---------------------------------------------------------------------------
// Config — only the fields the C++ selfplay path needs.
// Python side owns everything training-related.
// ---------------------------------------------------------------------------
struct SkyZeroConfig {
    int board_size = 15;

    // Gumbel MCTS
    int num_simulations = 64;
    int gumbel_m = 16;
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_noise_enabled = true;

    // Non-root (in-tree) action selection. This only switches how children
    // are picked at non-root nodes during descent:
    //   kPuct   — KataGo-style PUCT + FPU + variance-scaled cpuct (default).
    //   kGumbel — the Gumbel paper's deterministic "Full Gumbel" rule
    //             (Danihelka et al. 2022, eq. 14): argmax_a [ π'(a) −
    //             N(a)/(1+Σ_b N(b)) ], where π' is the completed-Q improved
    //             policy at that node (same construction as the Gumbel root
    //             target). See gumbel_deterministic_select() below.
    enum class NonRootSearchAlgo { kPuct, kGumbel };
    NonRootSearchAlgo non_root_search_algo = NonRootSearchAlgo::kPuct;
    static NonRootSearchAlgo parse_non_root_search_algo(const std::string& s) {
        return s == "gumbel" ? NonRootSearchAlgo::kGumbel : NonRootSearchAlgo::kPuct;
    }

    // Root action selection:
    //   kGumbel — Gumbel sequential halving (default; the historical behavior).
    //   kPuct   — KataGo-style per-simulation PUCT at the root. Selection uses
    //             root-specific FPU (root_fpu_reduction_max), optional
    //             Dirichlet noise / policy temperature on the root priors, and
    //             optional forced playouts; the training policy target becomes
    //             the *pruned* visit distribution and the chosen move is
    //             temperature-sampled from it (see puct_root_assemble()).
    // kPuct requires non_root_search_algo == kPuct (validate()).
    enum class RootSearchAlgo { kGumbel, kPuct };
    RootSearchAlgo root_search_algo = RootSearchAlgo::kGumbel;
    static RootSearchAlgo parse_root_search_algo(const std::string& s) {
        return s == "puct" ? RootSearchAlgo::kPuct : RootSearchAlgo::kGumbel;
    }

    // PUCT root knobs (only read when root_search_algo == kPuct). Defaults are
    // the match-safe values; selfplay_main overlays KataGo's selfplay values
    // (cpp/configs/training/selfplay*.cfg) as its cfg_get defaults.
    bool root_noise_enabled = false;                    // rootNoiseEnabled (Dirichlet)
    float root_dirichlet_noise_weight = 0.25f;          // rootDirichletNoiseWeight
    float root_dirichlet_total_concentration = 6.75f;   // rootDirichletNoiseTotalConcentration (0.03 * 15²; KataGo 19²: 10.83)
    float root_policy_temperature = 1.0f;               // rootPolicyTemperature
    float root_policy_temperature_early = 1.0f;         // rootPolicyTemperatureEarly
    float root_fpu_reduction_max = 0.0f;                // rootFpuReductionMax
    float root_desired_per_child_visits_coeff = 0.0f;   // rootDesiredPerChildVisitsCoeff (forced playouts; 0 = off)
    float chosen_move_temperature = 0.0f;               // chosenMoveTemperature (0 = argmax visits)
    float chosen_move_temperature_early = 0.0f;         // chosenMoveTemperatureEarly
    float chosen_move_temperature_halflife = 19.0f;     // chosenMoveTemperatureHalflife (19×19-equivalent turns)

    // useLcbForSelection — pick the final move by lower-confidence-bound among
    // adequately-visited root children (KataGo getPlaySelectionValues). EVAL /
    // PLAY ONLY: only the tree-parallel backend consults this; leaf-parallel
    // selfplay always passes use_lcb=false, so selfplay data is never affected.
    // Only active when root_search_algo == kPuct (gumbel root never assembles).
    bool lcb_for_selection = true;

    void validate() const {
        if (root_search_algo == RootSearchAlgo::kPuct &&
            non_root_search_algo == NonRootSearchAlgo::kGumbel) {
            throw std::runtime_error(
                "ROOT_SEARCH_ALGO=puct requires NON_ROOT_SEARCH_ALGO=puct");
        }
    }

    // Playout Cap Randomization (KataGo paper §4.1; KataGo's "cheap search"
    // is called fastSearch here). Selfplay-only: per move, with
    // fast_search_prob the search budget drops to 1/6 of the full budget
    // (fixed ratio, mirroring KataGo's 600/100; follows the warmup
    // NUM_SIMULATIONS stages automatically), root exploration is disabled
    // for that search (Gumbel noise for root=gumbel; Dirichlet noise /
    // forced playouts / root FPU & policy temperature for root=puct, KataGo
    // play.cpp:1193-1205), and the row's base training weight becomes
    // fast_search_target_weight (0 = the position only scaffolds the
    // value/TD-target chains; policy surprise can still resurrect it).
    // Mutually exclusive with the soft-resign visit reduction on the same
    // move (KataGo play.cpp:1076-1094). prob 0 = PCR off (the switch).
    float fast_search_prob = 0.0f;          // cheapSearchProb
    float fast_search_target_weight = 0.0f; // cheapSearchTargetWeight

    // KataGo side positions (sidePositionProb, play.cpp:1594). Selfplay-only:
    // per move, with side_position_prob, fork ONE *alternative* move off the
    // main line (chooseRandomForkingMove: 70% temp-1 / 25% temp-2 / 5% random,
    // the played move banned), search that forked position independently, and
    // emit it as an extra training row of off-policy policy + search-value data.
    // Because the row is off the game line it cannot carry the main-line-only
    // targets: futurepos is masked (weight 0 + zeroed planes, has_futurepos=
    // false) and opponent-policy is masked (has_opponent_policy=false), exactly
    // as KataGo passes NULL posHistForFutureBoards / next-policy for side rows
    // (trainingwrite.cpp:1245-1249). value + TD ARE trained at full weight from
    // the side search WDL (KataGo valueTargetWeight=tdValueTargetWeight=1.0).
    // The side search uses the SAME full settings as a main search (root noise
    // + forced playouts ON, cleaned by target pruning / improved policy), as
    // KataGo reuses its selfplay bot for side positions (play.cpp:1899). With
    // 25% probability a searched side position is continued two plies and
    // re-queued, so the fork can sit on an earlier turn too — KataGo's
    // geometric continuation (play.cpp:1885/1944). selfplay_main overlays the
    // KataGo selfplay default 0.02 as its cfg_get default (this struct default
    // is the inert match-mode value, never read outside selfplay); 0 disables.
    float side_position_prob = 0.0f;        // sidePositionProb
    // Visit budget for the side search; <=0 means the full num_simulations
    // (KataGo searches side positions at full strength, not cheaply).
    int side_position_visits = 0;

    // PUCT / FPU
    float c_puct = 1.1f;
    float c_puct_log = 0.45f;
    float c_puct_base = 500.0f;
    float fpu_pow = 1.0f;
    float fpu_reduction_max = 0.08f;
    float fpu_loss_prop = 0.0f;

    // Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297, v1.9+).
    // Multiplies cpuct by `1 + scale * (stdev/prior - 1)` where stdev is
    // Bayesian-shrunk per-visit utility stdev at the parent. High-variance
    // subtrees get more exploration. Applied at non-root only (Gumbel root
    // uses sequential halving, not PUCT). scale=0 disables.
    // KataGo SearchParams() ctor: 0.25 / 1.0 / 0.0; forBot()/forTestsV2(): 0.40 / 2.0 / 0.85 (used here).
    float cpuct_utility_stdev_prior = 0.40f;
    float cpuct_utility_stdev_prior_weight = 2.0f;
    float cpuct_utility_stdev_scale = 0.85f;

    // Stochastic transform at inference time
    bool enable_stochastic_transform_inference_for_root = true;
    bool enable_stochastic_transform_inference_for_child = true;

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

    torch::Device device = torch::kCPU;
};

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
// Virtual-loss weight: how many "losses" each in-flight visit charges to a node
// during parallel descent (KataGo numVirtualLossesPerThread = 3). The vloss
// counter is bumped by this amount per in-flight visit, so every score formula
// that reads `n + vloss` / `(v[2]-v[0]) - vloss` picks up the weighting with no
// further change.
constexpr int kVirtualLossWeight = 3;

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
    int vloss = 0;           // in-flight virtual losses, in units of kVirtualLossWeight

    MCTSNode() = default;
    MCTSNode(std::vector<int8_t> s, int p,
             float pr = 0.0f, MCTSNode* par = nullptr, int act = -1)
        : state(std::move(s)), to_play(p), prior(pr), parent(par), action_taken(act) {}

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
    std::vector<std::array<float, 3>> root_child_wdl;       // per-action root-perspective mean WDL ({0,0,0} if unvisited); canvas-stride
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

// ---------------------------------------------------------------------------
// PUCT + FPU helpers.
// Variance-scaled cPUCT is active when cfg.cpuct_utility_stdev_scale != 0
// (KataGo searchexplorehelpers.cpp:280-297); set scale=0 to fix factor at 1.0.
// ---------------------------------------------------------------------------

struct SelectParams {
    float explore_scaling;
    float fpu_value;
};

// effective_parent_n = node.n (single-thread) or node.n + node.vloss (parallel).
// is_root switches to the root-specific FPU reduction (KataGo
// rootFpuReductionMax) — only reachable when root_search_algo == kPuct or in
// the analysis/ponder root-PUCT path.
inline SelectParams compute_select_params(
    const MCTSNode& node,
    int effective_parent_n,
    float visited_policy_mass,
    const SkyZeroConfig& cfg,
    bool is_root = false
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

    const float fpu_reduction_max = is_root ? cfg.root_fpu_reduction_max : cfg.fpu_reduction_max;
    const float reduction = fpu_reduction_max * std::sqrt(visited_policy_mass);
    float fpu_value = parent_utility_for_fpu - reduction;
    fpu_value = fpu_value + ((-1.0f) - fpu_value) * cfg.fpu_loss_prop;

    return {explore_scaling, fpu_value};
}

// ---------------------------------------------------------------------------
// Gumbel deterministic non-root action selection (Danihelka et al. 2022,
// "Policy improvement by planning with Gumbel", eq. 14 — the "Full Gumbel"
// in-tree rule that replaces PUCT). Selects
//     argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ]
// where π'(a) = softmax_a( logit(a) + σ(completedQ(a)) ) is the completed-Q
// improved policy at this node, the same construction as the Gumbel root
// target:
//   * visited children (N>0) use their empirical node-perspective utility;
//   * unvisited children use v_mix, the prior-weighted blend of the node's NN
//     value and the visited children's utilities.
//   σ(q) = (c_visit + max_b N(b)) · c_scale · (q+1)/2,  q ∈ [−1,1].
// Subtracting N(a)/(1+Σ_b N(b)) makes the visit counts track π' over the
// search (the action just visited is penalised on the next descent), so this
// also provides path diversification when N includes virtual loss.
//
// `stats` holds one entry per child, gathered by the caller under whatever
// locking its backend needs; N(a) is the effective count (n + vloss).
// `node_nn_utility` = W−L of the node's own NN value (node-to-play view).
// Returns the index into `stats` of the chosen child, or -1 if empty.
// ---------------------------------------------------------------------------
struct GumbelChildStat {
    float prior;      // P(a) = node.nn_policy[action]
    int eff_n;        // N(a) = child.n + child.vloss
    float utility;    // node-perspective (W−L)/N for visited children; unused if eff_n==0
    float logit;      // node.nn_logits[action]
};

inline int gumbel_deterministic_select(
    const std::vector<GumbelChildStat>& stats,
    float node_nn_utility,
    float c_visit,
    float c_scale
) {
    const int k = static_cast<int>(stats.size());
    if (k <= 1) return k - 1;  // 0 children → -1; 1 child → 0

    int sum_n = 0;
    int max_n = 0;
    float wq_num = 0.0f;   // Σ_{N>0} π(a)·u(a)
    float wq_den = 0.0f;   // Σ_{N>0} π(a)
    for (const auto& s : stats) {
        sum_n += s.eff_n;
        if (s.eff_n > max_n) max_n = s.eff_n;
        if (s.eff_n > 0) {
            wq_num += s.prior * s.utility;
            wq_den += s.prior;
        }
    }

    // v_mix (node-perspective utility) completes the unvisited actions.
    float v_mix = node_nn_utility;
    if (sum_n > 0 && wq_den > 0.0f) {
        const float wq = wq_num / wq_den;
        v_mix = (node_nn_utility + static_cast<float>(sum_n) * wq)
                / (1.0f + static_cast<float>(sum_n));
    }

    const float sigma_coeff = (c_visit + static_cast<float>(max_n)) * c_scale;

    // improved logits = logit + σ(completedQ). softmax over children is the
    // same as over legal actions (children exist only for legal moves), so
    // π'(a) for a child equals its entry in the full improved policy.
    std::vector<float> improved(static_cast<size_t>(k));
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        const float u = (stats[i].eff_n > 0) ? stats[i].utility : v_mix;
        const float q_norm = (u + 1.0f) * 0.5f;
        improved[i] = stats[i].logit + sigma_coeff * q_norm;
        if (improved[i] > max_logit) max_logit = improved[i];
    }
    float exp_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        improved[i] = std::exp(improved[i] - max_logit);
        exp_sum += improved[i];
    }
    const float inv_sum = (exp_sum > 1e-20f) ? (1.0f / exp_sum) : 0.0f;

    // eq. 14: argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ].
    const float inv_total = 1.0f / (1.0f + static_cast<float>(sum_n));
    int best_i = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        const float pi_prime = (inv_sum > 0.0f)
            ? improved[i] * inv_sum
            : 1.0f / static_cast<float>(k);
        const float score = pi_prime - static_cast<float>(stats[i].eff_n) * inv_total;
        if (score > best_score) {
            best_score = score;
            best_i = i;
        }
    }
    return best_i;
}

// ---------------------------------------------------------------------------
// PUCT root (KataGo-style) shared helpers. Both MCTS backends use these when
// root_search_algo == kPuct; the per-simulation descent stays backend-local.
// ---------------------------------------------------------------------------

// Turn-number proxy: stones on board. Gomoku/Renju never remove stones, so
// the count equals the ply (balanced-opening stones count as played moves,
// matching KataGomo's turn number). `state` is board-stride {0,±1}.
inline int count_stones(const std::vector<int8_t>& state) {
    int n = 0;
    for (int8_t v : state) {
        if (v != 0) ++n;
    }
    return n;
}

// KataGo Search::interpolateEarly (searchhelpers.cpp:541): exponential decay
// from `early` to `late`; halflife is measured in 19×19-equivalent turns and
// rescaled by board area.
inline float interpolate_early(
    int turn_number, float halflife, int board_area, float early, float late
) {
    const float raw_halflives = static_cast<float>(turn_number) / std::max(1e-6f, halflife);
    const float halflives = raw_halflives * 19.0f
        / std::sqrt(static_cast<float>(std::max(1, board_area)));
    return late + (early - late) * std::pow(0.5f, halflives);
}

// Rewrite root child priors from the clean root.nn_policy: optional
// rootPolicyTemperature flattening (KataGo searchhelpers.cpp:170-205), then
// optional Dirichlet noise where half the total concentration is split
// uniformly and half shaped by log-policy (computeDirichletAlphaDistribution
// + addDirichletNoise). Always derives from nn_policy, so re-searching a
// reused root never compounds noise; root.nn_policy itself stays clean for
// the npz / surprise-weighting consumers.
inline void apply_root_policy_noise_and_temperature(
    MCTSNode& root, int turn_number, int board_area,
    const SkyZeroConfig& cfg, std::mt19937& rng
) {
    const size_t k = root.children.size();
    if (k == 0) return;
    const bool want_temp = cfg.root_policy_temperature != 1.0f
        || cfg.root_policy_temperature_early != 1.0f;
    if (!cfg.root_noise_enabled && !want_temp) return;

    // Clean priors; children exist exactly for the legal moves with p > 0 and
    // nn_policy sums to 1 over them.
    std::vector<double> p(k, 0.0);
    for (size_t i = 0; i < k; ++i) {
        const int a = root.children[i]->action_taken;
        if (a >= 0 && a < static_cast<int>(root.nn_policy.size())) {
            p[i] = static_cast<double>(root.nn_policy[static_cast<size_t>(a)]);
        }
    }

    if (want_temp) {
        const float temp = interpolate_early(
            turn_number, cfg.chosen_move_temperature_halflife, board_area,
            cfg.root_policy_temperature_early, cfg.root_policy_temperature);
        double max_p = 0.0;
        for (double v : p) max_p = std::max(max_p, v);
        if (max_p > 0.0 && temp > 0.0f) {
            const double log_max = std::log(max_p);
            const double inv_temp = 1.0 / static_cast<double>(temp);
            double sum = 0.0;
            for (size_t i = 0; i < k; ++i) {
                if (p[i] > 0.0) {
                    p[i] = std::exp((std::log(p[i]) - log_max) * inv_temp);
                    sum += p[i];
                }
            }
            if (sum > 0.0) {
                for (size_t i = 0; i < k; ++i) p[i] /= sum;
            }
        }
    }

    if (cfg.root_noise_enabled) {
        // Alpha proportions: 0.5 * (uniform + log-policy-shaped), capped at
        // policy 0.01 (KataGo computeDirichletAlphaDistribution).
        std::vector<double> alpha(k, 0.0);
        double log_sum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            alpha[i] = std::log(std::min(0.01, p[i]) + 1e-20);
            log_sum += alpha[i];
        }
        const double log_mean = log_sum / static_cast<double>(k);
        double prop_sum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            alpha[i] = std::max(0.0, alpha[i] - log_mean);
            prop_sum += alpha[i];
        }
        const double uniform = 1.0 / static_cast<double>(k);
        for (size_t i = 0; i < k; ++i) {
            alpha[i] = (prop_sum <= 0.0)
                ? uniform
                : 0.5 * (alpha[i] / prop_sum + uniform);
        }
        double draw_sum = 0.0;
        std::vector<double> draw(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            const double a = alpha[i] * static_cast<double>(cfg.root_dirichlet_total_concentration);
            if (a > 0.0) {
                std::gamma_distribution<double> gamma(a, 1.0);
                draw[i] = gamma(rng);
            }
            draw_sum += draw[i];
        }
        if (draw_sum > 0.0) {
            const double w = static_cast<double>(cfg.root_dirichlet_noise_weight);
            for (size_t i = 0; i < k; ++i) {
                p[i] = (draw[i] / draw_sum) * w + p[i] * (1.0 - w);
            }
        }
    }

    for (size_t i = 0; i < k; ++i) {
        root.children[i]->prior = static_cast<float>(p[i]);
    }
}

// Per-root-child snapshot for puct_root_assemble. The caller gathers these
// under whatever locking its backend needs.
struct PuctRootChildStat {
    int action = -1;
    int n = 0;                          // real visits (no vloss)
    float q = 0.0f;                     // root-perspective mean utility (W−L); valid when n > 0
    float prior = 0.0f;                 // selection prior (post noise/temperature)
    std::array<float, 3> wdl{0.0f, 0.0f, 0.0f};  // root-perspective mean W,D,L
    float q_sum_sq = 0.0f;              // Σ u_i² (u = value[2]-value[0]) for the LCB variance estimate
};

struct PuctRootMoveResult {
    std::vector<float> target_policy;   // canvas-stride pruned-visit distribution
    int chosen_action = -1;
    std::vector<std::vector<int>> phases;  // synthetic [visited..., chosen] for the front-end overlay
};

// Post-search root processing for PUCT root, mirroring KataGo
// getPlaySelectionValues (searchresults.cpp:64-195) + getChosenMoveLoc:
//   1. best child by weight with a small policy bonus / one-visit discount;
//   2. every other child's weight is capped at the PUCT-inverse amount the
//      best child's selection value retrospectively justifies
//      (getReducedPlaySelectionWeight) — strips forced-playout/noise excess;
//   3. the pruned weights are BOTH the training policy target and the
//      relative probabilities for temperature move sampling.
// explore_scaling comes from compute_select_params(root, root.n, ..., is_root)
// at the final root state.
// use_lcb (eval/play only — leaf-parallel selfplay passes false): after the
// pruned-visit target_policy is fixed, re-pick the chosen move by KataGo's
// lower-confidence-bound rule (getPlaySelectionValues + getSelfUtilityLCBAnd
// Radius). Only the chosen move shifts; target_policy stays the visit
// distribution, so selfplay data is unaffected even if it were ever set true.
inline PuctRootMoveResult puct_root_assemble(
    const std::vector<PuctRootChildStat>& stats,
    int action_size,
    float explore_scaling,
    int turn_number,
    int board_area,
    const SkyZeroConfig& cfg,
    std::mt19937& rng,
    bool use_lcb
) {
    PuctRootMoveResult out;
    out.target_policy.assign(static_cast<size_t>(action_size), 0.0f);
    if (stats.empty()) return out;

    // Best child (KataGo searchresults.cpp:119-139): goodness = weight with
    // one visit's worth discounted + small prior bonus. weight == n here.
    int best_i = 0;
    float best_g = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < stats.size(); ++i) {
        const float n = static_cast<float>(stats[i].n);
        const float g = std::max(0.0f, n - 1.0f) + 2.0f * stats[i].prior;
        if (g > best_g) {
            best_g = g;
            best_i = static_cast<int>(i);
        }
    }
    const auto& best = stats[static_cast<size_t>(best_i)];
    const float best_value = best.q
        + explore_scaling * best.prior / (1.0f + static_cast<float>(best.n));

    std::vector<float> w(stats.size(), 0.0f);
    for (size_t i = 0; i < stats.size(); ++i) {
        if (stats[i].n <= 0) continue;
        if (static_cast<int>(i) == best_i) {
            w[i] = static_cast<float>(stats[i].n);
            continue;
        }
        // Invert score = q + es*prior/(1+w) for w at score == best_value.
        const float denom = best_value - stats[i].q;
        float wanted = static_cast<float>(stats[i].n);
        if (denom > 1e-12f) {
            wanted = explore_scaling * stats[i].prior / denom - 1.0f;
        }
        wanted = std::min(wanted, static_cast<float>(stats[i].n));
        w[i] = std::ceil(std::max(0.0f, wanted));
    }

    float w_sum = 0.0f;
    float w_max = 0.0f;
    for (float v : w) {
        w_sum += v;
        w_max = std::max(w_max, v);
    }
    if (w_sum <= 0.0f) {
        // No visits at all (zero-budget search): fall back to the priors.
        int a_best = best.action;
        if (a_best >= 0 && a_best < action_size) {
            out.target_policy[static_cast<size_t>(a_best)] = 1.0f;
        }
        out.chosen_action = a_best;
        return out;
    }

    for (size_t i = 0; i < stats.size(); ++i) {
        const int a = stats[i].action;
        if (a >= 0 && a < action_size) {
            out.target_policy[static_cast<size_t>(a)] = w[i] / w_sum;
        }
    }

    // LCB move selection (eval/play only — runs AFTER target_policy is fixed, so
    // the training target stays the pure pruned-visit distribution). Mirrors
    // KataGo getPlaySelectionValues (searchresults.cpp:197-243) + getSelfUtility
    // LCBAndRadius (searchhelpers.cpp:555-598), specialized to SkyZero's WDL-only
    // utility (no score head → no scoreMean/scoreUtility terms, unit per-visit
    // weights → weightSum == weightSqSum == n). Bumps the best lower-confidence-
    // bound child's selection weight so it wins the argmax / temperature draw.
    if (use_lcb) {
        constexpr float kLcbStdevs = 4.0f;            // KataGo lcbStdevs
        constexpr float kMinVisitPropForLCB = 0.05f;  // KataGo minVisitPropForLCB
        constexpr float kUtilityRangeRadius = 1.0f;   // winLossUtilityFactor; WDL utility ∈ [−1,1]

        auto child_lcb_radius = [&](const PuctRootChildStat& s,
                                    float& out_lcb, float& out_radius) {
            out_radius = 2.0f * kUtilityRangeRadius * kLcbStdevs;
            out_lcb = -out_radius;
            if (s.n <= 0) return;
            float weight_sum = static_cast<float>(s.n);     // unit per-visit weights
            float weight_sq_sum = static_cast<float>(s.n);
            float ess = weight_sum * weight_sum / weight_sq_sum;
            const float utility_avg = s.q;                  // root-perspective mean
            float utility_sq_avg = s.q_sum_sq / static_cast<float>(s.n);
            // KataGo low-visit prior: add a small-weight prior that the variance
            // is the largest it can be, so the radius behaves at tiny sample sizes.
            const float prior_weight = weight_sum / (ess * ess * ess);
            utility_sq_avg = std::max(utility_sq_avg, utility_avg * utility_avg + 1e-8f);
            utility_sq_avg = (utility_sq_avg * weight_sum
                              + (utility_sq_avg + kUtilityRangeRadius * kUtilityRangeRadius) * prior_weight)
                             / (weight_sum + prior_weight);
            weight_sum += prior_weight;
            weight_sq_sum += prior_weight * prior_weight;
            ess = weight_sum * weight_sum / weight_sq_sum;
            const float variance = std::max(0.0f, utility_sq_avg - utility_avg * utility_avg);
            out_radius = std::sqrt(variance / ess) * kLcbStdevs;
            out_lcb = utility_avg - out_radius;
        };

        std::vector<float> lcb(stats.size(), 0.0f);
        std::vector<float> radius(stats.size(), 0.0f);
        float best_lcb = -1e10f;
        int best_lcb_i = -1;
        for (size_t i = 0; i < stats.size(); ++i) {
            child_lcb_radius(stats[i], lcb[i], radius[i]);
            // Eligible for "best LCB" only with enough pruned weight (KataGo gates
            // on minVisitPropForLCB * nonLCBBestChildWeight; w_max is that best).
            if (w[i] > 0.0f && w[i] >= kMinVisitPropForLCB * w_max && lcb[i] > best_lcb) {
                best_lcb = lcb[i];
                best_lcb_i = static_cast<int>(i);
            }
        }
        if (best_lcb_i >= 0) {
            float adjusted = w[static_cast<size_t>(best_lcb_i)];
            for (size_t i = 0; i < stats.size(); ++i) {
                if (static_cast<int>(i) == best_lcb_i) continue;
                const float excess = best_lcb - lcb[i];
                if (excess < 0.0f) continue;  // this child's gate failed, not actually better
                // How much wider would i's radius have to be before its lcb loses?
                // That factor² is the extra weight the best-lcb child deserves.
                const float radius_factor = (radius[i] + excess) / (radius[i] + 0.20f * excess);
                const float lbound = radius_factor * radius_factor * w[i];
                if (lbound > adjusted) adjusted = lbound;
            }
            w[static_cast<size_t>(best_lcb_i)] = adjusted;
            w_max = std::max(w_max, adjusted);
        }
    }

    // Chosen move: temperature sampling over the pruned weights (KataGo
    // chooseIndexWithTemperature with onlyBelowProb left at its default).
    const float temp = interpolate_early(
        turn_number, cfg.chosen_move_temperature_halflife, board_area,
        cfg.chosen_move_temperature_early, cfg.chosen_move_temperature);
    int chosen_i = best_i;
    if (temp > 1e-4f) {
        std::vector<double> rel(stats.size(), 0.0);
        const double log_max = std::log(static_cast<double>(w_max));
        double rel_sum = 0.0;
        for (size_t i = 0; i < stats.size(); ++i) {
            if (w[i] > 0.0f) {
                rel[i] = std::exp((std::log(static_cast<double>(w[i])) - log_max)
                                  / static_cast<double>(temp));
                rel_sum += rel[i];
            }
        }
        if (rel_sum > 0.0) {
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            double r = u01(rng) * rel_sum;
            for (size_t i = 0; i < stats.size(); ++i) {
                r -= rel[i];
                if (rel[i] > 0.0 && r <= 0.0) {
                    chosen_i = static_cast<int>(i);
                    break;
                }
            }
        }
    } else {
        float best_w = -1.0f;
        for (size_t i = 0; i < stats.size(); ++i) {
            if (w[i] > best_w) {
                best_w = w[i];
                chosen_i = static_cast<int>(i);
            }
        }
    }
    out.chosen_action = stats[static_cast<size_t>(chosen_i)].action;

    std::vector<int> visited;
    for (const auto& s : stats) {
        if (s.n > 0) visited.push_back(s.action);
    }
    if (!visited.empty()) out.phases.push_back(std::move(visited));
    if (out.chosen_action >= 0) out.phases.push_back({out.chosen_action});
    return out;
}

}  // namespace skyzero

#endif
