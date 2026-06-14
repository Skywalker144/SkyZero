#ifndef SKYZERO_2048_H
#define SKYZERO_2048_H

// Afterstate Stochastic-AlphaZero MCTS for 2048 (single-agent, stochastic,
// scalar reward). Correctness-first, single-threaded core. Parallel/batched
// inference and Gumbel root selection come later; this version is meant to be
// unit-testable with a stub inference callback (no LibTorch dependency).
//
// Tree structure alternates two node kinds:
//   DecisionNode(state)  --action a (PUCT over 4 dirs)-->  ChanceNode(afterstate, reward)
//   ChanceNode           --spawn outcome (env distribution)-->  DecisionNode(state')
//
// Value semantics (NO zero-sum flip — this is single-agent):
//   V(decision state) = expected discounted sum of future rewards.
//   Q(a) at a decision node = r(s,a) + gamma * V(afterstate)   [accumulated on
//   the ChanceNode], so backup pushes the full return G = r + gamma*G_child up
//   the path. Returns are large (2048 scores reach thousands), so PUCT compares
//   Q values after MuZero-style running min-max normalization.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "envs/game2048.h"

namespace skyzero {

struct SkyZero2048Config {
    int num_simulations = 200;
    float c_puct = 1.25f;
    // KataGo-style PUCT enhancements at non-root decision nodes (ported from
    // SkyZero V7.x; PUCTablation found the full combo is the one clear win).
    // All default OFF (recover the plain V7.1 PUCT); run.cfg turns them on.
    //   log-cpuct: c_puct grows with log(parent visits)
    float c_puct_log = 0.0f;             // 0 = constant c_puct
    float c_puct_base = 500.0f;
    //   FPU reduction: unvisited children get a pessimism penalty in the
    //   normalized [0,1] Q-space (on top of 2048's reward-aware FPU base).
    float fpu_reduction_max = 0.0f;      // 0 = pure reward+gamma*parent_value FPU
    //   Variance-scaled cpuct (KataGo searchexplorehelpers.cpp:280-297): scale
    //   c_puct by the parent's return stdev, measured in the NORMALIZED [0,1]
    //   space (raw stdev / MuZero min-max range) so the scale-free prior matches
    //   2048's unbounded returns. KataGo's [-1,1]-space prior 0.40 -> ~0.20 here.
    float cpuct_utility_stdev_scale = 0.0f;        // 0 = factor fixed at 1.0
    float cpuct_utility_stdev_prior = 0.20f;       // normalized [0,1] units
    float cpuct_utility_stdev_prior_weight = 2.0f;
    // Non-root (in-tree) selection rule. false = PUCT (above). true = the Gumbel
    // "Full Gumbel" eq.14 deterministic rule argmax_a[π'(a) − N(a)/(1+ΣN(b))]
    // (Danihelka et al. 2022); π' is the completed-Q improved policy at the node
    // (same construction as the Gumbel root target). ROOT always uses Gumbel SH.
    bool non_root_gumbel = false;
    // Tree reuse: carry the realized post-move subtree across plies as the next
    // root (warm-start — the reused visits/values seed the Q estimates while the
    // root still runs num_simulations fresh Gumbel-SH sims on top). Wired in the
    // parallel selfplay loop via detach_after_move() + begin(reuse).
    bool enable_tree_reuse = false;
    // ROOT search algorithm: false = Gumbel sequential halving (default).
    // true = KataGo-style per-simulation PUCT at the root (REQUIRES non_root
    // PUCT, i.e. non_root_gumbel=false). Then the root gets root-specific FPU +
    // forced playouts during selection, and the training target / chosen move
    // come from puct_root_assemble (forced-playout/noise excess pruned away).
    bool root_puct = false;
    float root_fpu_reduction_max = 0.0f;            // root-specific FPU (vs fpu_reduction_max)
    float root_desired_per_child_visits_coeff = 0.0f;  // forced playouts; 0 = off
    // Chosen-move temperature (root=puct selfplay move sampling over the pruned
    // visit weights). Interpolates from _early to chosen_move_temperature over
    // the game via _halflife (KataGo chosenMoveTemperature{,Early,Halflife});
    // halflife is in 19x19-equivalent turns, rescaled by board area (=16 here).
    // 0 = argmax pruned-visit weight.
    float chosen_move_temperature = 0.0f;           // the "late" temperature
    float chosen_move_temperature_early = 0.0f;     // early-game temperature
    float chosen_move_temperature_halflife = 19.0f; // turns (rescaled by board area)
    // Root policy temperature (root=puct): flatten the root PRIORS by ^(1/T)
    // before selection for extra exploration (KataGo rootPolicyTemperature{,Early};
    // interpolated by the same halflife). 1.0 = off. Applied to a clean prior
    // rebuilt from the NN logits, so re-searching a reused root never compounds.
    float root_policy_temperature = 1.0f;
    float root_policy_temperature_early = 1.0f;
    bool lcb_for_selection = true;                  // LCB chosen move (eval/play only; selfplay off)
    float gamma = 0.999f;     // discount on future reward
    // Value-target construction in self-play (see compute_value_targets):
    //   0  -> full Monte-Carlo discounted return-to-go (AlphaZero style).
    //   >0 -> n-step TD bootstrap on the MCTS search value (MuZero / Stochastic
    //         MuZero style): far lower variance than the full MC return, which on
    //         2048 carries the entire rest-of-game spawn randomness.
    int td_steps = 0;
    // Gumbel MCTS (Danihelka et al. 2022): root uses Gumbel-Top-k + sequential
    // halving + completed-Q improved policy. Matches az2048/mcts.py.
    float gumbel_c_visit = 50.0f;
    float gumbel_c_scale = 1.0f;
    bool gumbel_noise = true;            // add Gumbel noise at root (off for eval)
    // Dirichlet exploration noise at the root (optional; off by default).
    float root_dirichlet_alpha = 0.0f;   // <=0 disables
    float root_noise_frac = 0.25f;
    // Inference-time stochastic D4 transform (KataGo enable_stochastic_transform_
    // inference_for_root/_child): before each NN eval pick a random dihedral
    // transform, apply it to the input, undo it on the returned policy (action-
    // RELABELED via Game2048::ACTION_PERM, not a plain plane rotation). Cheap
    // test-time symmetry / robustness; the net is already D4-augment-trained.
    // Gated separately for the root eval vs in-tree (child/leaf) evals. Off by
    // default (mainline selfplay runs both on).
    bool stochastic_transform_root = false;
    bool stochastic_transform_child = false;
};

// Build per-step value targets for one finished self-play trajectory.
//   rewards[t] = realized merge reward of the move played at state_t.
//   vroot[t]   = MCTS search value V(state_t) (the backed-up root value).
//
//   td_steps <= 0 : full Monte-Carlo discounted return-to-go
//                     z_t = Σ_{k>=t} γ^{k-t} rewards[k]            (AlphaZero).
//   td_steps  > 0 : n-step TD bootstrap on the search value         (MuZero):
//                     z_t = Σ_{k=t}^{t+n-1} γ^{k-t} rewards[k] + γ^n · vroot[t+n],
//                   dropping the bootstrap term once t+n reaches terminal (the
//                   true post-terminal value is 0). vroot is unused when n<=0.
inline std::vector<float> compute_value_targets(const std::vector<int>& rewards,
                                                const std::vector<float>& vroot,
                                                float gamma, int td_steps) {
    const int T = static_cast<int>(rewards.size());
    std::vector<float> out(T, 0.0f);
    if (td_steps <= 0) {
        float g = 0.0f;
        for (int t = T - 1; t >= 0; --t) { g = rewards[t] + gamma * g; out[t] = g; }
    } else {
        for (int t = 0; t < T; ++t) {
            float z = 0.0f, disc = 1.0f;
            const int b = t + td_steps;
            const int end = std::min(b, T);
            for (int k = t; k < end; ++k) { z += disc * rewards[k]; disc *= gamma; }
            if (b < T) z += disc * vroot[b];   // disc == γ^td_steps here
            out[t] = z;
        }
    }
    return out;
}

// KataGo Search::interpolateEarly: exponential decay from `early` to `late` over
// the game; halflife is in 19x19-equivalent turns, rescaled by board area so a
// given halflife means the same fraction-of-game across board sizes.
inline float interpolate_early(int turn_number, float halflife, int board_area,
                               float early, float late) {
    const float raw_halflives = static_cast<float>(turn_number) / std::max(1e-6f, halflife);
    const float halflives = raw_halflives * 19.0f
        / std::sqrt(static_cast<float>(std::max(1, board_area)));
    return late + (early - late) * std::pow(0.5f, halflives);
}

// KL(p||q) over the 4 directions (KataGo policy surprise; clamp + renormalize).
inline float kl_div4(const std::array<float, 4>& p, const std::array<float, 4>& q) {
    const float eps = 1e-10f;
    float sp = 0.0f, sq = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sp += std::max(eps, std::min(p[i], 1.0f));
        sq += std::max(eps, std::min(q[i], 1.0f));
    }
    if (sp <= eps || sq <= eps) return 0.0f;
    float kl = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (p[i] <= 0.0f) continue;
        const float pi = std::max(eps, std::min(p[i], 1.0f)) / sp;
        const float qi = std::max(eps, std::min(q[i], 1.0f)) / sq;
        kl += pi * (std::log(pi) - std::log(qi));
    }
    return std::max(0.0f, kl);
}

// Policy-Surprise data weighting (KataGo policy_surprise_weighting.h) adapted to
// 2048: 4-direction policy KL(target||nn) + scalar value surprise
// min(|search-nn|/value_scale, 1). Returns a per-position final weight; the
// caller writes it as the row's sample_weight. 2048 trains weighted-loss, so
// this direct-weight form is equivalent to KataGo's stochastic-replication form
// (and resurrects fast-search rows whose base weight is 0 when surprising).
inline std::vector<float> compute_surprise_weights_2048(
    const std::vector<std::array<float, 4>>& target,
    const std::vector<std::array<float, 4>>& nn_pol,
    const std::vector<float>& search_value,
    const std::vector<float>& nn_value,
    const std::vector<float>& base_weight,
    float value_scale, float policy_w, float value_w) {
    const int n = static_cast<int>(target.size());
    std::vector<float> out(n, 0.0f);
    if (n == 0) return out;
    std::vector<float> ps(n), vs(n);
    float sumw = 0.0f;
    for (int i = 0; i < n; ++i) {
        ps[i] = kl_div4(target[i], nn_pol[i]);
        vs[i] = std::min(std::fabs(search_value[i] - nn_value[i]) / std::max(1.0f, value_scale), 1.0f);
        sumw += base_weight[i];
    }
    if (sumw <= 1e-8f) return out;
    float avg_p = 0.0f, avg_v = 0.0f;
    for (int i = 0; i < n; ++i) { avg_p += ps[i] * base_weight[i]; avg_v += vs[i] * base_weight[i]; }
    avg_p /= sumw; avg_v /= sumw;
    float vw = value_w;
    if (avg_v < 0.01f) vw *= (avg_v / 0.01f);
    const float base_ratio = std::max(0.0f, 1.0f - policy_w - vw);
    const float p_thresh = avg_p * 1.5f;
    std::vector<float> pp(n), vp(n);
    float sum_pp = 1e-10f, sum_vp = 1e-10f;
    for (int i = 0; i < n; ++i) {
        const float w = base_weight[i];
        pp[i] = w * ps[i] + (1.0f - w) * std::max(0.0f, ps[i] - p_thresh);
        vp[i] = w * vs[i];
        sum_pp += pp[i]; sum_vp += vp[i];
    }
    for (int i = 0; i < n; ++i)
        out[i] = base_ratio * base_weight[i]
               + policy_w * pp[i] * sumw / sum_pp
               + vw * vp[i] * sumw / sum_vp;
    return out;
}

// NN inference: encoded state (NUM_PLANES*AREA int8) -> (policy_logits[4], value).
// value is the network's estimate of V(state) (expected discounted future
// reward, RAW points). policy is raw LOGITS over the 4 directions; the search
// legal-masks and softmaxes internally (Gumbel needs the logits).
using Infer2048Fn = std::function<std::pair<std::array<float, 4>, float>(
    const std::vector<int8_t>&)>;

// Running min-max normalizer for Q values (MuZero appendix B). Keeps PUCT's
// Q term in [0,1] regardless of the raw reward scale.
struct MinMaxStats {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    void update(float v) {
        minimum = std::min(minimum, v);
        maximum = std::max(maximum, v);
    }
    float normalize(float v) const {
        if (maximum > minimum) return (v - minimum) / (maximum - minimum);
        return 0.5f;  // not enough spread yet (matches az2048/mcts.py)
    }
};

struct ChanceNode;

struct DecisionNode {
    std::vector<int8_t> state;
    bool terminal = false;
    bool expanded = false;

    std::array<float, 4> prior{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> logits{0.0f, 0.0f, 0.0f, 0.0f};  // masked logits (-inf illegal)
    std::array<std::unique_ptr<ChanceNode>, 4> children;  // null where illegal
    float nn_value = 0.0f;

    int n = 0;
    double w_sum = 0.0;     // sum of returns G observed from this node
    double w_sq_sum = 0.0;  // sum of G^2 (feeds variance-scaled cpuct)
    float value() const { return n > 0 ? static_cast<float>(w_sum / n) : nn_value; }
};

struct SpawnEdge {
    double prob = 0.0;
    int cell = -1;
    int exp = 0;
    std::unique_ptr<DecisionNode> child;  // created lazily
};

struct ChanceNode {
    std::vector<int8_t> afterstate;
    int reward = 0;

    int n = 0;
    double w_sum = 0.0;     // sum of returns G = reward + gamma*G_child
    double w_sq_sum = 0.0;  // sum of G^2 (root-PUCT LCB variance estimate)
    std::vector<SpawnEdge> edges;   // all spawn outcomes (built on expand)
    bool expanded = false;

    float q() const { return n > 0 ? static_cast<float>(w_sum / n) : 0.0f; }
};

class SkyZero2048MCTS {
public:
    SkyZero2048MCTS(Game2048& game, const SkyZero2048Config& cfg,
                    Infer2048Fn infer, uint64_t seed)
        : game_(game), cfg_(cfg), infer_(std::move(infer)), rng_(seed) {}

    struct SearchOutput {
        std::array<float, 4> visit_policy{0, 0, 0, 0};      // normalized visit counts
        std::array<float, 4> improved_policy{0, 0, 0, 0};   // Gumbel completed-Q target
        std::array<int, 4> visit_counts{0, 0, 0, 0};
        float root_value = 0.0f;                            // V(root) estimate
        int best_action = -1;                               // Gumbel-selected action
        std::array<float, 4> nn_policy{0, 0, 0, 0};         // raw NN policy (pre-noise; surprise weighting)
        float nn_value = 0.0f;                              // raw NN value at root (surprise weighting)
    };

    SearchOutput search(const std::vector<int8_t>& state, int sims_override = -1,
                        int turn_number = 0) {
        DecisionNode root;
        root.state = state;
        root.terminal = game_.is_terminal(state);
        MinMaxStats stats;
        const int budget = (sims_override > 0) ? sims_override : cfg_.num_simulations;
        sims_budget_ = budget;                 // sync search: no PCR
        root_explore_ = true;
        turn_number_ = turn_number;

        SearchOutput out;
        if (root.terminal) {
            out.best_action = -1;
            return out;
        }

        expand_decision(root, /*is_root=*/true);
        apply_root_exploration(root);
        backup_decision(&root, root.nn_value, stats);
        setup_root(root);

        for (int s = 0; s < budget && !active_.empty(); ++s) {
            const int a = next_root_action(root, stats);
            if (a < 0) break;
            simulate_from(root, a, stats);
        }

        for (int a = 0; a < 4; ++a) {
            out.visit_counts[a] = root.children[a] ? root.children[a]->n : 0;
        }
        const float total = static_cast<float>(std::max(1, root.n - 1));
        for (int a = 0; a < 4; ++a) out.visit_policy[a] = out.visit_counts[a] / total;
        out.root_value = root.value();
        assemble_output(root, stats, out, /*use_lcb=*/cfg_.lcb_for_selection);
        return out;
    }

    // ===== Deferred-eval stepping API (batched / parallel self-play) =====
    // Drive one game's search without inline inference, so a caller can batch
    // the NN evals of MANY concurrent games into one forward. Per game:
    //   begin(state);
    //   if (!root_terminal()) eval(root_state()) -> apply_root_eval(...)
    //   while (!done()) { auto e = select_leaf(); if (!e.empty()) eval -> apply_leaf(...) }
    //   result();
    // sims_override > 0 caps this search's budget (PCR cheap search); exploration
    // = false disables root noise for this search (PCR cheap search runs clean).
    void begin(const std::vector<int8_t>& state,
               std::unique_ptr<DecisionNode> reuse = nullptr,
               int sims_override = -1, bool exploration = true, int turn_number = 0) {
        stats_ = MinMaxStats();
        sims_done_ = 0;
        sims_budget_ = (sims_override > 0) ? sims_override : cfg_.num_simulations;
        root_explore_ = exploration;
        turn_number_ = turn_number;
        if (reuse && reuse->expanded && !reuse->terminal && reuse->state == state) {
            // Adopt the reused subtree: no NN re-eval / re-expand / re-backup —
            // its visit counts + values carry over as a warm start.
            root_ = std::move(reuse);
            // Re-seed min-max stats from the root level so PUCT normalize isn't
            // cold (deeper-node stats are lost; rebuilt as fresh sims back up).
            stats_.update(root_->value());
            for (auto& c : root_->children) if (c && c->n > 0) stats_.update(c->q());
            apply_root_exploration(*root_);
            setup_root(*root_);
        } else {
            root_ = std::make_unique<DecisionNode>();
            root_->state = state;
            root_->terminal = game_.is_terminal(state);
        }
    }
    bool root_terminal() const { return root_->terminal; }
    // A reused (already-expanded) root needs no NN eval; a fresh one does.
    bool root_needs_eval() const { return !root_->terminal && !root_->expanded; }
    const std::vector<int8_t>& root_state() const { return root_->state; }
    // Encoded root state for the deferred eval, with the stochastic root transform
    // applied (and remembered so apply_root_eval can undo it). Callers submit this
    // instead of encoding root_state() themselves.
    std::vector<int8_t> root_encoded() {
        root_transform_ = pick_transform(/*is_root=*/true);
        return encode_transformed(root_->state, root_transform_);
    }

    void apply_root_eval(const std::array<float, 4>& logits, float value) {
        expand_with(*root_, undo_action_perm(logits, root_transform_), value);
        apply_root_exploration(*root_);
        backup_decision(root_.get(), value, stats_);
        setup_root(*root_);
    }

    bool done() const {
        return root_->terminal || sims_done_ >= sims_budget_ || active_.empty();
    }

    // Advance one simulation. Returns the encoded leaf state to evaluate, or an
    // empty vector if this sim needed no NN call (terminal leaf already backed up).
    std::vector<int8_t> select_leaf() {
        ++sims_done_;
        const int a = next_root_action(*root_, stats_);
        if (a < 0) return {};
        pending_dec_.clear(); pending_chance_.clear(); pending_rewards_.clear();
        pending_dec_.push_back(root_.get());
        ChanceNode* cn = root_->children[a].get();
        pending_chance_.push_back(cn);
        pending_rewards_.push_back(cn->reward);
        DecisionNode* node = descend_chance(*cn);
        while (true) {
            pending_dec_.push_back(node);
            if (node->terminal) {
                backup_path(pending_dec_, pending_chance_, pending_rewards_, 0.0f, stats_);
                return {};
            }
            if (!node->expanded) {
                pending_leaf_ = node;
                pending_leaf_transform_ = pick_transform(/*is_root=*/false);
                return encode_transformed(node->state, pending_leaf_transform_);
            }
            const int a2 = select_nonroot(*node, stats_);
            ChanceNode* c2 = node->children[a2].get();
            pending_chance_.push_back(c2);
            pending_rewards_.push_back(c2->reward);
            node = descend_chance(*c2);
        }
    }

    void apply_leaf(const std::array<float, 4>& logits, float value) {
        expand_with(*pending_leaf_, undo_action_perm(logits, pending_leaf_transform_), value);
        backup_path(pending_dec_, pending_chance_, pending_rewards_, value, stats_);
    }

    SearchOutput result() {
        SearchOutput out;
        if (root_->terminal) { out.best_action = -1; return out; }
        for (int a = 0; a < 4; ++a)
            out.visit_counts[a] = root_->children[a] ? root_->children[a]->n : 0;
        const float total = std::max(1.0f, static_cast<float>(root_->n - 1));
        for (int a = 0; a < 4; ++a) out.visit_policy[a] = out.visit_counts[a] / total;
        out.root_value = root_->value();
        assemble_output(*root_, stats_, out, /*use_lcb=*/false);  // selfplay never uses LCB
        return out;
    }

    // Tree reuse: detach the subtree for (action, realized spawn outcome) so the
    // caller can hand it to the next begin(). Returns null when reuse is off or
    // that outcome was never created in the tree (caller then begins fresh).
    std::unique_ptr<DecisionNode> detach_after_move(int action, int spawn_cell, int spawn_exp) {
        if (!cfg_.enable_tree_reuse || !root_ || action < 0 || action >= 4
            || !root_->children[action]) return nullptr;
        ChanceNode* cn = root_->children[action].get();
        for (auto& e : cn->edges)
            if (e.cell == spawn_cell && e.exp == spawn_exp) return std::move(e.child);
        return nullptr;
    }

private:
    // One simulation: the ROOT action is chosen by the Gumbel scheduler; below
    // the root we descend with PUCT (mirrors az2048/mcts.py). Expand the leaf
    // decision node and back up the discounted return along the path.
    void simulate_from(DecisionNode& root, int root_action, MinMaxStats& stats) {
        std::vector<DecisionNode*> dec_path;
        std::vector<ChanceNode*> chance_path;
        std::vector<int> rewards;

        dec_path.push_back(&root);
        ChanceNode* cn = root.children[root_action].get();
        chance_path.push_back(cn);
        rewards.push_back(cn->reward);
        DecisionNode* node = descend_chance(*cn);

        while (true) {
            dec_path.push_back(node);
            if (node->terminal) {
                backup_path(dec_path, chance_path, rewards, 0.0f, stats);
                return;
            }
            if (!node->expanded) {
                expand_decision(*node, /*is_root=*/false);
                backup_path(dec_path, chance_path, rewards, node->nn_value, stats);
                return;
            }
            const int a = select_nonroot(*node, stats);
            ChanceNode* c2 = node->children[a].get();
            chance_path.push_back(c2);
            rewards.push_back(c2->reward);
            node = descend_chance(*c2);
        }
    }

    // ---- Gumbel root: sequential halving over legal candidates ----
    void setup_gumbel(DecisionNode& root) {
        for (int a = 0; a < 4; ++a) {
            g_[a] = 0.0f;
            if (cfg_.gumbel_noise && root_explore_ && root.children[a]) {
                std::extreme_value_distribution<float> gd(0.0f, 1.0f);
                g_[a] = gd(rng_);
            }
        }
        active_.clear();
        for (int a = 0; a < 4; ++a) if (root.children[a]) active_.push_back(a);
        std::sort(active_.begin(), active_.end(), [&](int x, int y) {
            return root.logits[x] + g_[x] > root.logits[y] + g_[y];
        });
        const int m = static_cast<int>(active_.size());
        const int phases = (m > 1) ? static_cast<int>(std::ceil(std::log2((double)m))) : 1;
        phase_budgets_.assign(phases, 0);
        const int base = sims_budget_ / std::max(1, phases);
        const int rem = sims_budget_ - base * phases;
        for (int i = 0; i < phases; ++i) phase_budgets_[i] = base + (i < rem ? 1 : 0);
        phase_ = 0; in_phase_ = 0; rr_ = 0;
    }

    // PUCT root: no sequential-halving scheduler — per-sim selection picks the
    // root child. active_ just holds the legal set so done()/iteration work.
    void setup_root_puct(DecisionNode& root) {
        active_.clear();
        for (int a = 0; a < 4; ++a) if (root.children[a]) active_.push_back(a);
    }

    void setup_root(DecisionNode& root) {
        if (cfg_.root_puct) setup_root_puct(root); else setup_gumbel(root);
    }

    int next_root_action(DecisionNode& root, const MinMaxStats& stats) {
        if (active_.empty()) return -1;
        if (cfg_.root_puct) return select_action(root, stats, /*is_root=*/true);
        const int m = static_cast<int>(active_.size());
        const int a = active_[rr_ % m];
        ++rr_; ++in_phase_;
        if (phase_ < static_cast<int>(phase_budgets_.size()) - 1
                && in_phase_ >= phase_budgets_[phase_]) {
            std::sort(active_.begin(), active_.end(), [&](int x, int y) {
                return root_score(root, x, stats) > root_score(root, y, stats);
            });
            const int keep = std::max(1, (m + 1) / 2);
            active_.resize(keep);
            ++phase_; in_phase_ = 0; rr_ = 0;
        }
        return a;
    }

    float sigma(const DecisionNode& root, float q, const MinMaxStats& stats) const {
        int max_n = 0;
        for (const auto& c : root.children) if (c) max_n = std::max(max_n, c->n);
        return (cfg_.gumbel_c_visit + static_cast<float>(max_n)) * cfg_.gumbel_c_scale
               * stats.normalize(q);
    }

    float root_score(const DecisionNode& root, int a, const MinMaxStats& stats) const {
        const ChanceNode* c = root.children[a].get();
        const float q = (c && c->n > 0) ? c->q() : root.value();
        return root.logits[a] + g_[a] + sigma(root, q, stats);
    }

    // Final action comes from the sequential-halving SURVIVORS only (paper
    // semantics, matches az2048/mcts.py): an eliminated action's q is frozen
    // at a low-visit estimate and must not re-enter the final argmax.
    int gumbel_best_action(const DecisionNode& root, const MinMaxStats& stats) const {
        int best = -1; float bs = -std::numeric_limits<float>::infinity();
        for (int a : active_) {
            const float s = root_score(root, a, stats);
            if (s > bs) { bs = s; best = a; }
        }
        return best;
    }

    float v_mix(const DecisionNode& root) const {
        int sum_n = 0;
        for (const auto& c : root.children) if (c) sum_n += c->n;
        if (sum_n == 0) return root.nn_value;
        float wq = 0.0f, psum = 1e-12f;
        for (int a = 0; a < 4; ++a) {
            const ChanceNode* c = root.children[a].get();
            if (c && c->n > 0) { wq += root.prior[a] * c->q(); psum += root.prior[a]; }
        }
        wq /= psum;
        return (root.nn_value + sum_n * wq) / (1.0f + sum_n);
    }

    std::array<float, 4> improved_policy(const DecisionNode& root, const MinMaxStats& stats) const {
        const float vm = v_mix(root);
        std::array<float, 4> comp;
        for (int a = 0; a < 4; ++a) {
            const ChanceNode* c = root.children[a].get();
            if (!c) { comp[a] = -std::numeric_limits<float>::infinity(); continue; }
            const float q = (c->n > 0) ? c->q() : vm;
            comp[a] = root.logits[a] + sigma(root, q, stats);
        }
        return softmax4(comp);
    }

    // ---- PUCT root: post-search assembly (vs Gumbel's gumbel_best_action) ----
    void assemble_output(DecisionNode& root, const MinMaxStats& stats,
                         SearchOutput& out, bool use_lcb) {
        out.nn_policy = softmax4(root.logits);   // raw NN policy (pre-noise)
        out.nn_value = root.nn_value;
        if (!cfg_.root_puct) {
            out.best_action = gumbel_best_action(root, stats);
            out.improved_policy = improved_policy(root, stats);
            return;
        }
        puct_root_assemble(root, stats, use_lcb, out);
    }

    // KataGo getPlaySelectionValues + getChosenMoveLoc, adapted to 2048's
    // normalized scalar Q. Best child by goodness; every other child's weight is
    // capped at the PUCT-inverse the best child's value retrospectively justifies
    // (strips forced-playout / Dirichlet excess). The pruned weights are BOTH the
    // training policy target and the temperature-sampling distribution. use_lcb
    // (eval/play only) just re-picks the chosen move by a lower-confidence bound;
    // the target stays the pruned-visit distribution, so selfplay data is intact.
    void puct_root_assemble(DecisionNode& root, const MinMaxStats& stats,
                            bool use_lcb, SearchOutput& out) {
        out.improved_policy = {0.0f, 0.0f, 0.0f, 0.0f};
        struct CS { int a; int n; float qn; float prior; double wsq; double wsum; };
        std::vector<CS> cs;
        for (int a = 0; a < 4; ++a) {
            ChanceNode* cn = root.children[a].get();
            if (cn) cs.push_back({a, cn->n, stats.normalize(cn->q()), root.prior[a],
                                  cn->w_sq_sum, cn->w_sum});
        }
        if (cs.empty()) { out.best_action = -1; return; }

        const float tcw = static_cast<float>(std::max(0, root.n - 1));
        const float c_puct = cfg_.c_puct + cfg_.c_puct_log
            * std::log((tcw + cfg_.c_puct_base) / cfg_.c_puct_base);
        const float explore = c_puct * std::sqrt(tcw + 0.01f) * parent_stdev_factor(root, stats);

        int best_i = 0; float best_g = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < cs.size(); ++i) {
            const float g = std::max(0.0f, static_cast<float>(cs[i].n) - 1.0f) + 2.0f * cs[i].prior;
            if (g > best_g) { best_g = g; best_i = static_cast<int>(i); }
        }
        const float best_value = cs[best_i].qn
            + explore * cs[best_i].prior / (1.0f + static_cast<float>(cs[best_i].n));

        std::vector<float> w(cs.size(), 0.0f);
        for (size_t i = 0; i < cs.size(); ++i) {
            if (cs[i].n <= 0) continue;
            if (static_cast<int>(i) == best_i) { w[i] = static_cast<float>(cs[i].n); continue; }
            const float denom = best_value - cs[i].qn;
            float wanted = static_cast<float>(cs[i].n);
            if (denom > 1e-12f) wanted = explore * cs[i].prior / denom - 1.0f;
            wanted = std::min(wanted, static_cast<float>(cs[i].n));
            w[i] = std::ceil(std::max(0.0f, wanted));
        }
        float w_sum = 0.0f, w_max = 0.0f;
        for (float v : w) { w_sum += v; w_max = std::max(w_max, v); }
        if (w_sum <= 0.0f) {           // zero-budget search: best child gets it all
            out.best_action = cs[best_i].a;
            out.improved_policy[cs[best_i].a] = 1.0f;
            return;
        }
        for (size_t i = 0; i < cs.size(); ++i) out.improved_policy[cs[i].a] = w[i] / w_sum;
        // ^ training target = pure pruned-visit distribution (LCB below only bumps
        //   the chosen move's selection weight, never this target).

        // Full KataGo LCB (eval/play only — selfplay passes use_lcb=false): bump
        // the best lower-confidence-bound child's selection weight by the radius-
        // factor amount (getSelfUtilityLCBAndRadius + the play-selection reweight),
        // worked in 2048's normalized [0,1] utility (range radius = 0.5). Composes
        // with the temperature draw below, exactly as in the mainline.
        if (use_lcb) {
            const float range = stats.maximum - stats.minimum;
            const float z = 4.0f, min_prop = 0.05f, R = 0.5f;   // R = normalized utility range radius
            std::vector<float> lcb(cs.size(), 0.0f), radius(cs.size(), 0.0f);
            float best_lcb = -1e30f; int best_lcb_i = -1;
            for (size_t i = 0; i < cs.size(); ++i) {
                float out_radius = 2.0f * R * z, out_lcb = -out_radius;
                if (cs[i].n > 0) {
                    const float avg = cs[i].qn;                 // normalized mean utility
                    float var_norm = 0.0f;
                    if (range > 0.0f && cs[i].n > 1) {
                        const double mean = cs[i].wsum / cs[i].n;
                        double var_raw = cs[i].wsq / cs[i].n - mean * mean;
                        if (var_raw < 0.0) var_raw = 0.0;
                        var_norm = static_cast<float>(var_raw) / (range * range);
                    }
                    float sq_avg = var_norm + avg * avg;        // normalized 2nd moment
                    float ws = static_cast<float>(cs[i].n), wsq = static_cast<float>(cs[i].n);
                    float ess = ws * ws / wsq;
                    // KataGo low-visit prior: blend in max-variance prior so the
                    // radius behaves at tiny sample sizes.
                    const float pw = ws / (ess * ess * ess);
                    sq_avg = std::max(sq_avg, avg * avg + 1e-8f);
                    sq_avg = (sq_avg * ws + (sq_avg + R * R) * pw) / (ws + pw);
                    ws += pw; wsq += pw * pw; ess = ws * ws / wsq;
                    const float variance = std::max(0.0f, sq_avg - avg * avg);
                    out_radius = std::sqrt(variance / ess) * z;
                    out_lcb = avg - out_radius;
                }
                lcb[i] = out_lcb; radius[i] = out_radius;
                if (w[i] > 0.0f && w[i] >= min_prop * w_max && out_lcb > best_lcb) {
                    best_lcb = out_lcb; best_lcb_i = static_cast<int>(i);
                }
            }
            if (best_lcb_i >= 0) {
                float adjusted = w[best_lcb_i];
                for (size_t i = 0; i < cs.size(); ++i) {
                    if (static_cast<int>(i) == best_lcb_i) continue;
                    const float excess = best_lcb - lcb[i];
                    if (excess < 0.0f) continue;
                    const float rf = (radius[i] + excess) / (radius[i] + 0.20f * excess);
                    const float lbound = rf * rf * w[i];
                    if (lbound > adjusted) adjusted = lbound;
                }
                w[best_lcb_i] = adjusted; w_max = std::max(w_max, adjusted);
            }
        }

        // Chosen move: temperature sampling over the pruned weights, with temp
        // interpolated early->late by turn (argmax when temp ~ 0).
        const float temp = interpolate_early(
            turn_number_, cfg_.chosen_move_temperature_halflife, Game2048::AREA,
            cfg_.chosen_move_temperature_early, cfg_.chosen_move_temperature);
        int chosen_i = best_i;
        if (temp > 1e-4f) {
            std::vector<double> rel(cs.size(), 0.0);
            const double log_max = std::log(static_cast<double>(w_max));
            double rel_sum = 0.0;
            for (size_t i = 0; i < cs.size(); ++i)
                if (w[i] > 0.0f) { rel[i] = std::exp((std::log((double)w[i]) - log_max) / temp); rel_sum += rel[i]; }
            if (rel_sum > 0.0) {
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                double r = u01(rng_) * rel_sum;
                for (size_t i = 0; i < cs.size(); ++i) { r -= rel[i]; if (rel[i] > 0.0 && r <= 0.0) { chosen_i = static_cast<int>(i); break; } }
            }
        } else {
            float bw = -1.0f;
            for (size_t i = 0; i < cs.size(); ++i) if (w[i] > bw) { bw = w[i]; chosen_i = static_cast<int>(i); }
        }
        out.best_action = cs[chosen_i].a;
    }

    static std::array<float, 4> softmax4(const std::array<float, 4>& x) {
        float mx = -std::numeric_limits<float>::infinity();
        for (float v : x) mx = std::max(mx, v);
        std::array<float, 4> e{}; float s = 0.0f;
        for (int a = 0; a < 4; ++a) { e[a] = std::isinf(x[a]) ? 0.0f : std::exp(x[a] - mx); s += e[a]; }
        for (int a = 0; a < 4; ++a) e[a] = (s > 0) ? e[a] / s : 0.0f;
        return e;
    }

    // Variance-scaled cpuct factor (KataGo searchexplorehelpers.cpp:280-297),
    // adapted to 2048's normalized Q-space: the shrinkage runs in raw return
    // units (w_sq_sum accumulates G^2) and the resulting stdev is divided by the
    // MuZero min-max range so it is comparable to cpuct_utility_stdev_prior,
    // which is expressed in normalized [0,1] units. scale=0 -> factor 1.
    float parent_stdev_factor(const DecisionNode& node, const MinMaxStats& stats) const {
        if (cfg_.cpuct_utility_stdev_scale == 0.0f) return 1.0f;
        const float range = stats.maximum - stats.minimum;
        if (!(range > 0.0f) || node.n <= 1) return 1.0f;   // not enough spread/visits
        const float prior_norm = cfg_.cpuct_utility_stdev_prior;
        const float wsum = static_cast<float>(node.n);
        const float pu = node.value();                        // raw parent utility
        float sq_avg = static_cast<float>(node.w_sq_sum / wsum);
        const float pu_sq = pu * pu;
        if (sq_avg < pu_sq) sq_avg = pu_sq;                   // 2nd moment >= mean^2
        const float var_prior_raw = (prior_norm * range) * (prior_norm * range);
        const float pw = cfg_.cpuct_utility_stdev_prior_weight;
        const float num = (pu_sq + var_prior_raw) * pw + sq_avg * wsum;
        const float den = pw + wsum - 1.0f;
        const float shrunk_var_raw = std::max(0.0f, num / den - pu_sq);
        const float stdev_norm = std::sqrt(shrunk_var_raw) / range;
        return 1.0f + cfg_.cpuct_utility_stdev_scale * (stdev_norm / prior_norm - 1.0f);
    }

    // PUCT over the 4 directions (illegal actions skipped). KataGo-style:
    // log-cpuct, variance-scaled cpuct, and FPU reduction layer on top of
    // 2048's reward-aware FPU base; all are no-ops at their default (off) config.
    int select_action(DecisionNode& node, const MinMaxStats& stats, bool is_root = false) {
        const float tcw = static_cast<float>(std::max(0, node.n - 1));  // total child weight
        const float c_puct = cfg_.c_puct + cfg_.c_puct_log
            * std::log((tcw + cfg_.c_puct_base) / cfg_.c_puct_base);
        const float explore = c_puct * std::sqrt(tcw + 0.01f)
            * parent_stdev_factor(node, stats);

        // FPU reduction scales with how much prior mass has already been visited.
        float visited_mass = 0.0f;
        for (int a = 0; a < 4; ++a)
            if (node.children[a] && node.children[a]->n > 0) visited_mass += node.prior[a];
        const float fpu_max = is_root ? cfg_.root_fpu_reduction_max : cfg_.fpu_reduction_max;
        const float fpu_red = fpu_max * std::sqrt(visited_mass);
        const bool forced = is_root && cfg_.root_desired_per_child_visits_coeff > 0.0f;

        float best = -std::numeric_limits<float>::infinity();
        int best_a = -1;
        for (int a = 0; a < 4; ++a) {
            ChanceNode* cn = node.children[a].get();
            if (!cn) continue;
            const int cn_n = cn->n;
            // KataGo forced playouts (root only): an under-visited root child is
            // forced this sim (searchexplorehelpers.cpp:166-169).
            if (forced && node.prior[a] > 0.0f && static_cast<float>(cn_n) < std::sqrt(
                    node.prior[a] * tcw * cfg_.root_desired_per_child_visits_coeff)) {
                return a;
            }
            // Q(a) = r(s,a) + gamma * V(afterstate). Unvisited actions use the
            // reward-aware FPU base minus the (normalized) reduction penalty.
            float q_norm;
            if (cn_n > 0) {
                q_norm = stats.normalize(cn->q());
            } else {
                const float fpu_raw = static_cast<float>(cn->reward) + cfg_.gamma * node.value();
                q_norm = stats.normalize(fpu_raw) - fpu_red;
            }
            const float u = explore * node.prior[a] / (1.0f + cn_n);
            const float score = q_norm + u;
            if (score > best) { best = score; best_a = a; }
        }
        return best_a;
    }

    // Gumbel "Full Gumbel" eq.14 non-root rule (Danihelka et al. 2022):
    // argmax_a [ π'(a) − N(a)/(1+Σ_b N(b)) ], where π' = improved_policy(node)
    // is the completed-Q policy (same construction as the Gumbel root target).
    int select_action_gumbel(DecisionNode& node, const MinMaxStats& stats) {
        const std::array<float, 4> pi = improved_policy(node, stats);
        int sum_n = 0;
        for (int a = 0; a < 4; ++a) if (node.children[a]) sum_n += node.children[a]->n;
        const float inv_total = 1.0f / (1.0f + static_cast<float>(sum_n));
        float best = -std::numeric_limits<float>::infinity();
        int best_a = -1;
        for (int a = 0; a < 4; ++a) {
            const ChanceNode* cn = node.children[a].get();
            if (!cn) continue;
            const float score = pi[a] - static_cast<float>(cn->n) * inv_total;
            if (score > best) { best = score; best_a = a; }
        }
        return best_a;
    }

    // Non-root descent dispatch (config NON_ROOT_SEARCH_ALGO).
    int select_nonroot(DecisionNode& node, const MinMaxStats& stats) {
        return cfg_.non_root_gumbel ? select_action_gumbel(node, stats)
                                    : select_action(node, stats);
    }

    // Descend a chance node by allocating the next visit to the spawn outcome
    // that is currently most under-represented vs its true probability. This
    // deterministically drives the empirical visit distribution toward the
    // known spawn distribution (lower variance than random sampling).
    DecisionNode* descend_chance(ChanceNode& cn) {
        int best = -1;
        float best_deficit = -std::numeric_limits<float>::infinity();
        const float total = static_cast<float>(cn.n);  // visits so far through cn
        for (int i = 0; i < static_cast<int>(cn.edges.size()); ++i) {
            const auto& e = cn.edges[i];
            const int child_n = e.child ? e.child->n : 0;
            const float frac = (total > 0.0f) ? (child_n / total) : 0.0f;
            const float deficit = static_cast<float>(e.prob) - frac;
            if (deficit > best_deficit) { best_deficit = deficit; best = i; }
        }
        SpawnEdge& edge = cn.edges[best];
        if (!edge.child) {
            edge.child = std::make_unique<DecisionNode>();
            edge.child->state = cn.afterstate;
            edge.child->state[edge.cell] = static_cast<int8_t>(edge.exp);
            edge.child->terminal = game_.is_terminal(edge.child->state);
        }
        return edge.child.get();
    }

    // Pick a stochastic D4 transform type (0..7) for one NN eval, or -1 (none).
    // is_root selects which cfg flag gates it.
    int pick_transform(bool is_root) {
        const bool on = is_root ? cfg_.stochastic_transform_root
                                : cfg_.stochastic_transform_child;
        if (!on) return -1;
        std::uniform_int_distribution<int> d(0, 7);
        return d(rng_);
    }
    // encode_state, optionally transformed by `type` (type<=0 => plain).
    std::vector<int8_t> encode_transformed(const std::vector<int8_t>& state, int type) {
        auto enc = game_.encode_state(state);
        if (type <= 0) return enc;     // -1 = off, 0 = identity transform
        return Game2048::transform_encoded(enc, type % 4, type >= 4);
    }
    // Undo the transform on policy logits: orig[a] = logits[ACTION_PERM[type][a]].
    std::array<float, 4> undo_action_perm(const std::array<float, 4>& lg, int type) {
        if (type <= 0) return lg;
        std::array<float, 4> out{};
        for (int a = 0; a < 4; ++a) out[a] = lg[Game2048::ACTION_PERM[type][a]];
        return out;
    }

    void expand_decision(DecisionNode& node, bool is_root) {
        const int t = pick_transform(is_root);
        auto [logits, value] = infer_(encode_transformed(node.state, t));
        expand_with(node, undo_action_perm(logits, t), value);   // back to original action frame
    }

    // Expand using an externally-supplied NN result (no inline inference) —
    // used by the deferred-eval stepping API for batched/parallel self-play.
    void expand_with(DecisionNode& node, const std::array<float, 4>& logits, float value) {
        node.expanded = true;
        node.nn_value = value;

        const auto legal = game_.get_legal_actions(node.state);
        std::array<float, 4> masked;
        for (int a = 0; a < 4; ++a) {
            masked[a] = legal[a] ? logits[a] : -std::numeric_limits<float>::infinity();
        }
        node.logits = masked;
        node.prior = softmax4(masked);
        for (int a = 0; a < 4; ++a) {
            if (legal[a]) {
                auto mr = game_.apply_move(node.state, a);
                auto cn = std::make_unique<ChanceNode>();
                cn->afterstate = std::move(mr.afterstate);
                cn->reward = mr.reward;
                cn->edges = build_edges(cn->afterstate);
                cn->expanded = true;
                node.children[a] = std::move(cn);
            }
        }
    }

    std::vector<SpawnEdge> build_edges(const std::vector<int8_t>& afterstate) {
        auto dist = game_.spawn_distribution(afterstate);
        std::vector<SpawnEdge> edges;
        edges.reserve(dist.size());
        for (const auto& o : dist) {
            SpawnEdge e;
            e.prob = o.prob;
            e.cell = o.cell;
            e.exp = o.exp;
            edges.push_back(std::move(e));
        }
        return edges;
    }

    // Back up the return from a freshly expanded/terminal leaf along the path.
    // leaf_value = V(leaf decision node) (0 for terminal). Walk the chance/
    // decision path back to the root, accumulating G = reward + gamma*G.
    void backup_path(const std::vector<DecisionNode*>& dec_path,
                     const std::vector<ChanceNode*>& chance_path,
                     const std::vector<int>& rewards,
                     float leaf_value, MinMaxStats& stats) {
        float g = leaf_value;
        // dec_path has one more entry than chance_path: dec_path[i] is followed
        // by chance_path[i] (reward rewards[i]) then dec_path[i+1].
        // Update the leaf decision node first.
        DecisionNode* leaf = dec_path.back();
        leaf->n += 1;
        leaf->w_sum += g;
        leaf->w_sq_sum += static_cast<double>(g) * g;
        stats.update(g);

        for (int i = static_cast<int>(chance_path.size()) - 1; i >= 0; --i) {
            g = static_cast<float>(rewards[i]) + cfg_.gamma * g;
            ChanceNode* cn = chance_path[i];
            cn->n += 1;
            cn->w_sum += g;
            cn->w_sq_sum += static_cast<double>(g) * g;
            stats.update(g);

            DecisionNode* dn = dec_path[i];
            dn->n += 1;
            dn->w_sum += g;
            dn->w_sq_sum += static_cast<double>(g) * g;
            stats.update(g);
        }
    }

    // Used only for the initial root expansion backup (no path yet).
    void backup_decision(DecisionNode* node, float value, MinMaxStats& stats) {
        node->n += 1;
        node->w_sum += value;
        node->w_sq_sum += static_cast<double>(value) * value;
        stats.update(value);
    }

    // Root prior exploration (root=puct): rootPolicyTemperature flatten +
    // Dirichlet noise, both rebuilt from the clean NN prior (softmax of the
    // noise-free root.logits) so re-searching a reused root never compounds.
    // No-op for the Gumbel root (which selects on logits, not priors) and when
    // both knobs are off. Temperature is interpolated early->late by turn.
    void apply_root_exploration(DecisionNode& root) {
        const bool want_temp = cfg_.root_puct
            && (cfg_.root_policy_temperature != 1.0f
                || cfg_.root_policy_temperature_early != 1.0f);
        const bool want_noise = root_explore_ && cfg_.root_dirichlet_alpha > 0.0f;
        if (!want_temp && !want_noise) return;

        std::vector<int> legal_idx;
        for (int a = 0; a < 4; ++a) if (root.children[a]) legal_idx.push_back(a);
        if (legal_idx.empty()) return;

        std::array<float, 4> p = softmax4(root.logits);   // clean priors (illegal -> 0)

        if (want_temp) {
            const float temp = interpolate_early(
                turn_number_, cfg_.chosen_move_temperature_halflife, Game2048::AREA,
                cfg_.root_policy_temperature_early, cfg_.root_policy_temperature);
            float max_p = 0.0f;
            for (int a : legal_idx) max_p = std::max(max_p, p[a]);
            if (max_p > 0.0f && temp > 0.0f) {
                const float log_max = std::log(max_p), inv = 1.0f / temp;
                float sum = 0.0f;
                for (int a : legal_idx) {
                    p[a] = (p[a] > 0.0f) ? std::exp((std::log(p[a]) - log_max) * inv) : 0.0f;
                    sum += p[a];
                }
                if (sum > 0.0f) for (int a : legal_idx) p[a] /= sum;
            }
        }

        if (want_noise && legal_idx.size() >= 2) {
            std::gamma_distribution<float> gd(cfg_.root_dirichlet_alpha, 1.0f);
            std::vector<float> noise(legal_idx.size());
            float nsum = 0.0f;
            for (auto& x : noise) { x = gd(rng_); nsum += x; }
            for (auto& x : noise) x /= std::max(1e-8f, nsum);
            for (size_t i = 0; i < legal_idx.size(); ++i) {
                const int a = legal_idx[i];
                p[a] = (1.0f - cfg_.root_noise_frac) * p[a] + cfg_.root_noise_frac * noise[i];
            }
        }
        root.prior = p;
    }

    Game2048& game_;
    SkyZero2048Config cfg_;
    Infer2048Fn infer_;
    std::mt19937 rng_;

    // Gumbel root scheduler state (reset per search()).
    std::array<float, 4> g_{0, 0, 0, 0};
    std::vector<int> active_;
    std::vector<int> phase_budgets_;
    int phase_ = 0, in_phase_ = 0, rr_ = 0;

    // Stepping-API state (deferred-eval / batched self-play).
    std::unique_ptr<DecisionNode> root_;
    MinMaxStats stats_;
    int sims_done_ = 0;
    int sims_budget_ = 0;        // this search's sim cap (PCR cheap search reduces it)
    bool root_explore_ = true;   // root noise on for this search (off for PCR cheap)
    int turn_number_ = 0;        // move index (for early/late temperature interpolation)
    std::vector<DecisionNode*> pending_dec_;
    std::vector<ChanceNode*> pending_chance_;
    std::vector<int> pending_rewards_;
    DecisionNode* pending_leaf_ = nullptr;
    // Stochastic-transform type chosen for the in-flight deferred root / leaf eval
    // (-1 = none), so apply_root_eval / apply_leaf can undo it on the logits.
    int root_transform_ = -1;
    int pending_leaf_transform_ = -1;
};

}  // namespace skyzero

#endif
