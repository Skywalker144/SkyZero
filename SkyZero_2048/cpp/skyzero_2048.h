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
    double w_sum = 0.0;   // sum of returns G observed from this node
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
    double w_sum = 0.0;   // sum of returns G = reward + gamma*G_child
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
    };

    SearchOutput search(const std::vector<int8_t>& state) {
        DecisionNode root;
        root.state = state;
        root.terminal = game_.is_terminal(state);
        MinMaxStats stats;

        SearchOutput out;
        if (root.terminal) {
            out.best_action = -1;
            return out;
        }

        expand_decision(root);
        add_root_noise(root);
        backup_decision(&root, root.nn_value, stats);
        setup_gumbel(root);

        for (int s = 0; s < cfg_.num_simulations && !active_.empty(); ++s) {
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
        out.best_action = gumbel_best_action(root, stats);
        out.improved_policy = improved_policy(root, stats);
        return out;
    }

    // ===== Deferred-eval stepping API (batched / parallel self-play) =====
    // Drive one game's search without inline inference, so a caller can batch
    // the NN evals of MANY concurrent games into one forward. Per game:
    //   begin(state);
    //   if (!root_terminal()) eval(root_state()) -> apply_root_eval(...)
    //   while (!done()) { auto e = select_leaf(); if (!e.empty()) eval -> apply_leaf(...) }
    //   result();
    void begin(const std::vector<int8_t>& state) {
        root_ = std::make_unique<DecisionNode>();
        root_->state = state;
        root_->terminal = game_.is_terminal(state);
        stats_ = MinMaxStats();
        sims_done_ = 0;
    }
    bool root_terminal() const { return root_->terminal; }
    const std::vector<int8_t>& root_state() const { return root_->state; }

    void apply_root_eval(const std::array<float, 4>& logits, float value) {
        expand_with(*root_, logits, value);
        add_root_noise(*root_);
        backup_decision(root_.get(), value, stats_);
        setup_gumbel(*root_);
    }

    bool done() const {
        return root_->terminal || sims_done_ >= cfg_.num_simulations || active_.empty();
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
                return game_.encode_state(node->state);
            }
            const int a2 = select_action(*node, stats_);
            ChanceNode* c2 = node->children[a2].get();
            pending_chance_.push_back(c2);
            pending_rewards_.push_back(c2->reward);
            node = descend_chance(*c2);
        }
    }

    void apply_leaf(const std::array<float, 4>& logits, float value) {
        expand_with(*pending_leaf_, logits, value);
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
        out.best_action = gumbel_best_action(*root_, stats_);
        out.improved_policy = improved_policy(*root_, stats_);
        return out;
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
                expand_decision(*node);
                backup_path(dec_path, chance_path, rewards, node->nn_value, stats);
                return;
            }
            const int a = select_action(*node, stats);
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
            if (cfg_.gumbel_noise && root.children[a]) {
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
        const int base = cfg_.num_simulations / std::max(1, phases);
        const int rem = cfg_.num_simulations - base * phases;
        for (int i = 0; i < phases; ++i) phase_budgets_[i] = base + (i < rem ? 1 : 0);
        phase_ = 0; in_phase_ = 0; rr_ = 0;
    }

    int next_root_action(DecisionNode& root, const MinMaxStats& stats) {
        if (active_.empty()) return -1;
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

    int gumbel_best_action(const DecisionNode& root, const MinMaxStats& stats) const {
        int best = -1; float bs = -std::numeric_limits<float>::infinity();
        for (int a = 0; a < 4; ++a) {
            if (!root.children[a]) continue;
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

    static std::array<float, 4> softmax4(const std::array<float, 4>& x) {
        float mx = -std::numeric_limits<float>::infinity();
        for (float v : x) mx = std::max(mx, v);
        std::array<float, 4> e{}; float s = 0.0f;
        for (int a = 0; a < 4; ++a) { e[a] = std::isinf(x[a]) ? 0.0f : std::exp(x[a] - mx); s += e[a]; }
        for (int a = 0; a < 4; ++a) e[a] = (s > 0) ? e[a] / s : 0.0f;
        return e;
    }

    // PUCT over the 4 directions (illegal actions skipped).
    int select_action(DecisionNode& node, const MinMaxStats& stats) {
        const float sqrt_parent = std::sqrt(static_cast<float>(std::max(1, node.n)));
        float best = -std::numeric_limits<float>::infinity();
        int best_a = -1;
        for (int a = 0; a < 4; ++a) {
            ChanceNode* cn = node.children[a].get();
            if (!cn) continue;
            const int cn_n = cn->n;
            // Q(a) = r(s,a) + gamma * V(afterstate); use FPU = parent value for
            // unvisited actions (optimistic-ish, normalized later).
            float q;
            if (cn_n > 0) {
                q = cn->q();
            } else {
                q = static_cast<float>(cn->reward) + cfg_.gamma * node.value();
            }
            const float q_norm = stats.normalize(q);
            const float u = cfg_.c_puct * node.prior[a] * sqrt_parent / (1.0f + cn_n);
            const float score = q_norm + u;
            if (score > best) { best = score; best_a = a; }
        }
        return best_a;
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

    void expand_decision(DecisionNode& node) {
        auto enc = game_.encode_state(node.state);
        auto [logits, value] = infer_(enc);   // logits over 4 dirs, raw value
        expand_with(node, logits, value);
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
        stats.update(g);

        for (int i = static_cast<int>(chance_path.size()) - 1; i >= 0; --i) {
            g = static_cast<float>(rewards[i]) + cfg_.gamma * g;
            ChanceNode* cn = chance_path[i];
            cn->n += 1;
            cn->w_sum += g;
            stats.update(g);

            DecisionNode* dn = dec_path[i];
            dn->n += 1;
            dn->w_sum += g;
            stats.update(g);
        }
    }

    // Used only for the initial root expansion backup (no path yet).
    void backup_decision(DecisionNode* node, float value, MinMaxStats& stats) {
        node->n += 1;
        node->w_sum += value;
        stats.update(value);
    }

    void add_root_noise(DecisionNode& root) {
        if (cfg_.root_dirichlet_alpha <= 0.0f) return;
        std::vector<int> legal_idx;
        for (int a = 0; a < 4; ++a) if (root.children[a]) legal_idx.push_back(a);
        if (legal_idx.size() < 2) return;
        std::gamma_distribution<float> gd(cfg_.root_dirichlet_alpha, 1.0f);
        std::vector<float> noise(legal_idx.size());
        float nsum = 0.0f;
        for (auto& x : noise) { x = gd(rng_); nsum += x; }
        for (auto& x : noise) x /= std::max(1e-8f, nsum);
        for (size_t i = 0; i < legal_idx.size(); ++i) {
            const int a = legal_idx[i];
            root.prior[a] = (1.0f - cfg_.root_noise_frac) * root.prior[a]
                          + cfg_.root_noise_frac * noise[i];
        }
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
    std::vector<DecisionNode*> pending_dec_;
    std::vector<ChanceNode*> pending_chance_;
    std::vector<int> pending_rewards_;
    DecisionNode* pending_leaf_ = nullptr;
};

}  // namespace skyzero

#endif
