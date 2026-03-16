// --- Utility ---

function softmaxArr(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > max) max = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        const e = Math.exp(logits[i] - max);
        exps[i] = e;
        sum += e;
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
}

/** Sample from Gumbel(0,1): g = -log(-log(U)), U ~ Uniform(0,1) */
function sampleGumbel(n) {
    const g = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        // Avoid exact 0 or 1
        let u = Math.random();
        while (u <= 0 || u >= 1) u = Math.random();
        g[i] = -Math.log(-Math.log(u));
    }
    return g;
}

// --- Node ---

class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];

        // WDL [win, draw, loss] from current player's perspective
        this.nnValue = new Float64Array(3);  // NN output WDL
        this.nnPolicy = null;                // NN policy (after softmax + legal mask)
        this.nnLogits = null;                // NN raw masked logits (before softmax)

        this.v = new Float64Array(3);        // cumulative WDL sum
        this.n = 0;                          // visit count
    }

    isExpanded() {
        return this.children.length > 0;
    }

    update(value) {
        // value is a 3-element WDL array
        this.v[0] += value[0];
        this.v[1] += value[1];
        this.v[2] += value[2];
        this.n += 1;
    }
}

// --- MCTS (Gumbel AlphaZero) ---

class MCTS {
    constructor(game, args) {
        this.game = game;
        this.args = Object.assign({
            c_puct: 1.1,
            c_puct_log: 0.45,
            c_puct_base: 500,
            fpu_reduction_max: 0.2,
            root_fpu_reduction_max: 0.0,
            fpu_pow: 1.0,
            // Gumbel parameters
            gumbel_m: 16,
            gumbel_c_visit: 50,
            gumbel_c_scale: 1.0,
        }, args);
    }

    /**
     * PUCT selection for non-root nodes (same as original but with WDL).
     * Returns the best child to explore.
     */
    select(node) {
        let visitedPolicyMass = 0;
        for (const child of node.children) {
            if (child.n > 0) visitedPolicyMass += child.prior;
        }

        const totalChildWeight = Math.max(0, node.n - 1);
        const c_puct = this.args.c_puct
            + this.args.c_puct_log * Math.log((totalChildWeight + this.args.c_puct_base) / this.args.c_puct_base);
        const exploreScaling = (c_puct / 2) * Math.sqrt(totalChildWeight + 0.01);

        // FPU - derive scalar Q from WDL: Q = W - L
        const parentQ = node.n > 0
            ? (node.v[0] / node.n - node.v[2] / node.n)
            : 0;
        const nnUtility = node.nnValue[0] - node.nnValue[2];

        const avgWeight = Math.min(1, Math.pow(visitedPolicyMass, this.args.fpu_pow));
        let parentUtility = avgWeight * parentQ + (1 - avgWeight) * nnUtility;

        const fpuReductionMax = (node.parent === null)
            ? this.args.root_fpu_reduction_max
            : this.args.fpu_reduction_max;
        const reduction = (fpuReductionMax / 2) * Math.sqrt(visitedPolicyMass);
        const fpuValue = parentUtility - reduction;

        let bestScore = -Infinity;
        let bestChild = null;

        for (const child of node.children) {
            let qValue;
            if (child.n === 0) {
                qValue = fpuValue;
            } else {
                // Child's WDL is from child's perspective; flip to parent's: parent_Q = child_L - child_W
                qValue = child.v[2] / child.n - child.v[0] / child.n;
            }
            const uValue = exploreScaling * child.prior / (1 + child.n);
            const score = qValue + uValue;

            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

    /**
     * Expand a node using NN outputs.
     * @param {Node} node
     * @param {Float32Array} nnPolicy - policy probabilities (after softmax + legal mask)
     * @param {Float64Array} nnValue - WDL [win, draw, loss]
     * @param {Float32Array|null} nnLogits - raw masked logits (optional, needed for root)
     */
    expand(node, nnPolicy, nnValue, nnLogits = null) {
        node.nnValue[0] = nnValue[0];
        node.nnValue[1] = nnValue[1];
        node.nnValue[2] = nnValue[2];
        node.nnPolicy = nnPolicy;
        if (nnLogits) node.nnLogits = nnLogits;

        const nextToPlay = -node.toPlay;
        for (let action = 0; action < nnPolicy.length; action++) {
            const prob = nnPolicy[action];
            if (prob > 0) {
                const child = new Node(
                    this.game.getNextState(node.state, action, node.toPlay),
                    nextToPlay,
                    prob,
                    node,
                    action
                );
                node.children.push(child);
            }
        }
    }

    /**
     * Backpropagate WDL value. Flips [W,D,L] -> [L,D,W] at each level.
     */
    backpropagate(node, value) {
        let curr = node;
        // Use a mutable copy to avoid modifying the original
        let w = value[0], d = value[1], l = value[2];
        while (curr !== null) {
            curr.update([w, d, l]);
            // Flip perspective: [W,D,L] -> [L,D,W]
            const tmp = w;
            w = l;
            l = tmp;
            curr = curr.parent;
        }
    }

    /**
     * Gumbel Sequential Halving — core Gumbel AlphaZero search algorithm.
     *
     * @param {Node} root - Already expanded root node
     * @param {number} numSimulations - Total simulation budget
     * @param {Function} inferFn - async (state, toPlay) => { policy, value, logits }
     * @param {Function|null} progressFn - optional (simsUsed, totalBudget) => void
     * @returns {Promise<{improvedPolicy: Float32Array, gumbelAction: number, vMix: Float64Array}>}
     */
    async gumbelSequentialHalving(root, numSimulations, inferFn, progressFn = null) {
        const boardArea = this.game.boardSize * this.game.boardSize;
        const logits = new Float64Array(root.nnLogits);  // copy
        const legalMask = this.game.getLegalActions(root.state, root.toPlay);

        // --- 1. Gumbel-Top-k sampling (eval mode: no noise) ---
        const g = new Float64Array(boardArea);  // zeros — eval mode, no Gumbel noise

        const scores = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            scores[i] = legalMask[i] ? (logits[i] + g[i]) : -Infinity;
        }

        // Select top-m actions
        const m0 = Math.min(numSimulations, this.args.gumbel_m);
        const indices = Array.from({ length: boardArea }, (_, i) => i);
        indices.sort((a, b) => scores[b] - scores[a]);
        let survivingActions = [];
        for (let i = 0; i < indices.length && survivingActions.length < m0; i++) {
            if (legalMask[indices[i]]) survivingActions.push(indices[i]);
        }
        let m = survivingActions.length;

        // --- 2. Sequential Halving ---
        if (m > 0) {
            const phases = m > 1 ? Math.ceil(Math.log2(m)) : 1;
            let simsBudget = numSimulations;
            let totalSimsUsed = 0;

            for (let phase = 0; phase < phases; phase++) {
                if (simsBudget <= 0) break;

                const remainingPhases = phases - phase;
                const simsThisPhase = Math.floor(simsBudget / remainingPhases);
                const numActions = survivingActions.length;
                const simsPerAction = Math.max(1, Math.floor(simsThisPhase / numActions));

                for (let s = 0; s < simsPerAction; s++) {
                    if (simsBudget <= 0) break;
                    for (const action of survivingActions) {
                        if (simsBudget <= 0) break;

                        // Find the child for this action
                        const child = root.children.find(c => c.actionTaken === action);
                        if (!child) continue;

                        // Select down from child
                        let node = child;
                        while (node.isExpanded()) {
                            node = this.select(node);
                        }

                        let value;
                        const winner = this.game.getWinner(node.state);
                        if (winner !== null) {
                            // Terminal: WDL one-hot
                            const result = winner * node.toPlay;
                            if (result === 1) {
                                value = [1.0, 0.0, 0.0];
                            } else if (result === -1) {
                                value = [0.0, 0.0, 1.0];
                            } else {
                                value = [0.0, 1.0, 0.0];
                            }
                        } else {
                            // Expand leaf with NN inference
                            const inf = await inferFn(node.state, node.toPlay);
                            this.expand(node, inf.policy, inf.value, inf.logits);
                            value = [inf.value[0], inf.value[1], inf.value[2]];
                        }

                        this.backpropagate(node, value);
                        simsBudget -= 1;
                        totalSimsUsed += 1;

                        if (progressFn) progressFn(totalSimsUsed, numSimulations);
                    }
                }

                // Halve surviving actions (except last phase)
                if (simsBudget <= 0) break;
                if (phase < phases - 1) {
                    const maxN = Math.max(...root.children.map(c => c.n), 0);
                    const c_visit = this.args.gumbel_c_visit;
                    const c_scale = this.args.gumbel_c_scale;

                    const evalAction = (a) => {
                        const c = root.children.find(ch => ch.actionTaken === a);
                        let q = 0.5;  // neutral
                        if (c && c.n > 0) {
                            // Parent's Q from child's WDL: parent_W = child_L, parent_L = child_W
                            const parentW = c.v[2] / c.n;
                            const parentL = c.v[0] / c.n;
                            q = ((parentW - parentL) + 1) / 2;  // normalize to [0, 1]
                        }
                        return logits[a] + g[a] + (c_visit + maxN) * c_scale * q;
                    };

                    survivingActions.sort((a, b) => evalAction(b) - evalAction(a));
                    survivingActions = survivingActions.slice(0, Math.max(1, Math.floor(survivingActions.length / 2)));
                }
            }
        }

        // --- 3. Compute improved policy via completed Q-values ---
        const c_visit = this.args.gumbel_c_visit;
        const c_scale = this.args.gumbel_c_scale;
        const maxN = root.children.length > 0
            ? Math.max(...root.children.map(c => c.n))
            : 0;

        // Gather per-action WDL Q-values and visit counts (parent's perspective)
        const qWdl = new Array(boardArea);
        const nValues = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) qWdl[i] = [0, 0, 0];

        for (const c of root.children) {
            if (c.n > 0) {
                const a = c.actionTaken;
                // Flip to parent perspective: [child_L, child_D, child_W]
                qWdl[a] = [c.v[2] / c.n, c.v[1] / c.n, c.v[0] / c.n];
                nValues[a] = c.n;
            }
        }

        let sumN = 0;
        for (let i = 0; i < boardArea; i++) sumN += nValues[i];

        const nnValueWdl = root.nnValue;  // [W, D, L]

        // v_mix: blend NN value with policy-weighted search Q-values
        let vMixWdl;
        if (sumN > 0) {
            const weightedQ = [0, 0, 0];
            let pVisitedSum = 1e-12;
            for (let i = 0; i < boardArea; i++) {
                if (nValues[i] > 0 && root.nnPolicy) {
                    const pw = root.nnPolicy[i];
                    pVisitedSum += pw;
                    weightedQ[0] += pw * qWdl[i][0];
                    weightedQ[1] += pw * qWdl[i][1];
                    weightedQ[2] += pw * qWdl[i][2];
                }
            }
            weightedQ[0] /= pVisitedSum;
            weightedQ[1] /= pVisitedSum;
            weightedQ[2] /= pVisitedSum;

            vMixWdl = new Float64Array([
                (nnValueWdl[0] + sumN * weightedQ[0]) / (1 + sumN),
                (nnValueWdl[1] + sumN * weightedQ[1]) / (1 + sumN),
                (nnValueWdl[2] + sumN * weightedQ[2]) / (1 + sumN),
            ]);
        } else {
            vMixWdl = new Float64Array([nnValueWdl[0], nnValueWdl[1], nnValueWdl[2]]);
        }

        // completed_q: actual Q for visited, v_mix for unvisited
        const completedQScalar = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            let w, l;
            if (nValues[i] > 0) {
                w = qWdl[i][0];
                l = qWdl[i][2];
            } else {
                w = vMixWdl[0];
                l = vMixWdl[2];
            }
            completedQScalar[i] = ((w - l) + 1) / 2;  // normalize to [0, 1]
        }

        const sigmaQ = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            sigmaQ[i] = (c_visit + maxN) * c_scale * completedQScalar[i];
        }

        // improved_policy = softmax(logits + sigma_q), masked for legal moves
        const improvedLogits = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            improvedLogits[i] = legalMask[i] ? (logits[i] + sigmaQ[i]) : -1e9;
        }
        const improvedPolicy = softmaxArr(Array.from(improvedLogits));

        // --- 4. Final action selection ---
        // Among surviving actions, pick the most-visited; break ties with g(a) + logits(a) + sigma_q(a)
        let maxNSurv = 0;
        for (const a of survivingActions) {
            if (nValues[a] > maxNSurv) maxNSurv = nValues[a];
        }
        const mostVisited = survivingActions.filter(a => nValues[a] === maxNSurv);

        let gumbelAction = mostVisited[0] || 0;
        let bestTiebreak = -Infinity;
        for (const a of mostVisited) {
            const tb = g[a] + logits[a] + sigmaQ[a];
            if (tb > bestTiebreak) {
                bestTiebreak = tb;
                gumbelAction = a;
            }
        }

        return { improvedPolicy, gumbelAction, vMix: vMixWdl };
    }

    /**
     * Legacy helper: extract visit-count policy from root (kept for compatibility).
     */
    getMCTSPolicy(root) {
        const boardArea = this.game.boardSize * this.game.boardSize;
        const policy = new Float32Array(boardArea).fill(0);
        let sumN = 0;
        for (const child of root.children) {
            policy[child.actionTaken] = child.n;
            sumN += child.n;
        }
        if (sumN > 0) {
            for (let i = 0; i < policy.length; i++) policy[i] /= sumN;
        }
        return policy;
    }
}

if (typeof module !== "undefined") {
    module.exports = { Node, MCTS };
}
