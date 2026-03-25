class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];

        // WDL value representation: [win, draw, loss]
        this.nnValue = new Float64Array(3);   // NN output WDL for this node
        this.nnPolicy = null;                  // NN softmax policy (Float32Array)
        this.nnLogits = null;                  // NN masked logits (Float32Array)
        this.v = new Float64Array(3);          // Cumulative WDL sum
        this.n = 0;                            // Visit count
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

// Gumbel(0,1) random sample
function sampleGumbel() {
    // g = -log(-log(U)), U ~ Uniform(0,1)
    let u = Math.random();
    // Clamp to avoid log(0)
    u = Math.max(1e-20, Math.min(1 - 1e-10, u));
    return -Math.log(-Math.log(u));
}

function softmax(logits) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    const scores = new Float64Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        scores[i] = Math.exp(logits[i] - maxLogit);
        sum += scores[i];
    }
    for (let i = 0; i < scores.length; i++) {
        scores[i] /= sum;
    }
    return scores;
}

class MCTS {
    constructor(game, args) {
        this.game = game;
        this.args = Object.assign({
            c_puct: 1.1,
            c_puct_log: 0.45,
            c_puct_base: 500,
            fpu_reduction_max: 0.08,
            root_fpu_reduction_max: 0.08,
            fpu_pow: 1.0,
            fpu_loss_prop: 0.0,
            gumbel_m: 16,
            gumbel_c_visit: 50,
            gumbel_c_scale: 1.0,
        }, args);
    }

    /**
     * PUCT selection for non-root nodes (matches Python alphazero.py select()).
     * Returns the child with the highest PUCT score.
     */
    select(node) {
        let visitedPolicyMass = 0;
        for (const child of node.children) {
            if (child.n > 0) visitedPolicyMass += child.prior;
        }

        const totalChildWeight = Math.max(0, node.n - 1);

        let c_puct = this.args.c_puct;
        c_puct += this.args.c_puct_log * Math.log((totalChildWeight + this.args.c_puct_base) / this.args.c_puct_base);

        const exploreScaling = c_puct * Math.sqrt(totalChildWeight + 0.01);

        // FPU - derive scalar Q from WDL: Q = W - L
        let parentQ;
        if (node.n > 0) {
            parentQ = node.v[0] / node.n - node.v[2] / node.n;  // (W - L) scalar
        } else {
            parentQ = 0;
        }
        const nnUtility = node.nnValue[0] - node.nnValue[2];  // W - L scalar

        const fpuPow = this.args.fpu_pow;
        const avgWeight = Math.min(1, Math.pow(visitedPolicyMass, fpuPow));
        let parentUtility = avgWeight * parentQ + (1 - avgWeight) * nnUtility;

        const fpuReductionMax = (node.parent === null) ? this.args.root_fpu_reduction_max : this.args.fpu_reduction_max;
        const reduction = fpuReductionMax * Math.sqrt(visitedPolicyMass);
        let fpuValue = parentUtility - reduction;

        const fpuLossProp = this.args.fpu_loss_prop;
        const lossValue = 0.0;
        fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;

        let bestScore = -Infinity;
        let bestChild = null;

        for (const child of node.children) {
            let qValue;
            if (child.n === 0) {
                qValue = fpuValue;
            } else {
                // Child's WDL is from child's perspective; parent's Q = child's L - child's W
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
     * Expand a node using NN policy and value.
     * nnPolicy: Float32Array of softmax probabilities (boardSize^2)
     * nnValue: Float64Array WDL [win, draw, loss]
     * nnLogits: Float32Array of masked logits (boardSize^2)
     */
    expand(node, nnPolicy, nnValue, nnLogits) {
        node.nnValue = new Float64Array(nnValue);
        node.nnPolicy = nnPolicy;
        node.nnLogits = nnLogits;

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
     * Backpropagate WDL value up the tree.
     * Flips [W,D,L] -> [L,D,W] at each level to switch perspective.
     */
    backpropagate(node, value) {
        let curr = node;
        // Work with a copy so we don't mutate the original
        let w = value[0], d = value[1], l = value[2];
        while (curr !== null) {
            curr.v[0] += w;
            curr.v[1] += d;
            curr.v[2] += l;
            curr.n += 1;
            // Flip perspective: [W,D,L] -> [L,D,W]
            const tmp = w;
            w = l;
            l = tmp;
            curr = curr.parent;
        }
    }

    /**
     * Gumbel Sequential Halving search at the root.
     *
     * This is the core of GumbelAlphaZero: instead of using Dirichlet noise + PUCT at root,
     * it uses Gumbel-Top-k sampling to select m candidate actions, then uses sequential
     * halving to allocate simulation budget across phases, eliminating half the candidates
     * each phase. Finally computes improved policy from completed Q-values.
     *
     * @param {Node} root - The expanded root node (must already be expanded with nnLogits set)
     * @param {number} numSimulations - Total simulation budget
     * @param {boolean} isEval - If true, no Gumbel noise (deterministic)
     * @param {Function} simulateOne - async function(action) that runs one simulation from a root child
     * @returns {Promise<{improvedPolicy, gumbelAction, vMix}>}
     */
    async gumbelSequentialHalving(root, numSimulations, isEval, simulateOne) {
        const boardArea = this.game.boardSize * this.game.boardSize;
        const logits = new Float64Array(root.nnLogits);
        const isLegal = this.game.getLegalActions(root.state, root.toPlay);

        // Sample Gumbel noise (or zeros for eval)
        const g = new Float64Array(boardArea);
        if (!isEval) {
            for (let i = 0; i < boardArea; i++) {
                g[i] = sampleGumbel();
            }
        }

        // Gumbel-Top-k: select m candidate actions with highest (logits + g)
        let m = Math.min(numSimulations, this.args.gumbel_m);
        const scores = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            scores[i] = isLegal[i] ? (logits[i] + g[i]) : -Infinity;
        }

        // Argsort descending, take top m
        const indices = Array.from({ length: boardArea }, (_, i) => i);
        indices.sort((a, b) => scores[b] - scores[a]);
        let survivingActions = [];
        for (let i = 0; i < indices.length && survivingActions.length < m; i++) {
            if (isLegal[indices[i]]) {
                survivingActions.push(indices[i]);
            }
        }
        m = survivingActions.length;

        if (m > 0) {
            const phases = m > 1 ? Math.ceil(Math.log2(m)) : 1;
            let simsBudget = numSimulations;

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
                        await simulateOne(action);
                        simsBudget--;
                    }
                }

                // Eliminate bottom half based on score
                if (simsBudget <= 0) break;
                if (phase < phases - 1) {
                    let maxN = 0;
                    for (const child of root.children) {
                        if (child.n > maxN) maxN = child.n;
                    }
                    const cVisit = this.args.gumbel_c_visit;
                    const cScale = this.args.gumbel_c_scale;

                    const evalAction = (a) => {
                        const c = root.children.find(ch => ch.actionTaken === a);
                        let q = 0.5; // neutral value in [0, 1]
                        if (c && c.n > 0) {
                            // Parent's Q = -(child's W - child's L) = child's L - child's W
                            const childW = c.v[0] / c.n;
                            const childL = c.v[2] / c.n;
                            q = (childL - childW + 1) / 2;  // normalize [-1,1] -> [0,1]
                        }
                        return logits[a] + g[a] + (cVisit + maxN) * cScale * q;
                    };

                    survivingActions.sort((a, b) => evalAction(b) - evalAction(a));
                    survivingActions = survivingActions.slice(0, Math.max(1, Math.floor(survivingActions.length / 2)));
                }
            }
        }

        // Compute improved policy from completed Q-values
        const cVisit = this.args.gumbel_c_visit;
        const cScale = this.args.gumbel_c_scale;

        let maxN = 0;
        for (const child of root.children) {
            if (child.n > maxN) maxN = child.n;
        }

        // Collect WDL Q-values and visit counts per action
        const qWdl = new Array(boardArea);
        const nValues = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            qWdl[i] = [0, 0, 0];
        }
        for (const c of root.children) {
            if (c.n > 0) {
                // Flip child's WDL [W,D,L] to parent's perspective [L,D,W]
                qWdl[c.actionTaken] = [c.v[2] / c.n, c.v[1] / c.n, c.v[0] / c.n];
                nValues[c.actionTaken] = c.n;
            }
        }

        let sumN = 0;
        for (let i = 0; i < boardArea; i++) sumN += nValues[i];

        const nnValueWdl = root.nnValue; // WDL [win, draw, loss]
        let vMixWdl;

        if (sumN > 0) {
            // Policy-weighted average of visited Q-values (WDL)
            let policyVisitedSum = 0;
            const weightedQ = [0, 0, 0];
            for (let i = 0; i < boardArea; i++) {
                if (nValues[i] > 0 && root.nnPolicy) {
                    const pw = root.nnPolicy[i];
                    policyVisitedSum += pw;
                    weightedQ[0] += pw * qWdl[i][0];
                    weightedQ[1] += pw * qWdl[i][1];
                    weightedQ[2] += pw * qWdl[i][2];
                }
            }
            policyVisitedSum = Math.max(policyVisitedSum, 1e-12);
            weightedQ[0] /= policyVisitedSum;
            weightedQ[1] /= policyVisitedSum;
            weightedQ[2] /= policyVisitedSum;

            vMixWdl = new Float64Array([
                (nnValueWdl[0] + sumN * weightedQ[0]) / (1 + sumN),
                (nnValueWdl[1] + sumN * weightedQ[1]) / (1 + sumN),
                (nnValueWdl[2] + sumN * weightedQ[2]) / (1 + sumN),
            ]);
        } else {
            vMixWdl = new Float64Array(nnValueWdl);
        }

        // Completed Q: use actual Q for visited actions, v_mix for unvisited
        const completedQScalar = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            let wdl;
            if (nValues[i] > 0) {
                wdl = qWdl[i];
            } else {
                wdl = [vMixWdl[0], vMixWdl[1], vMixWdl[2]];
            }
            // Scalar: W - L, normalized to [0, 1]
            completedQScalar[i] = (wdl[0] - wdl[2] + 1) / 2;
        }

        const sigmaQ = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            sigmaQ[i] = (cVisit + maxN) * cScale * completedQScalar[i];
        }

        // Improved logits = logits + sigma_q
        const improvedLogits = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            improvedLogits[i] = isLegal[i] ? (logits[i] + sigmaQ[i]) : -Infinity;
        }

        const improvedPolicy = new Float32Array(softmax(improvedLogits));

        // Gumbel action: among most-visited surviving actions, pick the one
        // maximizing logits[a] + g[a] + sigma_q[a]
        let maxNSurviving = 0;
        for (const a of survivingActions) {
            if (nValues[a] > maxNSurviving) maxNSurviving = nValues[a];
        }
        const mostVisited = survivingActions.filter(a => nValues[a] === maxNSurviving);

        let gumbelAction = mostVisited[0] || 0;
        let bestFinalScore = -Infinity;
        for (const a of mostVisited) {
            const s = logits[a] + g[a] + sigmaQ[a];
            if (s > bestFinalScore) {
                bestFinalScore = s;
                gumbelAction = a;
            }
        }

        return { improvedPolicy, gumbelAction, vMix: vMixWdl };
    }

    /**
     * Extract visit-count-based policy from root children (legacy, kept for reference).
     */
    getMCTSPolicy(root) {
        const policy = new Float32Array(this.game.boardSize * this.game.boardSize).fill(0);
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
    module.exports = { Node, MCTS, softmax, sampleGumbel };
}
