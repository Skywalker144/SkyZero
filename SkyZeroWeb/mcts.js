class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];
        this.nnValue = new Float64Array(3);   // WDL output from NN
        this.nnPolicy = null;                  // softmax policy (Float32Array)
        this.nnLogits = null;                  // masked logits (Float32Array)
        this.v = new Float64Array(3);          // cumulative WDL
        this.utilitySqSum = 0;                 // for variance-scaled cPUCT
        this.n = 0;
    }

    isExpanded() {
        return this.children.length > 0;
    }

    update(value) {
        this.v[0] += value[0];
        this.v[1] += value[1];
        this.v[2] += value[2];
        const u = value[2] - value[0];   // L - W (parent perspective utility)
        this.utilitySqSum += u * u;
        this.n += 1;
    }
}

function sampleGumbel() {
    let u = Math.random();
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
            fpu_reduction_max: 0.2,
            root_fpu_reduction_max: 0.1,
            fpu_pow: 1.0,
            fpu_loss_prop: 0.0,
            cpuct_utility_stdev_prior: 0.40,
            cpuct_utility_stdev_prior_weight: 2.0,
            cpuct_utility_stdev_scale: 0.85,
            gumbel_m: 16,
            gumbel_c_visit: 50,
            gumbel_c_scale: 1.0,
        }, args);
    }

    computeParentUtilityStdevFactor(node, parentUtility) {
        const prior = this.args.cpuct_utility_stdev_prior;
        const variancePrior = prior * prior;
        const variancePriorWeight = this.args.cpuct_utility_stdev_prior_weight;

        let parentStdev;
        if (node.n <= 1) {
            parentStdev = prior;
        } else {
            const effectiveN = node.n;
            const utilitySqAvg = node.utilitySqSum / effectiveN;
            const uSq = parentUtility * parentUtility;
            const adjSqAvg = Math.max(utilitySqAvg, uSq);
            parentStdev = Math.sqrt(Math.max(0,
                ((uSq + variancePrior) * variancePriorWeight + adjSqAvg * effectiveN)
                / (variancePriorWeight + effectiveN - 1)
                - uSq
            ));
        }
        return 1.0 + this.args.cpuct_utility_stdev_scale
            * (parentStdev / Math.max(1e-8, prior) - 1.0);
    }

    computeSelectParams(node, effectiveParentN, visitedPolicyMass) {
        const totalChildWeight = Math.max(0, effectiveParentN - 1);

        const cPuct = this.args.c_puct
            + this.args.c_puct_log * Math.log((totalChildWeight + this.args.c_puct_base) / this.args.c_puct_base);

        let parentUtility = 0;
        if (node.n > 0) {
            parentUtility = node.v[0] / node.n - node.v[2] / node.n;
        }

        const stdevFactor = this.computeParentUtilityStdevFactor(node, parentUtility);
        const exploreScaling = cPuct * Math.sqrt(totalChildWeight + 0.01) * stdevFactor;

        const nnUtility = node.nnValue[0] - node.nnValue[2];
        const avgWeight = Math.min(1, Math.pow(visitedPolicyMass, this.args.fpu_pow));
        const parentUtilityForFpu = avgWeight * parentUtility + (1 - avgWeight) * nnUtility;

        const fpuReductionMax = (node.parent === null) ? this.args.root_fpu_reduction_max : this.args.fpu_reduction_max;
        const reduction = fpuReductionMax * Math.sqrt(visitedPolicyMass);
        let fpuValue = parentUtilityForFpu - reduction;
        fpuValue = fpuValue + (-1.0 - fpuValue) * this.args.fpu_loss_prop;
        return { exploreScaling, fpuValue };
    }

    select(node) {
        let visitedPolicyMass = 0;
        for (const child of node.children) {
            if (child.n > 0) visitedPolicyMass += child.prior;
        }
        const sp = this.computeSelectParams(node, node.n, visitedPolicyMass);

        let bestScore = -Infinity;
        let bestChild = null;
        for (const child of node.children) {
            let qValue;
            if (child.n === 0) {
                qValue = sp.fpuValue;
            } else {
                qValue = child.v[2] / child.n - child.v[0] / child.n;
            }
            const uValue = sp.exploreScaling * child.prior / (1 + child.n);
            const score = qValue + uValue;
            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

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

    backpropagate(node, value) {
        let curr = node;
        let w = value[0], d = value[1], l = value[2];
        const buf = new Float64Array(3);
        while (curr !== null) {
            buf[0] = w; buf[1] = d; buf[2] = l;
            curr.update(buf);
            const tmp = w; w = l; l = tmp;
            curr = curr.parent;
        }
    }

    /**
     * Gumbel Sequential Halving (Danihelka et al., 2022).
     * Mirrors V5 cpp/alphazero.h::run_gumbel_search for the eval (no-noise) path.
     *
     * Records per-phase surviving action sets into root._gumbelPhases so the UI
     * can render the 16/8/4/2/1 colored overlay.
     *
     * @param {Node} root - expanded root with nnLogits set
     * @param {number} numSimulations - total simulation budget
     * @param {Function} simulateOne - async (action) => void; runs one sim from a root child
     * @returns {Promise<{improvedPolicy, gumbelAction, vMix}>}
     */
    async gumbelSequentialHalving(root, numSimulations, simulateOne) {
        const boardArea = this.game.boardSize * this.game.boardSize;
        const logits = new Float64Array(root.nnLogits);
        const isLegal = this.game.getLegalActions(root.state, root.toPlay);

        // Eval / no-noise (matches V5 play.cfg GUMBEL_NOISE_ENABLED=0).
        const g = new Float64Array(boardArea);   // all zeros

        // Gumbel-Top-k: pick top-m by logits (no noise = deterministic top-m).
        let m = Math.min(numSimulations, this.args.gumbel_m);
        const scores = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            scores[i] = isLegal[i] ? (logits[i] + g[i]) : -Infinity;
        }
        const indices = Array.from({ length: boardArea }, (_, i) => i);
        indices.sort((a, b) => scores[b] - scores[a]);
        let survivingActions = [];
        for (let i = 0; i < indices.length && survivingActions.length < m; i++) {
            if (isLegal[indices[i]]) survivingActions.push(indices[i]);
        }
        m = survivingActions.length;

        // Record initial surviving set as phase 0 (for UI overlay).
        const phases = [];
        const N = this.game.boardSize;
        const toPhaseRC = (acts) => acts.map(a => [(a / N) | 0, a % N]);
        if (m > 0) phases.push(toPhaseRC(survivingActions));

        if (m > 0) {
            const totalPhases = m > 1 ? Math.ceil(Math.log2(m)) : 1;
            let simsBudget = numSimulations;

            for (let phase = 0; phase < totalPhases; phase++) {
                if (simsBudget <= 0) break;

                const remainingPhases = totalPhases - phase;
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

                if (simsBudget <= 0) break;
                if (phase < totalPhases - 1) {
                    let maxN = 0;
                    for (const child of root.children) {
                        if (child.n > maxN) maxN = child.n;
                    }
                    const cVisit = this.args.gumbel_c_visit;
                    const cScale = this.args.gumbel_c_scale;

                    const evalAction = (a) => {
                        const c = root.children.find(ch => ch.actionTaken === a);
                        let q = 0.5;
                        if (c && c.n > 0) {
                            const childW = c.v[0] / c.n;
                            const childL = c.v[2] / c.n;
                            q = (childL - childW + 1) / 2;
                        }
                        return logits[a] + g[a] + (cVisit + maxN) * cScale * q;
                    };

                    survivingActions.sort((a, b) => evalAction(b) - evalAction(a));
                    survivingActions = survivingActions.slice(0, Math.max(1, Math.floor(survivingActions.length / 2)));
                    phases.push(toPhaseRC(survivingActions));
                }
            }
        }

        // Record final survivor (1) as the last phase if it isn't already.
        if (phases.length > 0
            && JSON.stringify(phases[phases.length - 1]) !== JSON.stringify(toPhaseRC(survivingActions))) {
            phases.push(toPhaseRC(survivingActions));
        }
        root._gumbelPhases = phases;

        // ----- Improved policy + vMix from completed Q -----
        const cVisit = this.args.gumbel_c_visit;
        const cScale = this.args.gumbel_c_scale;

        let maxN = 0;
        for (const child of root.children) {
            if (child.n > maxN) maxN = child.n;
        }

        const qWdl = new Array(boardArea);
        const nValues = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) qWdl[i] = [0, 0, 0];
        for (const c of root.children) {
            if (c.n > 0) {
                qWdl[c.actionTaken] = [c.v[2] / c.n, c.v[1] / c.n, c.v[0] / c.n];
                nValues[c.actionTaken] = c.n;
            }
        }

        let sumN = 0;
        for (let i = 0; i < boardArea; i++) sumN += nValues[i];

        const nnValueWdl = root.nnValue;
        let vMixWdl;
        if (sumN > 0) {
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

        const completedQScalar = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            const wdl = nValues[i] > 0 ? qWdl[i] : [vMixWdl[0], vMixWdl[1], vMixWdl[2]];
            completedQScalar[i] = (wdl[0] - wdl[2] + 1) / 2;
        }

        const sigmaQ = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            sigmaQ[i] = (cVisit + maxN) * cScale * completedQScalar[i];
        }

        const improvedLogits = new Float64Array(boardArea);
        for (let i = 0; i < boardArea; i++) {
            improvedLogits[i] = isLegal[i] ? (logits[i] + sigmaQ[i]) : -Infinity;
        }
        const improvedPolicy = new Float32Array(softmax(improvedLogits));

        // Pick gumbelAction = among most-visited surviving actions, max(logits + sigma_q).
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

    // Visit-count-based policy for UI heatmap (V5 "MCTS Visits (N(s,a)/sum)").
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

if (typeof module !== "undefined" && module.exports) {
    module.exports = { Node, MCTS, softmax, sampleGumbel };
}
