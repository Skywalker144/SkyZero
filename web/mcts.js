class Node {
    constructor(state, toPlay, prior = 0, parent = null, actionTaken = null) {
        this.state = state;
        this.toPlay = toPlay;
        this.prior = prior;
        this.parent = parent;
        this.actionTaken = actionTaken;
        this.children = [];
        this.nnValue = 0;
        this.v = 0; // Total value
        this.n = 0; // Visit count
    }

    isExpanded() {
        return this.children.length > 0;
    }

    update(value) {
        this.v += value;
        this.n += 1;
    }
}

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
        }, args);
    }

    select(node) {
        let totalChildWeight = 0;
        let visitedPolicyMass = 0;
        for (let child of node.children) {
            totalChildWeight += child.n;
            if (child.n > 0) visitedPolicyMass += child.prior;
        }

        let c_puct = this.args.c_puct;
        if (totalChildWeight > 0) {
            c_puct += this.args.c_puct_log * Math.log((totalChildWeight + this.args.c_puct_base) / this.args.c_puct_base);
        }
        const exploreScaling = c_puct * Math.sqrt(totalChildWeight + 0.01);

        // FPU
        let parentUtility = node.n > 0 ? node.v / node.n : 0;
        const avgWeight = Math.min(1, Math.pow(visitedPolicyMass, this.args.fpu_pow));
        parentUtility = avgWeight * parentUtility + (1 - avgWeight) * node.nnValue;
        
        const fpuReductionMax = (node.parent === null) ? this.args.root_fpu_reduction_max : this.args.fpu_reduction_max;
        const reduction = fpuReductionMax * Math.sqrt(visitedPolicyMass);
        const fpuValue = parentUtility - reduction;

        let bestScore = -Infinity;
        let bestChild = null;

        for (let child of node.children) {
            let qValue = child.n === 0 ? fpuValue : -child.v / child.n;
            let uValue = exploreScaling * child.prior / (1 + child.n);
            let score = qValue + uValue;

            if (score > bestScore) {
                bestScore = score;
                bestChild = child;
            }
        }
        return bestChild;
    }

    expand(node, nnPolicy, nnValue) {
        node.nnValue = nnValue;
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
        while (curr !== null) {
            curr.update(value);
            value = -value;
            curr = curr.parent;
        }
    }

    // Helper to extract policy from MCTS visit counts
    getMCTSPolicy(root) {
        const policy = new Float32Array(this.game.boardSize * this.game.boardSize).fill(0);
        let sumN = 0;
        for (let child of root.children) {
            policy[child.actionTaken] = child.n;
            sumN += child.n;
        }
        if (sumN > 0) {
            for (let i = 0; i < policy.length; i++) policy[i] /= sumN;
        }
        return policy;
    }
}

if (typeof module !== 'undefined') {
    module.exports = { Node, MCTS };
}
