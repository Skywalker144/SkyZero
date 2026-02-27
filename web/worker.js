importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
importScripts('gomoku.js');
importScripts('mcts.js');

// 强制 ONNX Runtime 从 CDN 加载 WASM 资源，避免 404
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

let session = null;
let game = null;
let mcts = null;
let root = null;

const boardSize = 15;
const historyStep = 2;

// --- Symmetry Utilities ---
function rotate90(data, C, H, W) {
    const newData = new Float32Array(data.length);
    const layerSize = H * W;
    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                // Counter-clockwise rotation: (h, w) -> (W-1-w, h)
                newData[c * layerSize + (W - 1 - w) * H + h] = data[c * layerSize + h * W + w];
            }
        }
    }
    return newData;
}

function flip(data, C, H, W) {
    const newData = new Float32Array(data.length);
    const layerSize = H * W;
    for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
                // Horizontal flip: (h, w) -> (h, W-1-w)
                newData[c * layerSize + h * W + (W - 1 - w)] = data[c * layerSize + h * W + w];
            }
        }
    }
    return newData;
}

function applySymmetry(data, C, H, W, doFlip, rot) {
    let res = data;
    for (let i = 0; i < rot; i++) res = rotate90(res, C, H, W);
    if (doFlip) res = flip(res, C, H, W);
    return res;
}

function applyInverseSymmetry(data, C, H, W, doFlip, rot) {
    let res = data;
    if (doFlip) res = flip(res, C, H, W);
    for (let i = 0; i < (4 - rot) % 4; i++) res = rotate90(res, C, H, W);
    return res;
}

async function init() {
    game = new Gomoku(boardSize, historyStep, true);
    mcts = new MCTS(game, {
        c_puct: 1.1,
        fpu_reduction_max: 0.2,
        root_fpu_reduction_max: 0.0,
        full_search_num_simulations: 600
    });
    
    try {
        session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['webgl', 'cpu']
        });
        postMessage({ type: 'ready' });
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
        postMessage({ type: 'error', message: e.message });
    }
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b);
    return scores.map(s => s / sum);
}

async function inference(state, toPlay, mode = 'single') {
    const encoded = game.encodeState(state, toPlay);
    const C = game.numPlanes, H = boardSize, W = boardSize;
    
    let batchSize = 1;
    let symmetries = [{ doFlip: false, rot: 0 }];

    if (mode === 'stochastic') {
        symmetries = [{ doFlip: Math.random() < 0.5, rot: Math.floor(Math.random() * 4) }];
    } else if (mode === 'full') {
        batchSize = 8;
        symmetries = [];
        for (let f of [false, true]) {
            for (let r = 0; r < 4; r++) symmetries.push({ doFlip: f, rot: r });
        }
    }

    const inputData = new Float32Array(batchSize * C * H * W);
    for (let i = 0; i < batchSize; i++) {
        const aug = applySymmetry(encoded, C, H, W, symmetries[i].doFlip, symmetries[i].rot);
        inputData.set(aug, i * C * H * W);
    }

    const input = new ort.Tensor('float32', inputData, [batchSize, C, H, W]);
    const results = await session.run({ input: input });

    const pLogits = results.policy_logits.data;
    const vLogits = results.value_logits.data;
    const ownershipData = results.ownership.data;
    const oppPLogitsData = results.opponent_policy_logits.data;

    // Average value
    const vLogitsAvg = new Float32Array(3).fill(0);
    for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < 3; j++) vLogitsAvg[j] += vLogits[i * 3 + j] / batchSize;
    }
    const vProbs = softmax(Array.from(vLogitsAvg));
    const value = vProbs[0] - vProbs[2];

    // Average policy & ownership
    const avgPLogits = new Float32Array(H * W).fill(0);
    const avgOppPLogits = new Float32Array(H * W).fill(0);
    const avgOwnership = new Float32Array(H * W).fill(0);

    for (let i = 0; i < batchSize; i++) {
        const p = applyInverseSymmetry(pLogits.slice(i * H * W, (i + 1) * H * W), 1, H, W, symmetries[i].doFlip, symmetries[i].rot);
        const oppP = applyInverseSymmetry(oppPLogitsData.slice(i * H * W, (i + 1) * H * W), 1, H, W, symmetries[i].doFlip, symmetries[i].rot);
        const own = applyInverseSymmetry(ownershipData.slice(i * H * W, (i + 1) * H * W), 1, H, W, symmetries[i].doFlip, symmetries[i].rot);
        
        for (let j = 0; j < H * W; j++) {
            avgPLogits[j] += p[j] / batchSize;
            avgOppPLogits[j] += oppP[j] / batchSize;
            avgOwnership[j] += own[j] / batchSize;
        }
    }

    const legalMask = game.getLegalActions(state, toPlay);
    const maskedLogits = Array.from(avgPLogits).map((l, i) => legalMask[i] ? l : -1e9);
    const policy = softmax(maskedLogits);

    return { policy, value, ownership: avgOwnership, oppPLogits: avgOppPLogits };
}

let latestSearchId = 0;

onmessage = async function(e) {
    const data = e.data;
    if (data.type === 'init') {
        await init();
    } else if (data.type === 'reset') {
        latestSearchId++;
        root = new Node(game.getInitialState(), 1);
    } else if (data.type === 'move') {
        latestSearchId++;
        // Tree reuse: apply action to root
        if (root) {
            let found = false;
            for (let child of root.children) {
                if (child.actionTaken === data.action) {
                    root = child;
                    root.parent = null;
                    found = true;
                    break;
                }
            }
            if (!found) {
                root = new Node(data.nextState, data.nextToPlay);
            }
        } else {
            root = new Node(data.nextState, data.nextToPlay);
        }
    } else if (data.type === 'search') {
        const numSimulations = data.simulations || 600;
        const searchId = data.searchId;
        latestSearchId = searchId;
        
        if (!root) {
            root = new Node(data.state, data.toPlay);
        }

        let aborted = false;
        for (let i = 0; i < numSimulations; i++) {
            // Check for abortion
            if (latestSearchId !== searchId) {
                aborted = true;
                break;
            }

            let node = root;
            while (node.isExpanded()) {
                node = mcts.select(node);
            }

            const winner = game.getWinner(node.state);
            let value;
            if (winner !== null) {
                value = winner * node.toPlay;
            } else {
                // Root expansion uses full symmetry, others use stochastic
                const mode = (node === root) ? 'full' : 'stochastic';
                const { policy, value: v } = await inference(node.state, node.toPlay, mode);
                
                // Double check after await
                if (latestSearchId !== searchId) {
                    aborted = true;
                    break;
                }

                mcts.expand(node, policy, v);
                value = v;
            }
            mcts.backpropagate(node, value);

            if (i % 20 === 0) {
                postMessage({ type: 'progress', progress: (i / numSimulations) * 100 });
            }
        }

        if (aborted) return;

        const mctsPolicy = mcts.getMCTSPolicy(root);
        // Final result uses full symmetry for best heatmap/eval
        const { ownership, oppPLogits, value: nnValue } = await inference(root.state, root.toPlay, 'full');

        // Final check
        if (latestSearchId !== searchId) return;

        postMessage({ 
            type: 'result', 
            policy: mctsPolicy, 
            rootValue: root.v / root.n,
            rootToPlay: root.toPlay,
            nnValue: nnValue,
            ownership: ownership,
            oppPLogits: oppPLogits,
            searchId: searchId
        });
    }
};
