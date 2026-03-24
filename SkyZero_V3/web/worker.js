importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js");
importScripts("gomoku.js");
importScripts("mcts.js");

// v1.17 uses no dynamic import(), so importScripts works reliably in classic Workers.
// Load WASM binary from the jsdelivr CDN to avoid Cloudflare Pages file size limits.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
ort.env.wasm.numThreads = 1; // 禁用多线程以避免 Worker 跨域加载失败导致的 TypeError

let session = null;
let game = null;
let mcts = null;
let root = null;

const boardSize = 15;

function concatChunks(chunks, total) {
    const size = total || chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}

async function fetchModelWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const total = Number(response.headers.get("Content-Length")) || 0;
    if (!response.body) {
        const buffer = await response.arrayBuffer();
        postMessage({ type: "model-progress", percent: 100, loaded: buffer.byteLength, total: buffer.byteLength });
        return new Uint8Array(buffer);
    }

    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    if (total > 0) {
        postMessage({ type: "model-progress", percent: 0, loaded: 0, total });
    }

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        const percent = total > 0 ? (loaded / total) * 100 : null;
        postMessage({ type: "model-progress", percent, loaded, total: total || null });
    }
    postMessage({ type: "model-progress", percent: 100, loaded, total: total || loaded });
    return concatChunks(chunks, total || loaded);
}

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

function mctsSoftmax(logits) {
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    const scores = new Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        scores[i] = Math.exp(logits[i] - maxLogit);
        sum += scores[i];
    }
    return scores.map(s => s / sum);
}

async function init() {
    game = new Gomoku(boardSize, true);
    mcts = new MCTS(game, {
        c_puct: 1.1,
        c_puct_log: 0.45,
        c_puct_base: 500,
        fpu_reduction_max: 0.2,
        root_fpu_reduction_max: 0.0,
        fpu_pow: 1.0,
        fpu_loss_prop: 0.0,
        gumbel_m: 4,
        gumbel_c_visit: 50,
        gumbel_c_scale: 1.0,
    });
    
    try {
        const executionProviders = ["wasm"];

        const modelBytes = await fetchModelWithProgress("model.onnx");
        session = await ort.InferenceSession.create(modelBytes, { executionProviders });
        postMessage({ type: "ready" });
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        postMessage({ type: "error", message: "模型加载失败: " + (e.message || String(e)) });
    }
}

/**
 * Run NN inference with optional symmetry augmentation.
 * Returns { policy, value, policyLogits, oppPLogits }
 *   - policy: Float32Array softmax probabilities (boardSize^2)
 *   - value: Float64Array WDL [win, draw, loss]
 *   - policyLogits: Float32Array masked logits (boardSize^2)
 *   - oppPLogits: Float32Array averaged opponent policy logits (boardSize^2)
 */
async function inference(state, toPlay, mode = "single") {
    if (!session) {
        throw new Error("ONNX session not initialized");
    }

    const encoded = game.encodeState(state, toPlay);
    const C = game.numPlanes, H = boardSize, W = boardSize;
    
    let batchSize = 1;
    let symmetries = [{ doFlip: false, rot: 0 }];

    if (mode === "stochastic") {
        symmetries = [{ doFlip: Math.random() < 0.5, rot: Math.floor(Math.random() * 4) }];
    } else if (mode === "full") {
        batchSize = 8;
        symmetries = [];
        for (const f of [false, true]) {
            for (let r = 0; r < 4; r++) symmetries.push({ doFlip: f, rot: r });
        }
    }

    const inputData = new Float32Array(batchSize * C * H * W);
    for (let i = 0; i < batchSize; i++) {
        const aug = applySymmetry(encoded, C, H, W, symmetries[i].doFlip, symmetries[i].rot);
        inputData.set(aug, i * C * H * W);
    }

    let results;
    try {
        const input = new ort.Tensor("float32", inputData, [batchSize, C, H, W]);
        results = await session.run({ input: input });
    } catch (e) {
        console.error("ONNX inference failed:", e);
        postMessage({ type: "error", message: "推理失败: " + (e.message || String(e)) });
        throw e;
    }

    const pLogits = results.policy_logits.data;
    const vLogits = results.value_logits.data;
    const oppPLogitsData = results.opponent_policy_logits.data;

    // Average value logits across symmetries, then softmax to WDL
    const vLogitsAvg = new Float32Array(3).fill(0);
    for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < 3; j++) vLogitsAvg[j] += vLogits[i * 3 + j] / batchSize;
    }
    const vProbs = mctsSoftmax(Array.from(vLogitsAvg));
    const value = new Float64Array([vProbs[0], vProbs[1], vProbs[2]]);  // WDL [win, draw, loss]

    // Average policy logits & opponent policy logits across symmetries (with inverse transform)
    const avgPLogits = new Float32Array(H * W).fill(0);
    const avgOppPLogits = new Float32Array(H * W).fill(0);

    for (let i = 0; i < batchSize; i++) {
        const p = applyInverseSymmetry(pLogits.slice(i * H * W, (i + 1) * H * W), 1, H, W, symmetries[i].doFlip, symmetries[i].rot);
        const oppP = applyInverseSymmetry(oppPLogitsData.slice(i * H * W, (i + 1) * H * W), 1, H, W, symmetries[i].doFlip, symmetries[i].rot);
        
        for (let j = 0; j < H * W; j++) {
            avgPLogits[j] += p[j] / batchSize;
            avgOppPLogits[j] += oppP[j] / batchSize;
        }
    }

    // Mask illegal moves and compute softmax policy
    const legalMask = game.getLegalActions(state, toPlay);
    const maskedLogits = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
        maskedLogits[i] = legalMask[i] ? avgPLogits[i] : -1e9;
    }
    const policy = new Float32Array(mctsSoftmax(Array.from(maskedLogits)));

    return { policy, value, policyLogits: maskedLogits, oppPLogits: avgOppPLogits };
}

let latestSearchId = 0;

// Track inference speed for time-to-simulations estimation
let inferenceTimeEma = null;  // Exponential moving average of single inference time (ms)

onmessage = async function(e) {
  try {
    const data = e.data;
    if (data.type === "init") {
        await init();
    } else if (data.type === "reset") {
        latestSearchId++;
        root = new Node(game.getInitialState(), 1);
    } else if (data.type === "move") {
        latestSearchId++;
        // Tree reuse: find child matching the played action and promote it to root
        if (root && root.children.length > 0) {
            let found = false;
            for (const child of root.children) {
                if (child.actionTaken === data.action) {
                    root = child;
                    root.parent = null; // detach from old tree for GC
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
    } else if (data.type === "search") {
        const thinkTimeMs = Number.isFinite(data.thinkTimeMs) ? data.thinkTimeMs : null;
        const fixedSims = Number.isFinite(data.numSimulations) ? data.numSimulations : null;
        const useFixedSims = fixedSims !== null;
        const searchId = data.searchId;
        latestSearchId = searchId;
        
        // Tree reuse: keep existing root if already expanded, otherwise create fresh
        if (!root || root.children.length === 0) {
            root = new Node(data.state, data.toPlay);
        }

        const searchStartTime = performance.now();
        const reusingTree = root.isExpanded();

        // === Step 1: Root expand + full symmetry inference ===
        // Always run inference for oppPLogits (needed for UI display)
        const rootInfStart = performance.now();
        const { policy: rootPolicy, value: rootValue, policyLogits: rootLogits, oppPLogits } =
            await inference(root.state, root.toPlay, "full");
        const rootInfTime = performance.now() - rootInfStart;

        // Check for abortion after async
        if (latestSearchId !== searchId) return;

        if (!reusingTree) {
            // Fresh root: expand and backpropagate as usual
            mcts.expand(root, rootPolicy, rootValue, rootLogits);
            mcts.backpropagate(root, rootValue);
        }
        // If reusing tree, root is already expanded with cached nnLogits/nnPolicy/nnValue

        // === Step 2: Determine simulation budget ===
        let numSimulations;
        if (useFixedSims) {
            // Fixed simulation count mode: use user-specified number directly
            numSimulations = Math.max(1, Math.min(fixedSims, 1600));
            // Subtract existing visits from tree reuse
            if (reusingTree && root.n > 0) {
                numSimulations = Math.max(1, numSimulations - root.n);
            }
        } else {
            // Time-based mode: estimate simulation budget from remaining time
            const effectiveThinkTimeMs = thinkTimeMs !== null ? thinkTimeMs : 3000;
            // Update inference time EMA (child inference uses stochastic mode = ~1/8 of full)
            const singleInfEstimate = rootInfTime / 8;  // Approximate stochastic inference time
            if (inferenceTimeEma === null) {
                inferenceTimeEma = singleInfEstimate;
            } else {
                inferenceTimeEma = 0.7 * inferenceTimeEma + 0.3 * singleInfEstimate;
            }

            const elapsed = performance.now() - searchStartTime;
            const remainingTime = Math.max(0, effectiveThinkTimeMs - elapsed);
            // Each simulation = 1 inference (approximately). Reserve some time for final computation.
            const reservedTime = Math.min(200, remainingTime * 0.05);
            numSimulations = Math.max(1, Math.floor((remainingTime - reservedTime) / Math.max(1, inferenceTimeEma)));
            // Cap simulations
            numSimulations = Math.min(numSimulations, 1600);
            // Subtract existing visits from tree reuse (consistent with Python alphazero.py:782)
            if (reusingTree && root.n > 0) {
                numSimulations = Math.max(1, numSimulations - root.n);
            }
        }

        // === Step 3: Run Gumbel Sequential Halving ===
        let lastProgressTime = performance.now();
        let totalSims = 0;

        const simulateOne = async (action) => {
            // Find the child for this action
            const child = root.children.find(c => c.actionTaken === action);
            if (!child) return;

            // Select down from child
            let node = child;
            while (node.isExpanded()) {
                node = mcts.select(node);
                if (!node) return;
            }

            // Evaluate leaf
            // node.toPlay is who plays NEXT; the last player was -node.toPlay
            const winner = game.getWinner(node.state, node.actionTaken, -node.toPlay);
            let value;
            if (winner !== null) {
                // Terminal: WDL one-hot from this node's perspective
                const result = winner * node.toPlay;
                if (result === 1) {
                    value = new Float64Array([1.0, 0.0, 0.0]);  // win
                } else if (result === -1) {
                    value = new Float64Array([0.0, 0.0, 1.0]);  // loss
                } else {
                    value = new Float64Array([0.0, 1.0, 0.0]);  // draw
                }
            } else {
                // NN inference (stochastic symmetry for non-root)
                const infStart = performance.now();
                const { policy, value: v, policyLogits } = await inference(node.state, node.toPlay, "stochastic");
                const infTime = performance.now() - infStart;

                // Update inference time EMA
                inferenceTimeEma = 0.7 * inferenceTimeEma + 0.3 * infTime;

                // Check abortion after async
                if (latestSearchId !== searchId) return;

                mcts.expand(node, policy, v, policyLogits);
                value = v;
            }

            mcts.backpropagate(node, value);
            totalSims++;

            // Report progress periodically
            const now = performance.now();
            if (now - lastProgressTime > 60) {
                lastProgressTime = now;
                let progress;
                if (useFixedSims) {
                    // Fixed sims mode: progress based on simulation count
                    progress = Math.min(100, (totalSims / numSimulations) * 100);
                } else {
                    // Time mode: progress based on elapsed time
                    const effectiveThinkTimeMs = thinkTimeMs !== null ? thinkTimeMs : 3000;
                    progress = Math.min(100, ((now - searchStartTime) / effectiveThinkTimeMs) * 100);
                }
                postMessage({ type: "progress", progress, searchId });
            }
        };

        const { improvedPolicy, gumbelAction, vMix } =
            await mcts.gumbelSequentialHalving(root, numSimulations, /* isEval */ true, simulateOne);

        // Final abortion check
        if (latestSearchId !== searchId) return;

        postMessage({ type: "progress", progress: 100, searchId });

        // Visit-count-based policy for heatmap display
        const visitPolicy = mcts.getMCTSPolicy(root);

        // Compute scalar root value from vMix WDL: W - L
        const rootValueScalar = vMix[0] - vMix[2];

        postMessage({
            type: "result",
            policy: visitPolicy,
            gumbelAction: gumbelAction,
            rootValue: rootValueScalar,
            rootToPlay: root.toPlay,
            nnValue: rootValue[0] - rootValue[2],  // scalar NN value for display
            oppPLogits: oppPLogits,
            iterations: totalSims,
            searchId: searchId
        });
    }
  } catch (err) {
    console.error("Worker error:", err);
    postMessage({ type: "error", message: err.message || String(err) });
  }
};
