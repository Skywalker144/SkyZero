importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
importScripts("gomoku.js");
importScripts("mcts.js");

// Force ONNX Runtime to load WASM from CDN
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session = null;
let game = null;
let mcts = null;

const boardSize = 15;
const historyStep = 2;

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

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b);
    return scores.map(s => s / sum);
}

async function init() {
    game = new Gomoku(boardSize, historyStep, true);
    mcts = new MCTS(game, {
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
    });

    try {
        const ua = (self.navigator && self.navigator.userAgent) ? self.navigator.userAgent : "";
        const isIOS = /iP(hone|ad|od)/.test(ua);
        const executionProviders = isIOS ? ["wasm", "cpu"] : ["webgl", "cpu"];

        if (isIOS) {
            session = await ort.InferenceSession.create("model.onnx", { executionProviders });
        } else {
            const modelBytes = await fetchModelWithProgress("model.onnx");
            session = await ort.InferenceSession.create(modelBytes, { executionProviders });
        }
        postMessage({ type: "ready" });
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        postMessage({ type: "error", message: e.message });
    }
}

/**
 * Run NN inference.
 * @param {Array} state
 * @param {number} toPlay
 * @param {string} mode - "single", "stochastic", or "full"
 * @returns {{ policy: Float32Array, value: Float64Array, logits: Float32Array, ownership: Float32Array, oppPLogits: Float32Array }}
 *   - policy: softmax probabilities (legal-masked)
 *   - value: WDL [win, draw, loss]
 *   - logits: raw averaged logits (legal-masked, before softmax)
 *   - ownership: averaged ownership map
 *   - oppPLogits: averaged opponent policy logits
 */
async function inference(state, toPlay, mode = "single") {
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

    const input = new ort.Tensor("float32", inputData, [batchSize, C, H, W]);
    const results = await session.run({ input: input });

    const pLogits = results.policy_logits.data;
    const vLogits = results.value_logits.data;
    const ownershipData = results.ownership.data;
    const oppPLogitsData = results.opponent_policy_logits.data;

    // Average value logits -> WDL probabilities
    const vLogitsAvg = new Float32Array(3).fill(0);
    for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < 3; j++) vLogitsAvg[j] += vLogits[i * 3 + j] / batchSize;
    }
    const vProbs = softmax(Array.from(vLogitsAvg));
    const value = new Float64Array([vProbs[0], vProbs[1], vProbs[2]]);  // WDL

    // Average policy logits, ownership, opponent policy
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

    // Legal mask + softmax for policy probabilities
    const legalMask = game.getLegalActions(state, toPlay);
    const maskedLogits = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
        maskedLogits[i] = legalMask[i] ? avgPLogits[i] : -1e9;
    }
    const policy = softmax(Array.from(maskedLogits));

    return {
        policy: new Float32Array(policy),
        value,
        logits: maskedLogits,       // raw masked logits (for Gumbel algorithm)
        ownership: avgOwnership,
        oppPLogits: avgOppPLogits,
    };
}

let latestSearchId = 0;
const MAX_SEARCH_ITERATIONS = 1600;

onmessage = async function(e) {
    const data = e.data;
    if (data.type === "init") {
        await init();
    } else if (data.type === "reset") {
        latestSearchId++;
    } else if (data.type === "move") {
        latestSearchId++;
        // No tree reuse in Gumbel AlphaZero — each search starts fresh
    } else if (data.type === "search") {
        const thinkTimeMs = Number.isFinite(data.thinkTimeMs) ? data.thinkTimeMs : 3000;
        const searchId = data.searchId;
        latestSearchId = searchId;

        const state = data.state;
        const toPlay = data.toPlay;

        const startTime = performance.now();
        const timeBudget = Math.max(0, thinkTimeMs);
        const deadline = startTime + timeBudget;

        // --- 1. Create fresh root and expand with full symmetry ---
        const root = new Node(state, toPlay);
        const rootInf = await inference(state, toPlay, "full");

        if (latestSearchId !== searchId) return;

        mcts.expand(root, rootInf.policy, rootInf.value, rootInf.logits);
        mcts.backpropagate(root, [rootInf.value[0], rootInf.value[1], rootInf.value[2]]);

        // --- 2. Calculate simulation budget from time ---
        // Estimate iterations from time budget: we use time-based adaptive budget
        // Start with a conservative budget, can be adjusted
        let numSimulations = MAX_SEARCH_ITERATIONS;

        // --- 3. Run Gumbel Sequential Halving ---
        let aborted = false;
        let totalIterations = 0;
        let lastProgressTime = startTime;

        const inferFn = async (nodeState, nodeToPlay) => {
            // Check abortion before inference
            if (latestSearchId !== searchId) {
                aborted = true;
                // Return dummy to avoid crash — caller should check aborted
                return {
                    policy: new Float32Array(boardSize * boardSize),
                    value: new Float64Array([0, 1, 0]),
                    logits: new Float32Array(boardSize * boardSize),
                };
            }
            // Non-root nodes use stochastic single-symmetry (matching Python behavior)
            const result = await inference(nodeState, nodeToPlay, "stochastic");
            if (latestSearchId !== searchId) {
                aborted = true;
            }
            return result;
        };

        const progressFn = (simsUsed, totalBudget) => {
            totalIterations = simsUsed;
            const now = performance.now();

            // Time-based abort: if we exceed the deadline, we let the current phase finish
            // but signal that we should stop (the search naturally stops when budget is exhausted)

            if (now - lastProgressTime > 60) {
                lastProgressTime = now;
                const timeProgress = timeBudget > 0
                    ? Math.min(100, ((now - startTime) / timeBudget) * 100)
                    : 100;
                const simProgress = (simsUsed / totalBudget) * 100;
                const progress = Math.max(timeProgress, simProgress);
                postMessage({ type: "progress", progress, searchId });
            }
        };

        const { improvedPolicy, gumbelAction, vMix } = await mcts.gumbelSequentialHalving(
            root, numSimulations, inferFn, progressFn
        );

        if (aborted || latestSearchId !== searchId) return;

        postMessage({ type: "progress", progress: 100, searchId });

        // --- 4. Get auxiliary outputs for display ---
        const { ownership, oppPLogits } = await inference(root.state, root.toPlay, "full");

        if (latestSearchId !== searchId) return;

        postMessage({
            type: "result",
            policy: improvedPolicy,
            gumbelAction: gumbelAction,
            rootValue: Array.from(vMix),          // WDL [win, draw, loss]
            rootToPlay: root.toPlay,
            nnValue: Array.from(rootInf.value),   // WDL from raw NN
            ownership: ownership,
            oppPLogits: oppPLogits,
            iterations: totalIterations + 1,       // +1 for root expansion
            searchId: searchId,
        });
    }
};
