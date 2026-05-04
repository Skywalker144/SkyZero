// Propagate main.js's `worker.js?v=<ts>` cache-bust to importScripts: without
// it, _headers' max-age=3600 makes the browser keep serving stale gomoku.js /
// mcts.js even when the worker itself is refreshed.
const _qs = self.location.search || ("?v=" + Date.now());
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js");
importScripts("gomoku.js" + _qs);
importScripts("mcts.js" + _qs);

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer cross-origin fragility; force single-thread

// --- module-level state ---
let session = null;
let game = null;
let mcts = null;
let root = null;
let currentBoardSize = 15;
let currentRule = "renju";
let currentPly = 0;
let latestSearchId = 0;

// --- helpers ---

function concatChunks(chunks, total) {
    const result = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) { result.set(c, offset); offset += c.length; }
    return result;
}

async function fetchModelWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${url} → ${response.status}`);
    const total = Number(response.headers.get("Content-Length")) || 0;
    if (!response.body) {
        const buf = await response.arrayBuffer();
        postMessage({ type: "model-progress", percent: 100, loaded: buf.byteLength, total: buf.byteLength });
        return new Uint8Array(buf);
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    if (total > 0) postMessage({ type: "model-progress", percent: 0, loaded: 0, total });
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

/**
 * Run one ONNX forward pass for `state` (board_size × board_size) with
 * current player `toPlay`. Returns the un-padded heatmap arrays the UI
 * wants, plus raw masked logits for MCTS.
 */
async function inference(state, toPlay) {
    if (!session) throw new Error("session not ready");
    const M = 17, A = M * M;                     // padded canvas
    const N = currentBoardSize, NA = N * N;      // game canvas

    const spatial = game.encodeState(state, toPlay);
    const globalF = game.computeGlobalFeatures(currentPly, toPlay);

    const feeds = {
        input_spatial: new ort.Tensor("float32", spatial, [1, 5, M, M]),
        input_global:  new ort.Tensor("float32", globalF, [1, 12]),
    };
    const out = await session.run(feeds);

    const policyAll = out.policy_logits.data;            // (1, 4, A)
    const wdlLogits = out.value_wdl_logits.data;         // (1, 3)
    const futureAll = out.value_futurepos_pretanh.data;  // (1, 2, M, M)

    // --- crop policy channels 0 (main) and 1 (opp) from padded → board area ---
    function cropChannel(channelIdx) {
        const cropped = new Float32Array(NA);
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                cropped[r * N + c] = policyAll[channelIdx * A + r * M + c];
            }
        }
        return cropped;
    }
    const policyMainRaw = cropChannel(0);
    const policyOppRaw  = cropChannel(1);

    // --- mask illegal + softmax ---
    const legal = game.getLegalActions(state, toPlay);
    const policyMainMasked = new Float32Array(NA);
    for (let i = 0; i < NA; i++) policyMainMasked[i] = legal[i] ? policyMainRaw[i] : -1e9;
    const policyMainSoft = new Float32Array(softmax(policyMainMasked));

    // Opp policy: don't mask (opponent's policy doesn't share legality), just softmax.
    const policyOppSoft = new Float32Array(softmax(policyOppRaw));

    // --- value WDL: softmax 3 logits ---
    const wdl = softmax(new Float64Array([wdlLogits[0], wdlLogits[1], wdlLogits[2]]));
    const wdlF64 = new Float64Array([wdl[0], wdl[1], wdl[2]]);

    // --- futurepos: tanh per cell, crop both channels ---
    function cropTanh(channelIdx) {
        const cropped = new Float32Array(NA);
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                cropped[r * N + c] = Math.tanh(futureAll[channelIdx * A + r * M + c]);
            }
        }
        return cropped;
    }
    const future8  = cropTanh(0);
    const future32 = cropTanh(1);

    return {
        policyMainSoft,                    // for MCTS expand prior
        policyMainMaskedLogits: policyMainMasked,   // for Gumbel halving
        policyOppSoft,                     // UI heatmap
        wdl: wdlF64,                       // root nn value
        future8, future32,                 // UI heatmaps
    };
}

async function initSession(modelUrl, boardSize, rule) {
    currentBoardSize = boardSize;
    currentRule = rule || "renju";
    game = new Gomoku(boardSize, currentRule);
    mcts = new MCTS(game, {
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
    });
    root = null;
    const bytes = await fetchModelWithProgress(modelUrl);
    session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
    });
    postMessage({ type: "ready" });
}

function resetGame(boardSize, ply, rule) {
    const sizeChanged = boardSize !== undefined && boardSize !== currentBoardSize;
    const ruleChanged = rule !== undefined && rule !== currentRule;
    if (sizeChanged) currentBoardSize = boardSize;
    if (ruleChanged) currentRule = rule;
    if (sizeChanged || ruleChanged) {
        game = new Gomoku(currentBoardSize, currentRule);
        if (mcts) mcts.game = game;   // keep MCTS bound to the live game
    }
    currentPly = ply || 0;
    root = null;
}

function applyMove(action, nextState, nextToPlay, ply) {
    currentPly = ply;
    if (root && root.children.length > 0) {
        const child = root.children.find(c => c.actionTaken === action);
        if (child) {
            root = child;
            root.parent = null;   // detach for GC
            return;
        }
    }
    root = new Node(nextState, nextToPlay);
}

async function runSearch(state, toPlay, ply, sims, gumbelM, gen, externalSearchId) {
    currentPly = ply;
    if (gumbelM != null) mcts.args.gumbel_m = gumbelM;
    if (!root) root = new Node(state, toPlay);

    // Root inference if not already expanded.
    let oppPolicy, future8, future32;
    let nnValueWDL;
    if (!root.isExpanded()) {
        const inf = await inference(root.state, root.toPlay);
        if (latestSearchId !== gen) return;
        mcts.expand(root, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
        mcts.backpropagate(root, inf.wdl);
        nnValueWDL = inf.wdl;
        oppPolicy = inf.policyOppSoft;
        future8 = inf.future8;
        future32 = inf.future32;
    } else {
        // Even on tree reuse we still run ONE inference to get fresh oppPolicy / future*.
        const inf = await inference(root.state, root.toPlay);
        if (latestSearchId !== gen) return;
        nnValueWDL = root.nnValue;   // cached
        oppPolicy = inf.policyOppSoft;
        future8 = inf.future8;
        future32 = inf.future32;
    }

    let totalSims = 0;
    let lastProgress = performance.now();

    const simulateOne = async (action) => {
        const child = root.children.find(c => c.actionTaken === action);
        if (!child) return;
        let node = child;
        while (node.isExpanded()) {
            node = mcts.select(node);
            if (!node) return;
        }
        // Evaluate leaf: terminal or NN.
        // node.toPlay is who moves NEXT; the actor of node.actionTaken is -node.toPlay.
        const winner = game.getWinner(node.state, node.actionTaken, -node.toPlay);
        let value;
        if (winner !== null) {
            const result = winner * node.toPlay;   // winner relative to node.toPlay's POV
            if      (result === 1)  value = new Float64Array([1, 0, 0]);
            else if (result === -1) value = new Float64Array([0, 0, 1]);
            else                    value = new Float64Array([0, 1, 0]);
        } else {
            const inf = await inference(node.state, node.toPlay);
            if (latestSearchId !== gen) return;
            mcts.expand(node, inf.policyMainSoft, inf.wdl, inf.policyMainMaskedLogits);
            value = inf.wdl;
        }
        mcts.backpropagate(node, value);
        totalSims++;

        const now = performance.now();
        if (now - lastProgress > 60) {
            lastProgress = now;
            const pct = Math.min(100, (totalSims / sims) * 100);
            postMessage({ type: "progress", progress: pct, searchId: externalSearchId });
        }
    };

    const { improvedPolicy, gumbelAction, vMix } =
        await mcts.gumbelSequentialHalving(root, sims, simulateOne);

    if (latestSearchId !== gen) return;
    postMessage({ type: "progress", progress: 100, searchId: externalSearchId });

    // Visit distribution N(s,a)/sum — matches V5 cpp label "MCTS Visits (N(s,a)/sum)".
    const visitDist = mcts.getMCTSPolicy(root);

    postMessage({
        type: "result",
        searchId: externalSearchId,
        gumbelAction,
        rootValueWDL: vMix,                   // [W, D, L] from vMix
        nnValueWDL,                           // [W, D, L] root NN
        mctsPolicy:    Array.from(improvedPolicy),  // V5 "MCTS Strategy (improved policy)"
        mctsVisits:    Array.from(visitDist),       // V5 "MCTS Visits (N(s,a)/sum)"
        nnPolicy:      Array.from(root.nnPolicy || new Float32Array(currentBoardSize * currentBoardSize)),  // V5 "NN Strategy"
        nnOppPolicy:   Array.from(oppPolicy),
        nnFuturepos8:  Array.from(future8),
        nnFuturepos32: Array.from(future32),
        gumbelPhases:  root._gumbelPhases || [],
        iterations:    totalSims,
    });
}

onmessage = async (e) => {
    const data = e.data;
    try {
        if (data.type === "init") {
            await initSession(data.modelUrl, data.boardSize, data.rule);
        } else if (data.type === "reset") {
            latestSearchId++;
            resetGame(data.boardSize, data.ply, data.rule);
        } else if (data.type === "move") {
            applyMove(data.action, data.nextState, data.nextToPlay, data.ply);
        } else if (data.type === "search") {
            latestSearchId++;
            const gen = latestSearchId;
            await runSearch(data.state, data.toPlay, data.ply, data.sims, data.gumbel_m, gen, data.searchId);
        } else if (data.type === "swap-model") {
            latestSearchId++;
            session = null;
            root = null;
            const bytes = await fetchModelWithProgress(data.modelUrl);
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: ["wasm"],
                intraOpNumThreads: 1,
                interOpNumThreads: 1,
            });
            postMessage({ type: "ready" });
        }
    } catch (err) {
        postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
    }
};
