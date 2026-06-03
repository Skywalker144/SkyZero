// Web Worker that runs the 2048 net (ONNX) off the main thread and picks moves
// via the full afterstate Stochastic Gumbel AlphaZero MCTS in mcts2048.js (a
// port of SkyZero_2048/python/mcts.py). Mirrors worker.js (the Gomoku worker):
// same onnxruntime-web build, single-threaded WASM, cache-bust propagated to
// importScripts.
//
// Protocol (postMessage):
//   in  { type:'init',  model:<url> }            -> out { type:'ready' }
//                                                   + { type:'model-progress', percent }
//   in  { type:'think', grid:number[][], id }    -> out { type:'move', id, action,
//                                                          dir, qs, value, terminal }
//   any failure                                  -> out { type:'error', message }
const _qs = self.location.search || ('?v=' + Date.now());
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js');
importScripts('ai2048.js' + _qs);
importScripts('mcts2048.js' + _qs);

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer cross-origin fragility; force single-thread

var session = null;
// Search config (overrides MCTS2048.DEFAULTS); mirrors configs/vtransform/run.cfg.
// Eval mode => gumbel_noise off (deterministic strongest play).
var SEARCH_CFG = { gamma: 0.999, num_simulations: 64, gumbel_noise: false };
var PLANES = AI2048.NUM_PLANES * AI2048.AREA;   // 16*16 = 256 floats / state

async function fetchModelWithProgress(url) {
  var response = await fetch(url);
  if (!response.ok) throw new Error('fetch ' + url + ' -> ' + response.status);
  var total = Number(response.headers.get('Content-Length')) || 0;
  if (!response.body) {
    var buf = await response.arrayBuffer();
    postMessage({ type: 'model-progress', percent: 100, loaded: buf.byteLength, total: buf.byteLength });
    return new Uint8Array(buf);
  }
  var reader = response.body.getReader();
  var chunks = [];
  var loaded = 0;
  while (true) {
    var r = await reader.read();
    if (r.done) break;
    chunks.push(r.value);
    loaded += r.value.length;
    postMessage({ type: 'model-progress', percent: total > 0 ? (loaded / total) * 100 : null, loaded: loaded, total: total || null });
  }
  var size = total || loaded;
  var out = new Uint8Array(size);
  var off = 0;
  for (var i = 0; i < chunks.length; i++) { out.set(chunks[i], off); off += chunks[i].length; }
  return out;
}

async function init(modelUrl) {
  var bytes = await fetchModelWithProgress(modelUrl);
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });
  postMessage({ type: 'ready' });
}

// Batched net eval: flat is B*256 float32 (plane-major encoded states). Returns
// both heads: { logits:Float32Array[B*4], values:Float32Array[B] } (value in
// raw 2048 points — the ONNX export folds value_scale + h_inv into the graph).
async function runNet(flat, B) {
  if (!session) throw new Error('session not ready');
  var feeds = { input: new ort.Tensor('float32', flat, [B, AI2048.NUM_PLANES, 4, 4]) };
  var out = await session.run(feeds);
  return { logits: out.policy_logits.data, values: out.value.data };
}

async function think(grid, id, sims) {
  var exps = AI2048.gridValuesToExps(grid);
  // sims (from the speed slider) overrides the default simulation budget; more
  // sims = stronger play (and slower). Falls back to SEARCH_CFG.num_simulations.
  var cfg = (sims > 0) ? Object.assign({}, SEARCH_CFG, { num_simulations: sims }) : SEARCH_CFG;
  var res = await MCTS2048.chooseMoveMCTS(exps, runNet, cfg);
  postMessage({
    type: 'move', id: id,
    action: res.action, dir: res.dir, qs: res.qs, value: res.value,
    sims: cfg.num_simulations, terminal: res.action < 0,
  });
}

self.onmessage = function (e) {
  var msg = e.data || {};
  var p;
  if (msg.type === 'init') p = init(msg.model);
  else if (msg.type === 'think') p = think(msg.grid, msg.id, msg.sims);
  else return;
  Promise.resolve(p).catch(function (err) {
    postMessage({ type: 'error', message: (err && err.message) || String(err) });
  });
};
