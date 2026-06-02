// Web Worker that runs the 2048 value network (ONNX) off the main thread and
// picks moves via the 1-ply expectimax in ai2048.js. Mirrors worker.js (the
// Gomoku worker): same onnxruntime-web build, single-threaded WASM, cache-bust
// propagated to importScripts.
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

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
ort.env.wasm.numThreads = 1;   // SharedArrayBuffer cross-origin fragility; force single-thread

var session = null;
var GAMMA = 0.999;
var PLANES = AI2048.NUM_PLANES * AI2048.AREA;   // 16*16 = 256 floats / state

async function fetchModelWithProgress(url) {
  var response = await fetch(url);
  if (!response.ok) throw new Error('fetch ' + url + ' -> ' + response.status);
  var total = Number(response.headers.get('Content-Length')) || 0;
  if (!response.body) {
    var buf = await response.arrayBuffer();
    postMessage({ type: 'model-progress', percent: 100 });
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
    postMessage({ type: 'model-progress', percent: total > 0 ? (loaded / total) * 100 : null });
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

// Batched value-net eval: flat is B*256 float32 (plane-major encoded states).
// Returns a Float32Array[B] of value-head outputs (raw 2048 points).
async function runBatch(flat, B) {
  if (!session) throw new Error('session not ready');
  var feeds = { input: new ort.Tensor('float32', flat, [B, AI2048.NUM_PLANES, 4, 4]) };
  var out = await session.run(feeds);
  return out.value.data;   // (B, 1) -> length-B Float32Array
}

async function think(grid, id) {
  var exps = AI2048.gridValuesToExps(grid);
  var res = await AI2048.chooseMove(exps, runBatch, GAMMA);
  postMessage({
    type: 'move', id: id,
    action: res.action, dir: res.dir, qs: res.qs, value: res.value,
    terminal: res.action < 0,
  });
}

self.onmessage = function (e) {
  var msg = e.data || {};
  var p;
  if (msg.type === 'init') p = init(msg.model);
  else if (msg.type === 'think') p = think(msg.grid, msg.id);
  else return;
  Promise.resolve(p).catch(function (err) {
    postMessage({ type: 'error', message: (err && err.message) || String(err) });
  });
};
