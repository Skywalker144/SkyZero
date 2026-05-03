# SkyZeroWeb

Static webpage that runs a SkyZero V5 model in the browser via
`onnxruntime-web`. Full UI parity with `SkyZero_V5/python/play_web.py`
but no server, no C++ engine — everything runs client-side.

## Quick start

### Local dev

```bash
cd /home/sky/RL/SkyZero/SkyZeroWeb
python3 -m http.server 8000
# Open http://localhost:8000
```

(`file://` won't work — Worker `importScripts` and `fetch('models/...')`
need an HTTP server.)

### Tests

```bash
export PATH=/home/sky/.nvm/versions/node/v24.15.0/bin:$PATH
npm test
```

Runs Node 18+ builtin test runner against `gomoku.js` and `mcts.js`
(pure-logic units; UI is verified manually). Should report `tests 37 | pass 37`.

### Deploy

This repo is set up for [Cloudflare Pages](https://pages.cloudflare.com/)
with no build step. Connect via git, point to repo root.

## Adding / updating a model

```bash
# Use V5's pytorch env (has torch/onnx/onnxscript installed)
/home/sky/anaconda3/envs/pytorch/bin/python tools/export_onnx.py \
    --ckpt /path/to/SkyZero_V5/data/.../models/model_iter_NNNNNN.pt \
    --out  models/levelN.onnx \
    --num-blocks 10 --num-channels 128

# Or use a TorchScript anchor file (the script handles both formats):
/home/sky/anaconda3/envs/pytorch/bin/python tools/export_onnx.py \
    --ckpt /path/to/SkyZero_V5/anchors/b10c128iter80.pt \
    --out  models/level3.onnx \
    --num-blocks 10 --num-channels 128

# Then edit models/manifest.json — add or update the entry's elo / label / file
# git add + commit + push → Cloudflare auto-deploys
```

The 5-tier ELO catalog is hand-curated. Each tier ships one ONNX
(~4 MB for `b10c128`, ~0.4 MB for `b4c64`).

### Python prerequisites for export

The export script needs these packages (in addition to torch):

```bash
pip install onnx onnxscript onnxruntime
```

## Architecture

- `index.html` / `style.css` — UI (ported from `play_web.py`)
- `main.js` — UI controller, canvas rendering, worker plumbing
- `worker.js` — runs `ort.InferenceSession` + MCTS in a Web Worker
- `mcts.js` — Sequential MCTS with variance-scaled cPUCT + Gumbel halving
- `gomoku.js` — RENJU game logic with multi-board-size + V5 5-plane encoding
- `tools/export_onnx.py` — V5 `.pt` → `.onnx` (drops UI-unused heads)
- `models/manifest.json` — 5-tier ELO catalog

## Differences from V5 `play_web.py`

These simplifications are intentional (browser constraints):

- No 8-fold symmetry ensemble (single forward pass)
- No stochastic transform
- No parallel MCTS (Worker is single-threaded)
- No root symmetry pruning toggle
- RENJU rule only (no STANDARD / FREESTYLE — only RENJU was trained)
- `value_td` and intermediate heads dropped from ONNX export
