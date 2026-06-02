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

## Adding / updating the 2048 model

The 2048 page runs the SkyZero_2048 Stochastic Gumbel AlphaZero value net in the
browser. Export a checkpoint with the dedicated script (uses the SkyZero_2048
`pytorch` env, which has torch/onnx/onnxruntime):

```bash
/home/sky/miniconda3/envs/pytorch/bin/python tools/export_onnx_2048.py \
    --ckpt ../SkyZero_2048/data2048_td/nets/b3c64/scripted_iter_000051.pt \
    --net  b3c64 --out models/2048.onnx
```

`--net b3c64` parses to blocks=3/channels=64; the value head is rescaled to raw
2048 points inside the ONNX graph (so the browser AI's expectimax Q is in
points). To swap in a stronger checkpoint just re-export to `models/2048.onnx`
and redeploy. After training a different arch, change `--net` accordingly.

The browser AI (`ai2048.js`) plays a **1-ply expectimax** over the value head —
`Q(a) = reward(a) + γ·E_spawn[V(next)]` — rather than the full Gumbel MCTS used
during self-play, so it's lighter but weaker than the engine's searched
strength. `tests/test_ai2048.mjs` cross-checks the JS slide/spawn/encode logic
against a fixture generated from `SkyZero_2048/python/game.py`.

## Architecture

- `index.html` — landing page (game picker → `/gomoku`, `/2048`)
- `gomoku.html` / `style.css` — five-in-a-row UI (ported from `play_web.py`)
- `2048.html` — 2048 game UI + in-page AI controls (AI 走子 / AI 托管)
- `main.js` — UI controller, canvas rendering, worker plumbing
- `worker.js` — runs `ort.InferenceSession` + MCTS in a Web Worker (gomoku)
- `worker2048.js` — runs the 2048 value net (ONNX) + expectimax off-thread
- `mcts.js` — Sequential MCTS with variance-scaled cPUCT + Gumbel halving
- `ai2048.js` — 2048 afterstate logic + 1-ply value-net expectimax
- `gomoku.js` — RENJU game logic with multi-board-size + V5 5-plane encoding
- `tools/export_onnx.py` — V5 `.pt` → `.onnx` (gomoku, drops UI-unused heads)
- `tools/export_onnx_2048.py` — SkyZero_2048 `.pt` → `models/2048.onnx`
- `models/manifest.json` — gomoku 5-tier ELO catalog

## Differences from V5 `play_web.py`

These simplifications are intentional (browser constraints):

- No 8-fold symmetry ensemble (single forward pass)
- No stochastic transform
- No parallel MCTS (Worker is single-threaded)
- No root symmetry pruning toggle
- RENJU rule only (no STANDARD / FREESTYLE — only RENJU was trained)
- `value_td` and intermediate heads dropped from ONNX export
