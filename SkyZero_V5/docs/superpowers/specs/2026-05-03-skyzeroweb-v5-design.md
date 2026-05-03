# SkyZeroWeb V5 — Static ONNX Web Port (Design)

Replace `SkyZero_V5/python/play_web.py` (Python HTTP server wrapping a C++ MCTS
subprocess) with a **static, deployable webpage** that runs the same V5 model
and MCTS in the user's browser via `onnxruntime-web`. UI parity with V5
`play_web.py`. New project lives at `/home/sky/RL/SkyZero/SkyZeroWeb/` (sibling
of `SkyZero_V5`, independent git repo). `/home/sky/RL/SkyZeroWeb/` is the older
V4-era prototype kept as architectural reference only.

## Goals

- **Static deployment** on Cloudflare Pages (free tier). No server-side runtime.
- **Full UI parity** with `play_web.py`'s `HTML_PAGE`: 6 heatmaps, dual WDL
  bars, value chart, Gumbel sequential-halving overlay, theme toggle, fullscreen
  heat modal, model selector, board size selector.
- **Algorithmic fidelity** with V5's `cpp/alphazero.h` MCTS and
  `cpp/envs/gomoku.h` game logic (Renju rules, multi-board-size, V5 5-plane
  encoding + 12-dim global features).
- **ELO-curated 5-tier** model selection ("新手 / 入门 / 进阶 / 高手 / 大师"
  with absolute ELO values shown). User manually picks which checkpoints map
  to which tier; export script + manifest do the plumbing.
- **Single-threaded** worker-side inference (no SharedArrayBuffer / cross-origin
  threading complications).

## Non-Goals (intentional simplifications vs V5)

These are deliberately dropped to keep the browser version tractable. Each
is documented inline in `worker.js` / `mcts.js` so future readers know it's
intentional, not an oversight.

- ❌ **8-fold symmetry ensemble** at root/child (V5 `play.cfg` defaults
  `ENABLE_SYMMETRY_ROOT=1, ENABLE_SYMMETRY_CHILD=1`). 8× forward passes per NN
  call would push browser inference to 10-60s per move; not workable. Single
  forward pass instead.
- ❌ **Stochastic transform** (V5 `play.cfg` defaults are 0; nothing to drop).
- ❌ **Parallel MCTS** (`alphazero_parallel.h`'s shared-tree threading). Worker
  is single-threaded; at <1000 sims the gains are marginal anyway.
- ❌ **Root symmetry pruning** toggle (depends on symmetry; moot when off).
- ❌ **Multi-rule support**: only RENJU, matching `RULE_RELPROBS="1, 0, 0"`
  in V5 `run.cfg`. STANDARD/FREESTYLE not exposed.
- ❌ **Time-based search budget**: only `sims` input (matching V5; not the
  older V4 web's "by time" tab).
- ❌ **TD value head** (`value_td`) and **intermediate heads**: present in
  the model but `play_web.py` doesn't display them. Drop from ONNX export.
- ❌ **Forbidden-point X marks on the board**: V5 `play_web.py` doesn't draw
  them either; user reads forbidden info via the NN Opp Policy / Futurepos
  heatmaps.

## Architecture

```
SkyZeroWeb/
├── index.html           # UI structure (extracted from play_web.py HTML_PAGE)
├── style.css            # Visual styling (extracted from play_web.py <style>)
├── main.js              # UI controller, canvas painting, worker plumbing
├── worker.js            # ONNX session + MCTS driver (off-main-thread)
├── mcts.js              # Pure MCTS algorithm (importScripts'd by worker)
├── gomoku.js            # Pure game logic + V5 encoding (importScripts'd)
├── models/
│   ├── manifest.json    # 5-tier ELO model catalog (hand-curated)
│   ├── level1.onnx      # 新手
│   ├── level2.onnx      # 入门
│   ├── level3.onnx      # 进阶
│   ├── level4.onnx      # 高手
│   └── level5.onnx      # 大师 (typically latest checkpoint)
├── tools/
│   └── export_onnx.py   # one V5 .pt ckpt → one .onnx
├── _headers             # Cloudflare cache directives
└── README.md            # build/deploy/add-model instructions
```

**Why a Worker:** ONNX inference + MCTS expansion are CPU-heavy. On the main
thread they freeze UI/canvas. Worker also lets us cleanly abort an in-flight
search via a `searchId` epoch when the user undoes / starts a new game / swaps
models.

**Why split mcts.js / gomoku.js:** Both are pure logic with no DOM/Worker
coupling. Splitting them makes them unit-testable from Node and lets the
worker `importScripts` them directly.

## Module Responsibilities

### `gomoku.js` — V5 game logic port

Mirrors `cpp/envs/gomoku.h` for the RENJU rule path. Pure JS, no dependencies.

**Class `Gomoku`** (constructed once with `boardSize ∈ [13..17]`):
- `getInitialState()` → `Int8Array(boardSize²)` (0 = empty, +1 = black, −1 = white)
- `getLegalActions(state, toPlay)` → `Uint8Array(boardSize²)` (Renju forbidden
  points for black are illegal; otherwise empty cells legal)
- `getNextState(state, action, toPlay)` → new `Int8Array`
- `getWinner(state, lastAction, lastPlayer)` → `+1 / -1 / 0 / null`
  (null = ongoing). Implements V5 `get_winner_v5`: Renju forbidden-on-black-
  loses + exactly-5 / overline-forbidden / white-wins-on-≥5.
- `encodeState(state, toPlay, ply)` → `{spatial: Float32Array, global:
  Float32Array}`. Spatial = 5 planes padded to 17×17 (mask / own / opp /
  fb_black / fb_white). Global = 12 dims (rule one-hot [3] + renju_color_sign
  + has_forbidden + ply/area + 6 zeros for VCF placeholder).

**Class `ForbiddenPointFinder`**: port of V5's FPF double-negation Renju
detector (overline, three-three, four-four). Used by `getLegalActions` and
within `encodeState` for the forbidden planes.

### `mcts.js` — V5 sequential MCTS port

Mirrors `cpp/alphazero.h` (sequential variant — not the parallel header).
Pure JS, depends only on `gomoku.js`.

**Class `Node`**: `{state, toPlay, prior, parent, actionTaken, children,
nnValue: Float64Array(3), nnPolicy, nnLogits, v: Float64Array(3),
utilitySqSum, n}`.

**Class `MCTS`**:
- Constants: `c_puct=1.1, c_puct_log=0.45, c_puct_base=500, fpu_reduction_max=0.2,
  fpu_pow=1, fpu_loss_prop=0, cpuct_utility_stdev_prior=0.4,
  cpuct_utility_stdev_prior_weight=2, cpuct_utility_stdev_scale=0.85,
  gumbel_m=16, gumbel_c_visit=50, gumbel_c_scale=1` (matching V5 `play.cfg`).
- `select(node)`: PUCT with variance-scaled cPUCT (per
  `compute_parent_utility_stdev_factor` in V5).
- `expand(node, policy, value, logits)`: builds children for all legal actions
  with prior = softmax probability.
- `backpropagate(node, value)`: WDL flip [W,D,L] ↔ [L,D,W] up the chain.
- `gumbelSequentialHalving(root, sims, simulateOne)`: ports
  `alphazero.h::run_gumbel_search` — Gumbel-Top-k of `m` candidates, halving
  phases, `vMix` mixed value, completed-Q improved policy. Records the
  surviving-action set per phase into `gumbelPhases: Array<Array<[r,c]>>` for
  UI overlay.

**No** parallel/shared-tree code. **No** symmetry. **No** stochastic transform.

### `worker.js` — ONNX runtime + driver

Imports `ort.min.js` from jsdelivr CDN (v1.17.0, classic worker via
`importScripts`), `mcts.js`, `gomoku.js`. WASM single-thread mode.

**State**: `session, game, mcts, root, latestSearchId, currentBoardSize`.

**Message handlers**:
- `init {modelUrl, boardSize}`: stream-fetch ONNX with progress events,
  `ort.InferenceSession.create`, post `ready`.
- `reset {boardSize}`: rebuild `game` if size changed; clear `root`.
- `move {action, nextState, nextToPlay}`: tree-reuse — promote matching child
  to root, otherwise rebuild root from `nextState`.
- `search {state, toPlay, ply, sims, gumbel_m, searchId}`:
  1. If root not expanded, root inference + expand + backprop.
  2. Run Gumbel halving for `sims` simulations. Each leaf inference checks
     `latestSearchId !== searchId` after the await and bails if stale.
  3. Pack result: `mctsPolicy` (visits/sum), `mctsVisits` (visits / max),
     `nnPolicy` (softmax of policy logits ch0), `nnOppPolicy` (softmax ch1),
     `futurepos8` (tanh of futurepos ch0), `futurepos32` (tanh ch1),
     `rootValueWDL` (vMix from gumbel halving), `nnValueWDL` (root.nnValue),
     `gumbelAction` (final pick), `gumbelPhases`, `iterations`.

**Inference helper**: encodes state → 2 input tensors, runs session, returns
`{policyLogits: Float32Array(boardSize²), policyChannels: Array<Float32Array>(4),
valueWDL: Float64Array(3), futurepos: [Float32Array, Float32Array]}`. Strips
the 17×17 padding back to `boardSize²` for board-relative arrays. Masks
illegal moves with `-1e9` before softmax.

### `main.js` — UI controller

Direct port of `play_web.py`'s embedded `<script>`. Replaces:
- `setInterval(refresh, 250)` polling `/state` → push-driven from `worker.onmessage`.
- `fetch('/move', ...)` engine commands → direct `gomoku.js` mutation +
  `worker.postMessage`.
- `/models`, `/config` HTTP → `fetch('models/manifest.json')`, board sizes
  hard-coded `[17, 16, 15, 14, 13]`.
- C++ stdout grid parsing → typed arrays from worker `result`.

**Kept verbatim**: tri-state theme toggle (auto/light/dark) with
`localStorage`, board canvas painting (DPR-aware, hoshi, last-move dot,
Gumbel phase circles + labels), 6 heatmap canvases (signed for futurepos,
unsigned for policies), value chart (root blue + nn red, undo carries
forward last value), WDL bar rendering with per-segment widths + W/D/L%
detail line, fullscreen heat modal with Esc/click-outside dismiss.

**Removed**: `prune_toggle`, `noise 0` / `prune N` send commands (no engine
to send to). Top-bar subtitle changed to `ONNX · static web`.

## Data Flow

```
[ user clicks board cell ]
        │
        ▼
[ main.js: validate via gomoku.js, mutate state ]
        │
        ▼
[ worker.postMessage('move') ] ──> [ worker: tree reuse on root ]
        │
        ▼
[ if AI's turn: searchId++; worker.postMessage('search') ]
        │
        ▼
[ worker: Gumbel halving loop ]
   ├── leaf inference (ort.run on 5-plane + 12-global)
   ├── expand + backprop
   └── periodic 'progress' postMessage  ─> [ main.js: status bar ]
        │
        ▼
[ worker.postMessage('result' with all heatmap + value data) ]
        │
        ▼
[ main.js: render board + 6 heatmaps + WDL + chart + Gumbel overlay ]
        │
        ▼
[ apply gumbelAction (AI's chosen move), back to top ]
```

**Abort path**: Any `reset` / `move` / model swap increments `searchId`. The
worker's inference helper checks `latestSearchId !== capturedSearchId` after
each `await ort.run` and returns silently. Stale `result` messages are
filtered in `main.js` by comparing the incoming `searchId` against current.

## ONNX Export

`tools/export_onnx.py` — adapts `python/export_model.py`. Per-checkpoint
invocation; user runs it once per model tier.

```bash
python tools/export_onnx.py \
    --ckpt /path/to/model_iter_NNNNNN.pt \
    --out  models/level5.onnx \
    --num-blocks 10 --num-channels 128 \
    --max-board-size 17 --num-planes 5
```

Steps:
1. Build `KataGoNet` via `nets.build_model(NetConfig(num_blocks, num_channels,
   board_size=max_board_size, num_planes))`.
2. Load checkpoint; prefer `swa_model_state_dict` (strip `module.` prefix);
   fall back to `model_state_dict`. Call `model.set_norm_scales()` (V5 trap 3 —
   without it WDL logits magnitude blows up ~10×).
3. `model.eval()`. Wrap in `ExportWrapper` whose `forward(spatial, global)`
   returns ONLY the 4 fields the UI needs (drops `value_td`, `intermediate_*`):
   - `policy_logits` (1, 4, 17·17)
   - `value_wdl_logits` (1, 3)
   - `value_futurepos_pretanh` (1, 2, 17, 17)
4. `torch.onnx.export` with `opset=15`, `dynamic_axes={"input_spatial":{0:"B"},
   "input_global":{0:"B"}, ...}`. Spatial dims fixed at `MAX_BOARD_SIZE` (17).

`onnx.checker.check_model` after write.

The script imports from V5 (`from nets import build_model`,
`from model_config import NetConfig`); we add `sys.path.insert` to
`SkyZero_V5/python/` so the script works regardless of CWD.

## Manifest & Model Curation

User manually picks which checkpoints map to the 5 tiers, exports each, and
edits `models/manifest.json`:

```json
{
  "default": "lv3",
  "models": [
    {"id":"lv1", "label":"新手", "elo":   0, "file":"level1.onnx", "params":"b4c64"},
    {"id":"lv2", "label":"入门", "elo": 300, "file":"level2.onnx", "params":"b10c128"},
    {"id":"lv3", "label":"进阶", "elo": 800, "file":"level3.onnx", "params":"b10c128"},
    {"id":"lv4", "label":"高手", "elo":1400, "file":"level4.onnx", "params":"b10c128"},
    {"id":"lv5", "label":"大师", "elo":2100, "file":"level5.onnx", "params":"b10c128"}
  ]
}
```

UI dropdown formats each entry as `Lv5 大师 · ELO +2100` (label + signed ELO).

The labels and ELO numbers are user-curated — there's no automation to
extract them from V5's `elo.py` output, since the user said they'd hand-pick.

## Deployment

**Cloudflare Pages** — git-connected to `SkyZeroWeb/`, no build step.

`_headers`:
```
/*.onnx
  Cache-Control: public, max-age=31536000, immutable
/models/manifest.json
  Cache-Control: public, max-age=300
/*
  Cache-Control: public, max-age=3600
```

**Capacity check**: each ONNX ≤ 25 MB single-file limit (b10c128 ≈ 4 MB ✓);
total deployment ≈ 5 × 4 MB + ~1 MB static = 22 MB, well below practical
limits. Bandwidth unmetered on free tier.

**Local dev**: `python3 -m http.server 8000` from `SkyZeroWeb/`. Required —
Worker `importScripts` and `fetch('models/...')` won't work over `file://`.

**Add-a-model flow** (documented in README):
1. Run `tools/export_onnx.py` with the chosen checkpoint.
2. Copy `.onnx` into `models/`.
3. Edit `models/manifest.json`.
4. `git push` → Cloudflare auto-deploys.

## Browser Compatibility

- ONNX Runtime Web v1.17.0 (proven in old SkyZeroWeb).
- WASM execution provider, single-threaded — avoids cross-origin
  `SharedArrayBuffer` headaches.
- iOS Safari: known to occasionally force-reload the page; carry over the
  iPhone-warning banner from old SkyZeroWeb (dismissible, remembered in
  `localStorage`).

## Testing

Manual checklist (no formal test runner — small enough to verify by hand):
- [ ] Each board size 13/14/15/16/17 plays through to a Renju win
- [ ] Renju forbidden moves (3-3, 4-4, overline) are illegal for black
- [ ] White wins on 5+ in any direction
- [ ] Theme toggle persists across reload
- [ ] Heat modal opens/closes on each of 6 heatmaps
- [ ] Gumbel phase overlay shows 16/8/4/2/1 colored circles after a search
- [ ] Undo mid-search aborts cleanly (no stale heatmap update)
- [ ] Model swap mid-game aborts and restarts (game state preserved)
- [ ] Value chart accumulates one point per ply (root + nn lines)
- [ ] First model load shows download progress bar

Optional Node test for `mcts.js` / `gomoku.js`: assert known Renju forbidden
positions, assert MCTS visit distribution stable on a fixed-seed Gumbel run.
Skipped from initial scope — defer until something regresses.

## Open Questions Deferred to Implementation

- Exact `c_p1/c_g1/c_v1/c_v2` derivation in `export_onnx.py` — `NetConfig`
  auto-derives from `num_channels`; should match the trained checkpoint
  without overrides as long as it was trained from a vanilla `run.cfg`.
- Worker-side memory growth across long games — JS GC should handle it via
  tree-reuse pruning (`root.parent = null` on promotion); revisit if a
  long game leaks.

## File-Size Estimates

| File | Approx |
|---|---|
| `index.html` | ~8 KB (HTML structure only, styles split out) |
| `style.css` | ~25 KB (porting `play_web.py`'s `<style>`) |
| `main.js` | ~50 KB (porting `play_web.py`'s `<script>`) |
| `worker.js` | ~10 KB |
| `mcts.js` | ~12 KB |
| `gomoku.js` | ~15 KB (with FPF) |
| Each `levelN.onnx` | ~4 MB (b10c128) |
| Total static (excluding models) | ~120 KB |
| Total deploy | ~22 MB |
