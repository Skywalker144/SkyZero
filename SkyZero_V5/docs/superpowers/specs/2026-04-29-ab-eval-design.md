# AB Hyperparameter Evaluation — Design

**Date:** 2026-04-29

**Goal:** evaluate the same model checkpoint(s) under two different MCTS / inference
hyperparameter configurations by running head-to-head matches, then summarise the
results as text and a plot of Elo difference vs. checkpoint.

## Motivation

`scripts/elo.sh` + `cpp/build/gomoku_elo` already pit two **models** against each
other under a **single** shared search config. That binary cannot answer the
question "given this trained model, does config A beat config B?", because both
sides of the match read MCTS knobs from the same `--config` file. We want a
sibling tool — `scripts/ab.sh` + `cpp/build/gomoku_ab` — dedicated to this
A/B comparison so we can:

- pick `NUM_SIMULATIONS`, `C_PUCT*`, symmetry / stochastic transform settings,
  `LCB_K`, FPU knobs, etc., with empirical evidence rather than guesses;
- track how the A-vs-B winrate evolves as the same trial is repeated across
  multiple checkpoints (a list of `.pt` files), to see if the answer is stable
  over training.

## Scope

- One run = one `.pt` model (or a list of them) playing itself, where one side
  uses search config `a.cfg` and the other side uses `b.cfg`.
- Match-level orchestration (model list, total games, concurrency, inference
  batching, board topology) lives in a third file `ab.cfg`.
- Pure head-to-head; no anchor / Elo-curve integration. `data/elo/games.jsonl`
  is **not** touched.
- Single GPU, single host. No distributed runs.
- No live plotting; the plot is regenerated at the end of every `scripts/ab.sh`
  invocation from the cumulative jsonl (so partial runs still produce a chart).

Out of scope:

- Comparing more than two configs in one run (no N-way tournament).
- Sweeping configs automatically (e.g. grid search over `NUM_SIMULATIONS`); a
  user wanting that runs `scripts/ab.sh` multiple times with different cfg
  pairs and different `OUT_NAME` values.
- Mixing different rules / board sizes within a single run.

## Layout

```
scripts/ab/
  ├── ab.cfg                  # match-level config
  ├── a.cfg                   # A-side MCTS / search hyperparams
  └── b.cfg                   # B-side MCTS / search hyperparams
scripts/ab.sh                 # shell runner (mirrors scripts/elo.sh)
cpp/gomoku_ab_main.cpp        # C++ main (forked from gomoku_elo_main.cpp)
cpp/build/gomoku_ab           # built binary (added to cpp/CMakeLists.txt)
python/ab.py                  # text report + Elo-diff plot
data/ab/<OUT_NAME>.jsonl      # cumulative per-game results
data/ab/<OUT_NAME>.png        # Elo-diff curve
```

## Configuration split

The cardinal rule: **anything that must be identical for both sides goes in
`ab.cfg`; anything per-side goes in `a.cfg` / `b.cfg`**.

### `ab.cfg`

| Key | Default | Notes |
|---|---|---|
| `MODELS` | `latest` | Space- or comma-separated. Accepts `latest`, `model_iter_NNNNNN.pt`, or absolute paths. Unset / empty defaults to `latest`. |
| `NUM_GAMES` | `200` | Per-model target. AB tests typically need more samples than anchor Elo to detect ~50/50 winrates with tight CIs. |
| `NUM_CONCURRENT_GAMES` | `20` | Worker threads inside a single `gomoku_ab` process. Each worker owns one `mcts_a` + one `mcts_b`. |
| `INFERENCE_BATCH_SIZE` | `64` | Shared inference server batch cap. |
| `INFERENCE_WAIT_US` | `500` | Server fill-up wait. |
| `BOARD_SIZE` | `15` | Must match the model. |
| `NUM_PLANES` | `5` | Must match the model (V5 padded encoding). |
| `RULE` | `renju` | Must match the model. |
| `OUT_NAME` | `ab` | Output basename: `data/ab/<OUT_NAME>.jsonl` and `.png`. Use distinct names to keep separate experiments from polluting each other. |

### `a.cfg` / `b.cfg`

Same MCTS / search keys as the existing `scripts/elo.cfg`, but **per side**:

- Search budget: `NUM_SIMULATIONS`, `SEARCH_THREADS_PER_TREE`.
- Gumbel: `GUMBEL_M`, `GUMBEL_C_VISIT`, `GUMBEL_C_SCALE` (`GUMBEL_NOISE_ENABLED`
  is forced false in C++ regardless of the file, matching `gomoku_elo`).
- PUCT family: `C_PUCT`, `C_PUCT_LOG`, `C_PUCT_BASE`, `CPUCT_UTILITY_STDEV_PRIOR`,
  `CPUCT_UTILITY_STDEV_PRIOR_WEIGHT`, `CPUCT_UTILITY_STDEV_SCALE`.
- FPU: `FPU_REDUCTION_MAX`, `ROOT_FPU_REDUCTION_MAX`, `FPU_POW`, `FPU_LOSS_PROP`.
- Final-move selection: `LCB_K`.
- Inference symmetry / transforms: `ENABLE_STOCHASTIC_TRANSFORM_ROOT`,
  `ENABLE_STOCHASTIC_TRANSFORM_CHILD`, `ENABLE_SYMMETRY_ROOT`,
  `ENABLE_SYMMETRY_CHILD`, `ROOT_SYMMETRY_PRUNING`.
- Misc: `HALF_LIFE`.

The starter `a.cfg` / `b.cfg` shipped in the repo will both be copies of the
MCTS section of `scripts/elo.cfg`, so the default run is effectively a no-op
A/B (winrate ≈ 0.5). Users edit `b.cfg` to change whichever knob they want
to A/B test against.

## C++ binary: `gomoku_ab`

Forked from `cpp/gomoku_elo_main.cpp`. Same skeleton (worker pool,
`BatchedInferenceServer`, jsonl append) with three meaningful differences:

### 1. CLI

```
gomoku_ab \
  --model PATH                  # single model file
  --config-ab scripts/ab/ab.cfg
  --config-a  scripts/ab/a.cfg
  --config-b  scripts/ab/b.cfg
  --output    data/ab/<OUT_NAME>.jsonl
  --num-games N
  [--seed S]
```

No `--num-simulations` override — that override exists in `gomoku_elo` for
quick batch tweaks where both sides scale together; in AB the whole point is
that the two sides are independent, so use the cfg files.

### 2. Two `AlphaZeroConfig` instances

Parse `a.cfg` and `b.cfg` separately into `cfg_a` and `cfg_b`. Each worker
constructs `TreeParallelMCTS<Gomoku> mcts_a(game, cfg_a, threads_a, ...)` and
`mcts_b(game, cfg_b, threads_b, ...)`. `SEARCH_THREADS_PER_TREE` is read from
each side's cfg independently (defaulting to 8 if missing).

### 3. Single shared `BatchedInferenceServer`

Both models are the same `.pt`, so we load the model **once** and run a
**single** server. Both `mcts_a` and `mcts_b` submit to it. This is the only
real divergence from `gomoku_elo`'s scaffolding — there's only one `ModelHandle`,
one server, and the `infer` / `fwd` lambdas are shared between both MCTS
instances. Halves GPU memory vs. naively loading the model twice and improves
batch fill-rate.

### 4. Output schema

One JSON line per finished game, appended under a write mutex:

```json
{"model":"data/models/model_iter_000300.pt",
 "cfg_a":"scripts/ab/a.cfg",
 "cfg_b":"scripts/ab/b.cfg",
 "a_black":true,
 "winner_a":1,
 "plies":42}
```

`model`, `cfg_a`, `cfg_b` are stored as the **paths the binary was invoked with**
(not absolute / canonical), so dedup (in shell) is purely string equality.

`winner_a` ∈ {-1, 0, 1}, with 0 = draw. Color alternation: even game index → A
plays black, odd → B plays black, identical to `gomoku_elo`.

`gumbel_noise_enabled` is force-disabled in C++ regardless of the cfg files.

## Shell runner: `scripts/ab.sh`

Mirrors `scripts/elo.sh`. Steps:

1. Resolve paths (`SCRIPT_DIR`, `ROOT`, `DATA_DIR`, `AB_BIN`, default cfg paths).
2. Verify `gomoku_ab` is built; verify all three cfg files exist.
3. Read `MODELS`, `NUM_GAMES`, `OUT_NAME` from `ab.cfg` via the same
   `cfg_get` helper as `elo.sh` (last-occurrence-wins, strips comments and
   quotes). Env vars of the same name override.
4. CLI arg `$1`, if present, overrides `MODELS`:
   - `latest` → `data/models/latest.pt`
   - `model_iter_NNNNNN.pt` → `data/models/model_iter_NNNNNN.pt`
   - `/abs/path/...` → as-is
5. For each model in the resolved list:
   - `count_existing` greps `OUT_FILE` for lines containing
     `"model":"<path>","cfg_a":"<a>","cfg_b":"<b>"` → integer count.
   - If `count >= NUM_GAMES`, skip and log.
   - Else run `gomoku_ab` with `--num-games (NUM_GAMES - count)`.
6. After all models done, run `python python/ab.py --games <jsonl> --plot
   data/ab/<OUT_NAME>.png`.

`scripts/ab.sh` does **not** stride / sweep checkpoints automatically (unlike
`elo.sh`'s batch mode). Users explicitly list the checkpoints they want in
`MODELS`. Rationale: AB experiments are usually targeted ("does this knob
matter for the latest model?"), so an implicit stride sweep adds confusion
without saving typing.

## Python report: `python/ab.py`

Reads `--games <jsonl>` and emits two outputs:

### Text report (stdout)

Per-model rows + a pooled summary:

```
model                          games   W-D-L         score    elo_diff (95% CI)
model_iter_000100.pt              200   89-22-89      0.500    +0   ( -50..+50 )
model_iter_000200.pt              200   105-10-85     0.600    +70  ( +18..+125)
model_iter_000300.pt              200   118-8-74      0.620    +85  ( +33..+143)
─────────────────────────────────────────────────────────────────────────────
pooled                            600   312-40-248    0.553    +37  ( +14..+62 )

by side (pooled): A-as-black 0.58 (185-12-103) | A-as-white 0.51 (127-28-145)
mean plies: 38.4
```

Score, Elo, and CI definitions:

- `score = (a_wins + 0.5 * draws) / total`
- `elo_diff = -400 * log10(1/score - 1)` for `0 < score < 1`; clamp to
  `±800` when `score ∈ {0, 1}` (avoids `inf`, signals a one-sided result).
- 95% CI: Wilson score interval on `score`, then transform both endpoints
  through the same Elo formula. Robust at small sample sizes and at the
  boundaries.

Pooled rows aggregate counts across all models (and use the same Elo
formula). The "by side" line splits on `a_black` to expose any
first-move advantage asymmetry.

### Plot (PNG)

- `--plot data/ab/<OUT_NAME>.png`.
- One point per model. Y = Elo diff (A − B); positive = A stronger.
- Error bars from the Wilson CI endpoints.
- X order: parse iteration number from `model_iter_NNNNNN.pt` if it matches;
  otherwise alphabetical. X-tick labels rotated 45° to avoid overlap.
- Horizontal dashed gray line at y=0 as the "tie" reference.
- Title includes both cfg basenames (e.g. `a.cfg vs b.cfg`) and total games.

CLI:

```
python python/ab.py --games data/ab/ab.jsonl --plot data/ab/ab.png
```

If `--plot` is omitted, only the text report is printed.

## Resume semantics

The shell runner counts existing `(model, cfg_a, cfg_b)` rows in the jsonl and
only fills in the gap. Three consequences worth flagging:

- Re-running with the same cfg files is idempotent; the script is a no-op
  if every model is already at `NUM_GAMES`.
- Editing `a.cfg` or `b.cfg` after a partial run **invalidates** the prior
  data only if it changes the file path used in `--config-a` / `--config-b`.
  If you edit `b.cfg` in place, prior rows still match `"cfg_b":"scripts/ab/b.cfg"`
  and will be reused — silently mixing two settings. **The runner therefore
  warns once per run** if any matching rows exist, suggesting either a fresh
  `OUT_NAME` or wiping the jsonl.
- `python/ab.py` does not enforce config integrity; it trusts whatever rows
  the jsonl contains.

## Determinism / fairness

- Color alternation removes first-move bias: across an even `NUM_GAMES`,
  each side plays black exactly half the time.
- Each game starts from a freshly randomized opening (`game.get_initial_state`),
  same RNG path as `gomoku_elo` (`seed + 0x9E3779B97F4A7C15 * (tid+1)`).
- Gumbel noise off; LCB-or-Gumbel argmax for final move selection (mirrors
  `gomoku_elo`).
- Both sides share the same model file → no model-load asymmetry.
- Both sides share the same inference server → no batching / latency
  asymmetry that could systematically favour one side.

## Build wiring

`cpp/CMakeLists.txt`: add `gomoku_ab` target alongside `gomoku_elo` and
`gomoku_play`. Same link deps (libtorch, alphazero headers, gomoku env).
Build command:

```
cmake --build cpp/build --target gomoku_ab
```

`scripts/ab.sh` exits with a clear "build first" message if the binary is
missing, identical to `elo.sh`'s pattern.

## Non-goals (re-stated)

- No anchor-based Elo. Use `scripts/elo.sh` for that.
- No automatic checkpoint stride sweep.
- No live plotting / progress bar beyond stderr lines from the C++ worker.
- No support for more than two configs per run.

## Default starter content

To make the first run smoke-test trivially:

- `scripts/ab/ab.cfg` — `MODELS=latest`, `NUM_GAMES=200`, otherwise mirrors the
  inference / topology section of `scripts/elo.cfg`.
- `scripts/ab/a.cfg` — copy of MCTS section of `scripts/elo.cfg`.
- `scripts/ab/b.cfg` — same, except `NUM_SIMULATIONS=128` (a visible
  perturbation so the smoke test produces a non-flat plot).
