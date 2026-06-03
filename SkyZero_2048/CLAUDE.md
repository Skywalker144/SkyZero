# SkyZero_2048 — Project Guide for Claude

**Stochastic Gumbel AlphaZero for 2048** — a single-agent, *stochastic* MDP fork
of the SkyZero AlphaZero framework. 2048 is not a two-player zero-sum game, so
the usual AlphaZero machinery (deterministic transitions, WDL value, perspective
flip) is replaced by an **afterstate** formulation with **chance nodes** and a
**scalar expected-discounted-score** value.

This directory was forked from `SkyZero_V7.1/` (the Gomoku/Renju framework) and
then specialized for 2048; the Gomoku code (Python pipeline, gomoku C++ env/MCTS,
run/elo/bench scripts, configs/) has been removed. See `../SkyZero_V7.1/` if you
need the original framework for reference.

---

## Layout: V7.1-style `python/` + `scripts/` + `cpp/` (no `az2048/` package)

The orchestration mirrors the inherited SkyZero V7.1 framework exactly — flat
`python/*.py` modules run as `python X.py` from `cwd=python/` (NOT a package),
driven by `scripts/run.sh` + `scripts/internal/*.sh`. Torch lives in the
`pytorch` conda env (`scripts/env_paths.cfg.local` → `PY`).

| file | role |
|---|---|
| `python/game.py` | 2048 env (afterstate model): `apply_move`→(afterstate,reward,changed), `spawn_distribution`, `encode_state` (16 one-hot exponent planes), legal/terminal. *(pure-Python debug path)* |
| `python/mcts.py` | afterstate **Gumbel** MCTS. DECISION vs CHANCE nodes; PUCT w/ MuZero min-max Q norm; chance descent = deterministic "most under-represented vs known spawn prob"; backup `G=r+γ·G_child`, **no flip**. *(debug path; the production search is `cpp/skyzero_2048.h`)* |
| `python/nets.py` | compact residual conv net (4×4, full res), heads: policy(4) + **scalar** value (softplus, ≥0). `build_net(cfg)`. |
| `python/model_config.py` | `Config` dataclass + `config_from_name("b6c96")` / `config_from_env()` (reads run.cfg env vars; widths auto-derive per net). |
| `python/train.py` | per-net CLI (`--data-dir --network --iter`, env `TRAIN_STEPS_PER_EPOCH`): soft-target policy CE + Huber value (target/`value_scale`) on the shuffle window → `nets/<net>/{model_latest.pt,train.tsv,state.json}`. |
| `python/export_model.py` | per-net `model_latest.pt` → TorchScript `nets/<net>/latest.pt` (+meta+`scripted_iter_*.pt`). |
| `python/init_model.py` | per-net random-init bootstrap. `python/schedule.py` | multi-network active-net resolver (NETWORKS + SELFPLAY_SCHEDULE). `python/warmup.py` | NUM_SIMULATIONS resolver (falls back to fixed `SIMS`). |
| `python/shuffle.py`,`bucket.py`,`data_processing.py` | power-law-window shuffle, token-bucket train cadence, npz I/O (faithful V7.1 ports). |
| `python/augment.py` | D4 augmentation that **also permutes the 4 actions** (perms derived by brute force + equivariance assert — the known trap). |
| `python/evaluate.py` | single-agent metrics CLI: avg score + tile reach-rates → `logs/eval.tsv` (no Elo). |
| `python/view_loss.py` | per-net loss + self-play + eval + probe PNGs. `python/loop.py`,`selfplay.py`,`selfplay_parallel.py` | pure-Python debug loop (`python loop.py`; slower, no C++). |
| `python/play_web.py` + `python/web/play2048.html` | live web demo (loads TorchScript `models/latest.pt`). |

Value semantics: value head regresses `h(target)/value_scale`, where `h` is the
optional invertible MuZero transform `h(x)=sign(x)(√(|x|+1)−1)+εx` (`VALUE_TRANSFORM=1`;
plain linear `target/value_scale` when `0`). `net_evaluator` applies `h⁻¹` and
multiplies back so MCTS Q = reward + γ·V stays in raw 2048 points. Two live
experiment data dirs: **`data2048_td`** (linear value-target baseline) and
**`data2048_vt`** (h()-transform). NEVER share a shuffle window or checkpoint
between them — the value-head output semantics differ.

---

## Entry points

| command | does |
|---|---|
| `CONFIG_DIR=configs/baseline bash scripts/run.sh [max_iters]` | **main entry** — V7.1-style bash loop: schedule→mirror→selfplay→shuffle→bucket→train×nets→export×nets→probe→eval→view_loss. Auto-builds + auto-resumes (per-net `state.json`). |
| `cd python && python loop.py --iters N --games G` | pure-Python self-play loop (slower, simpler, for debugging) |
| `CONFIG_DIR=configs/vtransform bash scripts/play_web.sh` | web demo — resolves `DATA_DIR` from the config like run.sh, loads `$DATA_DIR/models/latest.pt` (stub fallback if missing). `CKPT=` still overrides. |

`scripts/run.sh` sources `$CONFIG_DIR/{run.cfg,paths.cfg}` (+ `run.cfg.local`)
and exports every hyperparameter to the env, which the `internal/*.sh` steps and
`python/` scripts read (`model_config.config_from_env`). New experiment = copy
`configs/baseline/` and edit `run.cfg`. Per-machine tweaks → `run.cfg.local` /
`paths.cfg.local` (gitignored).

Multi-network: `NETWORKS="b6c96, b10c128"` + `SELFPLAY_SCHEDULE="0, 2e7"` — all
nets train each iter on the same shuffle; selfplay/active = largest threshold ≤
cumulative rows. Multi-GPU: `GPU_NUM>1` auto-launches `internal/selfplay_daemon.sh`
(one daemon process per spare GPU; hot-reloads `models/latest.pt`).

Data layout (`<DATA_DIR>/`, gitignored): `models/latest.pt` (active TorchScript
mirror), `nets/<net>/{model_latest.pt (state_dict), latest.pt (TS), state.json,
train.tsv}`, `selfplay/*.npz`, `shuffled/current/*.npz`,
`logs/{selfplay,selfplay_stats,schedule,eval,probe}.tsv + *.png`.

---

## C++ high-throughput self-play (~26× the Python loop)

The SkyZero-style C++ path is built and working — same architecture as the
inherited Gomoku framework (C++ LibTorch self-play + MCTS, Python training,
TorchScript bridge, npz data flow), with the game-specific afterstate Gumbel MCTS.

- `cpp/envs/game2048.h` — env (unit test `envs/game2048_test.cpp`, plain g++).
- `cpp/skyzero_2048.h` — afterstate **Gumbel** MCTS. Synchronous `search()` AND a
  deferred-eval stepping API (`begin`/`apply_root_eval`/`select_leaf`/`apply_leaf`/
  `result`) for batched play. Unit test `skyzero_2048_test.cpp` (plain g++, no torch).
- `cpp/infer_server_2048.h` — central batched inference server (N threads, one queue).
- `cpp/selfplay2048_par_main.cpp` — **the** self-play binary: `--threads` workers ×
  `--slot-games` lockstep games each, feeding the server; writes npz
  (state(N,16)/policy(N,4)/value(N,1)) that `train.py` reads directly.
- `cpp/selfplay2048_main.cpp` — single-thread bridge/eval binary (`--probe`, `--games`).

Build: `bash scripts/build.sh --target selfplay2048_par` (when ADDING a target,
reconfigure first: `source scripts/env_paths.cfg && cmake -S cpp -B cpp/build
-DCMAKE_PREFIX_PATH=$LIBTORCH -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=$NVCC`).

The C++ self-play is driven by `scripts/run.sh` → `scripts/internal/selfplay.sh`,
which runs the `selfplay2048_par_main` binary with the run.cfg knobs: `THREADS` ×
`SLOT_GAMES` lockstep games per worker, `SERVER_THREADS` inference threads,
`BATCH`/`WAIT_US` batch tuning, `GAMES_PER_ITER` games/iter. The 4×4 net is tiny so
the GPU sits ~20% regardless — games/s is the metric, not GPU%. Batches only fill
when total games >> THREADS×SLOT_GAMES. CRITICAL: the server takes ENCODED input
(NUM_PLANES×16); always submit `encode_state(state)`, never the raw 16-cell board.

The pure-Python loop (`cd python && python loop.py`) still works and is simpler for
debugging. Config flows: `configs/<exp>/run.cfg` (bash KEY=VALUE) → `scripts/run.sh`
sources + exports to env → `internal/*.sh` steps + `python/` scripts read it via
`model_config.config_from_env` (NO `az2048` package, NO `loop_cpp`/`config.py`).

Validate the C++ core:
```
cd cpp && g++ -std=c++17 -I . envs/game2048_test.cpp -o /tmp/t && /tmp/t
        g++ -std=c++17 -I . skyzero_2048_test.cpp  -o /tmp/m && /tmp/m
```

---

## Invariants / gotchas

- **Single-agent**: no `to_play`, no `flip_wdl`. Backup is plain discounted sum.
- **Stochastic**: one action → a chance node → a distribution of next states.
  Spawn probs are KNOWN (90% tile-2 / 10% tile-4, uniform empty cell) → enumerate,
  don't learn dynamics. This is why it's afterstate-AlphaZero, not MuZero.
- **D4 augmentation must relabel actions** (`augment.py`). Board-only D4 (rotating
  just the planes, as a naive Gomoku-style aug would) is WRONG for 2048.
- **Value scale**: returns reach tens of thousands → MuZero min-max norm in MCTS
  for PUCT; value head trained on `target/value_scale`.
- Torch lives in the `pytorch` conda env only (base python has none); always use
  `$PY` from `scripts/env_paths.cfg.local`.
