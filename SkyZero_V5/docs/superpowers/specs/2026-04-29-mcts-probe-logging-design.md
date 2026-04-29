# MCTS Probe Logging & Plotting — Design

**Date:** 2026-04-29

**Goal:** persist the per-iter `mcts_probe` result to a TSV and render two
diagnostic curves (first-stone distance from board center, and root-value
W−L) over training iters.

## Motivation

`cpp/mcts_probe_main.cpp` already runs once per training iter (post-export, in
`scripts/run.sh` step 5b) and prints `v_mix` (search root value) and
`nn_value_probs` (raw NN value) for an empty Gomoku board. The output goes to
stdout only — there is no on-disk log and nothing is plotted. We want to:

1. Track how confident the trained net is on the empty board over training
   (`v_mix` and `nn_value` as `W − L` scalars), to spot collapses or drift.
2. Track where black would actually open: as training progresses we expect
   the chosen first move to converge from "anywhere" to a small set of
   sensible center-area points. The Chebyshev distance from the board center
   captures that as a single integer per iter (0 = center, larger = farther
   out).

Both signals already exist inside `MCTSSearchOutput` — we only need to log and
plot them.

## Scope

- Add an optional `--log PATH` and `--iter N` CLI to `cpp/mcts_probe_main.cpp`.
  When `--log` is given, append one TSV row per probe to that path
  (auto-creating the file with a header on first write).
- Wire `scripts/run.sh` step 5b to pass `--iter "$iter" --log
  "$DATA_DIR/logs/probe.tsv"`.
- Extend `python/view_loss.py` with a `_plot_probe(data_dir, plt)` helper that
  is invoked from the existing `--plot` path (alongside `_plot_selfplay`) and
  writes `data/logs/probe.png`.

Out of scope:

- Adding probe data into `last_run.tsv` (different signal, different
  producer).
- Running additional probes per game inside selfplay.
- Plotting the full WDL triplet — chart is W−L only, per the user's request.
- Live / streaming plots — `view_loss.py` regenerates the PNG once per iter.

## Data Format

New file: `data/logs/probe.tsv`. Tab-separated, line-oriented, append-only,
written by `mcts_probe`.

Header (one line, written iff the file does not yet exist):

```
iter	gumbel_action	lcb_action	gumbel_dist	lcb_dist	vmix_W	vmix_D	vmix_L	nn_W	nn_D	nn_L
```

Per row:

- `iter` — integer, passed in via `--iter`.
- `gumbel_action`, `lcb_action` — flat board indices in `[0, board_size² - 1]`,
  taken straight from `MCTSSearchOutput::gumbel_action` /
  `MCTSSearchOutput::lcb_action`. `lcb_action` may equal `gumbel_action` when
  no child has `n ≥ 2` (existing fallback in the search).
- `gumbel_dist`, `lcb_dist` — Chebyshev distance from the board center,
  computed in C++ as `max(|r − c0|, |c − c0|)` where
  `r = a / board_size`, `c = a % board_size`, `c0 = board_size / 2`. Integer.
  For the default 15×15 board, `c0 = 7` and the value range is `[0, 7]`.
- `vmix_W`, `vmix_D`, `vmix_L` — `MCTSSearchOutput::v_mix` (the existing
  WDL triplet currently printed to stdout). Floats, 4-decimal precision (same
  formatting as the existing stdout print).
- `nn_W`, `nn_D`, `nn_L` — `MCTSSearchOutput::nn_value_probs` (the raw NN WDL
  the probe already prints).

The plot reduces these to W − L scalars; the full WDL is logged anyway so we
can re-derive things like draw rate later without re-running probes.

### Atomicity

`scripts/run.sh` is strictly serial (selfplay → shuffle → train → export →
probe → plot). `mcts_probe` is the only writer of `probe.tsv`, so a single
`std::ofstream` opened in `std::ios::app` mode with `<<` for the row plus a
flush is sufficient. No locking, no rotation.

If a probe invocation fails before the row is written, the file stays
consistent (one fewer row); the existing `|| echo "...failed (non-fatal)"`
handler in `run.sh` keeps the training loop running.

## C++ Changes — `cpp/mcts_probe_main.cpp`

1. Extend `CliArgs`:

   ```cpp
   struct CliArgs {
       std::string model;
       std::string config;
       int num_simulations_override = -1;
       int iter = -1;            // new
       std::string log_path;     // new
   };
   ```

2. Extend `parse_cli` with `--iter` (int, required iff `--log` is given) and
   `--log` (string, optional). Update the usage string.

3. After the existing `[mcts_probe] v_mix ...` / `[mcts_probe] nn_value ...`
   stdout prints, if `cli.log_path` is non-empty:

   - Compute `c0 = cfg.board_size / 2`.
   - For each of `sr.gumbel_action` and `sr.lcb_action`, derive the
     Chebyshev distance from `c0`.
   - Open the file in append mode. If `tellp() == 0` after open (i.e. brand
     new file), write the header line first.
   - Write one row using the same `std::fixed << std::setprecision(4)` for
     the WDL fields and plain ints for actions / distances. Tab-separated,
     trailing newline.

   This block lives in the same `try` so any I/O error is caught by the
   existing `catch` and turns into a non-zero exit, which `run.sh` already
   tolerates.

## Shell Changes — `scripts/run.sh` step 5b

Replace:

```bash
"$ROOT/cpp/build/mcts_probe" \
    --model "$DATA_DIR/models/latest.pt" \
    --config "$SCRIPT_DIR/run.cfg" \
    || echo "[run.sh] mcts_probe failed (non-fatal)"
```

with:

```bash
"$ROOT/cpp/build/mcts_probe" \
    --model "$DATA_DIR/models/latest.pt" \
    --config "$SCRIPT_DIR/run.cfg" \
    --iter "$iter" \
    --log "$DATA_DIR/logs/probe.tsv" \
    || echo "[run.sh] mcts_probe failed (non-fatal)"
```

No build-system changes — `mcts_probe` already builds via the existing
CMake target.

## Python Changes — `python/view_loss.py`

Add `_plot_probe(data_dir, plt)`, modeled on the existing
`_plot_selfplay`:

- Read `data/logs/probe.tsv` via the existing `_read_tsv` helper. Bail
  silently with a stderr note if the file is missing or empty.
- Required columns: `iter`, `gumbel_dist`, `lcb_dist`, `vmix_W`, `vmix_L`,
  `nn_W`, `nn_L`. If any is missing, log a "skipping probe plot" warning
  and return (matches existing pattern).
- Build a single figure with two stacked subplots, `sharex=True`,
  `figsize=(8, 6)`:

  - **Top**: first-stone Chebyshev distance vs iter.
    - Two lines, **distinct colors**:
      - `gumbel_dist` — color A, label `"gumbel"`
      - `lcb_dist` — color B, label `"lcb"`
    - `set_ylabel("dist from center")`, integer y-ticks `0..7`,
      `grid(alpha=0.3)`, `legend(loc="best")`.
  - **Bottom**: root value W − L vs iter.
    - Compute `vmix_wl = vmix_W − vmix_L`, `nn_wl = nn_W − nn_L`.
    - Two lines, **distinct colors**:
      - `vmix_wl` — color C, label `"v_mix W-L"`
      - `nn_wl` — color D, label `"nn_value W-L"`
    - `set_ylim(-1.0, 1.0)`, `axhline(0, ls="--", alpha=0.4)`,
      `set_xlabel("iter")`, `grid(alpha=0.3)`,
      `legend(loc="best")`.

  Pick four matplotlib default-cycle colors (e.g. C0/C1 for the top pair,
  C2/C3 for the bottom pair) so all four lines are visually distinct from
  each other; the user explicitly asked for different colors per pair.

- `fig.tight_layout()`; save to `data/logs/probe.png` at `dpi=200` and
  `print(f"saved plot to {out}")`. Same pattern as `_plot_selfplay`.

Wire it in `main()` next to the existing `_plot_selfplay(data_dir, plt)`
call inside the `if args.plot:` block.

## Testing & Verification

- Build: `cmake --build cpp/build --target mcts_probe`.
- Smoke run: invoke `cpp/build/mcts_probe --model
  data/models/latest.pt --config scripts/run.cfg --iter 0 --log
  /tmp/probe_test.tsv` against an existing checkpoint. Verify the file gets
  the header on first invocation and exactly one row per call afterwards.
- Spot-check a row: `gumbel_dist` and `lcb_dist` should be in `[0, 7]`;
  `vmix_W + vmix_D + vmix_L ≈ 1.0`; `nn_W + nn_D + nn_L ≈ 1.0`.
- Plot: with at least 2 rows in the TSV, run `python python/view_loss.py
  --data-dir data --plot` and confirm `data/logs/probe.png` is created with
  the expected two-pane layout.
- End-to-end: a single `bash scripts/run.sh 1` should produce the row, the
  PNG, and exit cleanly.

## Risks & Mitigations

- **Header drift** if columns get added later: tolerated — `_plot_probe`
  looks up columns by name (existing `_read_tsv` returns header + rows),
  and unknown columns are simply ignored.
- **Partial / failed probe runs** leave gaps in the TSV. Plot rendering
  uses per-row `float()` conversions, so any malformed row is skipped via
  the existing `nan` fallback path in `_read_tsv` consumers.
- **Backwards compatibility**: `--log` and `--iter` are optional from a CLI
  contract perspective (probe still works without them); only `run.sh`
  passes them. Anyone invoking `mcts_probe` by hand for ad-hoc inspection
  is unaffected.