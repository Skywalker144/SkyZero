# MCTS Probe Logging & Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist `mcts_probe` per-iter results to `data/logs/probe.tsv` and render a two-pane diagnostic plot (first-stone Chebyshev distance from center, root-value W−L) at `data/logs/probe.png`.

**Architecture:** Extend `cpp/mcts_probe_main.cpp` with optional `--iter N` / `--log PATH` CLI; the binary appends one TSV row per probe (auto-writing a header on first call). `scripts/run.sh` step 5b passes both args. `python/view_loss.py` gains a `_plot_probe()` helper that renders the PNG alongside the existing `_plot_selfplay()` call inside `--plot`.

**Tech Stack:** C++17 (libtorch), Bash, Python 3 + matplotlib. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-mcts-probe-logging-design.md`

**Note on TDD:** This codebase has no Python or C++ test harness; the existing pattern (cf. `cpp/mcts_probe_main.cpp`, `python/view_loss.py`) is smoke-test by running the binary / script and inspecting stdout/file output. Each task below uses that pattern: a concrete command and the expected observable result.

---

## File Structure

**Files modified (3):**

- `cpp/mcts_probe_main.cpp` — adds two CLI flags, computes Chebyshev distance for the chosen actions, appends a TSV row.
- `scripts/run.sh` — step 5b passes `--iter` and `--log` to the probe.
- `python/view_loss.py` — adds `_plot_probe(data_dir, plt)` and calls it from `main()` next to `_plot_selfplay`.

**Files created at runtime (not by the patch):**

- `data/logs/probe.tsv` — created by mcts_probe on first run with `--log`.
- `data/logs/probe.png` — created by view_loss.py on `--plot`.

---

## Task 1: Add `--iter` and `--log` CLI parsing to mcts_probe

**Files:**
- Modify: `cpp/mcts_probe_main.cpp` (struct `CliArgs` ~line 78; `parse_cli` ~line 84; usage string ~line 98)

- [ ] **Step 1: Extend `CliArgs` struct**

In `cpp/mcts_probe_main.cpp`, replace the current `CliArgs` struct (~line 78–82):

```cpp
struct CliArgs {
    std::string model;
    std::string config;
    int num_simulations_override = -1;
};
```

with:

```cpp
struct CliArgs {
    std::string model;
    std::string config;
    int num_simulations_override = -1;
    int iter = -1;            // optional; logged into probe.tsv when --log is given
    std::string log_path;     // optional; if non-empty, append a TSV row here
};
```

- [ ] **Step 2: Extend `parse_cli` to recognize the new flags**

Inside `parse_cli`, the existing if/else chain after the `--num-simulations` branch ends with `else throw ...`. Insert two new branches before the `else throw`:

```cpp
        else if (k == "--iter") a.iter = std::stoi(need("--iter"));
        else if (k == "--log") a.log_path = need("--log");
```

So the relevant block becomes:

```cpp
        if (k == "--model") a.model = need("--model");
        else if (k == "--config") a.config = need("--config");
        else if (k == "--num-simulations") a.num_simulations_override = std::stoi(need("--num-simulations"));
        else if (k == "--iter") a.iter = std::stoi(need("--iter"));
        else if (k == "--log") a.log_path = need("--log");
        else throw std::runtime_error("unknown arg: " + k);
```

- [ ] **Step 3: Update the usage string**

Replace the existing usage error (~line 98):

```cpp
        throw std::runtime_error("usage: mcts_probe --model PATH --config PATH [--num-simulations N]");
```

with:

```cpp
        throw std::runtime_error("usage: mcts_probe --model PATH --config PATH [--num-simulations N] [--iter N] [--log PATH]");
```

- [ ] **Step 4: Build to confirm it still compiles**

Run: `cmake --build cpp/build --target mcts_probe -j`

Expected: builds cleanly, no warnings or errors related to the changes.

- [ ] **Step 5: Sanity-run with the new flags ignored (no `--log`)**

Run (only if a model exists at `data/models/latest.pt`; otherwise skip — Task 3 covers a smoke test):

`cpp/build/mcts_probe --model data/models/latest.pt --config scripts/run.cfg --iter 0`

Expected: the existing `[mcts_probe] v_mix ...` and `[mcts_probe] nn_value ...` lines print to stdout. No file is created (no `--log` passed). Exit code 0.

- [ ] **Step 6: Commit**

```bash
git add cpp/mcts_probe_main.cpp
git commit -m "Add --iter and --log CLI flags to mcts_probe (parsing only; no behavior change yet)"
```

---

## Task 2: Append a TSV row when `--log` is given

**Files:**
- Modify: `cpp/mcts_probe_main.cpp` (the block right after the existing `[mcts_probe] nn_value ...` print, before `return 0;` ~line 253)
- Modify: `cpp/mcts_probe_main.cpp` (top of file, `#include` block ~lines 5–18)

- [ ] **Step 1: Add `<filesystem>` to the includes**

In the existing `#include` block (between `<fstream>` and `<iomanip>` to keep alphabetical-ish order), add:

```cpp
#include <filesystem>
```

- [ ] **Step 2: Insert the row-writing block before `return 0;`**

In `main()`, immediately before `return 0;` (right after the existing `std::cout << "[mcts_probe] nn_value W=" ... << "\n";` line ~253), insert:

```cpp
        if (!cli.log_path.empty()) {
            const int board_size = cfg.board_size;
            const int c0 = board_size / 2;
            auto cheb_dist = [&](int action) -> int {
                if (action < 0) return -1;
                const int r = action / board_size;
                const int c = action % board_size;
                const int dr = std::abs(r - c0);
                const int dc = std::abs(c - c0);
                return dr > dc ? dr : dc;
            };
            const int gumbel_dist = cheb_dist(sr.gumbel_action);
            const int lcb_dist = cheb_dist(sr.lcb_action);

            const bool need_header = !std::filesystem::exists(cli.log_path);
            std::ofstream out(cli.log_path, std::ios::app);
            if (!out) {
                throw std::runtime_error("cannot open log: " + cli.log_path);
            }
            if (need_header) {
                out << "iter\tgumbel_action\tlcb_action\tgumbel_dist\tlcb_dist"
                    << "\tvmix_W\tvmix_D\tvmix_L\tnn_W\tnn_D\tnn_L\n";
            }
            out << std::fixed << std::setprecision(4);
            out << cli.iter
                << "\t" << sr.gumbel_action
                << "\t" << sr.lcb_action
                << "\t" << gumbel_dist
                << "\t" << lcb_dist
                << "\t" << sr.v_mix[0]
                << "\t" << sr.v_mix[1]
                << "\t" << sr.v_mix[2]
                << "\t" << sr.nn_value_probs[0]
                << "\t" << sr.nn_value_probs[1]
                << "\t" << sr.nn_value_probs[2]
                << "\n";
        }
```

Notes for the implementer:
- `<cstdlib>` for `std::abs` is already pulled in transitively via `<random>`/`<sstream>`; if your toolchain warns, add `#include <cstdlib>`.
- `sr` is the `MCTSSearchOutput` already in scope from `const auto sr = mcts.search(...)` ~line 242.
- `std::ios::app` opens for append + create-if-missing. The `filesystem::exists` check happens before the open, so the header decision is correct even though `app` mode would otherwise also create an empty file.

- [ ] **Step 3: Build**

Run: `cmake --build cpp/build --target mcts_probe -j`

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add cpp/mcts_probe_main.cpp
git commit -m "Append a per-probe TSV row (iter, gumbel/lcb actions and their Chebyshev distance from board center, full v_mix and nn_value WDL triplets) to the path passed via --log, writing the header on first run"
```

---

## Task 3: Smoke-test the writer end-to-end

This task verifies the TSV is created and rows append correctly. Skip the actual runs if `data/models/latest.pt` does not exist on this machine (the patch is still correct; full verification will happen on the next training iter).

- [ ] **Step 1: Clean any stale test file**

Run: `rm -f /tmp/probe_smoke.tsv`

Expected: no error.

- [ ] **Step 2: Run probe once with `--log`**

Run:

```bash
cpp/build/mcts_probe \
    --model data/models/latest.pt \
    --config scripts/run.cfg \
    --iter 0 \
    --log /tmp/probe_smoke.tsv
```

Expected: stdout shows the existing `[mcts_probe] simulations=...`, `v_mix W=... D=... L=... scalar=...`, and `nn_value W=... D=... L=...` lines. Exit code 0.

- [ ] **Step 3: Inspect the file — header + 1 row**

Run: `cat /tmp/probe_smoke.tsv`

Expected: exactly two lines.
- Line 1: `iter\tgumbel_action\tlcb_action\tgumbel_dist\tlcb_dist\tvmix_W\tvmix_D\tvmix_L\tnn_W\tnn_D\tnn_L`
- Line 2: numeric, with `iter=0`, integer actions in `[0, 224]` (15×15 = 225 actions), distances in `[0, 7]`, and W/D/L floats with 4 decimals each summing to ≈ 1.0 for both `vmix_*` and `nn_*` columns.

- [ ] **Step 4: Run probe a second time — confirms append, no second header**

Run:

```bash
cpp/build/mcts_probe \
    --model data/models/latest.pt \
    --config scripts/run.cfg \
    --iter 1 \
    --log /tmp/probe_smoke.tsv
```

Then: `wc -l /tmp/probe_smoke.tsv`

Expected: `3 /tmp/probe_smoke.tsv` (header + 2 rows). No duplicated header line.

- [ ] **Step 5: Cleanup**

Run: `rm -f /tmp/probe_smoke.tsv`

Expected: no error. (No commit — this task is verification only.)

---

## Task 4: Wire `--iter` and `--log` into run.sh

**Files:**
- Modify: `scripts/run.sh` (step 5b, ~lines 88–92)

- [ ] **Step 1: Update the mcts_probe invocation**

In `scripts/run.sh`, locate the step 5b block:

```bash
    # (5b) post-export diagnostic: empty-board MCTS rootValue probe
    "$ROOT/cpp/build/mcts_probe" \
        --model "$DATA_DIR/models/latest.pt" \
        --config "$SCRIPT_DIR/run.cfg" \
        || echo "[run.sh] mcts_probe failed (non-fatal)"
```

Replace it with:

```bash
    # (5b) post-export diagnostic: empty-board MCTS rootValue probe
    "$ROOT/cpp/build/mcts_probe" \
        --model "$DATA_DIR/models/latest.pt" \
        --config "$SCRIPT_DIR/run.cfg" \
        --iter "$iter" \
        --log "$DATA_DIR/logs/probe.tsv" \
        || echo "[run.sh] mcts_probe failed (non-fatal)"
```

- [ ] **Step 2: Lint**

Run: `bash -n scripts/run.sh`

Expected: no output, exit 0 (script parses cleanly).

- [ ] **Step 3: Commit**

```bash
git add scripts/run.sh
git commit -m "Pass --iter and --log to mcts_probe in run.sh step 5b so each post-export probe appends a row to data/logs/probe.tsv"
```

---

## Task 5: Add `_plot_probe` helper to view_loss.py

**Files:**
- Modify: `python/view_loss.py` (add helper after `_plot_selfplay` ~line 65; call site inside `main()` ~line 134)

- [ ] **Step 1: Add the `_plot_probe` function**

In `python/view_loss.py`, immediately after the existing `_plot_selfplay` function (after line 65 ending with `print(f"saved plot to {out}")`), insert:

```python
def _plot_probe(data_dir: pathlib.Path, plt) -> None:
    log = data_dir / "logs" / "probe.tsv"
    if not log.exists():
        print(f"no probe log at {log}", file=sys.stderr)
        return
    header, rows = _read_tsv(log)
    if not rows:
        print("empty probe log", file=sys.stderr)
        return
    idx = {name: i for i, name in enumerate(header)}
    needed = ("iter", "gumbel_dist", "lcb_dist",
              "vmix_W", "vmix_L", "nn_W", "nn_L")
    if not all(k in idx for k in needed):
        print("probe.tsv missing expected columns; skipping probe plot",
              file=sys.stderr)
        return

    def col(name):
        out = []
        for r in rows:
            v = r[idx[name]] if idx[name] < len(r) else ""
            out.append(float(v) if v else float("nan"))
        return out

    x = col("iter")
    gumbel_dist = col("gumbel_dist")
    lcb_dist = col("lcb_dist")
    vmix_wl = [w - l for w, l in zip(col("vmix_W"), col("vmix_L"))]
    nn_wl = [w - l for w, l in zip(col("nn_W"), col("nn_L"))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(x, gumbel_dist, color="C0", label="gumbel")
    ax1.plot(x, lcb_dist, color="C1", label="lcb")
    ax1.set_ylabel("dist from center")
    ax1.set_yticks(range(0, 8))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(x, vmix_wl, color="C2", label="v_mix W-L")
    ax2.plot(x, nn_wl, color="C3", label="nn_value W-L")
    ax2.axhline(0.0, ls="--", color="gray", alpha=0.4)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_ylabel("root value (W-L)")
    ax2.set_xlabel("iter")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    out = data_dir / "logs" / "probe.png"
    fig.savefig(out, dpi=200)
    print(f"saved plot to {out}")
```

- [ ] **Step 2: Wire the call into `main()`**

In `python/view_loss.py`, find the `_plot_selfplay(data_dir, plt)` call (~line 134) inside `if args.plot:`. Append a call right after it:

Before:

```python
        _plot_selfplay(data_dir, plt)
    return 0
```

After:

```python
        _plot_selfplay(data_dir, plt)
        _plot_probe(data_dir, plt)
    return 0
```

- [ ] **Step 3: Quick syntax check**

Run: `python -c "import ast; ast.parse(open('python/view_loss.py').read())"`

Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add python/view_loss.py
git commit -m "Plot probe.tsv as data/logs/probe.png: top pane shows gumbel and lcb first-stone Chebyshev distance from center (different colors), bottom pane shows v_mix and nn_value W-L (different colors)"
```

---

## Task 6: End-to-end smoke test of the plot path

This task verifies plotting works against a probe.tsv with at least 2 rows.

- [ ] **Step 1: Confirm probe.tsv already has data, or fabricate test rows**

Run: `wc -l data/logs/probe.tsv 2>/dev/null || echo "missing"`

If the result is `missing` or `1` (header only), fabricate two test rows for plot verification:

```bash
mkdir -p data/logs
cat > data/logs/probe.tsv <<'EOF'
iter	gumbel_action	lcb_action	gumbel_dist	lcb_dist	vmix_W	vmix_D	vmix_L	nn_W	nn_D	nn_L
0	112	112	0	0	0.4000	0.2000	0.4000	0.5000	0.1000	0.4000
1	98	112	2	0	0.5000	0.3000	0.2000	0.4500	0.2000	0.3500
EOF
```

(If `data/logs/probe.tsv` already has ≥ 2 rows from a real run, skip this step.)

- [ ] **Step 2: Run view_loss with --plot**

Run: `cd python && python view_loss.py --data-dir ../data --plot >/dev/null`

Expected: prints something like `saved plot to ../data/logs/loss.png`, `saved plot to ../data/logs/selfplay.png` (if applicable), and `saved plot to ../data/logs/probe.png`. Exit 0.

- [ ] **Step 3: Confirm probe.png exists and is non-empty**

Run: `ls -l data/logs/probe.png`

Expected: a recently-modified PNG file with non-zero size (typically ≥ 30 KB).

- [ ] **Step 4: Visual spot-check (optional, manual)**

Open `data/logs/probe.png`. Expected layout:
- Top pane: two lines of distinct colors labeled "gumbel" and "lcb", y-axis with integer ticks 0–7, x-axis hidden (shared).
- Bottom pane: two lines of distinct colors labeled "v_mix W-L" and "nn_value W-L", a dashed gray y=0 reference line, y-range −1.0 to 1.0, x-axis labeled "iter".

- [ ] **Step 5: Cleanup if you fabricated rows**

If Task 6 Step 1 fabricated `data/logs/probe.tsv`, restore it:

```bash
rm -f data/logs/probe.tsv data/logs/probe.png
```

(If real probe rows were already present, skip — the next `bash scripts/run.sh` invocation will re-append cleanly.)

No commit — this task is verification only.

---

## Self-Review Checklist

(Author-side, run after writing this plan; included here so the executing engineer can double-check.)

- **Spec coverage:** every section of `2026-04-29-mcts-probe-logging-design.md` maps to a task —
  - "C++ Changes" → Tasks 1, 2, 3
  - "Shell Changes" → Task 4
  - "Python Changes" → Tasks 5, 6
  - "Data Format" → Task 2 Step 2 writes the exact header & row layout
  - "Atomicity" → Task 2 Step 2 uses single-writer `app` open, no locking
  - "Testing & Verification" → Task 3 (build + write smoke), Task 6 (plot smoke)
- **Distinct colors per pair (user requirement):** Task 5 Step 1 sets `color="C0"`/`"C1"` for the top pane and `"C2"`/`"C3"` for the bottom — four visually distinct matplotlib default colors.
- **No placeholders:** every step contains the exact code or command needed.
- **Type/name consistency:** column names (`gumbel_dist`, `lcb_dist`, `vmix_W`, `vmix_L`, `nn_W`, `nn_L`, …) match between Task 2's header line, Task 5's `needed` tuple, and Task 5's `col(...)` calls.
