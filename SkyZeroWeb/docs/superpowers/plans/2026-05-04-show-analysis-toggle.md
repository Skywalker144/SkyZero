# Show Analysis Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single icon-button toggle to the topbar that hides/shows analysis surfaces (right-column heatmaps, "网络" WDL row, "网络" line on the value chart). Default off, persisted in localStorage, no flash on first paint.

**Architecture:** Pure presentation toggle — `worker.js` is unchanged. State lives in `localStorage["skz_show_analysis"]`. A pre-paint inline script at the start of `<body>` sets `body.show-analysis` so the first render is correct. CSS rules under `body:not(.show-analysis)` hide the right column, network WDL row, and "网络" legend item. `main.js` adds `setShowAnalysis(on)` for click handling and calls `drawValueChart()` after toggling so the "网络" series appears/disappears immediately. The toggle button reuses the existing `.seg-row.compact` + `.seg-btn` styles so it visually matches the language and theme controls.

**Tech Stack:** Static HTML/CSS/JS (no framework), `localStorage`, native canvas drawing. Repo root for paths in this plan: `/home/sky/RL/SkyZero/SkyZeroWeb/`.

**Spec:** `docs/superpowers/specs/2026-05-04-show-analysis-toggle-design.md`

**Note on tests:** This feature is purely DOM/CSS/canvas behavior with no logic that's testable in Node. The existing test suite (`tests/test_gomoku.mjs`, `tests/test_mcts.mjs`) covers pure logic only. Verification is done manually in a browser per the explicit checklist in Task 5. Do not add a Node-based test for this feature.

---

## File map

- `i18n.js` — add `show_analysis_label` strings (zh + en).
- `index.html` — add pre-paint inline script at start of `<body>`; add toggle button in `.topbar-actions`; add marker classes `wdl-row--nn` (network WDL row) and `vc-item--nn` (network legend item) so CSS can hide them by selector without `:has()`.
- `style.css` — add a `body:not(.show-analysis)` rule block hiding `#right_col`, `.wdl-row--nn`, `#wdl_nn_detail`, and `.vc-item--nn`.
- `main.js` — add `setShowAnalysis(on)`, wire the button's `click` handler, call `setShowAnalysis(persisted)` once on init, and gate the "网络" series in `drawValueChart()` on `body.show-analysis`.

`worker.js`, `gomoku.js`, `mcts.js` are not touched.

---

### Task 1: Add i18n strings for the toggle label

**Files:**
- Modify: `i18n.js:5-114` (the `I18N_STRINGS` object — add one key in each of `en` and `zh`)

- [ ] **Step 1: Add `show_analysis_label` to the English block**

In `i18n.js`, find the line `theme_label_dark: "Dark",` inside the `en:` block (currently around line 58). Add a new entry immediately after it:

```javascript
        theme_label_dark: "Dark",
        show_analysis_label: "Show analysis",
    },
```

(The `},` on the next line closes the `en` block — keep it as is. Only the new line `show_analysis_label: "Show analysis",` is being inserted.)

- [ ] **Step 2: Add `show_analysis_label` to the Chinese block**

Find the line `theme_label_dark: "暗黑",` inside the `zh:` block (currently around line 112). Add a new entry immediately after it:

```javascript
        theme_label_dark: "暗黑",
        show_analysis_label: "显示分析",
    },
```

- [ ] **Step 3: Sanity-check i18n parses**

Run from the repo root:

```bash
node --check i18n.js && echo OK
```

Expected: `OK` printed. (`node --check` does a syntax-only check — it doesn't execute the file, so the browser-only `document.addEventListener` reference at the bottom is fine.)

- [ ] **Step 4: Commit**

```bash
git add i18n.js
git commit -m "Add show_analysis_label i18n strings (zh + en)"
```

---

### Task 2: Add toggle button, pre-paint script, and marker classes in index.html

**Files:**
- Modify: `index.html` (top of `<body>` for the inline script; `.topbar-actions` block for the button; `.wdl-row` and `.vc-item` for marker classes)

This task is structural HTML only. After it lands, the button is visible in the topbar but does nothing (no JS wiring yet) and clicking it has no effect on layout (no CSS yet). The page still works — simple-mode behavior is not yet active because the body class isn't set anywhere.

- [ ] **Step 1: Add the pre-paint inline script as the first child of `<body>`**

In `index.html`, find `<body>` (currently line 39). Insert the following block as its very first child, before the existing `<div id="loading_overlay">`:

```html
<body>
  <script>
    // Set show-analysis class before first paint so analysis surfaces don't
    // flash when the user has them turned off.
    (function(){
      try {
        var v = localStorage.getItem('skz_show_analysis');
        if (v === '1') document.body.classList.add('show-analysis');
      } catch(e) {}
    })();
  </script>

  <!-- Loading overlay shown while downloading the first ONNX model -->
```

(The existing `<!-- Loading overlay ... -->` line and everything after it stays unchanged.)

- [ ] **Step 2: Add the toggle button to `.topbar-actions`**

In `index.html`, find the `<div class="topbar-actions">` block (currently starting around line 57) and the closing `</div>` of that block (just before `</header>`, currently around line 81). Add a new button just before that closing `</div>`, after the theme `seg-row`:

```html
        <div class="seg-row compact" id="show_analysis_seg" role="group" aria-label="Show analysis">
          <button class="seg-btn" id="show_analysis_btn" type="button" aria-pressed="false"
                  data-i18n-title="show_analysis_label" data-i18n-aria="show_analysis_label"
                  aria-label="Show analysis" title="Show analysis">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="1.5" y="1.5"  width="5" height="3.5" rx="0.8"></rect>
              <rect x="9.5" y="1.5"  width="5" height="3.5" rx="0.8"></rect>
              <rect x="1.5" y="6.25" width="5" height="3.5" rx="0.8"></rect>
              <rect x="9.5" y="6.25" width="5" height="3.5" rx="0.8"></rect>
              <rect x="1.5" y="11"   width="5" height="3.5" rx="0.8"></rect>
              <rect x="9.5" y="11"   width="5" height="3.5" rx="0.8"></rect>
            </svg>
          </button>
        </div>
      </div>
```

The icon is a 2-column × 3-row grid of rounded rectangles, echoing the right-column heatmap grid (6 cards in a 2×3 layout). The `data-i18n-aria` and `data-i18n-title` attributes hook into the existing `applyI18n()` machinery in `i18n.js` (see `i18n.js:152-157`) — the static `aria-label="Show analysis"` and `title="Show analysis"` are fallbacks for first paint before `applyI18n()` runs.

- [ ] **Step 3: Add `wdl-row--nn` marker class to the network WDL row**

In `index.html`, find the network WDL row (currently line 157):

```html
            <div class="wdl-row" style="margin-top:4px;">
```

Replace with:

```html
            <div class="wdl-row wdl-row--nn" style="margin-top:4px;">
```

This is the only change to that line. The contents of the row (network label, `wdl_nn_bar`, `wdl_nn_wl`) stay as-is.

- [ ] **Step 4: Add `vc-item--nn` marker class to the "网络" legend item**

In `index.html`, find the legend block (currently around lines 168-172):

```html
              <div class="value-chart-legend">
                <span class="vc-item"><span class="vc-swatch" style="background:#0969da;"></span><span data-i18n="legend_root">根</span></span>
                <span class="vc-item"><span class="vc-swatch" style="background:#cf222e;"></span><span data-i18n="legend_nn">网络</span></span>
                <span class="vc-axis" data-i18n="legend_axis">胜负 · −1…+1</span>
              </div>
```

Change the second `<span class="vc-item">` to add the `vc-item--nn` modifier class:

```html
              <div class="value-chart-legend">
                <span class="vc-item"><span class="vc-swatch" style="background:#0969da;"></span><span data-i18n="legend_root">根</span></span>
                <span class="vc-item vc-item--nn"><span class="vc-swatch" style="background:#cf222e;"></span><span data-i18n="legend_nn">网络</span></span>
                <span class="vc-axis" data-i18n="legend_axis">胜负 · −1…+1</span>
              </div>
```

The first `<span class="vc-item">` (root legend) is unchanged.

- [ ] **Step 5: Verify HTML is well-formed**

From the repo root:

```bash
python3 -c "import html.parser as h, sys; p=h.HTMLParser(); p.feed(open('index.html').read()); p.close()" && echo OK
```

Expected: `OK` printed (no parse errors).

- [ ] **Step 6: Commit**

```bash
git add index.html
git commit -m "Add show-analysis toggle button, pre-paint script, and CSS markers"
```

---

### Task 3: Add CSS hide rules for simple mode

**Files:**
- Modify: `style.css` (append a new section)

After this task lands, the body never has `show-analysis` (Task 4 hasn't run yet), so the rules from this task are always active — the right column, network WDL row, and "网络" legend item are hidden. The page is now in "simple mode" by default. The toggle button is still inert (no JS wiring yet); the next task wires it.

- [ ] **Step 1: Append the hide rules to `style.css`**

Append the following block at the end of `style.css` (after the last existing rule):

```css
  /* ---------- Show-analysis toggle ---------- */
  /* Default state (no body.show-analysis): hide the right-column heatmaps,
     the network WDL row, and the "网络" legend item on the value chart.
     Toggle wiring lives in main.js (setShowAnalysis). */
  body:not(.show-analysis) #right_col,
  body:not(.show-analysis) .wdl-row--nn,
  body:not(.show-analysis) #wdl_nn_detail,
  body:not(.show-analysis) .vc-item--nn {
    display: none;
  }
```

- [ ] **Step 2: Commit**

```bash
git add style.css
git commit -m "Hide right column, network WDL row, and nn legend in simple mode"
```

---

### Task 4: Wire toggle behavior in main.js

**Files:**
- Modify: `main.js` (add `setShowAnalysis` near the theme controller block; gate "网络" line in `drawValueChart`; add init call)

After this task, the toggle button works: clicking it adds/removes `body.show-analysis`, persists the state, updates `aria-pressed`, and immediately redraws the value chart (so the "网络" line appears/disappears without waiting for the next move).

- [ ] **Step 1: Add the `setShowAnalysis` controller block**

In `main.js`, find the end of the theme controller block (currently the line with `} catch (_) {}` closing the `try` around `matchMedia`, around line 80). Insert the following block immediately after it, before the comment `// Drives sizing of right-column to match left-column height`:

```javascript
// Show-analysis toggle (heatmaps + network WDL + "nn" line on value chart).
// State persisted in localStorage("skz_show_analysis") as "1" / "0".
// A pre-paint inline script in index.html applies body.show-analysis before
// first paint to avoid flashing the analysis surfaces on load.
function getShowAnalysis() {
    try { return localStorage.getItem("skz_show_analysis") === "1"; }
    catch (_) { return false; }
}
function setShowAnalysis(on) {
    document.body.classList.toggle("show-analysis", !!on);
    try { localStorage.setItem("skz_show_analysis", on ? "1" : "0"); } catch (_) {}
    const btn = document.getElementById("show_analysis_btn");
    if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false");
    if (typeof drawValueChart === "function") drawValueChart();
}
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("show_analysis_btn");
    if (!btn) return;
    setShowAnalysis(getShowAnalysis());  // sync aria-pressed and chart on load
    btn.addEventListener("click", () => setShowAnalysis(!getShowAnalysis()));
});
```

The `setShowAnalysis(getShowAnalysis())` call on load matters: the pre-paint script already set the body class, but it didn't set `aria-pressed` (the button doesn't exist yet at that point) and didn't redraw the chart (the canvas isn't drawn yet at that point). Calling `setShowAnalysis` here makes the button's pressed state and the chart agree with the persisted state.

- [ ] **Step 2: Gate the "网络" line in `drawValueChart`**

In `main.js`, find the two `plot(...)` calls at the end of `drawValueChart` (currently lines 555-556):

```javascript
    plot("root", "#0969da");
    plot("nn",   "#cf222e");
}
```

Replace with:

```javascript
    plot("root", "#0969da");
    if (document.body.classList.contains("show-analysis")) {
        plot("nn", "#cf222e");
    }
}
```

This ensures the "网络" series isn't drawn in simple mode. The CSS rule from Task 3 already hides the legend item, so the chart's appearance is internally consistent.

- [ ] **Step 3: Verify main.js parses**

From the repo root:

```bash
node --check main.js && echo OK
```

Expected: `OK` printed.

- [ ] **Step 4: Commit**

```bash
git add main.js
git commit -m "Wire show-analysis toggle: setShowAnalysis, click handler, chart gate"
```

---

### Task 5: Manual browser verification

This task has no code changes — it confirms the feature works end-to-end in a real browser. Each check below is a hard requirement; if any fails, return to the relevant task and fix.

- [ ] **Step 1: Serve the static site locally**

From the repo root, in a terminal:

```bash
python3 -m http.server 8000
```

Leave it running. Open `http://localhost:8000/` in a browser.

- [ ] **Step 2: First-load default state (fresh, no localStorage)**

Open DevTools → Application → Local Storage → `http://localhost:8000` → delete the `skz_show_analysis` key if present, then hard-reload the page (Ctrl-Shift-R / Cmd-Shift-R).

Verify:
- Right column (6 heatmap cards) is **not visible**.
- "网络" / "nn" WDL row under "胜率估计" / "Value estimates" is **not visible**.
- The value chart shows only the "根" / "root" line and only "根" + axis label in the legend.
- The toggle icon button is visible in the topbar (right side, after the theme buttons), with `aria-pressed="false"` (you can confirm in DevTools → Elements). The icon is a 2×2 grid of small rounded squares.
- No flash of the right column or network surfaces during load.

- [ ] **Step 3: Toggle on**

Click the toggle icon button.

Verify:
- Right column appears (6 heatmap cards), showing data for whatever moves have been played (or empty if no moves yet).
- "网络" WDL row appears under the root row.
- "网络" line and legend item appear on the value chart.
- Button's `aria-pressed` is now `"true"` (DevTools → Elements). The button has the active visual style (background highlight matching the language/theme buttons in their pressed state).

- [ ] **Step 4: Persistence across reload**

Reload the page (Ctrl-R).

Verify:
- Page comes back up with all analysis surfaces still visible (right column, network WDL row, full value chart).
- No flash of simple-mode layout during load — the right column is there immediately.
- Button is still `aria-pressed="true"`.

- [ ] **Step 5: Toggle off**

Click the toggle button again.

Verify:
- All analysis surfaces disappear immediately (no waiting for next move).
- Value chart redraws without the "网络" line right away.
- Reloading the page comes back up in simple mode (no flash of analysis surfaces).

- [ ] **Step 6: Language switching updates aria/title**

Switch the language seg from 中 to EN, then back. With each switch:

Verify (DevTools → Elements on the toggle button):
- `aria-label` matches the current language: "Show analysis" in EN, "显示分析" in zh.
- `title` matches the same.

- [ ] **Step 7: Mid-game toggle (heatmaps populated)**

Start a new game and play one or two moves so the heatmaps and network WDL have real data.

Toggle off, then on. Verify:
- Toggling off: surfaces disappear, value chart redraws to root-only.
- Toggling on: heatmap canvases already show the current position's data (they were being updated while hidden — this is the "instant toggle" behavior from the spec).

- [ ] **Step 8: Mobile/narrow viewport**

In DevTools → Toggle device toolbar (or resize the window to < 1399px wide).

Verify:
- Simple mode: layout stacks (left column → board → no right column visible). Looks clean.
- Analysis mode: stack is left column → board → right column (heatmaps below). Existing responsive layout works in both states.

- [ ] **Step 9: Stop the server**

In the terminal running `python3 -m http.server 8000`, press Ctrl-C.

---

### Task 6: Final cleanup

- [ ] **Step 1: Verify all changes are committed**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: View the commit history for this feature**

```bash
git log --oneline -5
```

Expected (in reverse chronological order, the four most recent commits):

```
<hash> Wire show-analysis toggle: setShowAnalysis, click handler, chart gate
<hash> Hide right column, network WDL row, and nn legend in simple mode
<hash> Add show-analysis toggle button, pre-paint script, and CSS markers
<hash> Add show_analysis_label i18n strings (zh + en)
```

(The earlier spec commit `47d930f` is also in the history.)

---

## Acceptance criteria recap (from spec)

All of these are verified in Task 5:

1. ✅ Fresh visit: simple view, no flash. (Step 2)
2. ✅ Click reveals analysis surfaces immediately, persists across reload. (Steps 3–4)
3. ✅ Click again hides, persists. (Step 5)
4. ✅ `aria-pressed` and active visual state match persisted state. (Steps 2, 3, 4, 5)
5. ✅ Language switch updates aria/title. (Step 6)
6. ✅ Mobile renders cleanly in both states. (Step 8)
7. ✅ `worker.js` unchanged. (Verified by inspecting the plan: no file in the file map references `worker.js`.)
