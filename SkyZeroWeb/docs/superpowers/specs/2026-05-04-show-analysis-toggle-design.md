# 显示分析 (Show Analysis) Toggle — Design

**Date:** 2026-05-04
**Status:** Approved (awaiting implementation plan)

## Problem

The current SkyZero web UI exposes everything to every user: heatmaps (improved policy, visit distribution, NN policy, opponent policy, future-pos +8/+32), a network-side WDL bar, and a network-value curve overlaid on the root-value curve. Most users only want to play; the heatmaps and network internals are valuable for developers and curious players investigating model behavior, but they clutter the default experience.

We want a single toggle that hides the analysis surfaces by default and reveals them on demand. The default page should show only the board, the root WDL bar, the root-only value curve, and the existing controls (model, side, board size, rule, sims).

## Non-goals

- No separate "developer page" or alternate URL — single page, single bundle.
- No change to `worker.js` / MCTS pipeline. The worker keeps producing the same per-move data regardless of mode.
- No change to mobile layout rules other than what naturally falls out from hiding the right column.
- No change to the existing language and theme toggles.

## User-facing design

### Toggle widget

- Single icon button placed in `topbar-actions`, after the language and theme `seg-row`s.
- Reuses the existing `.seg-btn` visual style so active/inactive states match the rest of the topbar (active = background highlight, `aria-pressed="true"`).
- Icon: a 2×3 rounded-square grid SVG, visually echoing the right-column heatmap grid.
- Accessibility:
  - `aria-pressed` reflects state.
  - `aria-label` and `title` come from i18n key `show_analysis_label` ("显示分析" / "Show analysis").

### What the toggle reveals (when on)

- Right column `#right_col` (the 6-card heatmap grid).
- "网络" / "Network" WDL row (`wdl_nn_bar` and `wdl_nn_detail`, plus their label).
- The "网络" line on `#value_chart`, and the "网络" item in `.value-chart-legend`.

### What stays in default (off) state

- Status pill, model selector, side selector, board size, rule selector, sims input + pure-NN hint.
- Root WDL row (`wdl_root_bar`, `wdl_root_detail`).
- `#value_chart` with only the root line drawn; legend shows only "根" / "Root" and the axis label.
- Board, new game, undo.

### Default state

Off. New visitors see the simplified view.

## Architecture

### State model

- One localStorage key: `skz_show_analysis`, value `"1"` or `"0"`. Missing key = `"0"` (off).
- A pre-paint inline script in `<head>` (alongside the existing theme/lang setup) reads the key and sets `document.body` 's class so the first paint is correct and there is no flash of analysis surfaces.
  - Note: the existing theme/lang inline scripts run before `<body>` exists; the new one must be placed at the start of `<body>` or guarded so that it acts on `document.body` once available. Implementation can add a tiny inline script as the first child of `<body>`, mirroring the same try/catch pattern.
- Body class: `body.show-analysis`. Absence of the class = simple mode.

### `setShowAnalysis(on: boolean)` in `main.js`

A single function that:

1. Toggles `document.body.classList` 's `show-analysis`.
2. Writes `localStorage.setItem('skz_show_analysis', on ? '1' : '0')`.
3. Updates the icon button's `aria-pressed`.
4. Triggers a redraw of `#value_chart` (because the "网络" line's visibility depends on this state).

The toggle button's click handler calls `setShowAnalysis(!current)`.

On page load, after i18n and the rest of the UI are wired up, `setShowAnalysis` is called once with the persisted value to ensure the button's `aria-pressed` and the chart's drawn lines agree with the body class set by the pre-paint script.

### CSS rules (`style.css`)

Under `body:not(.show-analysis)`:

- `#right_col { display: none; }` — removes the right column from the grid; the existing flex/grid layout naturally widens the board column.
- The "网络" WDL row and `#wdl_nn_detail` are hidden.
- The `.value-chart-legend` "网络" item is hidden.

The current markup of the network WDL row is `<div class="wdl-row">` containing `#wdl_nn_bar`, followed by `<div class="wdl-detail" id="wdl_nn_detail">`. There is no wrapping container around the two pieces. To hide them, add a marker class to the row in `index.html` (e.g. `wdl-row--nn`) and target both with simple selectors: `body:not(.show-analysis) .wdl-row--nn, body:not(.show-analysis) #wdl_nn_detail { display: none; }`. Same approach for the "网络" legend item: add a marker class to its `<span class="vc-item">` so it can be hidden directly.

### Value-chart redraw

The chart's draw function reads `document.body.classList.contains('show-analysis')` and skips drawing the "网络" series when false. `setShowAnalysis` calls the redraw function explicitly so toggling produces an immediate visual update without waiting for the next move.

### Worker behavior

`worker.js` is not modified. It continues to compute and return heatmap data and NN value on every move. In simple mode `main.js` still receives the data and still writes it to the (hidden) heatmap canvases — that is acceptable and keeps toggle-on instantaneous: when the user enables analysis, the heatmap canvases for the current position are already painted.

### i18n

Add to `i18n.js`:

- `show_analysis_label`:
  - zh: "显示分析"
  - en: "Show analysis"

The icon button uses `data-i18n-aria` and `data-i18n-title` (matching the existing pattern used by the theme buttons) so it picks up the right language.

## Mobile

The existing layout already stacks columns on narrow viewports. Hiding `#right_col` in simple mode just removes the heatmap stack from below the board — no media-query changes needed. In analysis mode, the existing stacked layout is preserved.

## Edge cases

- **First load, no moves yet:** heatmap canvases are empty in both modes — no special handling.
- **Toggle during an active worker computation:** harmless — `setShowAnalysis` only changes CSS class, localStorage, and triggers a chart redraw. Worker results continue to flow into both visible and hidden canvases.
- **localStorage disabled / private mode:** `try/catch` around the read and write, falling back to off. Matches the pattern used for `skz_theme` and `skz_lang`.
- **Pre-paint script timing:** placed as the first child of `<body>` so `document.body` exists. If for some reason it runs before body is available, fall back to setting the class on `document.documentElement` and add a CSS mirror selector — but the simpler placement is preferred.

## Files touched

- `index.html` — add toggle button in `topbar-actions`, add pre-paint inline script at top of `<body>`, possibly wrap the "网络" WDL row in a hookable container.
- `style.css` — add `body:not(.show-analysis)` rules to hide right column, network WDL row, and "网络" legend item.
- `main.js` — add `setShowAnalysis`, wire the toggle click handler, call it once on init, make `#value_chart` redraw read the body class.
- `i18n.js` — add `show_analysis_label` zh/en strings.

`worker.js` is unchanged.

## Acceptance criteria

1. On a fresh visit (no localStorage), the page loads with no right column, no "网络" WDL row, and a single-line value chart — no flash of analysis surfaces during load.
2. Clicking the topbar icon button reveals the right column, the "网络" WDL row, and adds the "网络" line to the value chart immediately, and the change persists across reload.
3. Clicking again hides them and persists the off state.
4. The toggle's `aria-pressed` and visual active state match the persisted state on load and after every click.
5. Language switching updates the button's `aria-label` / `title`.
6. Mobile (narrow viewport) renders cleanly in both states.
7. `worker.js` is unchanged and still returns the same per-move data in both states.
