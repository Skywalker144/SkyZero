// Board geometry — N is mutable (board size dropdown can change it).
let N = 15;
const MARGIN_DESKTOP = 28;   // leaves room for top + left coordinate labels
const MARGIN_COMPACT = 10;   // mobile / portrait — labels hidden, just enough for stones
let MARGIN = MARGIN_DESKTOP;
let CELL = 36;
let BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
const MONO_FONT = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace';
const HEAT_LOGICAL = 240;
const DPR = window.devicePixelRatio || 1;

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setupCanvas(canvas, logicalW, logicalH, setStyle = true) {
    canvas.width = Math.round(logicalW * DPR);
    canvas.height = Math.round(logicalH * DPR);
    if (setStyle) {
        canvas.style.width = logicalW + "px";
        canvas.style.height = logicalH + "px";
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx._logicalW = logicalW;
    ctx._logicalH = logicalH;
    return ctx;
}
function clearLogical(ctx) { ctx.clearRect(0, 0, ctx._logicalW, ctx._logicalH); }

// Board canvas + 6 heat canvases set up later in Tasks 20-21.
const cv = document.getElementById("board");
const ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);

// Theme controller (tri-state segmented buttons).
const THEME_MODES = ["auto", "light", "dark"];
function resolveTheme(mode) {
    if (mode === "light" || mode === "dark") return mode;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function updateThemeSegPressed(mode) {
    const seg = document.getElementById("theme_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-theme]")) {
        b.setAttribute("aria-pressed", b.dataset.theme === mode ? "true" : "false");
    }
}
function applyTheme(mode) {
    if (!THEME_MODES.includes(mode)) mode = "auto";
    document.documentElement.dataset.theme = resolveTheme(mode);
    document.documentElement.dataset.themeMode = mode;
    updateThemeSegPressed(mode);
    if (typeof drawAll === "function") drawAll();
}
updateThemeSegPressed(document.documentElement.dataset.themeMode || "auto");
document.addEventListener("DOMContentLoaded", () => {
    const seg = document.getElementById("theme_seg");
    if (!seg) return;
    for (const b of seg.querySelectorAll(".seg-btn[data-theme]")) {
        b.addEventListener("click", () => {
            const next = b.dataset.theme;
            if (!THEME_MODES.includes(next)) return;
            try {
                if (next === "auto") localStorage.removeItem("skz_theme");
                else localStorage.setItem("skz_theme", next);
            } catch (_) {}
            applyTheme(next);
        });
    }
    updateThemeSegPressed(document.documentElement.dataset.themeMode || "auto");
});
try {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onSysChange = () => {
        if ((document.documentElement.dataset.themeMode || "auto") !== "auto") return;
        applyTheme("auto");
    };
    if (mql.addEventListener) mql.addEventListener("change", onSysChange);
    else if (mql.addListener) mql.addListener(onSysChange);
} catch (_) {}

// Show-analysis toggle (heatmaps + network WDL + "nn" line on value chart).
// State persisted in localStorage("skz_show_analysis") as "1" / "0".
// A pre-paint inline script in index.html applies body.show-analysis before
// first paint to avoid flashing the analysis surfaces on load.
//
// Simple mode: value-estimates card sits in the right column (default HTML
// position) so it mirrors the left controls column around the board.
// Analysis mode: card moves down into the left column so the right column
// can dedicate its full board-matching width to the 2x3 heatmap grid.
function getShowAnalysis() {
    try { return localStorage.getItem("skz_show_analysis") === "1"; }
    catch (_) { return false; }
}
function moveValueCard(toAnalysisMode) {
    const card = document.querySelector('[data-i18n="label_value_estimates"]')?.closest(".card");
    if (!card) return;
    const left = document.getElementById("left_col");
    const right = document.getElementById("right_col");
    if (!left || !right) return;
    if (toAnalysisMode) {
        if (card.parentElement !== left) left.appendChild(card);
    } else {
        if (card.parentElement !== right) right.insertBefore(card, right.firstElementChild);
    }
}
function setShowAnalysis(on) {
    document.body.classList.toggle("show-analysis", !!on);
    moveValueCard(!!on);
    try { localStorage.setItem("skz_show_analysis", on ? "1" : "0"); } catch (_) {}
    const btn = document.getElementById("show_analysis_btn");
    if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false");
    if (typeof syncBoardSize === "function") syncBoardSize();
    if (typeof drawValueChart === "function") drawValueChart();
}
document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("show_analysis_btn");
    if (!btn) return;
    setShowAnalysis(getShowAnalysis());
    btn.addEventListener("click", () => setShowAnalysis(!getShowAnalysis()));
});

// Drives sizing of right-column to match left-column height (port from play_web.py).
const leftCol = document.getElementById("left_col");
const rightCol = document.getElementById("right_col");
const boardCard = document.querySelector(".board-card");
const boardActions = document.querySelector(".board-actions");
const mainEl = document.querySelector(".main");

const topbarEl = document.querySelector(".topbar");
const appEl = document.querySelector(".app");
const boardColEl = document.querySelector(".board-col");

function syncBoardSize() {
    if (window.matchMedia("(max-width: 1399px)").matches) {
        rightCol.style.height = "";
        rightCol.style.width = "";
        // Stacked layout — fit board to the column's available width so it never
        // overflows the viewport on phones / narrow tablets. Cap at 560 so it
        // doesn't grow unreasonably large in tablet portrait. Use a compact
        // margin since labels are hidden in this mode (see draw()).
        const cardCS = getComputedStyle(boardCard);
        const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
        const cardBorderX = parseFloat(cardCS.borderLeftWidth) + parseFloat(cardCS.borderRightWidth);
        const availW = Math.max(0, mainEl.clientWidth - cardPadX - cardBorderX);
        const target = Math.min(560, Math.floor(availW));
        const newMargin = MARGIN_COMPACT;
        const cell = Math.max(12, Math.floor((target - 2 * newMargin) / (N - 1)));
        const logical = newMargin * 2 + cell * (N - 1);
        if (logical !== BOARD_LOGICAL || newMargin !== MARGIN) {
            MARGIN = newMargin;
            CELL = cell;
            BOARD_LOGICAL = logical;
            setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
            ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
            if (typeof draw === "function") draw();
        }
        return;
    }
    // Desktop: restore the wider margin used for coordinate labels. The
    // recompute below will pick this up via the cv.width !== ... check.
    if (MARGIN !== MARGIN_DESKTOP) MARGIN = MARGIN_DESKTOP;
    const cardCS = getComputedStyle(boardCard);
    const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const cardPadY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const legendH = 0;
    const mainCS = getComputedStyle(mainEl);
    const gap = parseFloat(mainCS.columnGap || mainCS.gap) || 20;
    const remaining = mainEl.clientWidth - leftCol.offsetWidth - 2 * gap;
    // Analysis mode: right col equals board width (heatmaps take the same
    // span as the board), so board gets half of the remaining space.
    // Simple mode: right col is a fixed 350px (mirroring the left col), so
    // the board fills everything else. Keep this constant in sync with the
    // CSS rule `body:not(.show-analysis) #right_col { width: 350px; }`.
    const SIMPLE_RIGHT_COL_PX = 350;
    const isSimple = !document.body.classList.contains("show-analysis");
    const sizeByWidth = isSimple
        ? Math.floor(remaining - SIMPLE_RIGHT_COL_PX - cardPadX)
        : Math.floor(remaining / 2 - cardPadX);

    // Cap by viewport height so board + action buttons stay fully visible.
    const topbarCS = getComputedStyle(topbarEl);
    const topbarH = topbarEl.offsetHeight + parseFloat(topbarCS.marginBottom || 0);
    const appCS = getComputedStyle(appEl);
    const appPadY = parseFloat(appCS.paddingTop) + parseFloat(appCS.paddingBottom);
    const mainMb = parseFloat(mainCS.marginBottom || 0);
    const colCS = getComputedStyle(boardColEl);
    const colGap = parseFloat(colCS.rowGap || colCS.gap) || 0;
    const actionsH = boardActions.offsetHeight;
    const reserved = appPadY + topbarH + mainMb + cardPadY + legendH + colGap + actionsH;
    const sizeByViewport = window.innerHeight - reserved;

    let size = Math.max(280, Math.min(sizeByWidth, sizeByViewport));
    CELL = Math.max(20, Math.floor((size - 2 * MARGIN) / (N - 1)));
    BOARD_LOGICAL = MARGIN * 2 + CELL * (N - 1);
    const need = cv.width !== Math.round(BOARD_LOGICAL * DPR);
    if (need) setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
    if (document.body.classList.contains("show-analysis")) {
        rightCol.style.height = boardCard.offsetHeight + "px";
        rightCol.style.width  = boardCard.offsetWidth  + "px";
    } else {
        // Simple mode: right column hosts only the value-estimates card.
        // Width is constrained by CSS to mirror the left controls column;
        // height is content-sized.
        rightCol.style.height = "";
        rightCol.style.width  = "";
    }
    if (need && typeof draw === "function") draw();
}
new ResizeObserver(syncBoardSize).observe(leftCol);
window.addEventListener("resize", syncBoardSize);

// Module-level game-display state. Updated by handlers in Task 24.
let state = null;        // { board: 2D N×N int, last_move: [r,c]|null, board_size: N }
let showGumbel = false;
let gumbelPhases = null; // last search's gumbel phases [[r,c]...] per phase

// Run once synchronously so the first paint is already correctly sized,
// instead of flashing the 560px default on small viewports. Must run after
// the `state` / `gumbelPhases` lets above, since syncBoardSize may invoke
// draw() which reads them (TDZ otherwise).
syncBoardSize();

function draw() {
    clearLogical(ctx);
    const boardLine = cssVar("--board-line") || "#6b5a3a";
    const boardStar = cssVar("--board-star") || "#3a2e1a";
    const stoneB0 = cssVar("--stone-black-0");
    const stoneB1 = cssVar("--stone-black-1");
    const stoneW0 = cssVar("--stone-white-0");
    const stoneW1 = cssVar("--stone-white-1");
    const stoneOutline = cssVar("--stone-outline");
    const stoneShadow = cssVar("--stone-shadow") || "rgba(0,0,0,0.18)";

    ctx.strokeStyle = boardLine; ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
        const p = Math.round(MARGIN + i * CELL) + 0.5;
        ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL * (N - 1), p); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL * (N - 1)); ctx.stroke();
    }
    ctx.fillStyle = boardStar;
    if (N >= 7) {
        const off = (N >= 13) ? 3 : 2;
        const pts = [[off, off], [off, N-1-off], [N-1-off, off], [N-1-off, N-1-off]];
        if (N % 2 === 1) pts.push([(N-1)/2, (N-1)/2]);
        for (const [r, c] of pts) {
            ctx.beginPath();
            ctx.arc(MARGIN + c * CELL, MARGIN + r * CELL, 3.5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    // Coordinate labels need ~14px of margin space; skip them in compact mode
    // so the board can use the saved real estate for larger cells.
    if (MARGIN >= 16) {
        ctx.fillStyle = boardLine;
        ctx.font = `11px ${MONO_FONT}`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        for (let i = 0; i < N; i++) {
            ctx.fillText(i, MARGIN + i * CELL, 12);
            ctx.fillText(i, 10, MARGIN + i * CELL);
        }
    }
    if (!state) return;

    const stoneR    = Math.max(6, Math.round(CELL * 0.39));
    const gumbelR   = Math.max(6, Math.round(CELL * 0.34));
    const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
    const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
    const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
    const gradInner = Math.max(1, Math.round(CELL * 0.11));
    const gumbelFontPx = Math.max(8, Math.round(CELL * 0.28));

    if (showGumbel && gumbelPhases && gumbelPhases.length > 0) {
        const COLORS = ["#9ca3af","#3b82f6","#10b981","#f59e0b","#ef4444"];
        const LABELS = ["16","8","4","2","1"];
        const deepest = new Map();
        for (let i = 0; i < gumbelPhases.length; i++) {
            for (const rc of gumbelPhases[i]) {
                const key = rc[0] * N + rc[1];
                if (!deepest.has(key) || deepest.get(key) < i) deepest.set(key, i);
            }
        }
        for (const [key, idx] of deepest) {
            const r = (key / N) | 0, c = key % N;
            const sizeLabel = String(gumbelPhases[idx].length);
            let bucket = LABELS.indexOf(sizeLabel);
            if (bucket < 0) bucket = Math.min(idx, COLORS.length - 1);
            const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
            ctx.beginPath(); ctx.arc(x, y, gumbelR, 0, Math.PI * 2);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = COLORS[bucket];
            ctx.stroke();
            ctx.lineWidth = 1;
            if (state.board[r][c] === 0) {
                ctx.fillStyle = COLORS[bucket];
                ctx.font = `bold ${gumbelFontPx}px ${MONO_FONT}`;
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(sizeLabel, x, y);
            }
        }
    }

    const b = state.board, lm = state.last_move;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const v = b[r][c]; if (!v) continue;
        const x = MARGIN + c * CELL, y = MARGIN + r * CELL;
        ctx.beginPath(); ctx.arc(x + shadowDx, y + shadowDy, stoneR, 0, Math.PI * 2);
        ctx.fillStyle = stoneShadow; ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI * 2);
        if (v === 1) {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneB0); grad.addColorStop(1, stoneB1);
            ctx.fillStyle = grad;
        } else {
            const grad = ctx.createRadialGradient(x - gradInner, y - gradInner, 2, x, y, stoneR);
            grad.addColorStop(0, stoneW0); grad.addColorStop(1, stoneW1);
            ctx.fillStyle = grad;
        }
        ctx.fill();
        ctx.strokeStyle = stoneOutline; ctx.lineWidth = 1; ctx.stroke();
        if (lm && lm[0] === r && lm[1] === c) {
            ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI * 2);
            ctx.fillStyle = "#ef4444"; ctx.fill();
        }
    }
}

// --- Six heatmap canvases ---
const heatCtxs = {
    h_mcts_policy:    setupCanvas(document.getElementById("h_mcts_policy"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_mcts_visits:    setupCanvas(document.getElementById("h_mcts_visits"),    HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_policy:      setupCanvas(document.getElementById("h_nn_policy"),      HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_opp_policy:  setupCanvas(document.getElementById("h_nn_opp_policy"),  HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_8: setupCanvas(document.getElementById("h_nn_futurepos_8"), HEAT_LOGICAL, HEAT_LOGICAL, false),
    h_nn_futurepos_32:setupCanvas(document.getElementById("h_nn_futurepos_32"),HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_GRID_KEYS = {
    h_mcts_policy:    "mcts_policy",
    h_mcts_visits:    "mcts_visits",
    h_nn_policy:      "nn_policy",
    h_nn_opp_policy:  "nn_opp_policy",
    h_nn_futurepos_8: "nn_futurepos_8",
    h_nn_futurepos_32:"nn_futurepos_32",
};
const SIGNED_HEAT_IDS = new Set(["h_nn_futurepos_8", "h_nn_futurepos_32"]);

function fitHeatCanvas(canvasId) {
    const c = document.getElementById(canvasId);
    const card = c.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const title = card.querySelector(".grid-title");
    let titleH = 0;
    if (title) {
        const tCS = getComputedStyle(title);
        titleH = title.offsetHeight + parseFloat(tCS.marginTop || 0) + parseFloat(tCS.marginBottom || 0);
    }
    const availW = card.clientWidth - padX;
    const availH = card.clientHeight - padY - titleH;
    const size = Math.max(60, Math.floor(Math.min(availW, availH > 0 ? availH : availW)));
    c.style.width = size + "px";
    c.style.height = size + "px";
    if (heatCtxs[canvasId]._logicalW === size) return false;
    heatCtxs[canvasId] = setupCanvas(c, size, size, false);
    return true;
}
for (const id of Object.keys(heatCtxs)) {
    const c = document.getElementById(id);
    new ResizeObserver(() => {
        if (!fitHeatCanvas(id)) return;
        const grid = state ? state[HEAT_GRID_KEYS[id]] : null;
        drawHeatById(id, grid);
    }).observe(c.parentElement);
    fitHeatCanvas(id);
}

function drawHeatById(id, grid) {
    if (SIGNED_HEAT_IDS.has(id)) drawHeatSigned(id, grid);
    else drawHeat(id, grid);
}

function drawHeat(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    let maxV = 0;
    if (grid) for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (grid[r][k] > maxV) maxV = grid[r][k];
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
        g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (v >= 0.01) {
            g.fillStyle = a > 0.5 ? "#fff" : (cssVar("--heat-text") || "#111");
            g.font = `${Math.floor(cell * 0.38)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            g.fillText((v * 100).toFixed(0), x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function drawHeatSigned(canvasId, grid) {
    const g = heatCtxs[canvasId];
    clearLogical(g);
    const W = g._logicalW;
    const cell = W / N;
    const gridCol = cssVar("--heat-grid") || "#e5e7eb";
    const heatText = cssVar("--heat-text") || "#111";
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const x = k * cell, y = r * cell;
        const v = grid ? grid[r][k] : 0;
        const a = Math.min(1, Math.abs(v));
        if (v > 0) g.fillStyle = `rgba(9,105,218,${a.toFixed(3)})`;
        else if (v < 0) g.fillStyle = `rgba(207,34,46,${a.toFixed(3)})`;
        else g.fillStyle = "rgba(0,0,0,0)";
        g.fillRect(x, y, cell, cell);
        g.strokeStyle = gridCol;
        g.strokeRect(x + 0.5, y + 0.5, cell, cell);
        if (Math.abs(v) >= 0.05) {
            g.fillStyle = a > 0.5 ? "#fff" : heatText;
            g.font = `${Math.floor(cell * 0.32)}px ${MONO_FONT}`;
            g.textAlign = "center"; g.textBaseline = "middle";
            const label = (v >= 0 ? "+" : "") + (v * 100).toFixed(0);
            g.fillText(label, x + cell / 2, y + cell / 2);
        }
    }
    if (state && state.board) overlayStones(g, cell);
}

function overlayStones(g, cell) {
    const r0 = cell * 0.32;
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
        const sv = state.board[r][k]; if (!sv) continue;
        const cx = k * cell + cell / 2, cy = r * cell + cell / 2;
        g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI * 2);
        if (sv === 1) {
            g.fillStyle = "rgba(0,0,0,0.7)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,0.6)"; g.stroke();
        } else {
            g.fillStyle = "rgba(255,255,255,0.85)"; g.fill();
            g.lineWidth = 1; g.strokeStyle = "rgba(0,0,0,0.5)"; g.stroke();
        }
    }
}

// --- Heat modal (fullscreen single heatmap) ---
let expandedSourceId = null;
function setupModalCanvas() {
    const canvas = document.getElementById("heat_modal_canvas");
    const card = canvas.parentElement;
    const cardCS = getComputedStyle(card);
    const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
    const padY = parseFloat(cardCS.paddingTop)  + parseFloat(cardCS.paddingBottom);
    const header = card.querySelector(".heat-modal-header");
    const headerH = header ? header.offsetHeight + 12 : 0;
    const availW = window.innerWidth  * 0.95 - padX;
    const availH = window.innerHeight * 0.95 - padY - headerH;
    const sz = Math.max(240, Math.floor(Math.min(availW, availH)));
    canvas.style.width  = sz + "px";
    canvas.style.height = sz + "px";
    heatCtxs.h_modal = setupCanvas(canvas, sz, sz, false);
}
function paintHeatModal() {
    if (!expandedSourceId) return;
    const grid = state ? state[HEAT_GRID_KEYS[expandedSourceId]] : null;
    if (SIGNED_HEAT_IDS.has(expandedSourceId)) drawHeatSigned("h_modal", grid);
    else drawHeat("h_modal", grid);
}
function openHeatModal(sourceId) {
    if (!HEAT_GRID_KEYS[sourceId]) return;
    expandedSourceId = sourceId;
    const card = document.getElementById(sourceId).parentElement;
    const titleEl = card.querySelector(".grid-title-text");
    document.getElementById("heat_modal_title").textContent = titleEl ? titleEl.textContent : t("heatmap_default_title");
    document.getElementById("heat_modal").classList.remove("hidden");
    setupModalCanvas();
    paintHeatModal();
}
function closeHeatModal() {
    if (expandedSourceId === null) return;
    expandedSourceId = null;
    document.getElementById("heat_modal").classList.add("hidden");
}
for (const btn of document.querySelectorAll(".expand-btn")) {
    btn.addEventListener("click", () => openHeatModal(btn.dataset.target));
}
document.getElementById("heat_modal_close").addEventListener("click", closeHeatModal);
document.getElementById("heat_modal").addEventListener("click", (ev) => {
    if (ev.target === ev.currentTarget) closeHeatModal();
});
document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && expandedSourceId !== null) closeHeatModal();
});
window.addEventListener("resize", () => {
    if (expandedSourceId !== null) {
        setupModalCanvas();
        paintHeatModal();
    }
});

// --- Value chart ---
// Sizing is handled by CSS (#value_chart in style.css) so that mobile media
// queries can shrink the canvas — JS just keeps the bitmap in sync via the
// ResizeObserver below.
const vcCanvas = document.getElementById("value_chart");
let vctx = setupCanvas(vcCanvas, 280, 160, false);
function resizeValueChart() {
    const rect = vcCanvas.getBoundingClientRect();
    const w = Math.max(120, Math.floor(rect.width));
    const h = Math.max(120, Math.floor(rect.height));
    if (vctx._logicalW === w && vctx._logicalH === h) return;
    vctx = setupCanvas(vcCanvas, w, h);
    drawValueChart();
}
new ResizeObserver(resizeValueChart).observe(vcCanvas);

let valueHistory = [];   // [{step, root, nn}]

function stoneCount(board2d) {
    let n = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board2d[r][c]) n++;
    return n;
}
function normWL(v) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s > 1e-4) return (v.w - v.l) / s;
    return v.wl;
}
function recordValues(rootValueWDL, nnValueWDL, board2d) {
    if (!board2d) return;
    const step = stoneCount(board2d);
    const rw = normWL(rootValueWDL);
    const nw = normWL(nnValueWDL);
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > step) {
        valueHistory.pop();
    }
    const last = valueHistory[valueHistory.length - 1];
    if (rw == null && nw == null) {
        if (last && step > last.step) valueHistory.push({ step, root: last.root, nn: last.nn });
        return;
    }
    if (last && last.step === step) {
        if (rw != null) last.root = rw;
        if (nw != null) last.nn = nw;
    } else if (!last || step > last.step) {
        valueHistory.push({ step, root: rw, nn: nw });
    }
}
function drawValueChart() {
    clearLogical(vctx);
    const W = vctx._logicalW, H = vctx._logicalH;
    const padL = 22, padR = 6, padT = 6, padB = 14;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const grid = cssVar("--heat-grid") || "#e5e7eb";
    const muted = cssVar("--fg-muted") || "#59636e";
    const subtle = cssVar("--fg-subtle") || "#8b949e";
    const axis = cssVar("--border") || "#d8dee4";
    vctx.strokeStyle = grid; vctx.lineWidth = 1;
    for (const v of [-1, 0, 1]) {
        const y = padT + ((1 - v) / 2) * innerH + 0.5;
        vctx.beginPath(); vctx.moveTo(padL, y); vctx.lineTo(W - padR, y); vctx.stroke();
    }
    vctx.fillStyle = subtle;
    vctx.font = `10px ${MONO_FONT}`;
    vctx.textAlign = "right"; vctx.textBaseline = "middle";
    for (const v of [1, 0, -1]) {
        const y = padT + ((1 - v) / 2) * innerH;
        vctx.fillText((v > 0 ? "+" : "") + v.toFixed(0), padL - 4, y);
    }
    vctx.strokeStyle = axis;
    vctx.beginPath();
    vctx.moveTo(padL + 0.5, padT); vctx.lineTo(padL + 0.5, H - padB);
    vctx.lineTo(W - padR, H - padB); vctx.stroke();
    if (valueHistory.length === 0) {
        vctx.fillStyle = subtle;
        vctx.font = `11px ${MONO_FONT}`;
        vctx.textAlign = "center"; vctx.textBaseline = "middle";
        vctx.fillText(t("chart_no_data"), padL + innerW / 2, padT + innerH / 2);
        return;
    }
    const maxStep = Math.max(1, valueHistory[valueHistory.length - 1].step);
    const xOf = (s) => padL + (s / maxStep) * innerW;
    const yOf = (v) => padT + ((1 - v) / 2) * innerH;
    vctx.fillStyle = muted;
    vctx.textAlign = "center"; vctx.textBaseline = "top";
    vctx.fillText("0", xOf(0), H - padB + 2);
    vctx.fillText(String(maxStep), xOf(maxStep), H - padB + 2);
    function plot(key, color) {
        const pts = valueHistory.filter(p => p[key] != null);
        if (pts.length === 0) return;
        vctx.strokeStyle = color; vctx.lineWidth = 1.5;
        vctx.beginPath();
        pts.forEach((p, i) => {
            const x = xOf(p.step), y = yOf(p[key]);
            if (i === 0) vctx.moveTo(x, y); else vctx.lineTo(x, y);
        });
        vctx.stroke();
        vctx.fillStyle = color;
        for (const p of pts) {
            vctx.beginPath();
            vctx.arc(xOf(p.step), yOf(p[key]), 2, 0, Math.PI * 2);
            vctx.fill();
        }
    }
    plot("root", "#0969da");
    if (document.body.classList.contains("show-analysis")) {
        plot("nn", "#cf222e");
    }
}

// --- WDL bars ---
let lastRootWDL = null;
let lastNNWDL = null;
function normalizeWDL(v) {
    if (!v) return null;
    const s = v.w + v.d + v.l;
    if (s > 1e-4) return { w: v.w / s * 100, d: v.d / s * 100, l: v.l / s * 100,
                           wl: (v.w - v.l) / s };
    return { w: v.w, d: v.d, l: v.l, wl: v.wl };
}
function renderWDL(prefix, vWDL) {
    if (prefix === "root") lastRootWDL = vWDL || null;
    else if (prefix === "nn") lastNNWDL = vWDL || null;
    const bar = document.getElementById("wdl_" + prefix + "_bar");
    const wlEl = document.getElementById("wdl_" + prefix + "_wl");
    const det = document.getElementById("wdl_" + prefix + "_detail");
    const n = normalizeWDL(vWDL);
    const segs = bar.querySelectorAll(".seg");
    if (!n) {
        segs[0].style.width = "0";
        segs[1].style.width = "100%";
        segs[2].style.width = "0";
        wlEl.textContent = t("wdl_dash");
        wlEl.classList.remove("pos", "neg");
        det.textContent = "";
        return;
    }
    segs[0].style.width = n.w.toFixed(2) + "%";
    segs[1].style.width = n.d.toFixed(2) + "%";
    segs[2].style.width = n.l.toFixed(2) + "%";
    wlEl.textContent = (n.wl >= 0 ? "+" : "") + n.wl.toFixed(2);
    wlEl.classList.toggle("pos", n.wl > 0.01);
    wlEl.classList.toggle("neg", n.wl < -0.01);
    det.innerHTML =
        '<span><span class="k">W</span> ' + n.w.toFixed(1) + "%</span>" +
        '<span><span class="k">D</span> ' + n.d.toFixed(1) + "%</span>" +
        '<span><span class="k">L</span> ' + n.l.toFixed(1) + "%</span>";
}

// --- Model dropdown (loads from models/manifest.json) ---
let manifest = { default: null, models: [] };
let currentModelId = null;

async function loadManifest() {
    const r = await fetch("models/manifest.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("manifest.json fetch failed: " + r.status);
    manifest = await r.json();
    const sel = document.getElementById("model_select");
    sel.innerHTML = "";
    // Sort by ELO ascending so "新手" sits at the top.
    const items = manifest.models.slice().sort((a, b) => a.elo - b.elo);
    for (const m of items) {
        const o = document.createElement("option");
        o.value = m.id;
        const eloStr = (m.elo >= 0 ? "+" : "") + m.elo;
        o.textContent = `${m.id.toUpperCase()} ${m.label} · ELO ${eloStr}`;
        sel.appendChild(o);
    }
    currentModelId = manifest.default || items[0].id;
    sel.value = currentModelId;
}

function modelById(id) {
    return manifest.models.find(m => m.id === id);
}

// --- Board size dropdown (hard-coded 13-17) ---
const BOARD_SIZES = [17, 16, 15, 14, 13];
function populateSizeSelect() {
    const sel = document.getElementById("size_select");
    sel.innerHTML = "";
    for (const sz of BOARD_SIZES) {
        const o = document.createElement("option");
        o.value = String(sz);
        o.textContent = String(sz);
        sel.appendChild(o);
    }
    sel.value = String(N);
}

// --- Loading overlay (only shown for first model load) ---
function setLoadingProgress(pct) {
    const fill = document.getElementById("loading_fill");
    const text = document.getElementById("loading_pct");
    if (!fill || !text) return;
    if (Number.isFinite(pct)) {
        fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
        text.textContent = Math.round(pct) + "%";
    }
}
function hideLoadingOverlay() {
    const o = document.getElementById("loading_overlay");
    if (o) o.style.display = "none";
    lastLoadingMsg = null;
}
let lastLoadingMsg = null; // { key, args }
function showLoadingOverlay(key, ...args) {
    const o = document.getElementById("loading_overlay");
    if (!o) return;
    o.style.display = "";
    lastLoadingMsg = { key, args };
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = t(key, ...args);
    setLoadingProgress(0);
}
function rerenderLoadingOverlay() {
    if (!lastLoadingMsg) return;
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = t(lastLoadingMsg.key, ...lastLoadingMsg.args);
}

// --- Game state (mirrors V5 gomoku semantics, in-browser) ---
let game = null;          // Gomoku instance (rebuilt on size change)
let boardState = null;    // Int8Array(N*N), +1 / -1 / 0
let toPlay = 1;           // 1 = black, -1 = white
let humanSide = 1;        // 1 = human is black; -1 = human is white
let currentRule = "renju"; // "renju" | "standard" | "freestyle"
let lastMove = null;      // { r, c }
let ply = 0;              // half-move counter
let gameOver = false;
let aiThinking = false;
let searchId = 0;
let history = [];         // [{ board: Int8Array, toPlay, lastMove, ply, gumbelPhases }]

const worker = new Worker("worker.js?v=" + Date.now());
worker.onerror = (e) => {
    const where = e.filename ? ` (${e.filename}:${e.lineno})` : "";
    const msg = t("err_worker_failed", e.message || t("err_unknown"), where);
    setStatusRaw(msg, "error");
    const tEl = document.getElementById("loading_text");
    if (tEl) tEl.textContent = msg;
    console.error("[worker.onerror]", e);
};
worker.onmessageerror = (e) => {
    setStatus("err_worker_msg", "error");
    console.error("[worker.onmessageerror]", e);
};

let lastStatus = { key: "status_idle", args: [], variant: "idle", raw: null };
function setStatus(key, variant, ...args) {
    lastStatus = { key, args, variant, raw: null };
    document.getElementById("status").textContent = t(key, ...args);
    document.getElementById("status_pill").dataset.variant = variant || "idle";
}
function setStatusRaw(text, variant) {
    lastStatus = { key: null, args: [], variant, raw: text };
    document.getElementById("status").textContent = text;
    document.getElementById("status_pill").dataset.variant = variant || "idle";
}
function rerenderStatus() {
    if (lastStatus.raw != null) return;
    if (!lastStatus.key) return;
    document.getElementById("status").textContent = t(lastStatus.key, ...lastStatus.args);
}

function board1Dto2D(b1d) {
    const b = [];
    for (let r = 0; r < N; r++) {
        b.push([]);
        for (let c = 0; c < N; c++) b[r].push(b1d[r * N + c]);
    }
    return b;
}

function publishStateForDrawing(extras = {}) {
    state = {
        board: board1Dto2D(boardState),
        last_move: lastMove ? [lastMove.r, lastMove.c] : null,
        board_size: N,
        // Heat data (set by handleResult; null otherwise — drawHeat handles nulls).
        mcts_policy:    extras.mcts_policy    || null,
        mcts_visits:    extras.mcts_visits    || null,
        nn_policy:      extras.nn_policy      || null,
        nn_opp_policy:  extras.nn_opp_policy  || null,
        nn_futurepos_8: extras.nn_futurepos_8 || null,
        nn_futurepos_32:extras.nn_futurepos_32|| null,
    };
}

function repaintAllHeatmaps() {
    drawHeat("h_mcts_policy",    state ? state.mcts_policy    : null);
    drawHeat("h_mcts_visits",    state ? state.mcts_visits    : null);
    drawHeat("h_nn_policy",      state ? state.nn_policy      : null);
    drawHeat("h_nn_opp_policy",  state ? state.nn_opp_policy  : null);
    drawHeatSigned("h_nn_futurepos_8",  state ? state.nn_futurepos_8  : null);
    drawHeatSigned("h_nn_futurepos_32", state ? state.nn_futurepos_32 : null);
    paintHeatModal();
}

function drawAll() {
    draw();
    drawValueChart();
    repaintAllHeatmaps();
}

// Convert flat Float32Array(N*N) → 2D N×N for heatmap render.
function flatToGrid(flat) {
    const g = [];
    for (let r = 0; r < N; r++) {
        g.push([]);
        for (let c = 0; c < N; c++) g[r].push(flat[r * N + c]);
    }
    return g;
}

function newGame() {
    game = new Gomoku(N, currentRule);
    boardState = game.getInitialState();
    toPlay = 1;
    lastMove = null;
    ply = 0;
    gameOver = false;
    history = [];
    valueHistory = [];
    gumbelPhases = null;
    renderWDL("root", null);
    renderWDL("nn", null);
    publishStateForDrawing();
    drawAll();
    worker.postMessage({ type: "reset", boardSize: N, ply: 0, rule: currentRule });
    if (humanSide === toPlay) {
        setStatus("status_your_turn", "active");
    } else {
        triggerAISearch();
    }
}

function triggerAISearch() {
    if (gameOver) return;
    aiThinking = true;
    searchId++;
    setStatus("status_ai_thinking", "thinking");
    const simsRaw = parseInt(document.getElementById("sims_input").value, 10);
    const sims = (Number.isFinite(simsRaw) && simsRaw >= 0) ? simsRaw : 32;
    worker.postMessage({
        type: "search",
        state: boardState,
        toPlay: toPlay,
        ply: ply,
        sims: sims,
        gumbel_m: 16,
        searchId: searchId,
    });
}

function applyMoveLocal(action) {
    history.push({
        board: new Int8Array(boardState),
        toPlay,
        lastMove: lastMove ? { ...lastMove } : null,
        ply,
        gumbelPhases,
        rootWDL: lastRootWDL,
        nnWDL: lastNNWDL,
    });
    const r = (action / N) | 0, c = action % N;
    boardState = game.getNextState(boardState, action, toPlay);
    lastMove = { r, c };
    const winner = game.getWinner(boardState, action, toPlay);
    const movedBy = toPlay;
    toPlay = -toPlay;
    ply++;
    if (winner !== null) {
        gameOver = true;
        let key;
        if      (winner === 1)  key = "status_black_wins";
        else if (winner === -1) key = "status_white_wins";
        else                    key = "status_draw";
        setStatus(key, "done");
    }
    publishStateForDrawing(state || {});
    drawAll();
    worker.postMessage({
        type: "move",
        action,
        nextState: boardState,
        nextToPlay: toPlay,
        ply,
    });
    return { winner, movedBy };
}

// --- Board click (human move) ---
cv.addEventListener("click", (ev) => {
    if (gameOver || aiThinking) return;
    if (toPlay !== humanSide) return;
    const rect = cv.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const c = Math.round((x - MARGIN) / CELL), r = Math.round((y - MARGIN) / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    if (boardState[r * N + c] !== 0) return;
    const legal = game.getLegalActions(boardState, toPlay);
    if (!legal[r * N + c]) return;   // Renju forbidden for black, etc.
    const { winner } = applyMoveLocal(r * N + c);
    if (winner === null) triggerAISearch();
});

// --- Worker message router ---
worker.onmessage = (e) => {
    const data = e.data;
    if (data.type === "model-progress") {
        if (Number.isFinite(data.percent)) setLoadingProgress(data.percent);
        return;
    }
    if (data.type === "ready") {
        hideLoadingOverlay();
        // First-ready means model is loaded; if it's a swap, just resume.
        if (!boardState) newGame();
        return;
    }
    if (data.type === "error") {
        setStatusRaw(t("err_prefix", data.message), "error");
        aiThinking = false;
        return;
    }
    if (data.type === "progress") {
        if (data.searchId !== searchId) return;
        // Could update a progress bar here; for now status pill stays "thinking".
        return;
    }
    if (data.type === "result") {
        if (data.searchId !== searchId) return;
        aiThinking = false;
        // Update gumbel overlay + heatmaps.
        gumbelPhases = data.gumbelPhases;
        publishStateForDrawing({
            mcts_policy:    flatToGrid(data.mctsPolicy),
            mcts_visits:    flatToGrid(data.mctsVisits),
            nn_policy:      flatToGrid(data.nnPolicy),
            nn_opp_policy:  flatToGrid(data.nnOppPolicy),
            nn_futurepos_8: flatToGrid(data.nnFuturepos8),
            nn_futurepos_32:flatToGrid(data.nnFuturepos32),
        });
        // WDL update.
        const rootWDL = data.rootValueWDL ? { w: data.rootValueWDL[0], d: data.rootValueWDL[1],
                                              l: data.rootValueWDL[2],
                                              wl: data.rootValueWDL[0] - data.rootValueWDL[2] } : null;
        const nnWDL   = data.nnValueWDL   ? { w: data.nnValueWDL[0], d: data.nnValueWDL[1],
                                              l: data.nnValueWDL[2],
                                              wl: data.nnValueWDL[0] - data.nnValueWDL[2] } : null;
        renderWDL("root", rootWDL);
        renderWDL("nn",   nnWDL);
        recordValues(rootWDL, nnWDL, state.board);
        drawAll();

        // AI plays its chosen move.
        const { winner } = applyMoveLocal(data.gumbelAction);
        if (winner === null && humanSide !== toPlay) {
            // AI vs AI? Should not happen in normal play. Stop.
        } else if (winner === null) {
            setStatus("status_your_turn", "active");
        }
    }
};

// Pure-NN mode hint: visible iff sims_input parses to 0.
function updateSimsModeHint() {
    const el = document.getElementById("sims_input");
    const hint = document.getElementById("sims_mode_hint");
    if (!el || !hint) return;
    const n = parseInt(el.value, 10);
    hint.hidden = !(Number.isFinite(n) && n === 0);
}
document.getElementById("sims_input").addEventListener("input", updateSimsModeHint);
updateSimsModeHint();

// --- Buttons ---
document.getElementById("new_btn").addEventListener("click", newGame);

document.getElementById("undo_btn").addEventListener("click", () => {
    if (history.length === 0) return;
    // Undo enough plies so the next move is the human's.
    let target = history.length;
    if (toPlay !== humanSide) target = Math.max(0, target - 1);
    else                      target = Math.max(0, target - 2);
    if (target === history.length) return;
    let restoredRootWDL = null, restoredNNWDL = null;
    while (history.length > target) {
        const prev = history.pop();
        boardState = prev.board;
        toPlay = prev.toPlay;
        lastMove = prev.lastMove;
        ply = prev.ply;
        gumbelPhases = prev.gumbelPhases;
        restoredRootWDL = prev.rootWDL || null;
        restoredNNWDL = prev.nnWDL || null;
    }
    gameOver = false;
    aiThinking = false;
    searchId++;   // abort any in-flight search
    while (valueHistory.length && valueHistory[valueHistory.length - 1].step > stoneCount(board1Dto2D(boardState))) {
        valueHistory.pop();
    }
    renderWDL("root", restoredRootWDL);
    renderWDL("nn", restoredNNWDL);
    publishStateForDrawing();
    drawAll();
    worker.postMessage({ type: "reset", boardSize: N, ply, rule: currentRule });
    if (toPlay === humanSide) {
        setStatus("status_your_turn", "active");
    } else {
        triggerAISearch();
    }
});

// Side toggle buttons.
function setSide(side) {
    if (side !== 1 && side !== -1) return;
    humanSide = side;
    document.getElementById("side_black").setAttribute("aria-pressed", side === 1 ? "true" : "false");
    document.getElementById("side_white").setAttribute("aria-pressed", side === -1 ? "true" : "false");
    newGame();
}
document.getElementById("side_black").addEventListener("click", () => setSide(1));
document.getElementById("side_white").addEventListener("click", () => setSide(-1));

// Rule toggle buttons (renju / standard / freestyle).
function setRule(rl) {
    if (rl !== "renju" && rl !== "standard" && rl !== "freestyle") return;
    if (rl === currentRule) return;
    currentRule = rl;
    for (const b of document.querySelectorAll(".seg-btn[data-rule]")) {
        b.setAttribute("aria-pressed", b.dataset.rule === rl ? "true" : "false");
    }
    newGame();
}
for (const b of document.querySelectorAll(".seg-btn[data-rule]")) {
    b.addEventListener("click", () => setRule(b.dataset.rule));
}

// Size dropdown.
document.getElementById("size_select").addEventListener("change", (ev) => {
    const sz = parseInt(ev.target.value, 10);
    if (!Number.isFinite(sz) || !BOARD_SIZES.includes(sz)) return;
    N = sz;
    syncBoardSize();
    newGame();
});

// Model dropdown.
document.getElementById("model_select").addEventListener("change", (ev) => {
    const id = ev.target.value;
    const m = modelById(id);
    if (!m) return;
    currentModelId = id;
    showLoadingOverlay("loading_model", m.label);
    searchId++;
    aiThinking = false;
    worker.postMessage({ type: "swap-model", modelUrl: "models/" + m.file });
});

// Re-render dynamic strings (status, loading overlay, modal title,
// value-chart 'no data') when the user switches language.
registerI18nCallback(() => {
    rerenderStatus();
    rerenderLoadingOverlay();
    if (expandedSourceId) {
        const card = document.getElementById(expandedSourceId).parentElement;
        const titleEl = card.querySelector(".grid-title-text");
        document.getElementById("heat_modal_title").textContent =
            titleEl ? titleEl.textContent : t("heatmap_default_title");
    }
    if (typeof drawValueChart === "function") drawValueChart();
});

// --- Bootstrap on load ---
(async function bootstrap() {
    populateSizeSelect();
    showLoadingOverlay("loading_manifest");
    try {
        await loadManifest();
    } catch (err) {
        setStatusRaw(t("err_manifest_load", err.message), "error");
        return;
    }
    const startModel = modelById(currentModelId) || manifest.models[0];
    if (!startModel) {
        setStatus("err_manifest_empty", "error");
        return;
    }
    showLoadingOverlay("loading_model", startModel.label);
    worker.postMessage({
        type: "init",
        modelUrl: "models/" + startModel.file,
        boardSize: N,
        rule: currentRule,
    });
    syncBoardSize();
})();
