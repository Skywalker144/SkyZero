#!/usr/bin/env python3
"""Minimal HTTP front-end for cpp/build/gomoku_play.

Single-user, single-process. Spawns the C++ engine as a subprocess, parses
its stdout into a JSON state blob, and serves a page that polls /state and
posts to /move. Intended for local use behind VSCode port forwarding.
"""
import argparse
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOARD_SIZE = 15

HEADER_RE = re.compile(r"^\s*0\s+1\s+2\s+3\s+4\s+5")
AI_MOVE_RE = re.compile(r"AI move:\s*\((\d+),\s*(\d+)\)")
RESULT_RE = re.compile(r"(Black wins!|White wins!|Draw!)")
ROOT_VALUE_RE = re.compile(
    r"root:\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([+-]?[\d.]+)"
)
NN_VALUE_RE = re.compile(
    r"nn:\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([+-]?[\d.]+)"
)
GUMBEL_PHASE_RE = re.compile(r"Gumbel Phase (\d+) \((\d+)\):(.*)$")


class EngineSession:
    """Wraps one gomoku_play subprocess; reader thread updates self.state."""

    def __init__(self, play_bin, model, config, human_side):
        self.play_bin = play_bin
        self.model = model
        self.config = config
        self.human_side = human_side

        self.lock = threading.Lock()
        self.version = 0
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.last_move = None
        self.status = "Launching engine..."
        self.root_value = None  # {w,d,l,wl}
        self.nn_value = None
        self.game_over = False
        self.mcts_policy = None  # 15x15
        self.mcts_visits = None
        self.nn_policy = None
        self.gumbel_phases = None  # list of list of [r,c], index 0 = initial 16, last = final 1

        self._pending_rows = None
        self._pending_grid_key = None
        self._pending_grid_rows = None

        self.proc = subprocess.Popen(
            [str(play_bin), "--model", str(model), "--config", str(config),
             "--human-side", str(human_side)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _bump(self):
        self.version += 1

    def _read_loop(self):
        try:
            for raw in self.proc.stdout:
                line = raw.rstrip("\n")
                with self.lock:
                    self._parse_line(line)
                    self._bump()
        except Exception:
            with self.lock:
                self._bump()
        with self.lock:
            self.status = "Engine exited."
            self._bump()

    @staticmethod
    def _try_parse_grid_row(line):
        toks = line.split()
        if len(toks) != BOARD_SIZE:
            return None
        row = []
        for t in toks:
            if t == ".":
                row.append(0.0)
            else:
                try:
                    row.append(float(t))
                except ValueError:
                    return None
        return row

    def _parse_line(self, line):
        if self._pending_rows is not None:
            self._pending_rows.append(line)
            if len(self._pending_rows) == BOARD_SIZE:
                self._apply_board(self._pending_rows)
                self._pending_rows = None
            return
        if HEADER_RE.search(line):
            self._pending_rows = []
            return
        if "MCTS Strategy" in line:
            self._pending_grid_key = "mcts_policy"
            self._pending_grid_rows = []
            return
        if "MCTS Visits" in line:
            self._pending_grid_key = "mcts_visits"
            self._pending_grid_rows = []
            return
        if "NN Strategy" in line:
            self._pending_grid_key = "nn_policy"
            self._pending_grid_rows = []
            return
        if self._pending_grid_key is not None:
            row = self._try_parse_grid_row(line)
            if row is not None:
                self._pending_grid_rows.append(row)
                if len(self._pending_grid_rows) == BOARD_SIZE:
                    setattr(self, self._pending_grid_key, self._pending_grid_rows)
                    self._pending_grid_key = None
                    self._pending_grid_rows = None
                return
            self._pending_grid_key = None
            self._pending_grid_rows = None
            # fall through to other matchers
        m = GUMBEL_PHASE_RE.search(line)
        if m:
            idx = int(m.group(1))
            coords = []
            for tok in m.group(3).split():
                rc = tok.split(",")
                if len(rc) == 2:
                    try:
                        coords.append([int(rc[0]), int(rc[1])])
                    except ValueError:
                        pass
            if self.gumbel_phases is None or idx == 0:
                self.gumbel_phases = []
            while len(self.gumbel_phases) <= idx:
                self.gumbel_phases.append([])
            self.gumbel_phases[idx] = coords
            return
        m = AI_MOVE_RE.search(line)
        if m:
            self.status = f"AI played ({m.group(1)}, {m.group(2)})"
            return
        m = RESULT_RE.search(line)
        if m:
            self.status = m.group(1)
            self.game_over = True
            return
        m = ROOT_VALUE_RE.search(line)
        if m:
            self.root_value = {
                "w": float(m.group(1)), "d": float(m.group(2)),
                "l": float(m.group(3)), "wl": float(m.group(4)),
            }
            return
        m = NN_VALUE_RE.search(line)
        if m:
            self.nn_value = {
                "w": float(m.group(1)), "d": float(m.group(2)),
                "l": float(m.group(3)), "wl": float(m.group(4)),
            }
            return
        if "Human step" in line:
            if not self.game_over:
                self.status = "Your turn"
            return
        if "AlphaZero thinking" in line:
            self.status = "AI thinking..."
            self.gumbel_phases = None
            return
        if "Invalid move" in line or "Invalid input" in line:
            self.status = line.strip()
            return

    def _apply_board(self, rows):
        new_board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        last = None
        for r, line in enumerate(rows):
            body = line[3:] if len(line) > 3 else ""
            for c in range(BOARD_SIZE):
                base = c * 3
                if base + 2 >= len(body):
                    break
                mid = body[base + 1]
                if mid == "X":
                    new_board[r][c] = 1
                elif mid == "O":
                    new_board[r][c] = -1
                if body[base] == "[":
                    last = (r, c)
        self.board = new_board
        self.last_move = last

    def send(self, text):
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.send("q")
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def snapshot(self):
        with self.lock:
            return {
                "version": self.version,
                "board": self.board,
                "last_move": list(self.last_move) if self.last_move else None,
                "status": self.status,
                "root_value": self.root_value,
                "nn_value": self.nn_value,
                "game_over": self.game_over,
                "human_side": self.human_side,
                "mcts_policy": self.mcts_policy,
                "mcts_visits": self.mcts_visits,
                "nn_policy": self.nn_policy,
                "gumbel_phases": self.gumbel_phases,
            }


class App:
    def __init__(self, play_bin, model, config):
        self.play_bin = play_bin
        self.model = model
        self.config = config
        self.session = None
        self.session_lock = threading.Lock()

    def start(self, human_side):
        with self.session_lock:
            if self.session is not None:
                self.session.stop()
            self.session = EngineSession(self.play_bin, self.model, self.config, human_side)

    def current(self):
        with self.session_lock:
            return self.session


HTML_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SkyZero Gomoku</title>
<style>
  :root {
    --bg: #f4f2ee;
    --panel: #ffffff;
    --ink: #1f2328;
    --muted: #57606a;
    --border: #d0d7de;
    --accent: #2563eb;
    --accent-ink: #ffffff;
    --board: #e8c583;
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.05);
  }
  * { box-sizing: border-box; }
  html, body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }
  body {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                 system-ui, sans-serif;
    font-size: 14px; line-height: 1.5;
    margin: 0; padding: 24px 20px 40px; background: var(--bg); color: var(--ink);
    font-feature-settings: "cv11", "ss01", "tnum";
  }
  .wrap { max-width: 1100px; margin: 0 auto; }
  h1 {
    font-size: 18px; font-weight: 600; margin: 0 0 14px;
    letter-spacing: -0.01em; text-align: center; color: var(--ink);
  }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; box-shadow: var(--shadow); margin-bottom: 14px;
  }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button {
    padding: 6px 14px; font-size: 13px; font-weight: 500;
    background: #fff; color: var(--ink);
    border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
  }
  button:hover { background: #f6f8fa; border-color: #adb5bd; }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover { background: #1d4ed8; border-color: #1d4ed8; }
  .sep { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }
  #status { font-weight: 600; font-size: 13px; color: var(--ink); }
  #values {
    font-family: "JetBrains Mono", "SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px; color: var(--muted); white-space: pre; line-height: 1.6;
    font-variant-numeric: tabular-nums;
  }
  .warn { color: #b45309; font-weight: 500; }
  /* toggle switch */
  .toggle { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; font-size: 13px; }
  .toggle input { display: none; }
  .toggle .track {
    width: 34px; height: 18px; background: #d0d7de; border-radius: 999px;
    position: relative; transition: background 0.15s;
  }
  .toggle .track::after {
    content: ""; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%; background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2); transition: transform 0.15s;
  }
  .toggle input:checked + .track { background: var(--accent); }
  .toggle input:checked + .track::after { transform: translateX(16px); }
  .num {
    width: 64px; padding: 4px 6px; font-size: 13px;
    border: 1px solid var(--border); border-radius: 6px;
    font-family: "JetBrains Mono", "SF Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }
  .field { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); }

  .board-panel {
    padding: 14px;
    display: inline-flex; flex-direction: column; align-items: center;
  }
  .board-wrap { display: flex; justify-content: center; margin-bottom: 14px; }
  #board { background: var(--board); border-radius: 6px; display: block; cursor: crosshair; }

  #gumbel_legend {
    font-family: "JetBrains Mono", "SF Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px; color: var(--muted); margin-top: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: 4px 14px;
  }
  #gumbel_legend .lbl { display: inline-flex; align-items: center; gap: 5px; }
  #gumbel_legend .dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  }

  .grids {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
  }
  .grid-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px; box-shadow: var(--shadow); text-align: center;
  }
  .grid-card .title {
    font-size: 12px; font-weight: 500; color: var(--muted); margin-bottom: 8px;
    font-family: "JetBrains Mono", "SF Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.01em;
  }
  .heat {
    background: #fff; border: 1px solid var(--border); border-radius: 4px;
    display: block; margin: 0 auto;
  }
  .hidden { display: none !important; }
</style></head>
<body>
<div class="wrap">
  <h1>SkyZero Gomoku</h1>

  <div class="panel">
    <div class="row">
      <button class="primary" onclick="newGame(1)">New · Black (first)</button>
      <button class="primary" onclick="newGame(-1)">New · White (second)</button>
      <button onclick="sendCmd('u')">Undo</button>
      <div class="sep"></div>
      <label class="toggle">
        <input type="checkbox" id="gumbel_toggle" checked>
        <span class="track"></span>
        <span>Gumbel overlay</span>
      </label>
      <div class="sep"></div>
      <label class="field">sims
        <input class="num" type="number" id="sims_input" min="1" step="1" value="800">
      </label>
      <button onclick="applySims()">Apply</button>
      <label class="field">gumbel_m
        <input class="num" type="number" id="gm_input" min="1" step="1" value="8">
      </label>
      <button onclick="applyGm()">Apply</button>
      <label class="toggle">
        <input type="checkbox" id="noise_toggle" checked onchange="applyNoise()">
        <span class="track"></span>
        <span>Gumbel noise</span>
      </label>
      <div class="sep"></div>
      <span id="status">idle</span>
    </div>
    <div id="values" style="margin-top:10px;"></div>
  </div>

  <div class="board-wrap">
    <div class="panel board-panel">
      <canvas id="board"></canvas>
      <div id="gumbel_legend">
        <span>Gumbel Sequential Halving:</span>
        <span class="lbl"><span class="dot" style="background:#9ca3af;"></span>16</span>
        <span class="lbl"><span class="dot" style="background:#3b82f6;"></span>8</span>
        <span class="lbl"><span class="dot" style="background:#10b981;"></span>4</span>
        <span class="lbl"><span class="dot" style="background:#f59e0b;"></span>2</span>
        <span class="lbl"><span class="dot" style="background:#ef4444;"></span>1 (picked)</span>
      </div>
    </div>
  </div>

  <div class="grids">
    <div class="grid-card"><div class="title">MCTS Strategy (improved policy)</div>
      <canvas class="heat" id="h_mcts_policy"></canvas></div>
    <div class="grid-card"><div class="title">MCTS Visits (N/sum)</div>
      <canvas class="heat" id="h_mcts_visits"></canvas></div>
    <div class="grid-card"><div class="title">NN Strategy</div>
      <canvas class="heat" id="h_nn_policy"></canvas></div>
  </div>
</div>

<script>
const N = 15, CELL = 36, MARGIN = 28;
const BOARD_LOGICAL = MARGIN*2 + CELL*(N-1); // 2*28 + 36*14 = 560
const HEAT_LOGICAL = 300;
const DPR = window.devicePixelRatio || 1;

function setupCanvas(canvas, logicalW, logicalH) {
  canvas.width = Math.round(logicalW * DPR);
  canvas.height = Math.round(logicalH * DPR);
  canvas.style.width = logicalW + 'px';
  canvas.style.height = logicalH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx._logicalW = logicalW;
  ctx._logicalH = logicalH;
  return ctx;
}
function clearLogical(ctx) { ctx.clearRect(0, 0, ctx._logicalW, ctx._logicalH); }

const cv = document.getElementById('board');
const ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
const heatCtxs = {
  h_mcts_policy: setupCanvas(document.getElementById('h_mcts_policy'), HEAT_LOGICAL, HEAT_LOGICAL),
  h_mcts_visits: setupCanvas(document.getElementById('h_mcts_visits'), HEAT_LOGICAL, HEAT_LOGICAL),
  h_nn_policy:   setupCanvas(document.getElementById('h_nn_policy'),   HEAT_LOGICAL, HEAT_LOGICAL),
};

let state = null;
let showGumbel = true;
const gumbelToggle = document.getElementById('gumbel_toggle');
const gumbelLegend = document.getElementById('gumbel_legend');
gumbelToggle.addEventListener('change', () => {
  showGumbel = gumbelToggle.checked;
  gumbelLegend.classList.toggle('hidden', !showGumbel);
  draw();
});

function drawHeat(canvasId, grid) {
  const g = heatCtxs[canvasId];
  clearLogical(g);
  const W = g._logicalW;
  const cell = W / N;
  let maxV = 0;
  if (grid) for (let r=0;r<N;r++) for (let k=0;k<N;k++) if (grid[r][k]>maxV) maxV=grid[r][k];
  for (let r=0;r<N;r++) for (let k=0;k<N;k++) {
    const x = k*cell, y = r*cell;
    let v = grid ? grid[r][k] : 0;
    const a = (maxV>0 && v>0) ? Math.min(1, v/maxV) : 0;
    g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
    g.fillRect(x, y, cell, cell);
    g.strokeStyle = '#e5e7eb';
    g.strokeRect(x + 0.5, y + 0.5, cell, cell);
    if (v >= 0.01) {
      g.fillStyle = a > 0.5 ? '#fff' : '#111';
      g.font = `${Math.floor(cell*0.38)}px ui-monospace, monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText((v*100).toFixed(0), x + cell/2, y + cell/2);
    }
  }
}

function draw() {
  clearLogical(ctx);
  // Grid lines: +0.5 for crisp 1px strokes.
  ctx.strokeStyle = '#6b5a3a'; ctx.lineWidth = 1;
  for (let i=0;i<N;i++){
    const p = Math.round(MARGIN + i*CELL) + 0.5;
    ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL*(N-1), p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL*(N-1)); ctx.stroke();
  }
  ctx.fillStyle = '#3a2e1a';
  for (const [r,c] of [[3,3],[3,11],[11,3],[11,11],[7,7]]) {
    ctx.beginPath(); ctx.arc(MARGIN+c*CELL, MARGIN+r*CELL, 3.5, 0, Math.PI*2); ctx.fill();
  }
  ctx.fillStyle = '#6b5a3a';
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i=0;i<N;i++){
    ctx.fillText(i, MARGIN + i*CELL, 12);
    ctx.fillText(i, 10, MARGIN + i*CELL);
  }
  if (!state) return;

  // Gumbel Sequential Halving overlay (toggleable).
  const phases = state.gumbel_phases;
  if (showGumbel && phases && phases.length > 0) {
    const COLORS = ['#9ca3af', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'];
    const LABELS = ['16','8','4','2','1'];
    const deepest = new Map();
    for (let i = 0; i < phases.length; i++) {
      for (const rc of phases[i]) {
        const key = rc[0]*N + rc[1];
        if (!deepest.has(key) || deepest.get(key) < i) deepest.set(key, i);
      }
    }
    for (const [key, idx] of deepest) {
      const r = Math.floor(key / N), c = key % N;
      const sizeLabel = String(phases[idx].length);
      let bucket = LABELS.indexOf(sizeLabel);
      if (bucket < 0) bucket = Math.min(idx, COLORS.length-1);
      const x = MARGIN+c*CELL, y = MARGIN+r*CELL;
      ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI*2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = COLORS[bucket];
      ctx.stroke();
      ctx.lineWidth = 1;
      if (state.board[r][c] === 0) {
        ctx.fillStyle = COLORS[bucket];
        ctx.font = 'bold 10px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(sizeLabel, x, y);
      }
    }
  }

  // Stones.
  const b = state.board, lm = state.last_move;
  for (let r=0;r<N;r++) for (let c=0;c<N;c++){
    const v = b[r][c]; if (!v) continue;
    const x = MARGIN+c*CELL, y = MARGIN+r*CELL;
    // subtle stone shadow
    ctx.beginPath(); ctx.arc(x+0.5, y+1.5, 14, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI*2);
    if (v === 1) {
      const grad = ctx.createRadialGradient(x-4, y-4, 2, x, y, 14);
      grad.addColorStop(0, '#555'); grad.addColorStop(1, '#000');
      ctx.fillStyle = grad;
    } else {
      const grad = ctx.createRadialGradient(x-4, y-4, 2, x, y, 14);
      grad.addColorStop(0, '#ffffff'); grad.addColorStop(1, '#d5d5d5');
      ctx.fillStyle = grad;
    }
    ctx.fill();
    ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1; ctx.stroke();
    if (lm && lm[0]===r && lm[1]===c) {
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2);
      ctx.fillStyle = '#ef4444'; ctx.fill();
    }
  }
}

// Renormalizes raw W/D/L so they display summing to ~100%. Returns
// {w,d,l,wl,sum} all in percent; `sum` is the *raw* sum (pre-normalize)
// so the caller can flag big deviations.
function normalizeVal(v) {
  if (!v) return null;
  const s = v.w + v.d + v.l;
  if (s > 1e-4) {
    return { w: v.w/s*100, d: v.d/s*100, l: v.l/s*100,
             wl: (v.w - v.l)/s, sum: s };
  }
  return { w: v.w, d: v.d, l: v.l, wl: v.wl, sum: s };
}

function fmtVal(v) {
  const n = normalizeVal(v);
  if (!n) return '—';
  const pad = x => x.toFixed(1).padStart(5);
  let s = `W:${pad(n.w)}%  D:${pad(n.d)}%  L:${pad(n.l)}%  (W-L ${n.wl>=0?'+':''}${n.wl.toFixed(2)})`;
  // Flag if the underlying three components don't sum to ~100%.
  if (n.sum < 0.9 || n.sum > 1.1) {
    s += `  [raw sum ${(n.sum*100).toFixed(1)}%]`;
  }
  return s;
}

async function refresh() {
  try {
    const r = await fetch('/state'); state = await r.json();
    document.getElementById('status').textContent = state.status;
    const rootN = normalizeVal(state.root_value);
    const warn = (rootN && (rootN.sum < 0.9 || rootN.sum > 1.1))
      ? '<span class="warn">⚠ root WDL raw sum ≠ 1 (likely v_mix bug); shown values are renormalized</span>\n' : '';
    document.getElementById('values').innerHTML =
      warn +
      'root: ' + fmtVal(state.root_value) + '\n  nn: ' + fmtVal(state.nn_value);
    draw();
    drawHeat('h_mcts_policy', state.mcts_policy);
    drawHeat('h_mcts_visits', state.mcts_visits);
    drawHeat('h_nn_policy',   state.nn_policy);
  } catch(e) { /* ignore */ }
}

cv.addEventListener('click', async (ev) => {
  if (!state || state.game_over) return;
  const rect = cv.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const c = Math.round((x - MARGIN)/CELL), r = Math.round((y - MARGIN)/CELL);
  if (r<0||r>=N||c<0||c>=N) return;
  if (state.board[r][c] !== 0) return;
  await fetch('/move', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({r, c})});
  refresh();
});

async function sendCmd(cmd) {
  await fetch('/move', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({cmd})});
  refresh();
}
function applySims() {
  const n = parseInt(document.getElementById('sims_input').value, 10);
  if (Number.isFinite(n) && n >= 1) sendCmd('sims ' + n);
}
function applyGm() {
  const m = parseInt(document.getElementById('gm_input').value, 10);
  if (Number.isFinite(m) && m >= 1) sendCmd('gm ' + m);
}
function applyNoise() {
  const v = document.getElementById('noise_toggle').checked ? 1 : 0;
  sendCmd('noise ' + v);
}
async function newGame(side) {
  await fetch('/new', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify({human_side: side})});
  refresh();
}

draw();
setInterval(refresh, 250);
refresh();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set on class before server starts

    def log_message(self, *a, **kw):
        pass  # quiet

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/state":
            sess = Handler.app.current()
            if sess is None:
                self._send_json(200, {"version": 0, "board": [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)],
                                      "last_move": None, "status": "No game. Click New.",
                                      "root_value": None, "nn_value": None,
                                      "game_over": False, "human_side": 1,
                                      "mcts_policy": None, "mcts_visits": None, "nn_policy": None,
                                      "gumbel_phases": None})
                return
            self._send_json(200, sess.snapshot())
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        body = self._read_json()
        if self.path == "/new":
            side = int(body.get("human_side", 1))
            if side not in (1, -1):
                side = 1
            Handler.app.start(side)
            self._send_json(200, {"ok": True})
            return
        if self.path == "/move":
            sess = Handler.app.current()
            if sess is None:
                self._send_json(409, {"ok": False, "err": "no session"})
                return
            if "cmd" in body:
                sess.send(str(body["cmd"]))
            else:
                r, c = int(body["r"]), int(body["c"])
                sess.send(f"{r} {c}")
            self._send_json(200, {"ok": True})
            return
        self.send_response(404); self.end_headers()


def main():
    root_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(root_dir / "data" / "models" / "latest.pt"))
    ap.add_argument("--bin", default=str(root_dir / "cpp" / "build" / "gomoku_play"))
    ap.add_argument("--config", default=str(root_dir / "scripts" / "play.cfg"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--human-side", type=int, default=1, choices=(1, -1))
    args = ap.parse_args()

    for p, name in [(args.model, "model"), (args.bin, "binary"), (args.config, "config")]:
        if not Path(p).exists():
            raise SystemExit(f"{name} not found: {p}")

    app = App(Path(args.bin), Path(args.model), Path(args.config))
    app.start(args.human_side)
    Handler.app = app

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[play_web] serving http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sess = app.current()
        if sess is not None:
            sess.stop()


if __name__ == "__main__":
    main()
