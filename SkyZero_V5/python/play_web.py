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

BOARD_SIZE = 15  # default only; per-session size lives on EngineSession.board_size

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

    def __init__(self, play_bin, model, config, human_side, board_size, rule):
        self.play_bin = play_bin
        self.model = model
        self.config = config
        self.human_side = human_side
        self.board_size = board_size
        self.rule = rule

        self.lock = threading.Lock()
        self.version = 0
        self.board = [[0] * board_size for _ in range(board_size)]
        self.last_move = None
        self.status = "Launching engine..."
        self.root_value = None  # {w,d,l,wl}
        self.nn_value = None
        self.game_over = False
        self.mcts_policy = None  # 15x15
        self.mcts_visits = None
        self.nn_policy = None
        self.nn_opp_policy = None
        self.nn_futurepos_8 = None   # 15x15, signed in [-1,+1]; +own / -opp future stone
        self.nn_futurepos_32 = None  # 15x15, signed in [-1,+1]; +32 step horizon
        self.gumbel_phases = None  # list of list of [r,c], index 0 = initial 16, last = final 1

        self._pending_rows = None
        self._pending_grid_key = None
        self._pending_grid_rows = None

        self.proc = subprocess.Popen(
            [str(play_bin), "--model", str(model), "--config", str(config),
             "--human-side", str(human_side), "--board-size", str(board_size),
             "--rule", str(rule)],
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

    def _try_parse_grid_row(self, line):
        toks = line.split()
        if len(toks) != self.board_size:
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
            if len(self._pending_rows) == self.board_size:
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
        if "NN Opp Strategy" in line:
            self._pending_grid_key = "nn_opp_policy"
            self._pending_grid_rows = []
            return
        if "NN Futurepos +8" in line:
            self._pending_grid_key = "nn_futurepos_8"
            self._pending_grid_rows = []
            return
        if "NN Futurepos +32" in line:
            self._pending_grid_key = "nn_futurepos_32"
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
                if len(self._pending_grid_rows) == self.board_size:
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
        if "[setting] human_side=" in line:
            try:
                self.human_side = int(line.strip().split("=")[-1])
            except ValueError:
                pass
            return
        if "Undo successful" in line:
            self.game_over = False
            self.status = "Undo"
            # The engine pops two plies but does not re-run a search, so any
            # cached search outputs still reflect the just-undone position.
            # Clear them so the chart/WDL/heatmaps don't attribute stale values
            # to the rolled-back state until the next AI think refills them.
            self.root_value = None
            self.nn_value = None
            self.mcts_policy = None
            self.mcts_visits = None
            self.nn_policy = None
            self.nn_opp_policy = None
            self.nn_futurepos_8 = None
            self.nn_futurepos_32 = None
            self.gumbel_phases = None
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
        new_board = [[0] * self.board_size for _ in range(self.board_size)]
        last = None
        for r, line in enumerate(rows):
            body = line[3:] if len(line) > 3 else ""
            for c in range(self.board_size):
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
                "board_size": self.board_size,
                "rule": self.rule,
                "last_move": list(self.last_move) if self.last_move else None,
                "status": self.status,
                "root_value": self.root_value,
                "nn_value": self.nn_value,
                "game_over": self.game_over,
                "human_side": self.human_side,
                "mcts_policy": self.mcts_policy,
                "mcts_visits": self.mcts_visits,
                "nn_policy": self.nn_policy,
                "nn_opp_policy": self.nn_opp_policy,
                "nn_futurepos_8": self.nn_futurepos_8,
                "nn_futurepos_32": self.nn_futurepos_32,
                "gumbel_phases": self.gumbel_phases,
            }


class App:
    def __init__(self, play_bin, config, models, current_model_id,
                 board_sizes, default_board_size, rules, default_rule):
        self.play_bin = play_bin
        self.config = config
        self.models = models  # id -> abs Path
        self.current_model_id = current_model_id
        self.board_sizes = board_sizes
        self.current_board_size = default_board_size
        self.rules = rules
        self.current_rule = default_rule
        self.session = None
        self.session_lock = threading.Lock()

    @property
    def model(self):
        return self.models[self.current_model_id]

    def start(self, human_side, model_id=None, board_size=None, rule=None):
        with self.session_lock:
            if model_id is not None and model_id in self.models:
                self.current_model_id = model_id
            if board_size is not None and board_size in self.board_sizes:
                self.current_board_size = board_size
            if rule is not None and rule in self.rules:
                self.current_rule = rule
            if self.session is not None:
                self.session.stop()
            self.session = EngineSession(self.play_bin, self.model, self.config,
                                         human_side, self.current_board_size,
                                         self.current_rule)

    def current(self):
        with self.session_lock:
            return self.session

    def model_listing(self):
        items = [{"id": mid, "label": Path(p).name, "group": mid.split("/", 1)[0]}
                 for mid, p in self.models.items()]
        return {"current": self.current_model_id, "items": items}


HTML_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkyZero Gomoku</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script>
  // Set theme before first paint to avoid flash. Tri-state: auto / light / dark.
  (function(){
    try {
      var saved = localStorage.getItem('skz_theme');
      var mode = (saved === 'light' || saved === 'dark') ? saved : 'auto';
      var resolved = (mode === 'auto')
        ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
        : mode;
      document.documentElement.dataset.theme = resolved;
      document.documentElement.dataset.themeMode = mode;
    } catch(e) {
      document.documentElement.dataset.theme = 'light';
      document.documentElement.dataset.themeMode = 'auto';
    }
  })();
</script>
<style>
  :root {
    --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB",
                 "Microsoft YaHei", system-ui, sans-serif;
    --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, "SF Mono",
                 Menlo, Monaco, Consolas, "Liberation Mono", "DejaVu Sans Mono",
                 "Noto Sans Mono", "Courier New", monospace;
    --bg: #ffffff;
    --surface: #ffffff;
    --surface-2: #f6f8fa;
    --surface-sunken: #fafbfc;
    --border: #d8dee4;
    --border-strong: #afb8c1;
    --fg: #1f2328;
    --fg-muted: #59636e;
    --fg-subtle: #8b949e;
    --accent: #0969da;
    --accent-hover: #0860c7;
    --accent-fg: #ffffff;
    --success: #1a7f37;
    --success-bg: #dafbe1;
    --info: #0969da;
    --info-bg: #ddf4ff;
    --warn: #9a6700;
    --warn-bg: #fff8c5;
    --danger: #cf222e;
    --danger-bg: #ffebe9;
    --done: #8250df;
    --done-bg: #fbefff;
    --board-bg: #e8c583;
    --board-line: #6b5a3a;
    --board-star: #3a2e1a;
    --stone-black-0: #555;
    --stone-black-1: #000;
    --stone-white-0: #ffffff;
    --stone-white-1: #d5d5d5;
    --stone-outline: #1a1a1a;
    --stone-shadow: rgba(0,0,0,0.18);
    --heat-bg: #ffffff;
    --heat-grid: #e5e7eb;
    --heat-text: #111;
    --heat-text-inv: #ffffff;
    --shadow-xs: 0 1px 0 rgba(27,31,36,0.04);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
  }
  html[data-theme="dark"] {
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #21262d;
    --surface-sunken: #010409;
    --border: #30363d;
    --border-strong: #6e7681;
    --fg: #e6edf3;
    --fg-muted: #9198a1;
    --fg-subtle: #6e7681;
    --accent: #4493f8;
    --accent-hover: #539bff;
    --accent-fg: #0d1117;
    --success: #3fb950;
    --success-bg: #0f2a1a;
    --info: #4493f8;
    --info-bg: #0f2238;
    --warn: #d29922;
    --warn-bg: #2d2200;
    --danger: #f85149;
    --danger-bg: #2d0d10;
    --done: #ab7df8;
    --done-bg: #1f1430;
    --board-bg: #c9a460;
    --board-line: #4a3a1f;
    --board-star: #2a1f0e;
    --stone-black-0: #2a2a2a;
    --stone-black-1: #000;
    --stone-white-0: #fafafa;
    --stone-white-1: #c2c2c2;
    --stone-outline: #000;
    --stone-shadow: rgba(0,0,0,0.35);
    --heat-bg: #0d1117;
    --heat-grid: #30363d;
    --heat-text: #e6edf3;
    --heat-text-inv: #ffffff;
    --shadow-xs: 0 1px 0 rgba(0,0,0,0.3);
  }
  * { box-sizing: border-box; }
  html, body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }
  body {
    font-family: var(--font-sans);
    font-size: 14px; line-height: 1.5;
    font-variant-numeric: tabular-nums;
    font-feature-settings: "cv11", "ss01", "tnum";
    margin: 0; background: var(--bg); color: var(--fg);
  }

  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
  }

  .app { max-width: 2000px; margin: 0 auto; padding: 16px 24px 48px; }

  /* ---------- Top bar ---------- */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .brand { display: flex; flex-direction: column; gap: 2px; }
  .brand-title {
    font-size: 16px; font-weight: 600; letter-spacing: -0.01em;
    color: var(--fg);
  }
  .brand-sub {
    font-size: 12px; color: var(--fg-muted); letter-spacing: 0.01em;
  }
  .icon-btn {
    width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center;
    background: transparent; border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--fg-muted); cursor: pointer;
    transition: background 0.12s, color 0.12s, border-color 0.12s;
  }
  .icon-btn:hover { background: var(--surface-2); color: var(--fg); border-color: var(--border-strong); }
  .icon-btn svg { width: 16px; height: 16px; display: block; }
  .icon-btn .sun-icon,
  .icon-btn .moon-icon,
  .icon-btn .auto-icon { display: none; }
  html[data-theme-mode="light"] .icon-btn .sun-icon,
  html[data-theme-mode="dark"]  .icon-btn .moon-icon,
  html[data-theme-mode="auto"]  .icon-btn .auto-icon { display: inline-block; }

  /* ---------- Layout ---------- */
  .main {
    display: grid;
    grid-template-columns: 260px minmax(0, auto) minmax(0, auto);
    gap: 20px;
    align-items: start;
    margin-bottom: 20px;
  }
  @media (max-width: 1399px) {
    .main { grid-template-columns: 1fr; }
  }
  .board-col {
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    min-width: 0;
  }
  .side-col {
    display: flex; flex-direction: column; gap: 12px;
    min-width: 0;
  }
  .seg-row { display: flex; gap: 8px; }
  .side-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .side-row .seg-row { flex: 1; max-width: 180px; }
  .seg-btn {
    flex: 1; height: 34px; font-size: 13px; font-weight: 500;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    transition: background 0.12s, border-color 0.12s, color 0.12s;
  }
  .seg-btn:hover { background: var(--surface-2); border-color: var(--border-strong); }
  .seg-btn[aria-pressed="true"] {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent);
  }
  .seg-btn:disabled {
    cursor: not-allowed; opacity: 0.45;
    background: var(--surface); color: var(--fg-muted); border-color: var(--border);
  }
  .seg-btn:disabled:hover { background: var(--surface); border-color: var(--border); }
  .seg-stone {
    width: 14px; height: 14px; border-radius: 50%;
    border: 1px solid var(--stone-outline);
    flex-shrink: 0;
  }
  .seg-stone.black { background: radial-gradient(circle at 30% 30%, var(--stone-black-0), var(--stone-black-1)); }
  .seg-stone.white { background: radial-gradient(circle at 30% 30%, var(--stone-white-0), var(--stone-white-1)); }
  .board-actions {
    display: flex; gap: 8px; width: 100%; max-width: 560px; justify-content: center;
  }

  /* ---------- Card ---------- */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xs);
  }
  .card-body { padding: 14px 16px; }
  .card-title {
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--fg-subtle);
    margin: 0 0 10px;
  }

  /* ---------- Status pill ---------- */
  .status-card { padding: 12px 16px; }
  .status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 4px 10px 4px 8px;
    border-radius: 999px;
    font-size: 12.5px; font-weight: 500;
    background: var(--surface-2); color: var(--fg);
    border: 1px solid var(--border);
  }
  .status-pill .dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: var(--fg-subtle);
  }
  .status-pill[data-variant="active"] { background: var(--success-bg); color: var(--success); border-color: transparent; }
  .status-pill[data-variant="active"] .dot { background: var(--success); }
  .status-pill[data-variant="thinking"] { background: var(--info-bg); color: var(--info); border-color: transparent; }
  .status-pill[data-variant="thinking"] .dot { background: var(--info); animation: pulse 1.2s ease-in-out infinite; }
  .status-pill[data-variant="info"] { background: var(--info-bg); color: var(--info); border-color: transparent; }
  .status-pill[data-variant="info"] .dot { background: var(--info); }
  .status-pill[data-variant="done"] { background: var(--done-bg); color: var(--done); border-color: transparent; }
  .status-pill[data-variant="done"] .dot { background: var(--done); }
  .status-pill[data-variant="warn"] { background: var(--warn-bg); color: var(--warn); border-color: transparent; }
  .status-pill[data-variant="warn"] .dot { background: var(--warn); }
  .status-pill[data-variant="error"] { background: var(--danger-bg); color: var(--danger); border-color: transparent; }
  .status-pill[data-variant="error"] .dot { background: var(--danger); }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.35); opacity: 0.55; }
  }
  @media (prefers-reduced-motion: reduce) {
    .status-pill .dot { animation: none !important; }
  }

  /* ---------- Buttons ---------- */
  .btn {
    height: 30px; padding: 0 12px; font-size: 13px; font-weight: 500;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    cursor: pointer; white-space: nowrap;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    transition: background 0.12s, border-color 0.12s, color 0.12s;
  }
  .btn:hover { background: var(--surface-2); border-color: var(--border-strong); }
  .btn:active { background: var(--surface-sunken); }
  .btn.primary {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent);
  }
  .btn.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  .btn.danger-ghost { color: var(--fg-muted); }
  .btn.danger-ghost:hover { color: var(--danger); border-color: var(--danger); }
  .btn-row { display: flex; flex-wrap: wrap; gap: 8px; }

  /* ---------- Toggle ---------- */
  .toggle {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; cursor: pointer; user-select: none;
    font-size: 13px; color: var(--fg);
    padding: 6px 0;
  }
  .toggle input { position: absolute; opacity: 0; pointer-events: none; }
  .toggle .track {
    width: 30px; height: 18px; background: var(--border-strong); border-radius: 999px;
    position: relative; transition: background 0.15s; flex-shrink: 0;
  }
  .toggle .track::after {
    content: ""; position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%; background: #fff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.25); transition: transform 0.15s;
  }
  .toggle input:checked + .track { background: var(--accent); }
  .toggle input:checked + .track::after { transform: translateX(12px); }
  @media (prefers-reduced-motion: reduce) {
    .toggle .track, .toggle .track::after { transition: none; }
  }

  /* ---------- Number input ---------- */
  .field-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 6px 0;
  }
  .field-row label {
    font-size: 13px; color: var(--fg-muted);
    font-family: var(--font-mono);
  }
  .num {
    width: 80px; height: 28px; padding: 0 8px; font-size: 13px;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    text-align: right;
    transition: border-color 0.12s, box-shadow 0.12s;
  }
  .num:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }

  .divider {
    height: 1px; background: var(--border); margin: 4px 0;
  }

  /* ---------- WDL ---------- */
  .wdl-row {
    display: grid;
    grid-template-columns: 42px minmax(0,1fr) 64px;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
  }
  .wdl-label {
    font-family: var(--font-mono);
    font-size: 11.5px; color: var(--fg-muted);
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .wdl-bar {
    display: flex; height: 8px; width: 100%;
    background: var(--surface-2);
    border-radius: 999px; overflow: hidden;
  }
  .wdl-bar .seg { height: 100%; transition: width 0.2s ease-out; }
  .wdl-bar .seg.w { background: var(--success); }
  .wdl-bar .seg.d { background: var(--fg-subtle); opacity: 0.45; }
  .wdl-bar .seg.l { background: var(--danger); }
  @media (prefers-reduced-motion: reduce) {
    .wdl-bar .seg { transition: none; }
  }
  .wdl-wl {
    font-family: var(--font-mono);
    font-size: 12px; text-align: right; color: var(--fg);
    font-variant-numeric: tabular-nums;
  }
  .wdl-wl.pos { color: var(--success); }
  .wdl-wl.neg { color: var(--danger); }
  .wdl-empty { color: var(--fg-subtle); font-size: 12px; padding: 8px 0; text-align: center; }
  .wdl-detail {
    margin-top: 6px; font-size: 11.5px; color: var(--fg-muted);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    display: flex; gap: 10px;
    min-height: 18px;
  }
  .wdl-detail .k { color: var(--fg-subtle); }
  .value-chart-wrap {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    min-height: 160px;
  }
  .value-chart-legend {
    display: flex; align-items: center; gap: 12px;
    font-size: 11px; color: var(--fg-muted);
    font-family: var(--font-mono);
    margin-bottom: 6px;
  }
  .vc-item { display: inline-flex; align-items: center; gap: 5px; }
  .vc-swatch { width: 10px; height: 2px; border-radius: 1px; display: inline-block; }
  .vc-axis { margin-left: auto; color: var(--fg-subtle); font-size: 10.5px; }
  #value_chart { display: block; width: 100%; height: 140px; min-height: 120px; }
  /* ---------- Board ---------- */
  .board-card {
    padding: 16px;
    display: inline-flex; flex-direction: column; align-items: center;
  }
  #board {
    background: var(--board-bg); border-radius: var(--radius-sm);
    display: block; cursor: crosshair;
  }
  #gumbel_legend {
    font-size: 11.5px; color: var(--fg-muted);
    margin-top: 14px;
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: 6px 10px;
  }
  #gumbel_legend .legend-head {
    font-weight: 600; color: var(--fg-subtle);
    text-transform: uppercase; letter-spacing: 0.06em; font-size: 10.5px;
    margin-right: 2px;
  }
  #gumbel_legend .chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 2px 8px; border-radius: 999px;
    background: var(--surface-2);
    font-family: var(--font-mono);
    font-size: 11px;
  }
  #gumbel_legend .dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  }

  /* ---------- Heat grid ---------- */
  #right_col { min-width: 0; }
  .grids {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-template-rows: repeat(3, minmax(0, 1fr));
    gap: 12px;
    height: 100%;
  }
  @media (max-width: 1399px) {
    .grids { grid-template-columns: repeat(2, minmax(0, 1fr)); height: auto; }
  }
  @media (max-width: 720px) {
    .grids { grid-template-columns: 1fr; }
  }
  .grid-card {
    padding: 12px; text-align: center;
    display: flex; flex-direction: column;
    min-height: 0;
  }
  .grid-card .grid-title {
    font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
    color: var(--fg-subtle);
    margin-bottom: 12px; text-align: left;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
  }
  .heat {
    background: var(--heat-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    display: block;
    margin: auto;
  }

  .expand-btn {
    background: transparent; border: none; cursor: pointer;
    color: var(--fg-subtle); padding: 0;
    width: 18px; height: 18px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 4px;
    transition: background 0.12s, color 0.12s;
    flex-shrink: 0;
  }
  .expand-btn:hover { background: var(--surface-2); color: var(--fg); }
  .expand-btn svg { width: 12px; height: 12px; display: block; }

  /* ---------- Heat modal ---------- */
  .heat-modal {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0, 0, 0, 0.55);
    display: flex; align-items: center; justify-content: center;
    padding: 24px;
  }
  .heat-modal-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
    padding: 16px 20px 20px;
    max-width: 95vw; max-height: 95vh;
    display: flex; flex-direction: column; gap: 12px;
    min-width: 0; min-height: 0;
  }
  .heat-modal-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px;
    flex-shrink: 0;
  }
  .heat-modal-header .heat-modal-title {
    font-size: 14px; font-weight: 600; color: var(--fg);
    letter-spacing: 0.01em;
  }
  .heat-modal-close {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--fg-muted); cursor: pointer;
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 0.12s, color 0.12s, border-color 0.12s;
  }
  .heat-modal-close:hover { background: var(--surface-2); color: var(--fg); border-color: var(--border-strong); }
  .heat-modal-close svg { width: 14px; height: 14px; }
  #heat_modal_canvas {
    background: var(--heat-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    display: block;
    margin: auto;
  }

  .hidden { display: none !important; }
</style></head>
<body>
<div class="app">

  <header class="topbar">
    <div class="brand">
      <div class="brand-title">SkyZero Gomoku</div>
      <div class="brand-sub">AlphaZero-style self-play · local inspector</div>
    </div>
    <button class="icon-btn" id="theme_toggle" aria-label="Toggle color theme" title="Toggle theme">
      <svg class="sun-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="8" cy="8" r="3"></circle>
        <path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M3.05 12.95l1.06-1.06M11.89 4.11l1.06-1.06"></path>
      </svg>
      <svg class="moon-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7z"></path>
      </svg>
      <svg class="auto-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
        <circle cx="8" cy="8" r="6"></circle>
        <path d="M8 2 a6 6 0 0 1 0 12 z" fill="currentColor" stroke="none"></path>
      </svg>
    </button>
  </header>

  <div class="main">
    <aside class="side-col" id="left_col">
      <div class="card status-card">
        <div class="status-pill" id="status_pill" data-variant="idle">
          <span class="dot"></span>
          <span id="status">idle</span>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Model</div>
          <select id="model_select" class="num" style="width:100%; height:32px; text-align:left; font-family: var(--font-mono);"></select>
        </div>
      </div>

      <div class="card">
        <div class="card-body side-row">
          <div class="card-title" style="margin:0;">Human side</div>
          <div class="seg-row">
            <button class="seg-btn" id="side_black" aria-pressed="true" onclick="setSide(1)">
              <span class="seg-stone black"></span>Black
            </button>
            <button class="seg-btn" id="side_white" aria-pressed="false" onclick="setSide(-1)">
              <span class="seg-stone white"></span>White
            </button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-body side-row">
          <div class="card-title" style="margin:0;">Board size</div>
          <select id="size_select" class="num"
                  style="width:auto; min-width:84px; height:32px; text-align:left; font-family: var(--font-mono);">
          </select>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Rule</div>
          <div class="seg-row">
            <button class="seg-btn" id="rule_renju"     data-rule="renju"     aria-pressed="false" onclick="setRule('renju')">Renju</button>
            <button class="seg-btn" id="rule_standard"  data-rule="standard"  aria-pressed="false" onclick="setRule('standard')">Standard</button>
            <button class="seg-btn" id="rule_freestyle" data-rule="freestyle" aria-pressed="false" onclick="setRule('freestyle')">Freestyle</button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Search</div>
          <div class="field-row">
            <label for="sims_input">sims</label>
            <input class="num" type="number" id="sims_input" min="1" step="1" value="800">
          </div>
          <div class="field-row">
            <label for="gm_input">gumbel_m</label>
            <input class="num" type="number" id="gm_input" min="1" step="1" value="16">
          </div>
          <div class="divider"></div>
          <label class="toggle">
            <span>Gumbel overlay</span>
            <span style="display:inline-flex;">
              <input type="checkbox" id="gumbel_toggle" checked>
              <span class="track"></span>
            </span>
          </label>
          <label class="toggle">
            <span>Root symmetry prune</span>
            <span style="display:inline-flex;">
              <input type="checkbox" id="prune_toggle">
              <span class="track"></span>
            </span>
          </label>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Value estimates</div>
          <div class="wdl-row">
            <span class="wdl-label">root</span>
            <div class="wdl-bar" id="wdl_root_bar">
              <span class="seg w" style="width:0"></span>
              <span class="seg d" style="width:0"></span>
              <span class="seg l" style="width:0"></span>
            </div>
            <span class="wdl-wl" id="wdl_root_wl">—</span>
          </div>
          <div class="wdl-detail" id="wdl_root_detail"></div>
          <div class="wdl-row" style="margin-top:4px;">
            <span class="wdl-label">nn</span>
            <div class="wdl-bar" id="wdl_nn_bar">
              <span class="seg w" style="width:0"></span>
              <span class="seg d" style="width:0"></span>
              <span class="seg l" style="width:0"></span>
            </div>
            <span class="wdl-wl" id="wdl_nn_wl">—</span>
          </div>
          <div class="wdl-detail" id="wdl_nn_detail"></div>
          <div class="value-chart-wrap">
            <div class="value-chart-legend">
              <span class="vc-item"><span class="vc-swatch" style="background:#0969da;"></span>root</span>
              <span class="vc-item"><span class="vc-swatch" style="background:#cf222e;"></span>nn</span>
              <span class="vc-axis">W/L · −1…+1</span>
            </div>
            <canvas id="value_chart"></canvas>
          </div>
        </div>
      </div>
    </aside>

    <section class="board-col">
      <div class="card board-card">
        <canvas id="board"></canvas>
        <div id="gumbel_legend">
          <span class="legend-head">Gumbel Sequential Halving</span>
          <span class="chip"><span class="dot" style="background:#9ca3af;"></span>16</span>
          <span class="chip"><span class="dot" style="background:#3b82f6;"></span>8</span>
          <span class="chip"><span class="dot" style="background:#10b981;"></span>4</span>
          <span class="chip"><span class="dot" style="background:#f59e0b;"></span>2</span>
          <span class="chip"><span class="dot" style="background:#ef4444;"></span>1 (picked)</span>
        </div>
      </div>
      <div class="board-actions">
        <button class="btn primary" onclick="newGame()">New game</button>
        <button class="btn danger-ghost" onclick="sendCmd('u')">Undo</button>
      </div>
    </section>

    <aside class="side-col" id="right_col">
      <div class="grids">
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">Improved Policy</span>
            <button class="expand-btn" data-target="h_mcts_policy" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_mcts_policy"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">Visits Dist</span>
            <button class="expand-btn" data-target="h_mcts_visits" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_mcts_visits"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">NN Policy</span>
            <button class="expand-btn" data-target="h_nn_policy" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_nn_policy"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">NN Opp Policy</span>
            <button class="expand-btn" data-target="h_nn_opp_policy" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_nn_opp_policy"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">NN Futurepos +8</span>
            <button class="expand-btn" data-target="h_nn_futurepos_8" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_nn_futurepos_8"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">NN Futurepos +32</span>
            <button class="expand-btn" data-target="h_nn_futurepos_32" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_nn_futurepos_32"></canvas>
        </div>
      </div>
    </aside>
  </div>

  <div class="heat-modal hidden" id="heat_modal" role="dialog" aria-modal="true" aria-labelledby="heat_modal_title">
    <div class="heat-modal-card">
      <div class="heat-modal-header">
        <span class="heat-modal-title" id="heat_modal_title">Heatmap</span>
        <button class="heat-modal-close" id="heat_modal_close" aria-label="Close">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M3 3l10 10M13 3L3 13"/>
          </svg>
        </button>
      </div>
      <canvas id="heat_modal_canvas"></canvas>
    </div>
  </div>
</div>

<script>
let N = 15;
const MARGIN = 28;
let CELL = 36;
let BOARD_LOGICAL = MARGIN*2 + CELL*(N-1); // recomputed in syncBoardSize()
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
    canvas.style.width = logicalW + 'px';
    canvas.style.height = logicalH + 'px';
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx._logicalW = logicalW;
  ctx._logicalH = logicalH;
  return ctx;
}
function clearLogical(ctx) { ctx.clearRect(0, 0, ctx._logicalW, ctx._logicalH); }

const cv = document.getElementById('board');
const ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
const vcCanvas = document.getElementById('value_chart');
let vctx = setupCanvas(vcCanvas, 280, 160);
vcCanvas.style.width = '100%';
vcCanvas.style.height = '100%';
function resizeValueChart() {
  const rect = vcCanvas.getBoundingClientRect();
  const w = Math.max(120, Math.floor(rect.width));
  const h = Math.max(120, Math.floor(rect.height));
  if (vctx._logicalW === w && vctx._logicalH === h) return;
  vctx = setupCanvas(vcCanvas, w, h);
  drawValueChart();
}
new ResizeObserver(resizeValueChart).observe(vcCanvas);

const leftCol = document.getElementById('left_col');
const rightCol = document.getElementById('right_col');
const boardCol = document.querySelector('.board-col');
const boardCard = document.querySelector('.board-card');
const boardActions = document.querySelector('.board-actions');
const mainEl = document.querySelector('.main');
function syncBoardSize() {
  if (window.matchMedia('(max-width: 1399px)').matches) {
    rightCol.style.height = '';
    rightCol.style.width = '';
    if (BOARD_LOGICAL !== 560) {
      CELL = 36;
      BOARD_LOGICAL = MARGIN*2 + CELL*(N-1);
      setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      draw();
    }
    return;
  }
  // Make .board-col total height match #left_col by reverse-computing the
  // canvas size from measured surroundings (no hard-coded constants).
  const cardCS = getComputedStyle(boardCard);
  const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const cardPadY = parseFloat(cardCS.paddingTop) + parseFloat(cardCS.paddingBottom);
  const legendCS = getComputedStyle(gumbelLegend);
  const legendH = gumbelLegend.classList.contains('hidden')
      ? 0
      : gumbelLegend.offsetHeight + parseFloat(legendCS.marginTop || 0);
  // Match .board-card height (canvas + legend + padding) to leftCol height.
  // Actions row sits below and is intentionally excluded.
  const sizeByHeight = leftCol.offsetHeight - cardPadY - legendH;
  // Cap by available width so the board doesn't overflow into adjacent
  // columns. right_col is sized to boardCard.offsetWidth, so the two auto
  // tracks together need 2 * (BOARD_LOGICAL + cardPadX) of room.
  const mainCS = getComputedStyle(mainEl);
  const gap = parseFloat(mainCS.columnGap || mainCS.gap) || 20;
  const remaining = mainEl.clientWidth - leftCol.offsetWidth - 2 * gap;
  const sizeByWidth = Math.floor(remaining / 2 - cardPadX);
  let size = Math.max(360, Math.min(sizeByHeight, sizeByWidth));
  CELL = Math.max(20, Math.floor((size - 2*MARGIN) / (N-1)));
  BOARD_LOGICAL = MARGIN*2 + CELL*(N-1);
  const canvasNeedsResize = cv.width !== Math.round(BOARD_LOGICAL * DPR);
  if (canvasNeedsResize) setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
  rightCol.style.height = boardCard.offsetHeight + 'px';
  rightCol.style.width = boardCard.offsetWidth + 'px';
  if (canvasNeedsResize) draw();
}
new ResizeObserver(syncBoardSize).observe(leftCol);
window.addEventListener('resize', syncBoardSize);

let valueHistory = []; // [{step, root, nn}]  step = stone count when recorded
const heatCtxs = {
  h_mcts_policy: setupCanvas(document.getElementById('h_mcts_policy'), HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_mcts_visits: setupCanvas(document.getElementById('h_mcts_visits'), HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_nn_policy:   setupCanvas(document.getElementById('h_nn_policy'),   HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_nn_opp_policy: setupCanvas(document.getElementById('h_nn_opp_policy'), HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_nn_futurepos_8:  setupCanvas(document.getElementById('h_nn_futurepos_8'),  HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_nn_futurepos_32: setupCanvas(document.getElementById('h_nn_futurepos_32'), HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_GRID_KEYS = {
  h_mcts_policy: 'mcts_policy',
  h_mcts_visits: 'mcts_visits',
  h_nn_policy: 'nn_policy',
  h_nn_opp_policy: 'nn_opp_policy',
  h_nn_futurepos_8: 'nn_futurepos_8',
  h_nn_futurepos_32: 'nn_futurepos_32',
};
const SIGNED_HEAT_IDS = new Set(['h_nn_futurepos_8', 'h_nn_futurepos_32']);
function drawHeatById(id, grid) {
  if (SIGNED_HEAT_IDS.has(id)) drawHeatSigned(id, grid);
  else drawHeat(id, grid);
}

/* ---------- Heat modal (fullscreen view of one heatmap) ---------- */
let expandedSourceId = null;
function setupModalCanvas() {
  const canvas = document.getElementById('heat_modal_canvas');
  // Square canvas sized to fit viewport with room for the header + card padding.
  const card = canvas.parentElement;
  const cardCS = getComputedStyle(card);
  const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const padY = parseFloat(cardCS.paddingTop) + parseFloat(cardCS.paddingBottom);
  const header = card.querySelector('.heat-modal-header');
  const headerH = header ? header.offsetHeight + 12 /* gap */ : 0;
  const availW = window.innerWidth * 0.95 - padX;
  const availH = window.innerHeight * 0.95 - padY - headerH;
  const sz = Math.max(240, Math.floor(Math.min(availW, availH)));
  canvas.style.width = sz + 'px';
  canvas.style.height = sz + 'px';
  heatCtxs.h_modal = setupCanvas(canvas, sz, sz, false);
}
function paintHeatModal() {
  if (!expandedSourceId) return;
  const grid = state ? state[HEAT_GRID_KEYS[expandedSourceId]] : null;
  if (SIGNED_HEAT_IDS.has(expandedSourceId)) drawHeatSigned('h_modal', grid);
  else drawHeat('h_modal', grid);
}
function openHeatModal(sourceId) {
  if (!HEAT_GRID_KEYS[sourceId]) return;
  expandedSourceId = sourceId;
  const card = document.getElementById(sourceId).parentElement;
  const titleEl = card.querySelector('.grid-title-text');
  document.getElementById('heat_modal_title').textContent =
      titleEl ? titleEl.textContent : 'Heatmap';
  document.getElementById('heat_modal').classList.remove('hidden');
  setupModalCanvas();
  paintHeatModal();
}
function closeHeatModal() {
  if (expandedSourceId === null) return;
  expandedSourceId = null;
  document.getElementById('heat_modal').classList.add('hidden');
}
function fitHeatCanvas(canvasId) {
  const c = document.getElementById(canvasId);
  const card = c.parentElement;
  const cardCS = getComputedStyle(card);
  const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const padY = parseFloat(cardCS.paddingTop) + parseFloat(cardCS.paddingBottom);
  const title = card.querySelector('.grid-title');
  let titleH = 0;
  if (title) {
    const tCS = getComputedStyle(title);
    titleH = title.offsetHeight + parseFloat(tCS.marginTop || 0) + parseFloat(tCS.marginBottom || 0);
  }
  const availW = card.clientWidth - padX;
  const availH = card.clientHeight - padY - titleH;
  const size = Math.max(60, Math.floor(Math.min(availW, availH > 0 ? availH : availW)));
  c.style.width = size + 'px';
  c.style.height = size + 'px';
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

let state = null;
let showGumbel = true;
// Side-swap perspective tracking. `side X` only flips human_side in the engine;
// it does not trigger a new search, so the cached root_value/nn_value in the
// snapshot remain in the OLD AI's perspective until the next AI think. To keep
// display consistent, we record (a) the snapshot values at the moment the user
// swapped and (b) the parity of pending swaps. While incoming snapshot values
// still match that baseline, we flip them for display iff parity is odd. Once
// the engine emits new values (mismatch), we drop the pending state.
let pendingSwapBaseline = null;
let pendingSwapParity = 0;
function valuesEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.w === b.w && a.d === b.d && a.l === b.l && a.wl === b.wl;
}
function flipWDL(v) {
  if (!v) return null;
  return {w: v.l, d: v.d, l: v.w, wl: -v.wl};
}
const gumbelToggle = document.getElementById('gumbel_toggle');
const gumbelLegend = document.getElementById('gumbel_legend');
gumbelToggle.addEventListener('change', () => {
  showGumbel = gumbelToggle.checked;
  gumbelLegend.classList.toggle('hidden', !showGumbel);
  syncBoardSize();
  draw();
});
const pruneToggle = document.getElementById('prune_toggle');
pruneToggle.addEventListener('change', () => {
  sendCmd('prune ' + (pruneToggle.checked ? 1 : 0));
});

/* ---------- Theme toggle (tri-state: auto / light / dark) ---------- */
const themeBtn = document.getElementById('theme_toggle');
const THEME_NEXT = { auto: 'light', light: 'dark', dark: 'auto' };
function resolveTheme(mode) {
  if (mode === 'light' || mode === 'dark') return mode;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function setThemeTooltip(mode) {
  const label = mode.charAt(0).toUpperCase() + mode.slice(1);
  const nxt = THEME_NEXT[mode];
  themeBtn.title = 'Theme: ' + label + ' (click for ' + nxt.charAt(0).toUpperCase() + nxt.slice(1) + ')';
}
function applyTheme(mode) {
  document.documentElement.dataset.theme = resolveTheme(mode);
  document.documentElement.dataset.themeMode = mode;
  setThemeTooltip(mode);
  draw();
  drawValueChart();
  if (state) {
    drawHeat('h_mcts_policy', state.mcts_policy);
    drawHeat('h_mcts_visits', state.mcts_visits);
    drawHeat('h_nn_policy',   state.nn_policy);
    drawHeat('h_nn_opp_policy', state.nn_opp_policy);
    drawHeatSigned('h_nn_futurepos_8',  state.nn_futurepos_8);
    drawHeatSigned('h_nn_futurepos_32', state.nn_futurepos_32);
  } else {
    drawHeat('h_mcts_policy', null);
    drawHeat('h_mcts_visits', null);
    drawHeat('h_nn_policy',   null);
    drawHeat('h_nn_opp_policy', null);
    drawHeatSigned('h_nn_futurepos_8',  null);
    drawHeatSigned('h_nn_futurepos_32', null);
  }
  paintHeatModal();
}
setThemeTooltip(document.documentElement.dataset.themeMode || 'auto');
themeBtn.addEventListener('click', () => {
  const cur = document.documentElement.dataset.themeMode || 'auto';
  const next = THEME_NEXT[cur] || 'auto';
  try {
    if (next === 'auto') localStorage.removeItem('skz_theme');
    else localStorage.setItem('skz_theme', next);
  } catch(e) {}
  applyTheme(next);
});
// Follow OS dark/light changes live, but only while in auto mode.
try {
  const mql = window.matchMedia('(prefers-color-scheme: dark)');
  const onSystemThemeChange = () => {
    if ((document.documentElement.dataset.themeMode || 'auto') !== 'auto') return;
    applyTheme('auto');
  };
  if (mql.addEventListener) mql.addEventListener('change', onSystemThemeChange);
  else if (mql.addListener) mql.addListener(onSystemThemeChange); // Safari < 14
} catch(e) {}

/* ---------- Heat map ---------- */
function drawHeat(canvasId, grid) {
  const g = heatCtxs[canvasId];
  clearLogical(g);
  const W = g._logicalW;
  const cell = W / N;
  const gridCol = cssVar('--heat-grid') || '#e5e7eb';
  let maxV = 0;
  if (grid) for (let r=0;r<N;r++) for (let k=0;k<N;k++) if (grid[r][k]>maxV) maxV=grid[r][k];
  for (let r=0;r<N;r++) for (let k=0;k<N;k++) {
    const x = k*cell, y = r*cell;
    let v = grid ? grid[r][k] : 0;
    const a = (maxV>0 && v>0) ? Math.min(1, v/maxV) : 0;
    g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
    g.fillRect(x, y, cell, cell);
    g.strokeStyle = gridCol;
    g.strokeRect(x + 0.5, y + 0.5, cell, cell);
    if (v >= 0.01) {
      g.fillStyle = a > 0.5 ? '#fff' : cssVar('--heat-text');
      g.font = `${Math.floor(cell*0.38)}px ${MONO_FONT}`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText((v*100).toFixed(0), x + cell/2, y + cell/2);
    }
  }
  if (state && state.board) {
    const r0 = cell * 0.32;
    for (let r=0;r<N;r++) for (let k=0;k<N;k++) {
      const sv = state.board[r][k]; if (!sv) continue;
      const cx = k*cell + cell/2, cy = r*cell + cell/2;
      g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI*2);
      if (sv === 1) {
        g.fillStyle = 'rgba(0,0,0,0.7)';
        g.fill();
        g.lineWidth = 1;
        g.strokeStyle = 'rgba(255,255,255,0.6)';
        g.stroke();
      } else {
        g.fillStyle = 'rgba(255,255,255,0.85)';
        g.fill();
        g.lineWidth = 1;
        g.strokeStyle = 'rgba(0,0,0,0.5)';
        g.stroke();
      }
    }
  }
}

/* ---------- Signed heat map (futurepos: +own / -opp ∈ [-1,+1]) ---------- */
function drawHeatSigned(canvasId, grid) {
  const g = heatCtxs[canvasId];
  clearLogical(g);
  const W = g._logicalW;
  const cell = W / N;
  const gridCol = cssVar('--heat-grid') || '#e5e7eb';
  const heatText = cssVar('--heat-text') || '#111';
  for (let r=0;r<N;r++) for (let k=0;k<N;k++) {
    const x = k*cell, y = r*cell;
    const v = grid ? grid[r][k] : 0;
    const a = Math.min(1, Math.abs(v));
    if (v > 0) {
      // Own future stone — accent blue.
      g.fillStyle = `rgba(9,105,218,${a.toFixed(3)})`;
    } else if (v < 0) {
      // Opponent future stone — danger red.
      g.fillStyle = `rgba(207,34,46,${a.toFixed(3)})`;
    } else {
      g.fillStyle = 'rgba(0,0,0,0)';
    }
    g.fillRect(x, y, cell, cell);
    g.strokeStyle = gridCol;
    g.strokeRect(x + 0.5, y + 0.5, cell, cell);
    if (Math.abs(v) >= 0.05) {
      g.fillStyle = a > 0.5 ? '#fff' : heatText;
      g.font = `${Math.floor(cell*0.32)}px ${MONO_FONT}`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      const label = (v >= 0 ? '+' : '') + (v*100).toFixed(0);
      g.fillText(label, x + cell/2, y + cell/2);
    }
  }
  if (state && state.board) {
    const r0 = cell * 0.32;
    for (let r=0;r<N;r++) for (let k=0;k<N;k++) {
      const sv = state.board[r][k]; if (!sv) continue;
      const cx = k*cell + cell/2, cy = r*cell + cell/2;
      g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI*2);
      if (sv === 1) {
        g.fillStyle = 'rgba(0,0,0,0.7)';
        g.fill();
        g.lineWidth = 1;
        g.strokeStyle = 'rgba(255,255,255,0.6)';
        g.stroke();
      } else {
        g.fillStyle = 'rgba(255,255,255,0.85)';
        g.fill();
        g.lineWidth = 1;
        g.strokeStyle = 'rgba(0,0,0,0.5)';
        g.stroke();
      }
    }
  }
}

/* ---------- Board ---------- */
function draw() {
  clearLogical(ctx);
  const boardLine = cssVar('--board-line') || '#6b5a3a';
  const boardStar = cssVar('--board-star') || '#3a2e1a';
  const stoneB0 = cssVar('--stone-black-0');
  const stoneB1 = cssVar('--stone-black-1');
  const stoneW0 = cssVar('--stone-white-0');
  const stoneW1 = cssVar('--stone-white-1');
  const stoneOutline = cssVar('--stone-outline');
  const stoneShadow = cssVar('--stone-shadow') || 'rgba(0,0,0,0.18)';

  ctx.strokeStyle = boardLine; ctx.lineWidth = 1;
  for (let i=0;i<N;i++){
    const p = Math.round(MARGIN + i*CELL) + 0.5;
    ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL*(N-1), p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL*(N-1)); ctx.stroke();
  }
  ctx.fillStyle = boardStar;
  // 4-4 corner hoshi for N>=13, 3-3 for smaller; tengen only on odd N (no
  // fake off-center tengen on even boards).
  if (N >= 7) {
    const off = (N >= 13) ? 3 : 2;
    const pts = [[off, off], [off, N-1-off], [N-1-off, off], [N-1-off, N-1-off]];
    if (N % 2 === 1) pts.push([(N-1)/2, (N-1)/2]);
    for (const [r, c] of pts) {
      ctx.beginPath();
      ctx.arc(MARGIN + c*CELL, MARGIN + r*CELL, 3.5, 0, Math.PI*2);
      ctx.fill();
    }
  }
  ctx.fillStyle = boardLine;
  ctx.font = `11px ${MONO_FONT}`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i=0;i<N;i++){
    ctx.fillText(i, MARGIN + i*CELL, 12);
    ctx.fillText(i, 10, MARGIN + i*CELL);
  }
  if (!state) return;

  const stoneR    = Math.max(6, Math.round(CELL * 0.39));
  const gumbelR   = Math.max(6, Math.round(CELL * 0.34));
  const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
  const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
  const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
  const gradInner = Math.max(1, Math.round(CELL * 0.11));
  const gumbelFontPx = Math.max(8, Math.round(CELL * 0.28));

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
      ctx.beginPath(); ctx.arc(x, y, gumbelR, 0, Math.PI*2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = COLORS[bucket];
      ctx.stroke();
      ctx.lineWidth = 1;
      if (state.board[r][c] === 0) {
        ctx.fillStyle = COLORS[bucket];
        ctx.font = `bold ${gumbelFontPx}px ${MONO_FONT}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(sizeLabel, x, y);
      }
    }
  }

  const b = state.board, lm = state.last_move;
  for (let r=0;r<N;r++) for (let c=0;c<N;c++){
    const v = b[r][c]; if (!v) continue;
    const x = MARGIN+c*CELL, y = MARGIN+r*CELL;
    ctx.beginPath(); ctx.arc(x+shadowDx, y+shadowDy, stoneR, 0, Math.PI*2);
    ctx.fillStyle = stoneShadow; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI*2);
    if (v === 1) {
      const grad = ctx.createRadialGradient(x-gradInner, y-gradInner, 2, x, y, stoneR);
      grad.addColorStop(0, stoneB0); grad.addColorStop(1, stoneB1);
      ctx.fillStyle = grad;
    } else {
      const grad = ctx.createRadialGradient(x-gradInner, y-gradInner, 2, x, y, stoneR);
      grad.addColorStop(0, stoneW0); grad.addColorStop(1, stoneW1);
      ctx.fillStyle = grad;
    }
    ctx.fill();
    ctx.strokeStyle = stoneOutline; ctx.lineWidth = 1; ctx.stroke();
    if (lm && lm[0]===r && lm[1]===c) {
      ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI*2);
      ctx.fillStyle = '#ef4444'; ctx.fill();
    }
  }
}

/* ---------- Value chart ---------- */
function stoneCount(board) {
  let n = 0;
  for (let r=0;r<N;r++) for (let c=0;c<N;c++) if (board[r][c]) n++;
  return n;
}
function normWL(v) {
  if (!v) return null;
  const s = v.w + v.d + v.l;
  if (s > 1e-4) return (v.w - v.l) / s;
  return v.wl;
}
function recordValues(st) {
  if (!st || !st.board) return;
  const step = stoneCount(st.board);
  const rw = normWL(st.root_value);
  const nw = normWL(st.nn_value);
  while (valueHistory.length && valueHistory[valueHistory.length - 1].step > step) {
    valueHistory.pop();
  }
  const last = valueHistory[valueHistory.length - 1];
  if (rw == null && nw == null) {
    // No fresh AI values. After an undo, root_value/nn_value are cleared, so
    // we still want each subsequent move to extend the chart with a
    // carry-forward of the last known eval — matching the normal-play rhythm
    // where every ply produces a point. (Pre-game has no `last` and is left
    // empty until the first AI think.)
    if (last && step > last.step) {
      valueHistory.push({step, root: last.root, nn: last.nn});
    }
    return;
  }
  if (last && last.step === step) {
    if (rw != null) last.root = rw;
    if (nw != null) last.nn = nw;
  } else if (!last || step > last.step) {
    valueHistory.push({step, root: rw, nn: nw});
  }
}
function drawValueChart() {
  clearLogical(vctx);
  const W = vctx._logicalW, H = vctx._logicalH;
  const padL = 22, padR = 6, padT = 6, padB = 14;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const axis = cssVar('--border') || '#d8dee4';
  const grid = cssVar('--heat-grid') || '#e5e7eb';
  const muted = cssVar('--fg-muted') || '#59636e';
  const subtle = cssVar('--fg-subtle') || '#8b949e';
  vctx.strokeStyle = grid; vctx.lineWidth = 1;
  for (const v of [-1, 0, 1]) {
    const y = padT + ((1 - v) / 2) * innerH + 0.5;
    vctx.beginPath(); vctx.moveTo(padL, y); vctx.lineTo(W - padR, y); vctx.stroke();
  }
  vctx.fillStyle = subtle;
  vctx.font = `10px ${MONO_FONT}`;
  vctx.textAlign = 'right'; vctx.textBaseline = 'middle';
  for (const v of [1, 0, -1]) {
    const y = padT + ((1 - v) / 2) * innerH;
    vctx.fillText((v > 0 ? '+' : '') + v.toFixed(0), padL - 4, y);
  }
  vctx.strokeStyle = axis;
  vctx.beginPath();
  vctx.moveTo(padL + 0.5, padT); vctx.lineTo(padL + 0.5, H - padB);
  vctx.lineTo(W - padR, H - padB); vctx.stroke();

  if (valueHistory.length === 0) {
    vctx.fillStyle = subtle;
    vctx.font = `11px ${MONO_FONT}`;
    vctx.textAlign = 'center'; vctx.textBaseline = 'middle';
    vctx.fillText('no data', padL + innerW/2, padT + innerH/2);
    return;
  }
  const maxStep = Math.max(1, valueHistory[valueHistory.length - 1].step);
  const xOf = s => padL + (s / maxStep) * innerW;
  const yOf = v => padT + ((1 - v) / 2) * innerH;
  vctx.fillStyle = muted;
  vctx.textAlign = 'center'; vctx.textBaseline = 'top';
  vctx.fillText('0', xOf(0), H - padB + 2);
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
      vctx.arc(xOf(p.step), yOf(p[key]), 2, 0, Math.PI*2);
      vctx.fill();
    }
  }
  plot('root', '#0969da');
  plot('nn', '#cf222e');
}

/* ---------- WDL ---------- */
// Renormalizes raw W/D/L so they display summing to ~100%.
function normalizeVal(v) {
  if (!v) return null;
  const s = v.w + v.d + v.l;
  if (s > 1e-4) {
    return { w: v.w/s*100, d: v.d/s*100, l: v.l/s*100,
             wl: (v.w - v.l)/s, sum: s };
  }
  return { w: v.w, d: v.d, l: v.l, wl: v.wl, sum: s };
}

function renderWDL(prefix, v) {
  const bar = document.getElementById('wdl_' + prefix + '_bar');
  const wlEl = document.getElementById('wdl_' + prefix + '_wl');
  const det = document.getElementById('wdl_' + prefix + '_detail');
  const n = normalizeVal(v);
  const segs = bar.querySelectorAll('.seg');
  if (!n) {
    segs[0].style.width = '0';
    segs[1].style.width = '100%';
    segs[2].style.width = '0';
    wlEl.textContent = '—';
    wlEl.classList.remove('pos', 'neg');
    det.textContent = '';
    return;
  }
  segs[0].style.width = n.w.toFixed(2) + '%';
  segs[1].style.width = n.d.toFixed(2) + '%';
  segs[2].style.width = n.l.toFixed(2) + '%';
  const wl = n.wl;
  wlEl.textContent = (wl >= 0 ? '+' : '') + wl.toFixed(2);
  wlEl.classList.toggle('pos', wl > 0.01);
  wlEl.classList.toggle('neg', wl < -0.01);
  det.innerHTML =
    '<span><span class="k">W</span> ' + n.w.toFixed(1) + '%</span>' +
    '<span><span class="k">D</span> ' + n.d.toFixed(1) + '%</span>' +
    '<span><span class="k">L</span> ' + n.l.toFixed(1) + '%</span>';
}

/* ---------- Status pill ---------- */
function statusVariant(s) {
  if (!s) return 'idle';
  const t = s.toLowerCase();
  if (t.includes('thinking')) return 'thinking';
  if (t.includes('your turn')) return 'active';
  if (t.includes('wins') || t.includes('draw')) return 'done';
  if (t.includes('invalid') || t.includes('exited')) return 'error';
  if (t.includes('launching')) return 'warn';
  if (t.includes('played')) return 'info';
  return 'idle';
}

async function refresh() {
  try {
    const r = await fetch('/state'); state = await r.json();
    if (state && state.board_size && state.board_size !== N) {
      N = state.board_size;
      valueHistory = [];
      syncBoardSize();
      const ss = document.getElementById('size_select');
      if (ss && ss.value !== String(N)) ss.value = String(N);
    }
    if (pendingSwapBaseline) {
      const stale = valuesEqual(state.root_value, pendingSwapBaseline.root)
                 && valuesEqual(state.nn_value,   pendingSwapBaseline.nn);
      if (stale) {
        if (pendingSwapParity === 1) {
          state.root_value = flipWDL(state.root_value);
          state.nn_value   = flipWDL(state.nn_value);
        }
      } else {
        pendingSwapBaseline = null;
        pendingSwapParity = 0;
      }
    }
    const statusText = state.status || 'idle';
    document.getElementById('status').textContent = statusText;
    document.getElementById('status_pill').dataset.variant = statusVariant(statusText);

    if (!sideSynced && (state.human_side === 1 || state.human_side === -1)) {
      selectedSide = state.human_side;
      updateSideButtons();
      sideSynced = true;
    }
    renderWDL('root', state.root_value);
    renderWDL('nn', state.nn_value);
    recordValues(state);
    drawValueChart();

    draw();
    drawHeat('h_mcts_policy', state.mcts_policy);
    drawHeat('h_mcts_visits', state.mcts_visits);
    drawHeat('h_nn_policy',   state.nn_policy);
    drawHeat('h_nn_opp_policy', state.nn_opp_policy);
    drawHeatSigned('h_nn_futurepos_8',  state.nn_futurepos_8);
    drawHeatSigned('h_nn_futurepos_32', state.nn_futurepos_32);
    paintHeatModal();
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
  const el = document.getElementById('sims_input');
  const n = parseInt(el.value, 10);
  if (Number.isFinite(n) && n >= 1) sendCmd('sims ' + n);
}
function applyGm() {
  const el = document.getElementById('gm_input');
  const m = parseInt(el.value, 10);
  if (Number.isFinite(m) && m >= 1) sendCmd('gm ' + m);
}
let selectedSide = 1;
let sideSynced = false;
function updateSideButtons() {
  document.getElementById('side_black').setAttribute('aria-pressed', selectedSide === 1 ? 'true' : 'false');
  document.getElementById('side_white').setAttribute('aria-pressed', selectedSide === -1 ? 'true' : 'false');
}
function setSide(side) {
  if (side !== 1 && side !== -1) return;
  if (selectedSide === side) return;
  // If no session yet or game over, start fresh. Otherwise swap mid-game.
  if (!state || state.game_over) {
    selectedSide = side;
    updateSideButtons();
    newGame(side);
    return;
  }
  selectedSide = side;
  updateSideButtons();
  // Historical values were from the previous AI's perspective; flip so chart stays AI-relative.
  for (const p of valueHistory) {
    if (p.root != null) p.root = -p.root;
    if (p.nn != null) p.nn = -p.nn;
  }
  if (pendingSwapBaseline === null && state) {
    pendingSwapBaseline = {
      root: state.root_value ? {...state.root_value} : null,
      nn:   state.nn_value   ? {...state.nn_value}   : null,
    };
  }
  pendingSwapParity ^= 1;
  drawValueChart();
  sendCmd('side ' + side);
}
async function newGame(side, modelId, boardSize, rule) {
  if (side === undefined) side = selectedSide;
  else { selectedSide = side; updateSideButtons(); }
  valueHistory = [];
  pendingSwapBaseline = null;
  pendingSwapParity = 0;
  drawValueChart();
  const payload = {human_side: side};
  if (modelId) payload.model = modelId;
  if (Number.isFinite(boardSize) && boardSize > 0) payload.board_size = boardSize;
  if (typeof rule === 'string' && rule.length > 0) payload.rule = rule;
  await fetch('/new', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(payload)});
  sendCmd('noise 0');
  sendCmd('prune ' + (pruneToggle.checked ? 1 : 0));
  const sims = parseInt(document.getElementById('sims_input').value, 10);
  if (Number.isFinite(sims) && sims >= 1) sendCmd('sims ' + sims);
  const gm = parseInt(document.getElementById('gm_input').value, 10);
  if (Number.isFinite(gm) && gm >= 1) sendCmd('gm ' + gm);
  refresh();
}

async function loadModels() {
  try {
    const r = await fetch('/models');
    const data = await r.json();
    const sel = document.getElementById('model_select');
    sel.innerHTML = '';
    const groups = {};
    for (const it of data.items) {
      (groups[it.group] = groups[it.group] || []).push(it);
    }
    const order = ['models', 'anchors', 'custom'];
    const seen = new Set();
    for (const g of order) {
      if (!groups[g]) continue;
      seen.add(g);
      const og = document.createElement('optgroup');
      og.label = g;
      for (const it of groups[g]) {
        const o = document.createElement('option');
        o.value = it.id; o.textContent = it.label;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    for (const g of Object.keys(groups)) {
      if (seen.has(g)) continue;
      const og = document.createElement('optgroup');
      og.label = g;
      for (const it of groups[g]) {
        const o = document.createElement('option');
        o.value = it.id; o.textContent = it.label;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    sel.value = data.current;
  } catch(e) { /* ignore */ }
}
document.getElementById('model_select').addEventListener('change', (ev) => {
  newGame(selectedSide, ev.target.value);
});

async function loadConfig() {
  try {
    const r = await fetch('/config');
    const data = await r.json();
    const sel = document.getElementById('size_select');
    sel.innerHTML = '';
    for (const sz of data.board_sizes) {
      const o = document.createElement('option');
      o.value = String(sz); o.textContent = String(sz);
      sel.appendChild(o);
    }
    sel.value = String(data.current_board_size);
    const allowed = new Set(data.rules || []);
    selectedRule = data.current_rule;
    for (const btn of document.querySelectorAll('.seg-btn[data-rule]')) {
      const rl = btn.dataset.rule;
      btn.disabled = !allowed.has(rl);
      btn.setAttribute('aria-pressed', rl === selectedRule ? 'true' : 'false');
    }
  } catch(e) { /* ignore */ }
}
document.getElementById('size_select').addEventListener('change', (ev) => {
  const sz = parseInt(ev.target.value, 10);
  if (!Number.isFinite(sz)) return;
  newGame(selectedSide, undefined, sz);
});
let selectedRule = null;
function setRule(rl) {
  if (!rl || rl === selectedRule) return;
  const btn = document.querySelector('.seg-btn[data-rule="' + rl + '"]');
  if (!btn || btn.disabled) return;
  selectedRule = rl;
  for (const b of document.querySelectorAll('.seg-btn[data-rule]')) {
    b.setAttribute('aria-pressed', b.dataset.rule === rl ? 'true' : 'false');
  }
  newGame(selectedSide, undefined, undefined, rl);
}

// Auto-apply sims/gumbel_m on change (blur) or Enter, replacing Apply buttons.
function bindNumInput(id, apply) {
  const el = document.getElementById(id);
  let last = el.value;
  el.addEventListener('change', () => { if (el.value !== last) { last = el.value; apply(); } });
  el.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); el.blur(); }
  });
}
bindNumInput('sims_input', applySims);
bindNumInput('gm_input', applyGm);
sendCmd('noise 0'); // ensure engine starts with Gumbel noise off
sendCmd('prune ' + (pruneToggle.checked ? 1 : 0));

draw();
drawValueChart();
drawHeat('h_mcts_policy', null);
drawHeat('h_mcts_visits', null);
drawHeat('h_nn_policy', null);
drawHeat('h_nn_opp_policy', null);
drawHeatSigned('h_nn_futurepos_8', null);
drawHeatSigned('h_nn_futurepos_32', null);

/* ---------- Heat modal events ---------- */
for (const btn of document.querySelectorAll('.expand-btn')) {
  btn.addEventListener('click', () => openHeatModal(btn.dataset.target));
}
document.getElementById('heat_modal_close').addEventListener('click', closeHeatModal);
document.getElementById('heat_modal').addEventListener('click', (ev) => {
  if (ev.target === ev.currentTarget) closeHeatModal();
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && expandedSourceId !== null) closeHeatModal();
});
window.addEventListener('resize', () => {
  if (expandedSourceId !== null) {
    setupModalCanvas();
    paintHeatModal();
  }
});

syncBoardSize();
setInterval(refresh, 250);
loadModels();
loadConfig();
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
        if self.path == "/models":
            self._send_json(200, Handler.app.model_listing())
            return
        if self.path == "/config":
            self._send_json(200, {
                "board_sizes": Handler.app.board_sizes,
                "current_board_size": Handler.app.current_board_size,
                "rules": Handler.app.rules,
                "current_rule": Handler.app.current_rule,
            })
            return
        if self.path == "/state":
            sess = Handler.app.current()
            if sess is None:
                bs = Handler.app.current_board_size
                self._send_json(200, {"version": 0, "board": [[0]*bs for _ in range(bs)],
                                      "board_size": bs,
                                      "rule": Handler.app.current_rule,
                                      "last_move": None, "status": "No game. Click New.",
                                      "root_value": None, "nn_value": None,
                                      "game_over": False, "human_side": 1,
                                      "mcts_policy": None, "mcts_visits": None, "nn_policy": None,
                                      "nn_opp_policy": None, "nn_futurepos_8": None,
                                      "nn_futurepos_32": None, "gumbel_phases": None})
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
            model_id = body.get("model")
            if model_id is not None and not isinstance(model_id, str):
                model_id = None
            bs = body.get("board_size")
            if not (isinstance(bs, int) and bs in Handler.app.board_sizes):
                bs = None
            rule = body.get("rule")
            if not (isinstance(rule, str) and rule in Handler.app.rules):
                rule = None
            Handler.app.start(side, model_id=model_id, board_size=bs, rule=rule)
            self._send_json(200, {"ok": True,
                                  "model": Handler.app.current_model_id,
                                  "board_size": Handler.app.current_board_size,
                                  "rule": Handler.app.current_rule})
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


def parse_board_sizes(run_cfg_path):
    """Read BOARD_SIZES="17, 16, ..." from run.cfg → sorted list[int] (desc)."""
    try:
        for line in Path(run_cfg_path).read_text().splitlines():
            s = line.strip()
            if not s.startswith("BOARD_SIZES"):
                continue
            _, _, val = s.partition("=")
            val = val.strip().strip('"').strip("'").split("#", 1)[0]
            sizes = [int(x.strip()) for x in val.split(",") if x.strip()]
            return sorted(set(sizes), reverse=True)
    except (OSError, ValueError):
        pass
    return []


SUPPORTED_RULES = ("renju", "standard", "freestyle")


def parse_rules(run_cfg_path):
    """Read RULES="renju, standard, ..." from run.cfg → list[str] preserving cfg order."""
    try:
        for line in Path(run_cfg_path).read_text().splitlines():
            s = line.strip()
            if not s.startswith("RULES"):
                continue
            _, _, val = s.partition("=")
            val = val.strip().strip('"').strip("'").split("#", 1)[0]
            seen = []
            for tok in val.split(","):
                r = tok.strip().lower()
                if r in SUPPORTED_RULES and r not in seen:
                    seen.append(r)
            return seen
    except OSError:
        pass
    return []


def read_cfg_int(path, key, default):
    try:
        for line in Path(path).read_text().splitlines():
            s = line.strip()
            if s.startswith(f"{key}=") or s.startswith(f"{key} ="):
                _, _, val = s.partition("=")
                return int(val.strip().split("#", 1)[0])
    except (OSError, ValueError):
        pass
    return default


def read_cfg_str(path, key, default):
    try:
        for line in Path(path).read_text().splitlines():
            s = line.strip()
            if s.startswith(f"{key}=") or s.startswith(f"{key} ="):
                _, _, val = s.partition("=")
                return val.strip().strip('"').strip("'").split("#", 1)[0].strip()
    except OSError:
        pass
    return default


def discover_models(root_dir):
    """Scan data/models/*.pt and anchors/*.pt; returns ordered dict id -> abs Path."""
    out = {}
    models_dir = root_dir / "data" / "models"
    if models_dir.is_dir():
        files = sorted(models_dir.glob("*.pt"), key=lambda p: p.name, reverse=True)
        # surface latest.pt first if present
        files.sort(key=lambda p: 0 if p.name == "latest.pt" else 1)
        for p in files:
            out[f"models/{p.name}"] = p.resolve()
    anchors_dir = root_dir / "anchors"
    if anchors_dir.is_dir():
        for p in sorted(anchors_dir.glob("*.pt"), key=lambda p: p.name):
            out[f"anchors/{p.name}"] = p.resolve()
    return out


def main():
    root_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(root_dir / "data" / "models" / "latest.pt"))
    ap.add_argument("--bin", default=str(root_dir / "cpp" / "build" / "gomoku_play"))
    ap.add_argument("--config", default=str(root_dir / "scripts" / "play.cfg"))
    ap.add_argument("--run-config", default=str(root_dir / "scripts" / "run.cfg"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--human-side", type=int, default=1, choices=(1, -1))
    args = ap.parse_args()

    for p, name in [(args.model, "model"), (args.bin, "binary"), (args.config, "config")]:
        if not Path(p).exists():
            raise SystemExit(f"{name} not found: {p}")

    models = discover_models(root_dir)
    init_path = Path(args.model).resolve()
    init_id = None
    for mid, p in models.items():
        if p == init_path:
            init_id = mid
            break
    if init_id is None:
        init_id = f"custom/{init_path.name}"
        models = {init_id: init_path, **models}

    board_sizes = parse_board_sizes(args.run_config) or [BOARD_SIZE]
    default_board_size = read_cfg_int(args.config, "BOARD_SIZE", BOARD_SIZE)
    if default_board_size not in board_sizes:
        board_sizes = [default_board_size] + board_sizes

    rules = parse_rules(args.run_config) or list(SUPPORTED_RULES)
    default_rule = read_cfg_str(args.config, "RULE", rules[0]).lower()
    if default_rule not in rules:
        rules = [default_rule] + rules

    app = App(Path(args.bin), Path(args.config), models, init_id,
              board_sizes, default_board_size, rules, default_rule)
    app.start(args.human_side, board_size=default_board_size, rule=default_rule)
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
