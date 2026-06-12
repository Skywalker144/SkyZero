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
        self.gumbel_winrate = None  # 15x15, per-candidate win rate in [0,1] (expected score, draws=0.5)
        self.analyzing = False        # True while the engine is pondering (analyze loop)
        self.analyze_sims = None      # accumulated sims in the current ponder (None when idle)
        self.value_persp = None       # perspective of root/nn values: +1 Black, -1 White
        self.analysis_mode = False    # True in free-analysis board mode (human drives both colors)
        self.awaiting_human = False   # True while the engine waits at the human-move prompt
        self.last_move_winrate = None     # win rate (0..1) of the move just played, mover's view
        self.last_move_winrate_rc = None  # [r,c] it belongs to; drawn only when == last_move

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
        if "Gumbel WinRate" in line:
            self._pending_grid_key = "gumbel_winrate"
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
        if line.startswith("Analyze sims:"):
            try:
                self.analyze_sims = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
            return
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
            r, c = int(m.group(1)), int(m.group(2))
            self.status = f"AI played ({r}, {c})"
            # Pin the chosen move's win rate (mover's view) to its cell so the
            # board can label that one stone; cleared automatically once the
            # last move moves on (rc no longer matches).
            wr = None
            if self.gumbel_winrate and 0 <= r < len(self.gumbel_winrate) \
                    and 0 <= c < len(self.gumbel_winrate[r]):
                wr = self.gumbel_winrate[r][c]
            self.last_move_winrate = wr
            self.last_move_winrate_rc = [r, c] if wr is not None else None
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
            # Lock the perspective NOW: the value table is emitted before the
            # searched move is applied/printed, so self.board still holds the
            # search position and its side-to-move IS this value's perspective
            # (+1 Black / -1 White). Deriving it later from the advanced board
            # would mis-sign a stale value after the opponent's reply.
            self.value_persp = 1 if self._black_to_move() else -1
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
        if "[setting] mode=" in line:
            self.analysis_mode = line.strip().split("=")[-1] == "analysis"
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
            self.gumbel_winrate = None
            self.last_move_winrate = None
            self.last_move_winrate_rc = None
            return
        if "Human step" in line:
            self.awaiting_human = True
            if not self.game_over:
                self.status = "Your turn"
            return
        if "SkyZero thinking" in line:
            self.status = "AI thinking..."
            self.awaiting_human = False
            self.gumbel_phases = None
            self.gumbel_winrate = None
            return
        if "Analyze start" in line:
            self.analyzing = True
            self.analyze_sims = 0
            self.awaiting_human = False
            self.status = "Analyzing… (engine pondering)"
            return
        if "Analyze stopped" in line:
            self.analyzing = False
            self.analyze_sims = None
            self.awaiting_human = True
            self.status = "Your turn"
            # Analyze values are side-to-move (human) perspective; drop them so
            # a later poll cannot record them into the AI-perspective chart.
            self.root_value = None
            self.nn_value = None
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

    def _black_to_move(self):
        # Black plays first and colors strictly alternate, so equal stone counts
        # mean it is Black's turn.
        b = w = 0
        for row in self.board:
            for v in row:
                if v == 1:
                    b += 1
                elif v == -1:
                    w += 1
        return b == w

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

    def reset(self, human_side):
        """Start a fresh game in the same engine process. Saves the ~2.5s
        respawn cost; the engine's loaded model + MCTS threads stay alive.
        Returns False if the process is no longer running."""
        if self.proc.poll() is not None:
            return False
        bs = self.board_size
        with self.lock:
            self.human_side = human_side
            self.board = [[0] * bs for _ in range(bs)]
            self.last_move = None
            self.status = "New game starting..."
            self.root_value = None
            self.nn_value = None
            self.game_over = False
            self.mcts_policy = None
            self.mcts_visits = None
            self.nn_policy = None
            self.nn_opp_policy = None
            self.nn_futurepos_8 = None
            self.nn_futurepos_32 = None
            self.gumbel_phases = None
            self.gumbel_winrate = None
            self.analyzing = False
            self.analyze_sims = None
            self.value_persp = None
            self.awaiting_human = False
            self.last_move_winrate = None
            self.last_move_winrate_rc = None
            self._pending_rows = None
            self._pending_grid_key = None
            self._pending_grid_rows = None
            self._bump()
        self.send(f"newgame {human_side}")
        return True

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
                "gumbel_winrate": self.gumbel_winrate,
                "analyzing": self.analyzing,
                "analyze_sims": self.analyze_sims,
                "analysis_mode": self.analysis_mode,
                "value_persp": self.value_persp,
                "awaiting_human": self.awaiting_human,
                "last_move_winrate": self.last_move_winrate,
                "last_move_winrate_rc": list(self.last_move_winrate_rc) if self.last_move_winrate_rc else None,
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
            # Reuse the running engine when model/board/rule are unchanged —
            # avoids the ~2.5s gomoku_play respawn for each new game.
            if self.session is not None:
                same_engine = (
                    self.session.model == self.model
                    and self.session.board_size == self.current_board_size
                    and self.session.rule == self.current_rule
                    and self.session.proc.poll() is None
                )
                if same_engine and self.session.reset(human_side):
                    return
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
  // Resolve light/dark from the OS before first paint to avoid a flash.
  (function(){
    try {
      document.documentElement.dataset.theme =
        window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } catch(e) {
      document.documentElement.dataset.theme = 'light';
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

  .app {
    max-width: 2100px; margin: 0 auto; padding: 12px 16px;
    display: flex; flex-direction: column; gap: 12px;
    min-height: 100vh;
  }

  /* ---------- Top toolbar ---------- */
  .topbar {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 8px 16px; padding: 8px 12px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); box-shadow: var(--shadow-xs);
  }
  .tb-group { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .tb-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--fg-subtle); white-space: nowrap;
  }
  .tb-select { height: 30px; text-align: left; max-width: 200px;
    font-family: var(--font-mono); }
  .tb-actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 auto; }
  .tb-sep { width: 1px; align-self: stretch; background: var(--border); margin: 2px 0; }

  /* ---------- Main: win-rate chart + board + analysis ---------- */
  .main {
    display: grid;
    grid-template-columns: 320px minmax(0, 1fr) 340px;
    gap: 12px; align-items: start;
  }
  @media (max-width: 1180px) {
    .main { grid-template-columns: 1fr; }
    .board-col { order: -1; }
  }
  .board-col {
    display: flex; flex-direction: column; align-items: center;
    min-width: 0;
  }
  .analysis-col {
    display: flex; flex-direction: column; gap: 12px;
    min-width: 0; min-height: 0; overflow-y: auto;
  }
  /* Left column: the win-rate-over-moves chart fills the board's height. */
  .winrate-col { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .winrate-col .card { display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
  .winrate-col .card-body {
    display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0;
    padding: 10px 14px;
  }
  .seg-row { display: flex; gap: 6px; }
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

  /* ---------- Number input ---------- */
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
  .mode-hint {
    font-size: 11px; color: var(--accent);
    font-family: var(--font-mono);
    text-align: right;
    padding: 2px 0 4px 0;
    letter-spacing: 0.4px;
  }

  /* ---------- Win-rate stacked-area chart (left column) ---------- */
  .vc-tabs { margin-bottom: 8px; }
  .vc-legend {
    display: flex; align-items: center; gap: 12px;
    font-size: 11px; color: var(--fg-muted);
    font-family: var(--font-mono);
    margin-bottom: 6px;
  }
  .vc-legend .sw {
    width: 10px; height: 10px; border-radius: 3px;
    display: inline-block; vertical-align: middle; margin-right: 5px;
  }
  .vc-legend .sw.blk { background: var(--stone-black-1); }
  .vc-legend .sw.drw { background: var(--fg-subtle); }
  .vc-legend .sw.wht { background: var(--stone-white-0); box-shadow: inset 0 0 0 1px var(--border-strong); }
  .vc-legend b { color: var(--fg); font-weight: 600; }
  #value_chart { display: block; width: 100%; flex: 1 1 auto; min-height: 0; }
  /* ---------- Board ---------- */
  .board-card {
    padding: 16px;
    display: inline-flex; flex-direction: column; align-items: center;
  }
  #board {
    background: var(--board-bg); border-radius: var(--radius-sm);
    display: block; cursor: crosshair;
  }
  /* ---------- Candidate move list ---------- */
  .cand-card { display: flex; flex-direction: column; min-height: 0; flex: 1 1 auto; }
  .cand-body { display: flex; flex-direction: column; min-height: 0; flex: 1 1 auto; padding: 12px 14px; }
  .cand-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 8px; margin-bottom: 8px;
  }
  .cand-legend {
    font-size: 10.5px; color: var(--fg-subtle); font-family: var(--font-mono);
    text-transform: none; letter-spacing: 0; font-weight: 500;
  }
  .cand-list {
    display: flex; flex-direction: column; gap: 2px;
    overflow-y: auto; min-height: 0; flex: 1 1 auto;
  }
  .cand-empty { color: var(--fg-subtle); font-size: 12px; text-align: center; padding: 18px 0; }
  .cand-row {
    display: grid; grid-template-columns: 16px 42px 40px minmax(0,1fr) 46px;
    align-items: center; gap: 8px;
    padding: 5px 8px; border-radius: var(--radius-sm); cursor: pointer;
    font-family: var(--font-mono); font-size: 12.5px;
    border: 1px solid transparent;
  }
  .cand-row:hover { background: var(--surface-2); border-color: var(--border); }
  .cand-row.best { background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .cand-rank { font-weight: 600; color: var(--fg-muted); text-align: center; }
  .cand-row.best .cand-rank { color: var(--accent); }
  .cand-coord { color: var(--fg-muted); }
  .cand-wr { font-weight: 600; color: var(--fg); text-align: right; }
  .cand-track { height: 6px; background: var(--surface-2); border-radius: 999px; overflow: hidden; }
  .cand-track > span { display: block; height: 100%; background: var(--fg-subtle); }
  .cand-row.best .cand-track > span { background: var(--accent); }
  .cand-visits { color: var(--fg-subtle); text-align: right; font-size: 11.5px; }

  /* ---------- Heatmap drawer ---------- */
  .drawer { overflow: hidden; flex: 0 0 auto; }
  .drawer-toggle {
    width: 100%; display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; background: transparent; border: none; cursor: pointer;
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--fg-subtle); font-family: var(--font-sans);
  }
  .drawer-toggle:hover { color: var(--fg); }
  .drawer-toggle .chev { width: 12px; height: 12px; transition: transform 0.15s; flex-shrink: 0; }
  .drawer-toggle[aria-expanded="true"] .chev { transform: rotate(90deg); }
  .drawer-count { margin-left: auto; color: var(--fg-subtle); font-weight: 500; }
  .drawer-body { padding: 0 12px 12px; }
  @media (prefers-reduced-motion: reduce) { .drawer-toggle .chev { transition: none; } }

  /* ---------- Heat grid ---------- */
  .grids {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }
  .drawer .grid-card { height: 168px; }
  @media (max-width: 520px) {
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
    <div class="status-pill" id="status_pill" data-variant="idle">
      <span class="dot"></span>
      <span id="status">idle</span>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <span class="tb-label">Mode</span>
      <div class="seg-row">
        <button class="seg-btn" id="mode_play" aria-pressed="true" onclick="setMode('play')">对弈</button>
        <button class="seg-btn" id="mode_analysis" aria-pressed="false" onclick="setMode('analysis')">分析</button>
      </div>
    </div>
    <div class="tb-group">
      <span class="tb-label">Model</span>
      <select id="model_select" class="num tb-select"></select>
    </div>
    <div class="tb-group" id="side_row">
      <span class="tb-label">Side</span>
      <div class="seg-row">
        <button class="seg-btn" id="side_black" aria-pressed="true" onclick="setSide(1)">
          <span class="seg-stone black"></span>Black
        </button>
        <button class="seg-btn" id="side_white" aria-pressed="false" onclick="setSide(-1)">
          <span class="seg-stone white"></span>White
        </button>
      </div>
    </div>
    <div class="tb-group">
      <span class="tb-label">Board</span>
      <select id="size_select" class="num tb-select" style="min-width:72px;"></select>
    </div>
    <div class="tb-group">
      <span class="tb-label">Rule</span>
      <div class="seg-row">
        <button class="seg-btn" id="rule_renju"     data-rule="renju"     aria-pressed="false" onclick="setRule('renju')">Renju</button>
        <button class="seg-btn" id="rule_standard"  data-rule="standard"  aria-pressed="false" onclick="setRule('standard')">Standard</button>
        <button class="seg-btn" id="rule_freestyle" data-rule="freestyle" aria-pressed="false" onclick="setRule('freestyle')">Freestyle</button>
      </div>
    </div>
    <div class="tb-group">
      <span class="tb-label">sims</span>
      <input class="num" type="number" id="sims_input" min="0" step="1" value="800" style="width:72px;">
      <span id="sims_mode_hint" class="mode-hint" hidden>Pure NN</span>
    </div>
    <div class="tb-actions">
      <button class="btn primary" id="newgame_btn" onclick="newGame()">New game</button>
      <button class="btn danger-ghost" id="undo_btn" onclick="sendCmd('u')">Undo</button>
      <button class="btn" id="analyze_btn" onclick="toggleAnalyze()">Analyze</button>
    </div>
  </header>

  <div class="main">
    <aside class="winrate-col" id="winrate_col">
      <div class="card">
        <div class="card-body">
          <div class="seg-row vc-tabs">
            <button class="seg-btn" id="vc_tab_root" aria-pressed="true" onclick="setValueTab('root')">root</button>
            <button class="seg-btn" id="vc_tab_nn" aria-pressed="false" onclick="setValueTab('nn')">nn</button>
          </div>
          <div class="vc-legend">
            <span><span class="sw blk"></span>黑胜 <b id="vc_w_black">—</b></span>
            <span><span class="sw drw"></span>平局 <b id="vc_w_draw">—</b></span>
            <span><span class="sw wht"></span>白胜 <b id="vc_w_white">—</b></span>
          </div>
          <canvas id="value_chart"></canvas>
        </div>
      </div>
    </aside>

    <section class="board-col">
      <div class="card board-card">
        <canvas id="board"></canvas>
      </div>
    </section>

    <aside class="analysis-col" id="analysis_col">
      <div class="card cand-card">
        <div class="card-body cand-body">
          <div class="card-title cand-head">
            <span>Candidate moves</span>
            <span class="cand-legend">win% · visits</span>
          </div>
          <div class="cand-list" id="cand_list">
            <div class="cand-empty">No analysis yet</div>
          </div>
        </div>
      </div>

      <div class="card drawer" id="heat_drawer">
        <button class="drawer-toggle" id="heat_drawer_btn" aria-expanded="false" aria-controls="heat_drawer_body">
          <svg class="chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M6 3l5 5-5 5"/>
          </svg>
          <span>NN 热力图</span>
          <span class="drawer-count">6</span>
        </button>
        <div class="drawer-body hidden" id="heat_drawer_body">
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
// setStyle=false: the canvas is CSS-sized (width:100% / flex height); only its
// backing store is managed in JS (see resizeValueChart).
let vctx = setupCanvas(vcCanvas, 280, 160, false);
function resizeValueChart() {
  const rect = vcCanvas.getBoundingClientRect();
  const w = Math.max(120, Math.floor(rect.width));
  const h = Math.max(80, Math.floor(rect.height));
  if (vctx._logicalW === w && vctx._logicalH === h) return;
  vctx = setupCanvas(vcCanvas, w, h, false);
  drawValueChart();
}
new ResizeObserver(resizeValueChart).observe(vcCanvas);

const boardCol = document.querySelector('.board-col');
const boardCard = document.querySelector('.board-card');
const winrateCol = document.getElementById('winrate_col');
const analysisCol = document.getElementById('analysis_col');
const topbarEl = document.querySelector('.topbar');
const mainEl = document.querySelector('.main');
const appEl = document.querySelector('.app');
// Fit the board into the space below the top toolbar, capped by the board
// column's width — so the whole page never scrolls. The win-rate chart (left)
// and analysis panel (right) are both pinned to the board's height; the
// analysis panel scrolls internally.
function syncBoardSize() {
  const narrow = window.matchMedia('(max-width: 1180px)').matches;
  const cardCS = getComputedStyle(boardCard);
  const cardPadX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const cardPadY = parseFloat(cardCS.paddingTop) + parseFloat(cardCS.paddingBottom);
  const appCS = getComputedStyle(appEl);
  const appGap = parseFloat(appCS.rowGap || appCS.gap) || 12;
  const appPadY = parseFloat(appCS.paddingTop) + parseFloat(appCS.paddingBottom);
  // Vertical budget: viewport − app padding − topbar − the gap between the two
  // app rows (topbar / main) − this card's own padding.
  const chromeY = appPadY + topbarEl.offsetHeight + appGap + cardPadY;
  const sizeByHeight = window.innerHeight - chromeY;
  const sizeByWidth = narrow
      ? mainEl.clientWidth - cardPadX
      : boardCol.clientWidth - cardPadX;
  let size = Math.max(360, Math.min(sizeByHeight, sizeByWidth));
  CELL = Math.max(20, Math.floor((size - 2*MARGIN) / (N-1)));
  BOARD_LOGICAL = MARGIN*2 + CELL*(N-1);
  const canvasNeedsResize = cv.width !== Math.round(BOARD_LOGICAL * DPR);
  if (canvasNeedsResize) setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
  const colH = narrow ? '' : boardCard.offsetHeight + 'px';
  winrateCol.style.height = colH;
  analysisCol.style.height = colH;
  if (canvasNeedsResize) draw();
}
new ResizeObserver(syncBoardSize).observe(mainEl);
new ResizeObserver(syncBoardSize).observe(topbarEl);
window.addEventListener('resize', syncBoardSize);

let valueTab = 'root'; // which evaluation the chart shows: 'root' | 'nn'
let valueHistory = []; // [{step, root:{b,d,w}|null, nn:{b,d,w}|null}] — per-ply WDL (Black frame)
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
let hoverCand = null;       // {r,c} candidate row under the pointer → board highlight
const MAX_CANDS = 12;       // board overlay + list cap (lizzie shows the strongest few)
// Value display is fixed to BLACK's frame (Black ahead → +, drawn up; White
// ahead → −, drawn down), so no per-side flipping is ever needed. The engine
// wrapper stamps each root/nn value with the perspective of the side that
// searched it (value_persp: +1 Black / -1 White) at the moment it was produced
// — deriving it from the live board would mis-sign a stale value after the
// opponent's reply.
function valuePerspective(st) {
  return (st && (st.value_persp === 1 || st.value_persp === -1)) ? st.value_persp : 1;
}
// Re-express a side-to-move {w,d,l,wl} in Black's frame.
function toBlack(v, persp) {
  if (!v) return null;
  return persp === 1 ? v : { w: v.l, d: v.d, l: v.w, wl: -v.wl };
}
/* ---------- Candidate moves (lizzie-style overlay + side list) ---------- */
// Rank the side-to-move's candidates from the engine's visit distribution
// (mcts_visits = N(s,a)/sum) and per-candidate win rate (gumbel_winrate, root
// player's view ∈ [0,1]). Sorted by visits desc; capped at MAX_CANDS.
function computeCandidates() {
  if (!state || !state.board) return [];
  const vis = state.mcts_visits, wrG = state.gumbel_winrate;
  if (!vis && !wrG) return [];
  const out = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    if (state.board[r][c] !== 0) continue;
    const vf = (vis && vis[r]) ? (vis[r][c] || 0) : 0;
    const wr = (wrG && wrG[r] && wrG[r][c] != null) ? wrG[r][c] : null;
    if (vf > 0 || (wr != null && wr > 0)) out.push({r, c, vf, wr});
  }
  if (out.length === 0) return [];
  out.sort((a, b) => (b.vf - a.vf) || ((b.wr ?? -1) - (a.wr ?? -1)));
  const maxV = out[0].vf || 0;
  out.forEach((o, i) => { o.frac = maxV > 0 ? o.vf / maxV : 1; o.best = (i === 0); });
  return out.slice(0, MAX_CANDS);
}
// Total root visits — turns a visit fraction back into a count for display.
function totalVisits() {
  if (state && state.analyzing && Number.isFinite(state.analyze_sims) && state.analyze_sims > 0)
    return state.analyze_sims;
  const s = parseInt(document.getElementById('sims_input').value, 10);
  return (Number.isFinite(s) && s > 0) ? s : 0;
}
function fmtVisits(vf) {
  const tot = totalVisits();
  if (tot > 0) {
    const n = Math.round(vf * tot);
    return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n);
  }
  return Math.round(vf * 100) + '%';
}
// Visit-rank color: best move blue, others lerp gray → green by visit share.
function candColor(frac, best) {
  if (best) return {fill: '#2b7fff', text: '#ffffff'};
  const g0 = [156, 163, 175], g1 = [34, 165, 89];  // gray → green
  const t = Math.max(0, Math.min(1, frac));
  const c = g0.map((a, i) => Math.round(a + (g1[i] - a) * t));
  return {fill: `rgb(${c[0]},${c[1]},${c[2]})`, text: t > 0.45 ? '#ffffff' : '#1f2328'};
}

/* ---------- Heatmap drawer (collapsed by default) ---------- */
const heatDrawerBtn = document.getElementById('heat_drawer_btn');
const heatDrawerBody = document.getElementById('heat_drawer_body');
heatDrawerBtn.addEventListener('click', () => {
  const open = heatDrawerBody.classList.toggle('hidden') === false;
  heatDrawerBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) {
    // The canvases measure 0 while hidden — refit + repaint now that they show.
    for (const id of Object.keys(heatCtxs)) {
      if (id === 'h_modal') continue;
      fitHeatCanvas(id);
      drawHeatById(id, state ? state[HEAT_GRID_KEYS[id]] : null);
    }
  }
});

/* ---------- Candidate move list (right panel) ---------- */
const candListEl = document.getElementById('cand_list');
let candSig = '';
// 0-indexed (r,c), matching the board's edge labels and the engine's "AI move".
function coordLabel(r, c) { return r + ',' + c; }
function canPlaceNow() {
  return state && !state.game_over && !(state.analyzing && !state.analysis_mode);
}
function renderCandidates() {
  const cands = state ? computeCandidates() : [];
  // Only rebuild when the data actually changed, so a row's :hover stays put.
  const sig = cands.map(o =>
    o.r+','+o.c+':'+Math.round((o.wr ?? -1)*1000)+':'+Math.round(o.vf*1000)).join('|');
  if (sig === candSig) return;
  candSig = sig;
  if (cands.length === 0) {
    candListEl.innerHTML = '<div class="cand-empty">No analysis yet</div>';
    return;
  }
  candListEl.innerHTML = cands.map((o, i) => {
    const rank = String.fromCharCode(65 + i);            // A, B, C, …
    const wr = o.wr != null ? Math.round(o.wr * 100) + '%' : '—';
    const barW = o.wr != null ? Math.round(o.wr * 100) : 0;
    return '<div class="cand-row' + (o.best ? ' best' : '') +
             '" data-r="' + o.r + '" data-c="' + o.c + '">' +
             '<span class="cand-rank">' + rank + '</span>' +
             '<span class="cand-coord">' + coordLabel(o.r, o.c) + '</span>' +
             '<span class="cand-wr">' + wr + '</span>' +
             '<span class="cand-track"><span style="width:' + barW + '%"></span></span>' +
             '<span class="cand-visits">' + fmtVisits(o.vf) + '</span>' +
           '</div>';
  }).join('');
}
candListEl.addEventListener('mouseover', (ev) => {
  const row = ev.target.closest('.cand-row'); if (!row) return;
  hoverCand = {r: +row.dataset.r, c: +row.dataset.c};
  draw();
});
candListEl.addEventListener('mouseout', (ev) => {
  if (!ev.target.closest('.cand-row')) return;
  if (hoverCand) { hoverCand = null; draw(); }
});
candListEl.addEventListener('click', async (ev) => {
  const row = ev.target.closest('.cand-row'); if (!row) return;
  if (!canPlaceNow()) return;
  const r = +row.dataset.r, c = +row.dataset.c;
  if (!state.board[r] || state.board[r][c] !== 0) return;
  await fetch('/move', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({r, c})});
  refresh();
});

/* ---------- Follow OS dark/light theme ---------- */
// CSS variables flip automatically via [data-theme]; the canvases bake colors
// in at draw time, so they need an explicit repaint when the OS theme changes.
function applySystemTheme() {
  document.documentElement.dataset.theme =
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
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
try {
  const mql = window.matchMedia('(prefers-color-scheme: dark)');
  if (mql.addEventListener) mql.addEventListener('change', applySystemTheme);
  else if (mql.addListener) mql.addListener(applySystemTheme); // Safari < 14
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
  const candR     = Math.max(7, Math.round(CELL * 0.44));
  const lastDotR  = Math.max(2, Math.round(CELL * 0.11));
  const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
  const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
  const gradInner = Math.max(1, Math.round(CELL * 0.11));
  const wrFontPx  = Math.max(9, Math.round(CELL * 0.30));
  const visFontPx = Math.max(7, Math.round(CELL * 0.21));

  // Lizzie-style candidate overlay: one colored disc per candidate move (best =
  // blue, others fade gray→green by visit share), big win% + small visit count.
  for (const o of computeCandidates()) {
    const x = MARGIN + o.c*CELL, y = MARGIN + o.r*CELL;
    const col = candColor(o.frac, o.best);
    ctx.globalAlpha = o.best ? 0.92 : (0.45 + 0.45 * o.frac);
    ctx.beginPath(); ctx.arc(x, y, candR, 0, Math.PI*2);
    ctx.fillStyle = col.fill; ctx.fill();
    ctx.globalAlpha = 1;
    if (o.best) {
      ctx.beginPath(); ctx.arc(x, y, candR, 0, Math.PI*2);
      ctx.lineWidth = 2; ctx.strokeStyle = '#1d4ed8'; ctx.stroke(); ctx.lineWidth = 1;
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = col.text;
    const hasWr = o.wr != null, hasVis = o.vf > 0;
    if (hasWr && hasVis) {
      ctx.font = `bold ${wrFontPx}px ${MONO_FONT}`;
      ctx.fillText(Math.round(o.wr * 100), x, y - wrFontPx * 0.42);
      ctx.font = `${visFontPx}px ${MONO_FONT}`;
      ctx.globalAlpha = 0.85;
      ctx.fillText(fmtVisits(o.vf), x, y + visFontPx * 0.75);
      ctx.globalAlpha = 1;
    } else {
      ctx.font = `bold ${wrFontPx}px ${MONO_FONT}`;
      ctx.fillText(hasWr ? Math.round(o.wr * 100) : fmtVisits(o.vf), x, y);
    }
  }
  // Hover highlight driven by the candidate list on the right.
  if (hoverCand && state.board[hoverCand.r] && state.board[hoverCand.r][hoverCand.c] === 0) {
    const x = MARGIN + hoverCand.c*CELL, y = MARGIN + hoverCand.r*CELL;
    ctx.beginPath(); ctx.arc(x, y, candR + 2, 0, Math.PI*2);
    ctx.lineWidth = 2.5; ctx.strokeStyle = cssVar('--accent') || '#0969da';
    ctx.stroke(); ctx.lineWidth = 1;
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
      const lmwr = state.last_move_winrate;
      const rc = state.last_move_winrate_rc;
      if (lmwr != null && rc && rc[0]===r && rc[1]===c) {
        // Win% of this move for the side that played it (mover's view), drawn
        // on the stone in a contrasting color in place of the last-move dot.
        ctx.fillStyle = (v === 1) ? '#ffffff' : '#111111';
        ctx.font = `bold ${Math.max(9, Math.round(CELL * 0.30))}px ${MONO_FONT}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(lmwr * 100), x, y);
      } else {
        ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI*2);
        ctx.fillStyle = '#ef4444'; ctx.fill();
      }
    }
  }
}


/* ---------- Win-rate stacked-area chart (over moves, Black's frame) ---------- */
function stoneCount(board) {
  let n = 0;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (board[r][c]) n++;
  return n;
}
// Side-to-move {w,d,l} → Black-frame fractions {b: black-win, d: draw, w: white-win}.
function wdlBlack(v, persp) {
  const bv = toBlack(v, persp);
  if (!bv) return null;
  const s = bv.w + bv.d + bv.l;
  if (s <= 1e-4) return null;
  return { b: bv.w / s, d: bv.d / s, w: bv.l / s };
}
// Append the current ply's root/nn WDL to the history (one point per stone count).
function recordValues(st) {
  if (!st || !st.board) return;
  const step = stoneCount(st.board);
  const persp = valuePerspective(st);
  const rv = st.root_value ? wdlBlack(st.root_value, persp) : null;
  const nv = st.nn_value   ? wdlBlack(st.nn_value,   persp) : null;
  while (valueHistory.length && valueHistory[valueHistory.length - 1].step > step) {
    valueHistory.pop();
  }
  const last = valueHistory[valueHistory.length - 1];
  if (!rv && !nv) {
    // No fresh AI values (e.g. after an undo). Carry the last eval forward so the
    // chart keeps one point per ply, matching normal play.
    if (last && step > last.step) valueHistory.push({step, root: last.root, nn: last.nn});
    return;
  }
  if (last && last.step === step) {
    if (rv) last.root = rv;
    if (nv) last.nn = nv;
  } else if (!last || step > last.step) {
    valueHistory.push({step, root: rv, nn: nv});
  }
}
// Stacked area over moves for the active tab: white (bottom) / draw / black (top),
// the two band boundaries being the "two lines" the user asked for.
function drawValueChart() {
  clearLogical(vctx);
  const W = vctx._logicalW, H = vctx._logicalH;
  const padL = 26, padR = 6, padT = 6, padB = 14;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const axis = cssVar('--border') || '#d8dee4';
  const grid = cssVar('--heat-grid') || '#e5e7eb';
  const subtle = cssVar('--fg-subtle') || '#8b949e';
  const muted = cssVar('--fg-muted') || '#59636e';
  const yOf = f => padT + (1 - f) * innerH;
  vctx.strokeStyle = grid; vctx.lineWidth = 1;
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const y = yOf(f) + 0.5;
    vctx.beginPath(); vctx.moveTo(padL, y); vctx.lineTo(W - padR, y); vctx.stroke();
  }
  vctx.fillStyle = subtle; vctx.font = `10px ${MONO_FONT}`;
  vctx.textAlign = 'right'; vctx.textBaseline = 'middle';
  for (const f of [1, 0.5, 0]) vctx.fillText(Math.round(f * 100), padL - 4, yOf(f));
  vctx.strokeStyle = axis;
  vctx.beginPath();
  vctx.moveTo(padL + 0.5, padT); vctx.lineTo(padL + 0.5, H - padB);
  vctx.lineTo(W - padR, H - padB); vctx.stroke();

  const pts = valueHistory.filter(p => p[valueTab]).map(p => ({step: p.step, ...p[valueTab]}));
  if (pts.length === 0) {
    vctx.fillStyle = subtle; vctx.font = `11px ${MONO_FONT}`;
    vctx.textAlign = 'center'; vctx.textBaseline = 'middle';
    vctx.fillText('no data', padL + innerW / 2, padT + innerH / 2);
    return;
  }
  const n = pts.length;
  // x = data-point index, so the first eval sits at the left edge. (The stone
  // count at first eval is ≥1, so mapping by absolute step starts mid-chart.)
  const xAt = i => n <= 1 ? padL : padL + (i / (n - 1)) * innerW;
  vctx.fillStyle = muted; vctx.textAlign = 'center'; vctx.textBaseline = 'top';
  vctx.fillText(String(pts[0].step), xAt(0), H - padB + 2);
  if (n > 1) vctx.fillText(String(pts[n - 1].step), xAt(n - 1), H - padB + 2);

  // Fill one stacked band between two cumulative-fraction accessors.
  function band(lowerFn, upperFn, color) {
    vctx.fillStyle = color;
    vctx.beginPath();
    pts.forEach((p, i) => { const x = xAt(i), y = yOf(upperFn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
    for (let i = n - 1; i >= 0; i--) vctx.lineTo(xAt(i), yOf(lowerFn(pts[i])));
    vctx.closePath(); vctx.fill();
  }
  const colBlk = cssVar('--stone-black-1') || '#000';
  const colWht = cssVar('--stone-white-0') || '#fff';
  const colDrw = cssVar('--fg-subtle') || '#8b949e';
  // Faint fills hint at the three bands; the boundary lines carry the structure.
  vctx.globalAlpha = 0.18;
  band(p => 0,             p => p.w,       colWht);  // white win — bottom
  band(p => p.w,           p => p.w + p.d, colDrw);  // draw — middle
  band(p => p.w + p.d,     p => 1,         colBlk);  // black win — top
  vctx.globalAlpha = 1;
  // The two band boundaries (white/draw and draw/black), drawn over the fills.
  function line(fn) {
    vctx.beginPath();
    pts.forEach((p, i) => { const x = xAt(i), y = yOf(fn(p)); i === 0 ? vctx.moveTo(x, y) : vctx.lineTo(x, y); });
    vctx.stroke();
  }
  vctx.strokeStyle = muted; vctx.lineWidth = 1.5;
  line(p => p.w);
  line(p => p.w + p.d);
}

/* ---------- Win-rate legend (current ply, Black's frame) ---------- */
// Renormalizes raw W/D/L so the legend numbers sum to ~100%.
function normalizeVal(v) {
  if (!v) return null;
  const s = v.w + v.d + v.l;
  if (s > 1e-4) return { w: v.w/s*100, d: v.d/s*100, l: v.l/s*100 };
  return { w: v.w, d: v.d, l: v.l };
}
// v is in Black's frame: w = Black win, d = draw, l = White win.
function renderWinLegend(v) {
  const bEl = document.getElementById('vc_w_black');
  const dEl = document.getElementById('vc_w_draw');
  const wEl = document.getElementById('vc_w_white');
  const n = normalizeVal(v);
  if (!n) {
    bEl.textContent = dEl.textContent = wEl.textContent = '—';
    return;
  }
  bEl.textContent = n.w.toFixed(1) + '%';
  dEl.textContent = n.d.toFixed(1) + '%';
  wEl.textContent = n.l.toFixed(1) + '%';
}

// Switch the chart between the root and nn evaluations.
function setValueTab(tab) {
  if (tab !== 'root' && tab !== 'nn') return;
  valueTab = tab;
  document.getElementById('vc_tab_root').setAttribute('aria-pressed', tab === 'root' ? 'true' : 'false');
  document.getElementById('vc_tab_nn').setAttribute('aria-pressed', tab === 'nn' ? 'true' : 'false');
  renderValuePanel();
}

// Repaint the active tab's legend (current ply) + stacked-area history.
function renderValuePanel() {
  const persp = valuePerspective(state);
  const raw = state ? (valueTab === 'root' ? state.root_value : state.nn_value) : null;
  renderWinLegend(toBlack(raw, persp));
  drawValueChart();
}

/* ---------- Status pill ---------- */
function statusVariant(s) {
  if (!s) return 'idle';
  const t = s.toLowerCase();
  if (t.includes('analyz')) return 'thinking';
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
    const analysisMode = !!state.analysis_mode;
    if (!modeSynced) {
      currentMode = analysisMode ? 'analysis' : 'play';
      updateModeButtons();
      modeSynced = true;
    }
    const analyzing = !!state.analyzing;
    let statusText = state.status || 'idle';
    // Surface how deep the ponder is so analysis mode isn't a black box.
    if (analyzing && Number.isFinite(state.analyze_sims)) {
      statusText = '分析中 · 模拟 ' + state.analyze_sims.toLocaleString();
    }
    document.getElementById('status').textContent = statusText;
    document.getElementById('status_pill').dataset.variant = statusVariant(statusText);

    // Play-mode ponder freezes the board; the analysis board stays interactive
    // (a click places the next stone and restarts the ponder).
    const frozen = analyzing && !analysisMode;
    const canAnalyze = !!state.awaiting_human && !analyzing && !state.game_over;
    const setDisabled = (id, v) => { const el = document.getElementById(id); if (el) el.disabled = v; };
    const analyzeBtn = document.getElementById('analyze_btn');
    analyzeBtn.textContent = analyzing ? 'Stop' : 'Analyze';
    analyzeBtn.disabled = analyzing ? false : !canAnalyze;
    setDisabled('newgame_btn', frozen);
    setDisabled('undo_btn', frozen);

    if (!sideSynced && (state.human_side === 1 || state.human_side === -1)) {
      selectedSide = state.human_side;
      updateSideButtons();
      sideSynced = true;
    }
    recordValues(state);
    renderValuePanel();

    draw();
    renderCandidates();
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
  // Analysis board: a click places the next stone even while pondering.
  if (state.analyzing && !state.analysis_mode) return;
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
function startAnalyze() {
  if (!state || state.game_over || state.analyzing || !state.awaiting_human) return;
  sendCmd('analyze');
}
function stopAnalyze() {
  if (!state || !state.analyzing) return;
  sendCmd('stop');
}
// The single Analyze button doubles as Stop while a ponder is running.
function toggleAnalyze() {
  if (state && state.analyzing) stopAnalyze();
  else startAnalyze();
}
function updateSimsModeHint() {
  const el = document.getElementById('sims_input');
  const hint = document.getElementById('sims_mode_hint');
  if (!el || !hint) return;
  const n = parseInt(el.value, 10);
  hint.hidden = !(Number.isFinite(n) && n === 0);
}
function applySims() {
  const el = document.getElementById('sims_input');
  const n = parseInt(el.value, 10);
  if (Number.isFinite(n) && n >= 0) sendCmd('sims ' + n);
  updateSimsModeHint();
}
let currentMode = 'play';
let modeSynced = false;
function updateModeButtons() {
  document.getElementById('mode_play').setAttribute('aria-pressed', currentMode === 'play' ? 'true' : 'false');
  document.getElementById('mode_analysis').setAttribute('aria-pressed', currentMode === 'analysis' ? 'true' : 'false');
  const inAnalysis = currentMode === 'analysis';
  // Free-analysis board drives both colors and ponders automatically, so the
  // human-side picker and the manual Analyze/Stop buttons are meaningless.
  document.getElementById('side_row').classList.toggle('hidden', inAnalysis);
  document.getElementById('analyze_btn').classList.toggle('hidden', inAnalysis);
}
function setMode(m) {
  if (m !== 'play' && m !== 'analysis') return;
  if (currentMode === m) return;
  currentMode = m;
  updateModeButtons();
  sendCmd('mode ' + m);
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
  // Black-framed values don't depend on which color the human controls, so a
  // side swap needs no history rewrite — just tell the engine.
  sendCmd('side ' + side);
}
async function newGame(side, modelId, boardSize, rule) {
  if (side === undefined) side = selectedSide;
  else { selectedSide = side; updateSideButtons(); }
  valueHistory = [];
  renderWinLegend(null);
  drawValueChart();
  const payload = {human_side: side};
  if (modelId) payload.model = modelId;
  if (Number.isFinite(boardSize) && boardSize > 0) payload.board_size = boardSize;
  if (typeof rule === 'string' && rule.length > 0) payload.rule = rule;
  await fetch('/new', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(payload)});
  sendCmd('noise 0');
  const sims = parseInt(document.getElementById('sims_input').value, 10);
  if (Number.isFinite(sims) && sims >= 0) sendCmd('sims ' + sims);
  // A model/board/rule change spawns a fresh engine that defaults to play mode;
  // re-assert the client's mode so the analysis board survives a New game.
  sendCmd('mode ' + currentMode);
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
document.getElementById('sims_input').addEventListener('input', updateSimsModeHint);
updateSimsModeHint();
sendCmd('noise 0'); // ensure engine starts with Gumbel noise off

draw();
renderWinLegend(null);
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
                                      "nn_futurepos_32": None, "gumbel_phases": None,
                                      "gumbel_winrate": None, "analyzing": False,
                                      "analyze_sims": None, "analysis_mode": False,
                                      "value_persp": None,
                                      "awaiting_human": False, "last_move_winrate": None,
                                      "last_move_winrate_rc": None})
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


def _read_cfg_map(path):
    """Parse path then path+'.local' (later wins) → dict[str, str].

    Mirrors scripts/run.sh's run.cfg.local handling: base values first, then
    server-local overrides. Values are stripped of inline #comments and
    surrounding quotes; CSV-style values are returned verbatim for callers
    to split.
    """
    out = {}
    for p in (str(path), str(path) + ".local"):
        try:
            text = Path(p).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            s = line.split("#", 1)[0].strip()
            if "=" not in s:
                continue
            k, _, v = s.partition("=")
            v = v.strip().strip('"').strip("'")
            out[k.strip()] = v
    return out


def parse_board_sizes(run_cfg_path):
    """Read BOARD_SIZES="17, 16, ..." from run.cfg → sorted list[int] (desc)."""
    v = _read_cfg_map(run_cfg_path).get("BOARD_SIZES", "")
    try:
        sizes = [int(x.strip()) for x in v.split(",") if x.strip()]
    except ValueError:
        return []
    return sorted(set(sizes), reverse=True)


SUPPORTED_RULES = ("renju", "standard", "freestyle")


def parse_rules(run_cfg_path):
    """Read RULES="renju, standard, ..." from run.cfg → list[str] preserving cfg order."""
    v = _read_cfg_map(run_cfg_path).get("RULES", "")
    seen = []
    for tok in v.split(","):
        r = tok.strip().lower()
        if r in SUPPORTED_RULES and r not in seen:
            seen.append(r)
    return seen


def read_cfg_int(path, key, default):
    v = _read_cfg_map(path).get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def read_cfg_str(path, key, default):
    return _read_cfg_map(path).get(key, default)


def discover_models(root_dir, data_dir):
    """Scan <data>/models, <data>/nets/<arch>/, and <root>/anchors for TorchScript .pt files.

    gomoku_play loads via torch::jit::load, so only TorchScript files work:
    latest.pt and scripted_iter_*.pt (per export_model.py). State-dict files
    (model_latest.pt, model_iter_*.pt) are excluded.
    """
    out = {}
    models_dir = data_dir / "models"
    if models_dir.is_dir():
        files = sorted(models_dir.glob("*.pt"), key=lambda p: p.name, reverse=True)
        files.sort(key=lambda p: 0 if p.name == "latest.pt" else 1)
        for p in files:
            out[f"models/{p.name}"] = p.resolve()
    nets_dir = data_dir / "nets"
    if nets_dir.is_dir():
        def _net_key(p):
            # bNcM convention: sort small→large by (blocks, channels). Non-matching
            # names fall back to alphabetical, after the bNcM ones.
            m = re.match(r"b(\d+)c(\d+)", p.name)
            return (0, int(m.group(1)), int(m.group(2))) if m else (1, 0, 0, p.name)
        for net in sorted((p for p in nets_dir.iterdir() if p.is_dir()), key=_net_key):
            cands = [p for p in net.glob("*.pt")
                     if p.name == "latest.pt" or p.name.startswith("scripted_")]
            cands.sort(key=lambda p: (0 if p.name == "latest.pt" else 1, p.name), reverse=False)
            # latest.pt first, then scripted_iter_NNNNNN.pt newest-first.
            scripted = sorted([p for p in cands if p.name.startswith("scripted_")],
                              key=lambda p: p.name, reverse=True)
            ordered = [p for p in cands if p.name == "latest.pt"] + scripted
            for p in ordered:
                out[f"{net.name}/{p.name}"] = p.resolve()
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
    ap.add_argument("--config", default=str(root_dir / "configs" / "baseline" / "play.cfg"))
    ap.add_argument("--run-config", default=str(root_dir / "configs" / "baseline" / "run.cfg"))
    ap.add_argument("--data-dir", default=str(root_dir / "data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--human-side", type=int, default=1, choices=(1, -1))
    args = ap.parse_args()

    for p, name in [(args.model, "model"), (args.bin, "binary"), (args.config, "config")]:
        if not Path(p).exists():
            raise SystemExit(f"{name} not found: {p}")

    models = discover_models(root_dir, Path(args.data_dir))
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
    max_board_size = read_cfg_int(args.run_config, "MAX_BOARD_SIZE", 0)
    if max_board_size > 0:
        board_sizes = [s for s in board_sizes if s <= max_board_size]
    default_board_size = read_cfg_int(args.run_config, "MAIN_BOARD_SIZE", BOARD_SIZE)
    if default_board_size not in board_sizes:
        board_sizes = [default_board_size] + board_sizes

    rules = parse_rules(args.run_config) or list(SUPPORTED_RULES)
    default_rule = read_cfg_str(args.run_config, "MAIN_RULE", rules[0]).lower()
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
