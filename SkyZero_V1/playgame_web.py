#!/usr/bin/env python3
"""HTTP front-end for human-vs-AlphaZero play (V1, pure-Python MCTS).

Single-user, in-process. Loads the latest checkpoint for the requested game
and serves a page that lets a human play against MCTS (PUCT or Gumbel). Two
heatmaps are shown per AI move: NN policy and MCTS policy (visit-count
distribution under PUCT, improved policy under Gumbel). A WDL trend chart
tracks AlphaZero's evaluation at the root over the course of the game.

Usually invoked via the per-game entry points
(`tictactoe/tictactoe_play.py`, `gomoku/gomoku_play.py`), which call
``run_server(...)`` directly.
"""
import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.optim as optim

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from skyzero import AlphaZero
from nets import ResNet


# ---------------------------------------------------------------------------
# Game wiring
# ---------------------------------------------------------------------------

GAME_TITLES = {
    "gomoku": "Gomoku",
    "tictactoe": "Tic-Tac-Toe",
}


def _load_game(name, train_args_override=None):
    if name == "gomoku":
        from envs.gomoku import Gomoku
        if train_args_override is None:
            from gomoku.gomoku_train import train_args as ta
        else:
            ta = train_args_override
        game = Gomoku(board_size=ta["board_size"], rule=ta.get("rule", "freestyle"))
        meta = {"rule": ta.get("rule", "freestyle")}
    elif name == "tictactoe":
        from envs.tictactoe import TicTacToe
        if train_args_override is None:
            from tictactoe.tictactoe_train import train_args as ta
        else:
            ta = train_args_override
        game = TicTacToe()
        meta = {"rule": None}
    else:
        raise ValueError(f"unknown game {name!r}")
    return game, ta, meta


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class GameSession:
    """Owns the AlphaZero instance and one in-progress game."""

    def __init__(self, game, alphazero, args, meta, game_name):
        self.game = game
        self.alphazero = alphazero
        self.args = dict(args)
        self.meta = meta
        self.game_name = game_name
        self.board_size = game.board_size

        self.lock = threading.Lock()
        self._reset_internal(
            human_side=1,
            algo=self.args.get("algo", "puct"),
            num_simulations=self.args["num_simulations"],
        )

    # ----- mutators (call with self.lock held) -----

    def _reset_internal(self, human_side, algo, num_simulations):
        self.state = self.game.get_initial_state()
        self.to_play = 1
        self.human_side = int(human_side)
        self.algo = algo
        self.num_simulations = int(num_simulations)
        self.last_move = None
        self.last_ai_move = None
        self.last_nn_policy = None
        self.last_mcts_policy = None
        self.last_nn_value = None
        self.last_root_wdl = None
        self.last_nn_value_black = None
        self.last_root_wdl_black = None
        self.value_history = []
        self._move_history = []
        self.game_over = False
        self.winner = None
        self.status = "Your turn." if self.human_side == 1 else "AlphaZero is thinking..."

    @staticmethod
    def _wdl_to_black(wdl, to_play):
        if to_play == 1:
            return [float(wdl[0]), float(wdl[1]), float(wdl[2])]
        return [float(wdl[2]), float(wdl[1]), float(wdl[0])]

    def _run_ai_move(self):
        if self.algo == "puct":
            mcts_policy, info = self.alphazero.mcts.search(
                self.state, self.to_play, self.num_simulations, add_noise=False
            )
            action = int(np.argmax(mcts_policy))
        elif self.algo == "gumbel":
            mcts_policy, action, _v_mix, info = self.alphazero.mcts.gumbel_sequential_halving(
                self.state, self.to_play, self.num_simulations
            )
            action = int(action)
        else:
            raise ValueError(f"unknown algo {self.algo!r}")

        side_at_search = self.to_play
        self.last_nn_policy = np.asarray(info["nn_policy"], dtype=np.float32)
        self.last_mcts_policy = np.asarray(mcts_policy, dtype=np.float32)
        self.last_nn_value = np.asarray(info["nn_value"], dtype=np.float32)
        self.last_root_wdl = np.asarray(info["root_wdl"], dtype=np.float32)
        self.last_root_wdl_black = self._wdl_to_black(self.last_root_wdl, side_at_search)
        self.last_nn_value_black = self._wdl_to_black(self.last_nn_value, side_at_search)

        ply = int(np.count_nonzero(self.state))
        self.value_history.append({
            "ply": ply,
            "to_play": int(side_at_search),
            "wdl_black": list(self.last_root_wdl_black),
            "nn_wdl_black": list(self.last_nn_value_black),
        })

        r, c = action // self.board_size, action % self.board_size
        self.last_ai_move = (int(r), int(c))
        self._apply_action(action)

    def _apply_action(self, action):
        # Save pre-move snapshot for undo
        self._move_history.append({
            'state': self.state.copy(),
            'to_play': self.to_play,
            'last_move': self.last_move,
            'last_ai_move': self.last_ai_move,
            'last_nn_policy': self.last_nn_policy,
            'last_mcts_policy': self.last_mcts_policy,
            'last_nn_value': self.last_nn_value,
            'last_root_wdl': self.last_root_wdl,
            'last_nn_value_black': self.last_nn_value_black,
            'last_root_wdl_black': self.last_root_wdl_black,
            'game_over': self.game_over,
            'winner': self.winner,
            'status': self.status,
            'value_history': list(self.value_history),
        })
        self.state = self.game.get_next_state(self.state, action, self.to_play)
        r, c = action // self.board_size, action % self.board_size
        self.last_move = (int(r), int(c))
        if self.game.is_terminal(self.state):
            self.game_over = True
            w = int(self.game.get_winner(self.state))
            self.winner = w
            if w == 1:
                self.status = "Black wins."
            elif w == -1:
                self.status = "White wins."
            else:
                self.status = "Draw."
            return
        self.to_play = -self.to_play
        self.status = "Your turn." if self.to_play == self.human_side else "AlphaZero is thinking..."

    # ----- public API -----

    def undo_move(self):
        with self.lock:
            if not self._move_history:
                return {'ok': False, 'err': 'no moves to undo'}
            while self._move_history:
                entry = self._move_history.pop()
                self.state = entry['state']
                self.to_play = entry['to_play']
                self.last_move = entry['last_move']
                self.last_ai_move = entry['last_ai_move']
                self.last_nn_policy = entry['last_nn_policy']
                self.last_mcts_policy = entry['last_mcts_policy']
                self.last_nn_value = entry['last_nn_value']
                self.last_root_wdl = entry['last_root_wdl']
                self.last_nn_value_black = entry['last_nn_value_black']
                self.last_root_wdl_black = entry['last_root_wdl_black']
                self.game_over = entry['game_over']
                self.winner = entry['winner']
                self.status = entry['status']
                self.value_history = entry['value_history']
                if self.to_play == self.human_side and not self.game_over:
                    break
            self.last_nn_policy = None
            self.last_mcts_policy = None
            self.last_nn_value = None
            self.last_root_wdl = None
            self.last_nn_value_black = None
            self.last_root_wdl_black = None
            self.status = 'Your turn (undid last move).'
            return {'ok': True}

    def new_game(self, human_side, algo, num_simulations):
        with self.lock:
            self._reset_internal(human_side, algo, num_simulations)
            if not self.game_over and self.to_play != self.human_side:
                self._run_ai_move()

    def human_move(self, r, c):
        with self.lock:
            if self.game_over:
                return {"ok": False, "err": "game already over"}
            if self.to_play != self.human_side:
                return {"ok": False, "err": "not your turn"}
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                return {"ok": False, "err": "out of bounds"}
            action = r * self.board_size + c
            is_legal = self.game.get_is_legal_actions(self.state, self.to_play)
            if not is_legal[action]:
                return {"ok": False, "err": "illegal move"}
            self._apply_action(action)
            if not self.game_over:
                self._run_ai_move()
            return {"ok": True}

    def snapshot(self):
        with self.lock:
            return {
                "game": self.game_name,
                "game_title": GAME_TITLES.get(self.game_name, self.game_name),
                "board_size": self.board_size,
                "board": self.state.astype(int).tolist(),
                "to_play": int(self.to_play),
                "human_side": int(self.human_side),
                "algo": self.algo,
                "num_simulations": int(self.num_simulations),
                "last_move": list(self.last_move) if self.last_move else None,
                "last_ai_move": list(self.last_ai_move) if self.last_ai_move else None,
                "game_over": bool(self.game_over),
                "winner": (None if self.winner is None else int(self.winner)),
                "status": self.status,
                "rule": self.meta.get("rule"),
                "nn_policy": (None if self.last_nn_policy is None else
                              self.last_nn_policy.reshape(self.board_size, self.board_size).tolist()),
                "mcts_policy": (None if self.last_mcts_policy is None else
                                self.last_mcts_policy.reshape(self.board_size, self.board_size).tolist()),
                "nn_value_black": (None if self.last_nn_value_black is None
                                   else list(self.last_nn_value_black)),
                "root_wdl_black": (None if self.last_root_wdl_black is None
                                   else list(self.last_root_wdl_black)),
                "value_history": list(self.value_history),
            }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    session = None

    def log_message(self, fmt, *args):
        return

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            return json.loads(raw.decode("utf-8")) if raw else {}
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
            self._send_json(200, Handler.session.snapshot())
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        body = self._read_json()
        try:
            if self.path == "/new":
                sess = Handler.session
                side = int(body.get("human_side", sess.human_side))
                if side not in (1, -1):
                    side = 1
                algo = body.get("algo", sess.algo)
                if algo not in ("puct", "gumbel"):
                    algo = "puct"
                sims = body.get("num_simulations", sess.num_simulations)
                try:
                    sims = max(1, int(sims))
                except (TypeError, ValueError):
                    sims = sess.num_simulations
                sess.new_game(side, algo, sims)
                self._send_json(200, sess.snapshot())
                return
            if self.path == "/move":
                r = int(body.get("r"))
                c = int(body.get("c"))
                result = Handler.session.human_move(r, c)
                if not result["ok"]:
                    self._send_json(409, {"ok": False, "err": result.get("err"),
                                          "state": Handler.session.snapshot()})
                    return
                self._send_json(200, Handler.session.snapshot())
                return
            if self.path == '/undo':
                result = Handler.session.undo_move()
                if not result['ok']:
                    self._send_json(409, {'ok': False, 'err': result.get('err'),
                                          'state': Handler.session.snapshot()})
                    return
                self._send_json(200, Handler.session.snapshot())
                return
        except Exception:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "err": "server error"})
            return
        self.send_response(404); self.end_headers()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkyZero V1</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script>
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
    margin: 0; background: var(--bg); color: var(--fg);
  }
  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
  }
  .app { max-width: 1800px; margin: 0 auto; padding: 16px 24px 48px; }

  /* topbar */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .brand { display: flex; flex-direction: column; gap: 2px; }
  .brand-title { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
  .brand-sub { font-size: 12px; color: var(--fg-muted); }
  .icon-btn {
    width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center;
    background: transparent; border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--fg-muted); cursor: pointer;
    transition: background .12s, color .12s, border-color .12s;
  }
  .icon-btn:hover { background: var(--surface-2); color: var(--fg); border-color: var(--border-strong); }
  .icon-btn svg { width: 16px; height: 16px; display: block; }
  .icon-btn .sun-icon, .icon-btn .moon-icon, .icon-btn .auto-icon { display: none; }
  html[data-theme-mode="light"] .icon-btn .sun-icon,
  html[data-theme-mode="dark"]  .icon-btn .moon-icon,
  html[data-theme-mode="auto"]  .icon-btn .auto-icon { display: inline-block; }

  /* layout */
  .main {
    display: grid;
    grid-template-columns: 260px minmax(0, auto) minmax(0, 1fr);
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 1199px) { .main { grid-template-columns: 1fr; } }
  .board-col { display: flex; flex-direction: column; align-items: center; gap: 12px; min-width: 0; }
  .side-col { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .seg-row { display: flex; gap: 8px; }
  .side-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .seg-btn {
    flex: 1; height: 34px; font-size: 13px; font-weight: 500;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    transition: background .12s, border-color .12s, color .12s;
  }
  .seg-btn:hover { background: var(--surface-2); border-color: var(--border-strong); }
  .seg-btn[aria-pressed="true"] {
    background: var(--accent); color: var(--accent-fg); border-color: var(--accent);
  }
  .seg-stone {
    width: 14px; height: 14px; border-radius: 50%;
    border: 1px solid var(--stone-outline); flex-shrink: 0;
  }
  .seg-stone.black { background: radial-gradient(circle at 30% 30%, var(--stone-black-0), var(--stone-black-1)); }
  .seg-stone.white { background: radial-gradient(circle at 30% 30%, var(--stone-white-0), var(--stone-white-1)); }
  .board-actions {
    display: flex; gap: 8px; width: 100%; max-width: 560px; justify-content: center;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xs);
  }
  .card-body { padding: 14px 16px; }
  .card-title {
    font-size: 11px; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: var(--fg-subtle);
    margin: 0 0 10px;
  }

  /* status */
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
  .status-pill[data-variant="active"]   { background: var(--success-bg); color: var(--success); border-color: transparent; }
  .status-pill[data-variant="active"]   .dot { background: var(--success); }
  .status-pill[data-variant="thinking"] { background: var(--info-bg); color: var(--info); border-color: transparent; }
  .status-pill[data-variant="thinking"] .dot { background: var(--info); animation: pulse 1.2s ease-in-out infinite; }
  .status-pill[data-variant="done"]     { background: var(--done-bg); color: var(--done); border-color: transparent; }
  .status-pill[data-variant="done"]     .dot { background: var(--done); }
  .status-pill[data-variant="error"]    { background: var(--danger-bg); color: var(--danger); border-color: transparent; }
  .status-pill[data-variant="error"]    .dot { background: var(--danger); }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.35); opacity: 0.55; }
  }
  @media (prefers-reduced-motion: reduce) { .status-pill .dot { animation: none !important; } }

  /* buttons */
  .btn {
    height: 30px; padding: 0 12px; font-size: 13px; font-weight: 500;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    cursor: pointer; white-space: nowrap;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    transition: background .12s, border-color .12s, color .12s;
  }
  .btn:hover { background: var(--surface-2); border-color: var(--border-strong); }
  .btn.primary { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
  .btn.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  .btn.undo-btn { color: var(--fg-muted); }
  .btn.undo-btn:hover:not(:disabled) { color: var(--danger); border-color: var(--danger); }
  .btn.undo-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* field-row + number input */
  .field-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 6px 0;
  }
  .field-row label { font-size: 13px; color: var(--fg-muted); font-family: var(--font-mono); }
  .num {
    width: 92px; height: 28px; padding: 0 8px; font-size: 13px;
    background: var(--surface); color: var(--fg);
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    text-align: right;
    transition: border-color .12s, box-shadow .12s;
  }
  .num:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
  }
  .meta-line {
    font-size: 11.5px; color: var(--fg-muted); font-family: var(--font-mono);
    margin-top: 8px; line-height: 1.6;
  }
  .meta-line .k { color: var(--fg-subtle); }

  /* WDL */
  .wdl-row {
    display: grid;
    grid-template-columns: 42px minmax(0,1fr) 64px;
    align-items: center; gap: 10px; padding: 6px 0;
  }
  .wdl-label {
    font-family: var(--font-mono); font-size: 11.5px; color: var(--fg-muted);
    text-transform: uppercase; letter-spacing: .04em;
  }
  .wdl-bar {
    display: flex; height: 8px; width: 100%;
    background: var(--surface-2);
    border-radius: 999px; overflow: hidden;
  }
  .wdl-bar .seg { height: 100%; transition: width .2s ease-out; }
  .wdl-bar .seg.w { background: var(--success); }
  .wdl-bar .seg.d { background: var(--fg-subtle); opacity: 0.45; }
  .wdl-bar .seg.l { background: var(--danger); }
  .wdl-wl {
    font-family: var(--font-mono); font-size: 12px; text-align: right; color: var(--fg);
    font-variant-numeric: tabular-nums;
  }
  .wdl-wl.pos { color: var(--success); }
  .wdl-wl.neg { color: var(--danger); }
  .wdl-detail {
    margin-top: 6px; font-size: 11.5px; color: var(--fg-muted);
    font-family: var(--font-mono);
    display: flex; gap: 10px; min-height: 18px;
  }
  .wdl-detail .k { color: var(--fg-subtle); }
  .value-chart-wrap {
    margin-top: 12px; padding-top: 12px;
    border-top: 1px solid var(--border);
    display: flex; flex-direction: column; min-height: 160px;
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

  /* board */
  .board-card {
    padding: 16px;
    display: inline-flex; flex-direction: column; align-items: center;
  }
  #board {
    background: var(--board-bg); border-radius: var(--radius-sm);
    display: block; cursor: crosshair;
  }

  /* heatmaps */
  #right_col { min-width: 0; }
  .grids {
    display: grid; grid-template-columns: minmax(0, 1fr);
    gap: 12px;
    max-width: 420px;
  }
  .grid-card {
    padding: 12px; text-align: center;
    display: flex; flex-direction: column;
  }
  .grid-card .grid-title {
    font-size: 12px; font-weight: 600; letter-spacing: .02em;
    color: var(--fg-subtle);
    margin-bottom: 12px; text-align: left;
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
  }
  .heat {
    background: var(--heat-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    display: block; margin: auto;
  }
  .expand-btn {
    background: transparent; border: none; cursor: pointer;
    color: var(--fg-subtle); padding: 0;
    width: 18px; height: 18px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 4px;
    transition: background .12s, color .12s;
  }
  .expand-btn:hover { background: var(--surface-2); color: var(--fg); }
  .expand-btn svg { width: 12px; height: 12px; display: block; }

  /* modal */
  .heat-modal {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.55);
    display: flex; align-items: center; justify-content: center;
    padding: 24px;
  }
  .heat-modal-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: 0 16px 48px rgba(0,0,0,0.35);
    padding: 16px 20px 20px;
    max-width: 95vw; max-height: 95vh;
    display: flex; flex-direction: column; gap: 12px;
  }
  .heat-modal-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px;
  }
  .heat-modal-title { font-size: 14px; font-weight: 600; color: var(--fg); }
  .heat-modal-close {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--fg-muted); cursor: pointer;
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .heat-modal-close:hover { background: var(--surface-2); color: var(--fg); border-color: var(--border-strong); }
  .heat-modal-close svg { width: 14px; height: 14px; }
  #heat_modal_canvas {
    background: var(--heat-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    display: block; margin: auto;
  }
  .hidden { display: none !important; }
</style></head>
<body>
<div class="app">

  <header class="topbar">
    <div class="brand">
      <div class="brand-title" id="brand_title">SkyZero V1</div>
      <div class="brand-sub">Human vs AlphaZero · local inspector</div>
    </div>
    <button class="icon-btn" id="theme_toggle" aria-label="Toggle color theme" title="Toggle theme">
      <svg class="sun-icon"  viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
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
        <div class="card-body side-row">
          <div class="card-title" style="margin:0;">Human side</div>
          <div class="seg-row" style="flex: 1; max-width: 180px;">
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
        <div class="card-body">
          <div class="card-title">Algorithm</div>
          <div class="seg-row">
            <button class="seg-btn" id="algo_puct"   aria-pressed="true"  onclick="setAlgo('puct')">PUCT</button>
            <button class="seg-btn" id="algo_gumbel" aria-pressed="false" onclick="setAlgo('gumbel')">Gumbel</button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Search</div>
          <div class="field-row">
            <label for="sims_input">sims</label>
            <input class="num" type="number" id="sims_input" min="1" step="1" value="400">
          </div>
          <div class="meta-line" id="meta_line"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-body">
          <div class="card-title">Value estimates (Black perspective)</div>
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
              <span class="vc-axis">B−W · −1…+1</span>
            </div>
            <canvas id="value_chart"></canvas>
          </div>
        </div>
      </div>
    </aside>

    <section class="board-col">
      <div class="card board-card">
        <canvas id="board"></canvas>
      </div>
      <div class="board-actions">
        <button class="btn primary" onclick="newGame()">New game</button>
        <button class="btn undo-btn" id="undo_btn" onclick="undoMove()" disabled>Undo</button>
      </div>
    </section>

    <aside class="side-col" id="right_col">
      <div class="grids">
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text">NN Policy</span>
            <button class="expand-btn" data-target="h_nn" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_nn"></canvas>
        </div>
        <div class="card grid-card">
          <div class="grid-title">
            <span class="grid-title-text" id="title_mcts">MCTS Policy</span>
            <button class="expand-btn" data-target="h_mcts" aria-label="Expand" title="Expand">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 6V2.5h3.5M13.5 6V2.5H10M2.5 10v3.5h3.5M13.5 10v3.5H10"/>
              </svg>
            </button>
          </div>
          <canvas class="heat" id="h_mcts"></canvas>
        </div>
      </div>
    </aside>
  </div>

  <div class="heat-modal hidden" id="heat_modal" role="dialog" aria-modal="true">
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
const MONO_FONT = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace';
const DPR = window.devicePixelRatio || 1;

let N = 15;
let MARGIN = 28;
let CELL = 36;
let BOARD_LOGICAL = MARGIN*2 + CELL*(N-1);
let HEAT_LOGICAL = 280;

let state = null;
let busy = false;
let selectedSide = 1;
let selectedAlgo = 'puct';

function $(id) { return document.getElementById(id); }
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

const cv = $('board');
let ctx = setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);

const heatCanvases = {
  h_nn: $('h_nn'),
  h_mcts: $('h_mcts'),
};
const heatCtxs = {
  h_nn: setupCanvas(heatCanvases.h_nn, HEAT_LOGICAL, HEAT_LOGICAL, false),
  h_mcts: setupCanvas(heatCanvases.h_mcts, HEAT_LOGICAL, HEAT_LOGICAL, false),
};
const HEAT_KEYS = { h_nn: 'nn_policy', h_mcts: 'mcts_policy' };

const vcCanvas = $('value_chart');
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

function computeBoardLayout() {
  // Pick a comfortable board size: bigger for smaller N (tic-tac-toe), capped for gomoku.
  const targetTotal = 560;
  const wantCell = Math.max(40, Math.min(120, Math.floor((targetTotal - 2*28) / Math.max(1, N - 1))));
  CELL = N <= 5 ? 90 : wantCell;
  MARGIN = N <= 5 ? 32 : 28;
  BOARD_LOGICAL = MARGIN*2 + CELL*(N-1);
  setupCanvas(cv, BOARD_LOGICAL, BOARD_LOGICAL);
}

function fitHeatCanvas(canvasId) {
  const c = heatCanvases[canvasId];
  const card = c.parentElement;
  const cardCS = getComputedStyle(card);
  const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const availW = card.clientWidth - padX;
  const size = Math.max(160, Math.min(BOARD_LOGICAL, Math.min(380, Math.floor(availW))));
  c.style.width = size + 'px';
  c.style.height = size + 'px';
  if (heatCtxs[canvasId]._logicalW === size) return false;
  heatCtxs[canvasId] = setupCanvas(c, size, size, false);
  return true;
}
for (const id of Object.keys(heatCanvases)) {
  new ResizeObserver(() => {
    if (!fitHeatCanvas(id)) return;
    drawHeat(id, state ? state[HEAT_KEYS[id]] : null);
  }).observe(heatCanvases[id].parentElement);
  fitHeatCanvas(id);
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
  for (let i = 0; i < N; i++) {
    const p = Math.round(MARGIN + i*CELL) + 0.5;
    ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + CELL*(N-1), p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + CELL*(N-1)); ctx.stroke();
  }
  ctx.fillStyle = boardStar;
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
  for (let i = 0; i < N; i++) {
    ctx.fillText(String(i), MARGIN + i*CELL, 12);
    ctx.fillText(String(i), 10, MARGIN + i*CELL);
  }
  if (!state) return;

  const stoneR    = Math.max(8, Math.round(CELL * (N <= 5 ? 0.35 : 0.42)));
  const lastDotR  = Math.max(3, Math.round(CELL * 0.11));
  const shadowDx  = Math.max(0, Math.round(CELL * 0.015));
  const shadowDy  = Math.max(1, Math.round(CELL * 0.045));
  const gradInner = Math.max(1, Math.round(CELL * 0.11));

  const b = state.board, lm = state.last_move;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const v = b[r][c]; if (!v) continue;
    const x = MARGIN + c*CELL, y = MARGIN + r*CELL;
    ctx.beginPath(); ctx.arc(x+shadowDx, y+shadowDy, stoneR, 0, Math.PI*2);
    ctx.fillStyle = stoneShadow; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, stoneR, 0, Math.PI*2);
    const grad = ctx.createRadialGradient(x-gradInner, y-gradInner, 2, x, y, stoneR);
    if (v === 1) { grad.addColorStop(0, stoneB0); grad.addColorStop(1, stoneB1); }
    else { grad.addColorStop(0, stoneW0); grad.addColorStop(1, stoneW1); }
    ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = stoneOutline; ctx.lineWidth = 1; ctx.stroke();
    if (lm && lm[0] === r && lm[1] === c) {
      ctx.beginPath(); ctx.arc(x, y, lastDotR, 0, Math.PI*2);
      ctx.fillStyle = '#ef4444'; ctx.fill();
    }
  }
}

/* ---------- Heatmaps ---------- */
function drawHeat(canvasId, grid) {
  const g = heatCtxs[canvasId];
  clearLogical(g);
  const W = g._logicalW;
  const cell = W / N;
  const gridCol = cssVar('--heat-grid') || '#e5e7eb';
  let maxV = 0;
  if (grid) for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (grid[r][k] > maxV) maxV = grid[r][k];
  for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
    const x = k*cell, y = r*cell;
    let v = grid ? grid[r][k] : 0;
    const a = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
    g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
    g.fillRect(x, y, cell, cell);
    g.strokeStyle = gridCol;
    g.strokeRect(x + 0.5, y + 0.5, cell, cell);
    if (v >= 0.01) {
      g.fillStyle = a > 0.5 ? '#fff' : (cssVar('--heat-text') || '#111');
      g.font = `${Math.max(8, Math.floor(cell*0.34))}px ${MONO_FONT}`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText((v*100).toFixed(0), x + cell/2, y + cell/2);
    }
  }
  if (state && state.board) {
    const r0 = cell * 0.32;
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
      const sv = state.board[r][k]; if (!sv) continue;
      const cx = k*cell + cell/2, cy = r*cell + cell/2;
      g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI*2);
      if (sv === 1) {
        g.fillStyle = 'rgba(0,0,0,0.7)'; g.fill();
        g.lineWidth = 1; g.strokeStyle = 'rgba(255,255,255,0.6)'; g.stroke();
      } else {
        g.fillStyle = 'rgba(255,255,255,0.85)'; g.fill();
        g.lineWidth = 1; g.strokeStyle = 'rgba(0,0,0,0.5)'; g.stroke();
      }
    }
  }
}

/* ---------- Heat modal ---------- */
let expandedSourceId = null;
function setupModalCanvas() {
  const canvas = $('heat_modal_canvas');
  const card = canvas.parentElement;
  const cardCS = getComputedStyle(card);
  const padX = parseFloat(cardCS.paddingLeft) + parseFloat(cardCS.paddingRight);
  const padY = parseFloat(cardCS.paddingTop) + parseFloat(cardCS.paddingBottom);
  const header = card.querySelector('.heat-modal-header');
  const headerH = header ? header.offsetHeight + 12 : 0;
  const availW = window.innerWidth * 0.95 - padX;
  const availH = window.innerHeight * 0.95 - padY - headerH;
  const sz = Math.max(240, Math.floor(Math.min(availW, availH)));
  canvas.style.width = sz + 'px';
  canvas.style.height = sz + 'px';
  heatCtxs.h_modal = setupCanvas(canvas, sz, sz, false);
}
function paintHeatModal() {
  if (!expandedSourceId) return;
  const grid = state ? state[HEAT_KEYS[expandedSourceId]] : null;
  drawHeatToModal(grid);
}
function drawHeatToModal(grid) {
  const g = heatCtxs.h_modal;
  if (!g) return;
  clearLogical(g);
  const W = g._logicalW;
  const cell = W / N;
  const gridCol = cssVar('--heat-grid') || '#e5e7eb';
  let maxV = 0;
  if (grid) for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) if (grid[r][k] > maxV) maxV = grid[r][k];
  for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
    const x = k*cell, y = r*cell;
    let v = grid ? grid[r][k] : 0;
    const a = (maxV > 0 && v > 0) ? Math.min(1, v / maxV) : 0;
    g.fillStyle = `rgba(220,38,38,${a.toFixed(3)})`;
    g.fillRect(x, y, cell, cell);
    g.strokeStyle = gridCol;
    g.strokeRect(x + 0.5, y + 0.5, cell, cell);
    if (v >= 0.01) {
      g.fillStyle = a > 0.5 ? '#fff' : (cssVar('--heat-text') || '#111');
      g.font = `${Math.max(10, Math.floor(cell*0.34))}px ${MONO_FONT}`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText((v*100).toFixed(0), x + cell/2, y + cell/2);
    }
  }
  if (state && state.board) {
    const r0 = cell * 0.32;
    for (let r = 0; r < N; r++) for (let k = 0; k < N; k++) {
      const sv = state.board[r][k]; if (!sv) continue;
      const cx = k*cell + cell/2, cy = r*cell + cell/2;
      g.beginPath(); g.arc(cx, cy, r0, 0, Math.PI*2);
      if (sv === 1) {
        g.fillStyle = 'rgba(0,0,0,0.7)'; g.fill();
        g.strokeStyle = 'rgba(255,255,255,0.6)'; g.lineWidth = 1; g.stroke();
      } else {
        g.fillStyle = 'rgba(255,255,255,0.85)'; g.fill();
        g.strokeStyle = 'rgba(0,0,0,0.5)'; g.lineWidth = 1; g.stroke();
      }
    }
  }
}
function openHeatModal(id) {
  if (!HEAT_KEYS[id]) return;
  expandedSourceId = id;
  const card = heatCanvases[id].parentElement;
  const titleEl = card.querySelector('.grid-title-text');
  $('heat_modal_title').textContent = titleEl ? titleEl.textContent : 'Heatmap';
  $('heat_modal').classList.remove('hidden');
  setupModalCanvas();
  paintHeatModal();
}
function closeHeatModal() {
  if (expandedSourceId === null) return;
  expandedSourceId = null;
  $('heat_modal').classList.add('hidden');
}

/* ---------- WDL ---------- */
function normWDL(wdl) {
  if (!wdl) return null;
  const s = wdl[0] + wdl[1] + wdl[2];
  if (s > 1e-4) {
    return { w: wdl[0]/s*100, d: wdl[1]/s*100, l: wdl[2]/s*100,
             wl: (wdl[0] - wdl[2]) / s };
  }
  return { w: wdl[0], d: wdl[1], l: wdl[2], wl: wdl[0] - wdl[2] };
}
function renderWDL(prefix, wdl) {
  const bar = $('wdl_' + prefix + '_bar');
  const wlEl = $('wdl_' + prefix + '_wl');
  const det = $('wdl_' + prefix + '_detail');
  const n = normWDL(wdl);
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
    '<span><span class="k">B</span> ' + n.w.toFixed(1) + '%</span>' +
    '<span><span class="k">D</span> ' + n.d.toFixed(1) + '%</span>' +
    '<span><span class="k">W</span> ' + n.l.toFixed(1) + '%</span>';
}

/* ---------- Value chart ---------- */
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

  const hist = (state && state.value_history) || [];
  if (hist.length === 0) {
    vctx.fillStyle = subtle;
    vctx.font = `11px ${MONO_FONT}`;
    vctx.textAlign = 'center'; vctx.textBaseline = 'middle';
    vctx.fillText('no data', padL + innerW/2, padT + innerH/2);
    return;
  }
  const maxX = Math.max(1, hist[hist.length - 1].ply);
  const xOf = s => padL + (s / maxX) * innerW;
  const yOf = v => padT + ((1 - v) / 2) * innerH;
  vctx.fillStyle = muted;
  vctx.textAlign = 'center'; vctx.textBaseline = 'top';
  vctx.fillText('0', xOf(0), H - padB + 2);
  vctx.fillText(String(maxX), xOf(maxX), H - padB + 2);

  function plot(key, color) {
    vctx.strokeStyle = color; vctx.lineWidth = 1.5;
    vctx.beginPath();
    hist.forEach((p, i) => {
      const v = p[key];
      const wl = v[0] - v[2];
      const x = xOf(p.ply), y = yOf(wl);
      if (i === 0) vctx.moveTo(x, y); else vctx.lineTo(x, y);
    });
    vctx.stroke();
    vctx.fillStyle = color;
    for (const p of hist) {
      const v = p[key];
      const wl = v[0] - v[2];
      vctx.beginPath(); vctx.arc(xOf(p.ply), yOf(wl), 2, 0, Math.PI*2); vctx.fill();
    }
  }
  plot('wdl_black', '#0969da');
  plot('nn_wdl_black', '#cf222e');
}

/* ---------- Status pill ---------- */
function statusVariant(s) {
  if (!s) return 'idle';
  const t = s.toLowerCase();
  if (t.includes('thinking')) return 'thinking';
  if (t.includes('your turn')) return 'active';
  if (t.includes('wins') || t.includes('draw')) return 'done';
  if (t.includes('invalid') || t.includes('illegal')) return 'error';
  return 'idle';
}

/* ---------- UI sync ---------- */
function updateSideButtons() {
  $('side_black').setAttribute('aria-pressed', selectedSide === 1 ? 'true' : 'false');
  $('side_white').setAttribute('aria-pressed', selectedSide === -1 ? 'true' : 'false');
}
function updateAlgoButtons() {
  $('algo_puct').setAttribute('aria-pressed', selectedAlgo === 'puct' ? 'true' : 'false');
  $('algo_gumbel').setAttribute('aria-pressed', selectedAlgo === 'gumbel' ? 'true' : 'false');
  $('title_mcts').textContent =
      selectedAlgo === 'gumbel' ? 'MCTS Improved Policy' : 'MCTS Visit Distribution';
}

function refresh(s) {
  if (s) state = s;
  if (!state) return;
  if (state.board_size !== N) {
    N = state.board_size;
    computeBoardLayout();
    for (const id of Object.keys(heatCanvases)) fitHeatCanvas(id);
  }
  $('brand_title').textContent = 'SkyZero V1 — ' + (state.game_title || '');
  document.title = 'SkyZero V1 — ' + (state.game_title || '');
  $('status').textContent = state.status || '';
  $('status_pill').dataset.variant = statusVariant(state.status);
  $('meta_line').innerHTML =
    `<span class="k">board</span> ${state.board_size}×${state.board_size}` +
    (state.rule ? ` &nbsp;·&nbsp; <span class="k">rule</span> ${state.rule}` : '') +
    ` &nbsp;·&nbsp; <span class="k">algo</span> ${state.algo}`;

  selectedSide = state.human_side;
  selectedAlgo = state.algo;
  updateSideButtons();
  updateAlgoButtons();
  if (parseInt($('sims_input').value, 10) !== state.num_simulations) {
    $('sims_input').value = state.num_simulations;
  }
  renderWDL('root', state.root_wdl_black);
  renderWDL('nn',   state.nn_value_black);
  draw();
  drawHeat('h_nn', state.nn_policy);
  drawHeat('h_mcts', state.mcts_policy);
  drawValueChart();
  paintHeatModal();
  // Enable undo button only when it's human's turn, game isn't over, and board has moves
  const hasMoves = state.board.some(row => row.some(v => v !== 0));
  $('undo_btn').disabled = !(state.to_play === state.human_side && !state.game_over && hasMoves);
}

function setBusy(b) {
  busy = b;
  document.body.style.cursor = b ? 'progress' : '';
}

async function fetchJSON(url, body) {
  const opts = body
    ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}
    : {method: 'GET'};
  const r = await fetch(url, opts);
  return r.json();
}

async function submitMove(r, c) {
  setBusy(true);
  $('status').textContent = 'AlphaZero is thinking...';
  $('status_pill').dataset.variant = 'thinking';
  try {
    const resp = await fetchJSON('/move', {r, c});
    if (resp && resp.ok === false) {
      // illegal move; show err briefly
      $('status').textContent = resp.err || 'Move rejected.';
      $('status_pill').dataset.variant = 'error';
      if (resp.state) refresh(resp.state);
      return;
    }
    refresh(resp);
  } finally {
    setBusy(false);
  }
}

async function undoMove() {
  setBusy(true);
  try {
    const resp = await fetchJSON('/undo', {});
    if (resp && resp.ok === false) {
      $('status').textContent = resp.err || 'Cannot undo.';
      $('status_pill').dataset.variant = 'error';
      return;
    }
    refresh(resp);
  } finally {
    setBusy(false);
  }
}

async function newGame() {
  setBusy(true);
  $('status').textContent = 'Starting...';
  $('status_pill').dataset.variant = 'thinking';
  try {
    const body = {
      human_side: selectedSide,
      algo: selectedAlgo,
      num_simulations: parseInt($('sims_input').value, 10) || undefined,
    };
    const s = await fetchJSON('/new', body);
    refresh(s);
  } finally {
    setBusy(false);
  }
}

function setSide(side) {
  if (side !== 1 && side !== -1) return;
  selectedSide = side;
  updateSideButtons();
  newGame();
}
function setAlgo(algo) {
  if (algo !== 'puct' && algo !== 'gumbel') return;
  selectedAlgo = algo;
  updateAlgoButtons();
  newGame();
}

cv.addEventListener('click', (ev) => {
  if (busy || !state || state.game_over) return;
  if (state.to_play !== state.human_side) return;
  const rect = cv.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const c = Math.round((x - MARGIN) / CELL);
  const r = Math.round((y - MARGIN) / CELL);
  if (r < 0 || r >= N || c < 0 || c >= N) return;
  if (state.board[r][c] !== 0) return;
  submitMove(r, c);
});

/* ---------- Theme ---------- */
const themeBtn = $('theme_toggle');
const THEME_NEXT = { auto: 'light', light: 'dark', dark: 'auto' };
function resolveTheme(mode) {
  if (mode === 'light' || mode === 'dark') return mode;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function setThemeTooltip(mode) {
  const label = mode.charAt(0).toUpperCase() + mode.slice(1);
  const nxt = THEME_NEXT[mode];
  themeBtn.title = 'Theme: ' + label + ' (click for ' + nxt + ')';
}
function applyTheme(mode) {
  document.documentElement.dataset.theme = resolveTheme(mode);
  document.documentElement.dataset.themeMode = mode;
  setThemeTooltip(mode);
  draw();
  drawValueChart();
  drawHeat('h_nn', state ? state.nn_policy : null);
  drawHeat('h_mcts', state ? state.mcts_policy : null);
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
try {
  const mql = window.matchMedia('(prefers-color-scheme: dark)');
  const onSystemThemeChange = () => {
    if ((document.documentElement.dataset.themeMode || 'auto') !== 'auto') return;
    applyTheme('auto');
  };
  if (mql.addEventListener) mql.addEventListener('change', onSystemThemeChange);
} catch(e) {}

/* ---------- Modal events ---------- */
for (const btn of document.querySelectorAll('.expand-btn')) {
  btn.addEventListener('click', () => openHeatModal(btn.dataset.target));
}
$('heat_modal_close').addEventListener('click', closeHeatModal);
$('heat_modal').addEventListener('click', (ev) => {
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

$('sims_input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') { ev.preventDefault(); ev.target.blur(); newGame(); }
});

// initial fetch
fetchJSON('/state').then(refresh);
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Server entry points
# ---------------------------------------------------------------------------

def run_server(game_name, *, host="127.0.0.1", port=8765, device=None,
               algo=None, sims=None, ckpt=None, train_args_override=None):
    """Build session, load checkpoint, serve forever. Used by entry points."""
    game, train_args, meta = _load_game(game_name, train_args_override)

    chosen_device = device or train_args.get("device", "cpu")
    if chosen_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        chosen_device = "cpu"

    model = ResNet(game,
                   num_blocks=train_args["num_blocks"],
                   num_channels=train_args["num_channels"]).to(chosen_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    az_args = dict(train_args)
    az_args["device"] = chosen_device
    az_args["mode"] = "eval"
    if sims is not None:
        az_args["num_simulations"] = sims
    if algo is not None:
        az_args["algo"] = algo
    az_args.setdefault("algo", "puct")

    alphazero = AlphaZero(game, model, optimizer, az_args)
    loaded = alphazero.load_checkpoint(ckpt) if ckpt else alphazero.load_checkpoint()
    if not loaded:
        print("WARNING: no checkpoint loaded; using random weights.")
    model.eval()

    session = GameSession(game, alphazero, az_args, meta, game_name)
    Handler.session = session

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  "
          f"(game={game_name}, algo={az_args['algo']}, "
          f"sims={az_args['num_simulations']}, device={chosen_device})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["gomoku", "tictactoe"], default="gomoku")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default=None, help="cuda|cpu (defaults to train cfg)")
    p.add_argument("--algo", choices=["puct", "gumbel"], default=None,
                   help="initial algorithm; switchable in the UI")
    p.add_argument("--sims", type=int, default=None,
                   help="MCTS simulations per move; overrides train cfg")
    p.add_argument("--ckpt", default=None, help="path to specific checkpoint .pth")
    args = p.parse_args()
    run_server(args.game, host=args.host, port=args.port, device=args.device,
               algo=args.algo, sims=args.sims, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
