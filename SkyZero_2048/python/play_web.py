"""Web demo server: the trained Stochastic Gumbel AlphaZero agent plays 2048
live in the browser.

Stateless HTTP (Python stdlib, no extra deps):
    GET  /                 -> the demo page
    GET  /api/new          -> a fresh board (two random tiles)
    POST /api/step {board} -> run MCTS, apply the chosen move + spawn a tile,
                              return the new board plus the agent's reasoning
                              (chosen direction, per-direction visits, value).

Run via scripts/play_web.sh, or:
    python play_web.py --ckpt data2048_nbt/models/latest.pt --port 8848 --sims 128
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import game as G
from model_config import Config
from mcts import GameSearch, batch_search, net_evaluator, stub_evaluator

_WEB_DIR = pathlib.Path(__file__).parent / "web"

# Populated in main().
_STATE: dict = {}


def _decide(board: list[int]):
    """Decide the move for a board (exponents) WITHOUT spawning — the web UI's
    own animation engine applies the slide + spawn. Returns the chosen direction
    plus the raw network policy, the Gumbel improved policy, visits and value."""
    cfg: Config = _STATE["cfg"]
    eval_fn = _STATE["eval_fn"]
    rng = _STATE["rng"]
    state = np.asarray(board, dtype=np.int8)
    if G.is_terminal(state):
        return {"terminal": True}
    gs = GameSearch(state, cfg, rng)
    batch_search([gs], eval_fn, cfg)
    return {
        "terminal": False,
        "action": int(gs.best_action()),          # 0=up 1=right 2=down 3=left
        "nn_policy": gs.nn_policy().tolist(),
        "improved_policy": gs.improved_policy().tolist(),
        "visits": gs.visit_counts().tolist(),
        "value": float(gs.root_value()),
    }


def _think(board: list[int]):
    cfg: Config = _STATE["cfg"]
    eval_fn = _STATE["eval_fn"]
    rng = _STATE["rng"]
    state = np.asarray(board, dtype=np.int8)
    if G.is_terminal(state):
        return {"terminal": True}
    gs = GameSearch(state, cfg, rng)
    batch_search([gs], eval_fn, cfg)
    action = gs.best_action()
    visits = gs.visit_counts().tolist()
    value = gs.root_value()
    after, reward, _ = G.apply_move(state, action)
    nxt = G.spawn_random(after, rng)
    return {
        "terminal": False,
        "action": int(action),              # 0=up 1=right 2=down 3=left
        "reward": int(reward),
        "value": float(value),
        "visits": visits,
        "nn_policy": gs.nn_policy().tolist(),            # raw policy-head prior
        "improved_policy": gs.improved_policy().tolist(),  # Gumbel completed-Q target
        "policy": gs.improved_policy().tolist(),         # backward-compat alias
        "board": nxt.astype(int).tolist(),
        "afterstate": after.astype(int).tolist(),
        "maxtile": int(1 << G.max_tile_exp(nxt)),
        "next_terminal": bool(G.is_terminal(nxt)),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._serve_file("play2048.html")
        elif self.path.startswith("/api/new"):
            rng = _STATE["rng"]
            board = G.initial_state(rng).astype(int).tolist()
            self._json({"board": board})
        elif self.path.startswith("/api/info"):
            self._json({"sims": _STATE["cfg"].num_simulations,
                        "ckpt": _STATE["ckpt"], "device": _STATE["cfg"].device})
        elif self.path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
        else:
            # Static assets (style.css, fonts/, etc.) from the web dir.
            self._serve_file(self.path.lstrip("/").split("?")[0])

    _CT = {".css": "text/css", ".js": "application/javascript", ".html": "text/html",
           ".woff2": "font/woff2", ".woff": "font/woff", ".svg": "image/svg+xml",
           ".png": "image/png", ".json": "application/json"}

    def _serve_file(self, rel: str) -> None:
        path = (_WEB_DIR / rel).resolve()
        if _WEB_DIR not in path.parents and path != _WEB_DIR or not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = self._CT.get(path.suffix, "application/octet-stream")
        if ctype.startswith("text") or ctype.endswith(("javascript", "json")):
            ctype += "; charset=utf-8"
        self._send(200, path.read_bytes(), ctype)

    def do_POST(self):
        if self.path.startswith("/api/decide"):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            self._json(_decide(data["board"]))
        elif self.path.startswith("/api/step"):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            self._json(_think(data["board"]))
        else:
            self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    # Loads the active TorchScript mirror written by run.sh (data/models/latest.pt)
    # plus its meta sidecar (value_scale, network). No state_dict / Config needed.
    ap.add_argument("--ckpt", type=str, default="data2048_nbt/models/latest.pt")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import json
    import torch

    ckpt = pathlib.Path(args.ckpt)
    if ckpt.exists():
        device = args.device if torch.cuda.is_available() else "cpu"
        meta = {}
        meta_path = ckpt.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
        cfg = Config(value_scale=float(meta.get("value_scale", Config.value_scale)),
                     num_simulations=args.sims, gumbel_noise=False, device=device)
        ts = torch.jit.load(str(ckpt), map_location=device)
        ts.eval()
        eval_fn = net_evaluator(ts, cfg, device)   # callable module; returns (logits, scaled value)
        print(f"[web] loaded {ckpt} (iter {meta.get('iter','?')}, net {meta.get('network','?')}) "
              f"on {device}, sims={args.sims}")
    else:
        cfg = dataclasses.replace(Config(), num_simulations=args.sims, gumbel_noise=False, device="cpu")
        eval_fn = stub_evaluator(cfg)
        print(f"[web] no checkpoint at {ckpt} — running with an untrained stub policy")

    _STATE.update(cfg=cfg, eval_fn=eval_fn, rng=np.random.default_rng(),
                  ckpt=str(ckpt))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[web] serving on http://localhost:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
