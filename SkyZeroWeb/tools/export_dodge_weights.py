#!/usr/bin/env python3
"""Export a trained PPO Channel-Dodge policy to a tiny browser-runnable JS file.

Unlike gomoku/2048 (big conv nets + MCTS → onnxruntime-web in a Worker), the
dodge policy is a trivial MLP (271 → 256 → 256 → 9, tanh). So instead of shipping
the whole ONNX/WASM runtime to run ~90k params, we just pack the actor weights
into ``dodge-policy.js`` and do a ~30-line synchronous forward pass in the page.
This keeps channel-dodge.html self-contained (no CDN, no async, no WASM).

Only the ACTOR path is exported (torso + policy head); the critic/value head is
inference-irrelevant and dropped.

Layout of the packed float32 blob (little-endian, the order the JS reads):
    L0.weight [256,271] (row-major), L0.bias [256] ,
    L1.weight [256,256]            , L1.bias [256] ,
    L2.weight [9,256]              , L2.bias [9]
A PyTorch Linear stores weight as [out,in] and computes y = x @ Wᵀ + b, i.e.
y[o] = Σ_i W[o,i]·x[i] + b[o] — so row-major flatten + the JS dot product agree.

Usage (from the pytorch conda env):
    python tools/export_dodge_weights.py                       # run=dodge_v2 → dodge-policy.js
    python tools/export_dodge_weights.py --run dodge_v3
    python tools/export_dodge_weights.py --ckpt /abs/best.pt --version dodge_v3-1

After re-exporting, bump the ?v= on the <script src="dodge-policy.js?v=…"> tag in
channel-dodge.html (the file is served max-age=3600; see _headers) — the script
prints the exact line to paste.
"""

import argparse
import base64
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.dirname(HERE)                      # …/SkyZeroWeb
DEFAULT_PPO = "/home/sky/RL/PPO"

# state_dict keys for the actor path (see networks.ActorCritic)
LAYERS = [
    ("torso.net.0.weight", "torso.net.0.bias", 271, 256, "tanh"),
    ("torso.net.2.weight", "torso.net.2.bias", 256, 256, "tanh"),
    ("actor.weight",       "actor.bias",       256, 9,   "linear"),
]
OBS_DIM = LAYERS[0][2]                              # network input width = obs size
NUM_ACTIONS = LAYERS[-1][3]                         # final layer output = #discrete actions
HIDDEN = tuple(n_out for *_, n_out, _act in LAYERS[:-1])   # (256, 256)


def _forward_numpy(blob, x):
    """Mirror of the JS forward pass — used to self-check the packed blob."""
    off = 0
    h = np.asarray(x, dtype=np.float32)
    for _wk, _bk, n_in, n_out, act in LAYERS:
        W = blob[off:off + n_out * n_in].reshape(n_out, n_in); off += n_out * n_in
        b = blob[off:off + n_out];                              off += n_out
        h = W @ h + b
        if act == "tanh":
            h = np.tanh(h)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppo", default=DEFAULT_PPO, help="PPO training repo root")
    ap.add_argument("--run", default="dodge_v2", help="run name under <ppo>/runs/")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (overrides --run)")
    ap.add_argument("--out", default=os.path.join(WEB_ROOT, "dodge-policy.js"))
    ap.add_argument("--version", default=None, help="version tag baked into the file (default: run name)")
    args = ap.parse_args()

    ckpt_path = args.ckpt or os.path.join(args.ppo, "runs", args.run, "best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"[export] checkpoint not found: {ckpt_path}")
    version = args.version or args.run

    ck = torch.load(ckpt_path, map_location="cpu")
    obs_mode = ck.get("obs_mode", ck.get("config", {}).get("obs_mode"))
    action_mode = ck.get("action_mode", ck.get("config", {}).get("action_mode"))
    obs_shape = tuple(ck.get("obs_shape", ()))
    hidden = tuple(ck.get("config", {}).get("hidden", ()))
    if obs_mode != "vector" or action_mode != "discrete":
        sys.exit(f"[export] this exporter only handles vector+discrete; got "
                 f"obs_mode={obs_mode!r} action_mode={action_mode!r}")
    if obs_shape != (OBS_DIM,):
        sys.exit(f"[export] expected obs_shape ({OBS_DIM},), got {obs_shape}")
    if hidden not in (HIDDEN, ()):  # () = older ckpt without config; assume HIDDEN
        print(f"[export] WARNING: hidden={hidden} (this exporter assumes {HIDDEN})")

    sd = ck["model"]
    parts = []
    for wk, bk, n_in, n_out, _act in LAYERS:
        W = sd[wk].detach().cpu().numpy().astype(np.float32)
        b = sd[bk].detach().cpu().numpy().astype(np.float32)
        assert W.shape == (n_out, n_in), f"{wk}: {W.shape} != {(n_out, n_in)}"
        assert b.shape == (n_out,), f"{bk}: {b.shape} != {(n_out,)}"
        parts.append(W.reshape(-1))   # row-major [out,in]
        parts.append(b)
    blob = np.concatenate(parts).astype("<f4")   # explicit little-endian float32

    # --- self-check: packed-blob forward MUST match the real torch model ---
    sys.path.insert(0, args.ppo)
    from networks import ActorCritic  # noqa: E402  (only needed for the check)
    agent = ActorCritic((OBS_DIM,), num_actions=NUM_ACTIONS, hidden=HIDDEN)
    agent.load_state_dict(sd)
    agent.eval()
    rng = np.random.default_rng(0)
    max_logit_err = 0.0
    mismatches = 0
    for _ in range(64):
        x = rng.standard_normal(OBS_DIM).astype(np.float32)
        with torch.no_grad():
            ref = agent.actor(agent.torso(torch.from_numpy(x))).numpy()
        got = _forward_numpy(blob, x)
        max_logit_err = max(max_logit_err, float(np.abs(ref - got).max()))
        mismatches += int(ref.argmax() != got.argmax())
    if mismatches or max_logit_err > 1e-3:
        sys.exit(f"[export] SELF-CHECK FAILED: argmax mismatches={mismatches} "
                 f"max_logit_err={max_logit_err:.2e}")

    b64 = base64.b64encode(blob.tobytes()).decode("ascii")
    raw_bytes = blob.nbytes

    # arch/dims/layout strings derived from LAYERS so the JS never hardcodes obs size
    arch_js = ", ".join(f'[{ni}, {no}, "{act}"]' for _, _, ni, no, act in LAYERS)
    dims = f"{OBS_DIM} -> " + " -> ".join(str(no) for *_, no, _a in LAYERS)
    blob_layout = " ".join(f"L{li}.W({no}x{ni}) L{li}.b({no})"
                           for li, (_, _, ni, no, _a) in enumerate(LAYERS))

    js = f"""// AUTO-GENERATED by tools/export_dodge_weights.py — DO NOT EDIT BY HAND.
// Trained PPO Channel-Dodge policy (actor only): obs {dims}, tanh.
// Source checkpoint: {os.path.relpath(ckpt_path, WEB_ROOT)}   version: {version}
// Forward pass + observation builder live in channel-dodge.html (pure JS, no deps).
window.DODGE_POLICY = {{
  version: {version!r},
  obsDim: {OBS_DIM},
  numActions: {NUM_ACTIONS},
  // [in, out, activation] per layer, in packed order:
  arch: [{arch_js}],
  // little-endian float32: {blob_layout}
  weightsB64: "{b64}",
}};
"""
    with open(args.out, "w") as f:
        f.write(js)

    print(f"[export] {ckpt_path}")
    print(f"[export]   self-check OK (64 random inputs, max logit err {max_logit_err:.1e}, argmax all match)")
    print(f"[export]   wrote {args.out}  ({raw_bytes/1024:.0f} KB float32 -> {len(js)/1024:.0f} KB js)")
    print(f"[export]   version tag: {version!r}")
    print(f"[export] >>> bump the cache-bust in channel-dodge.html to:")
    print(f'[export] >>>   <script src="dodge-policy.js?v={version}"></script>')


if __name__ == "__main__":
    main()
