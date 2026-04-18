#!/usr/bin/env python3
"""
Read a C++ AlphaZero checkpoint (.ckpt, libtorch format) and generate
the same three plots that SkyZero_V2.1-main/alphazero.py produces:

  1. {file_name}_total_loss.png      -- Total training loss curve
  2. {file_name}_loss_components.png  -- Individual loss components
  3. {file_name}_win_rates.png        -- Win rates + avg game length ratio

Usage:
    python plot_from_checkpoint.py                          # auto-find latest .ckpt
    python plot_from_checkpoint.py <path_to_checkpoint>     # specify .ckpt file
    python plot_from_checkpoint.py --board-size 9 <ckpt>    # override board size
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Load checkpoint via torch.jit.load ───────────────────────────────

def load_metrics_from_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load training metrics from a C++ AlphaZero checkpoint.

    The C++ side saves via ``torch::serialize::OutputArchive``, which
    produces a TorchScript-compatible archive.  ``torch.jit.load`` can
    read it and all top-level tensors become attributes on the returned
    ``RecursiveScriptModule``.
    """
    module = torch.jit.load(path, map_location="cpu")

    metric_keys = [
        "game_count",
        "total_samples",
        "loss_total",
        "loss_policy",
        "loss_opp_policy",
        "loss_soft_policy",
        "loss_soft_opp_policy",
        "loss_value",
        "avg_game_len_history",
        "winrate_history",
    ]

    result: Dict[str, torch.Tensor] = {}
    for key in metric_keys:
        try:
            val = getattr(module, key)
            if isinstance(val, torch.Tensor):
                result[key] = val.detach().cpu()
        except AttributeError:
            pass

    return result


# ── Plotting ─────────────────────────────────────────────────────────

def plot_metrics(
    losses_dict: Dict[str, List[float]],
    winrate_history: Optional[np.ndarray],
    avg_game_len_history: Optional[np.ndarray],
    game_count: int,
    board_size: int,
    output_dir: str,
    file_name: str,
):
    """Generate the same 3 plots as AlphaZero.plot_metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Total Loss ────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(losses_dict["total_loss"], label="Total Loss")
    plt.title(f"Total Training Loss (Game {game_count})")
    plt.xlabel("Training Generation")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"{file_name}_total_loss.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    # ── 2. Loss Components ───────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    for key, values in losses_dict.items():
        if key == "total_loss":
            continue
        label = key.replace("_", " ").title()
        plt.plot(values, label=label)
    plt.title(f"Loss Components (Game {game_count})")
    plt.xlabel("Training Generation")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f"{file_name}_loss_components.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    # ── 3. Win Rates ─────────────────────────────────────────────────
    if winrate_history is not None and len(winrate_history) > 0:
        games = winrate_history[:, 0]
        b_rates = winrate_history[:, 1]
        w_rates = winrate_history[:, 2]
        d_rates = winrate_history[:, 3]

        plt.figure(figsize=(10, 6))
        plt.plot(games, b_rates, label="Black Win Rate", color="black")
        plt.plot(games, w_rates, label="White Win Rate", color="red")
        plt.plot(games, d_rates, label="Draw Rate", color="gray")

        if (
            avg_game_len_history is not None
            and len(avg_game_len_history) == len(games)
        ):
            plt.plot(
                games,
                avg_game_len_history / (board_size ** 2),
                label="Avg Game Length Ratio",
                color="blue",
                linestyle="--",
            )

        plt.title(f"Win Rates (Last {len(b_rates)} Statistics)")
        plt.xlabel("Game Count")
        plt.ylabel("Rate")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output_dir, f"{file_name}_win_rates.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        print("  (No winrate history found, skipping win rates plot)")


# ── Main ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(data_dir: str = "data/gomoku/checkpoints") -> Optional[str]:
    """Find the most recently modified .ckpt file in the given directory."""
    ckpt_dir = Path(data_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from a C++ AlphaZero checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to the .ckpt file (default: auto-find latest in data/gomoku/checkpoints/)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        help="Board size for avg game length normalization (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as checkpoint's data dir)",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Base file name prefix for PNG files (default: derived from checkpoint name)",
    )
    args = parser.parse_args()

    # ── Locate checkpoint ────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("Error: No checkpoint found. Specify a path or ensure "
                  "data/gomoku/checkpoints/ contains .ckpt files.")
            sys.exit(1)
        print(f"Auto-selected latest checkpoint: {ckpt_path}")

    if not os.path.isfile(ckpt_path):
        print(f"Error: file not found: {ckpt_path}")
        sys.exit(1)

    # ── Derive names ─────────────────────────────────────────────────
    ckpt_name = Path(ckpt_path).stem  # e.g. "gomoku_checkpoint_2026-03-31_18-39-20"
    # Extract game name (everything before "_checkpoint_")
    if "_checkpoint_" in ckpt_name:
        game_name = ckpt_name.split("_checkpoint_")[0]
    else:
        game_name = ckpt_name

    file_name = args.file_name or game_name

    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: data/<game>/  (parent's parent of checkpoints/)
        output_dir = str(Path(ckpt_path).parent.parent)

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    data = load_metrics_from_checkpoint(ckpt_path)

    if not data:
        print("Error: failed to extract any tensors from the checkpoint.")
        sys.exit(1)

    print(f"Found {len(data)} metric keys: {sorted(data.keys())}")

    # ── Extract metrics ──────────────────────────────────────────────
    game_count = int(data["game_count"].item()) if "game_count" in data else 0

    losses_dict: Dict[str, List[float]] = {}

    key_map = {
        "loss_total": "total_loss",
        "loss_policy": "policy_loss",
        "loss_opp_policy": "opponent_policy_loss",
        "loss_soft_policy": "soft_policy_loss",
        "loss_soft_opp_policy": "soft_opponent_policy_loss",
        "loss_value": "value_loss",
    }

    for cpp_key, py_key in key_map.items():
        if cpp_key in data:
            losses_dict[py_key] = data[cpp_key].float().numpy().tolist()

    if not losses_dict:
        print("Error: no loss history found in checkpoint.")
        sys.exit(1)

    print(f"Game count: {game_count}")
    for k, v in losses_dict.items():
        print(f"  {k}: {len(v)} entries")

    # Winrate history: Nx4 tensor [game_count, b_rate, w_rate, d_rate]
    winrate_history = None
    if "winrate_history" in data:
        wh = data["winrate_history"].float().numpy()
        if wh.ndim == 2 and wh.shape[1] == 4:
            winrate_history = wh
            print(f"  winrate_history: {wh.shape[0]} entries")
        elif wh.ndim == 1 and wh.shape[0] % 4 == 0:
            winrate_history = wh.reshape(-1, 4)
            print(f"  winrate_history: {winrate_history.shape[0]} entries (reshaped)")

    # Avg game length history
    avg_game_len_history = None
    if "avg_game_len_history" in data:
        agl = data["avg_game_len_history"].float().numpy()
        avg_game_len_history = agl
        print(f"  avg_game_len_history: {len(agl)} entries")

    # ── Plot ─────────────────────────────────────────────────────────
    print(f"\nGenerating plots in: {output_dir}")
    plot_metrics(
        losses_dict=losses_dict,
        winrate_history=winrate_history,
        avg_game_len_history=avg_game_len_history,
        game_count=game_count,
        board_size=args.board_size,
        output_dir=output_dir,
        file_name=file_name,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
