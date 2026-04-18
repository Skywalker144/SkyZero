"""
Plot training and validation losses from JSON-lines log files.

Usage:
    python view_loss.py                          # defaults: ../data/train/skyzero
    python view_loss.py --traindir ../data/train/skyzero --output loss.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path, keys):
    """Read a JSON-lines file, extracting specified keys."""
    data = {"x": []}
    for k in keys:
        data[k] = []
    if not os.path.exists(path):
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) < 5:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            x = j.get("global_samples", j.get("step", 0))
            data["x"].append(x)
            for k in keys:
                data[k].append(j.get(k, 0.0))
    return data


def main():
    parser = argparse.ArgumentParser(description="Plot SkyZero training losses")
    parser.add_argument("--traindir", default="../data/train/skyzero",
                        help="Training directory containing *_metrics.json files")
    parser.add_argument("--output", default="../loss.png",
                        help="Output image path")
    args = parser.parse_args()

    train_keys = ["total_loss", "policy_loss", "opp_policy_loss", "value_loss"]
    val_keys = ["val_total", "val_policy", "val_opp_policy", "val_value"]

    train_path = os.path.join(args.traindir, "train_metrics.json")
    val_path = os.path.join(args.traindir, "val_metrics.json")

    train_data = read_jsonl(train_path, train_keys)
    val_data = read_jsonl(val_path, val_keys)

    # Plot layout: total, policy, opp_policy, value
    plot_configs = [
        ("Total Loss", "total_loss", "val_total"),
        ("Policy Loss", "policy_loss", "val_policy"),
        ("Opp Policy Loss", "opp_policy_loss", "val_opp_policy"),
        ("Value Loss", "value_loss", "val_value"),
    ]

    fig, axes = plt.subplots(len(plot_configs), 1, figsize=(8, 3.5 * len(plot_configs)), dpi=150)
    fig.suptitle("SkyZero V4 Training", fontsize=14)
    plt.subplots_adjust(hspace=0.4)

    for i, (title, train_key, val_key) in enumerate(plot_configs):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel("samples")
        ax.set_ylabel("loss")

        if train_data["x"]:
            ax.plot(train_data["x"], train_data[train_key], label="train", alpha=0.8, linewidth=0.8)
        if val_data["x"]:
            ax.plot(val_data["x"], val_data[val_key], label="val", alpha=0.8, linewidth=1.2, linestyle="--")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved loss plot to {args.output}")


if __name__ == "__main__":
    main()
