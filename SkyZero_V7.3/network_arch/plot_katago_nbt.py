"""Plot KataGo nested-bottleneck (nbt) configs: outer_blocks vs trunk_channels."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (name, outer_blocks, trunk_channels, tier, label_dx, label_dy)
configs = [
    ("b1c6nbt",     1,   6,   "micro",       8,   8),
    ("b5c192nbt",   5,   192, "small",       8,   8),
    ("b8c192nbt",   8,   192, "small",       8, -18),
    ("b10c256nbt",  10,  256, "medium",     -65, -18),
    ("b10c384nbt",  10,  384, "AlphaZero", -85,  10),
    ("b12c384nbt",  12,  384, "AlphaZero",   8,  -6),
    ("b10c480nb3t", 10,  480, "AlphaZero", -90,   8),
    ("b18c384nbt",  18,  384, "AlphaGoZero", 8, -22),
    ("b14c448nbt",  14,  448, "AlphaGoZero", 8,   8),
    ("b41c384nbt",  41,  384, "large",     -80,   8),
    ("b32c448nbt",  32,  448, "large",       8,   8),
    ("b28c512nbt",  28,  512, "large",      10,  10),
    ("b20c640nbt",  20,  640, "large",       8,   8),
    ("b40c768nbt",  40,  768, "large",     -85,   8),
]

recommended = {"b10c384nbt", "b18c384nbt", "b28c512nbt"}

tier_colors = {
    "micro":       "#888888",
    "small":       "#1f77b4",
    "medium":      "#2ca02c",
    "AlphaZero":   "#ff7f0e",
    "AlphaGoZero": "#d62728",
    "large":       "#9467bd",
}
tier_order = ["micro", "small", "medium", "AlphaZero", "AlphaGoZero", "large"]

fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)

# Iso-ratio reference lines (c = k * b)
b_line = [0.5, 45]
for k, style in [(12.8, ":"), (20, "--"), (30, "--"), (40, ":")]:
    ax.plot(b_line, [k * b_line[0], k * b_line[1]], style,
            color="gray", alpha=0.45, linewidth=1, zorder=1)
    # label at right edge
    ax.text(44.5, k * 44.5, f" c/b={k}",
            color="gray", fontsize=9, ha="left", va="center", alpha=0.7)

# Plot points grouped by tier (so legend works)
for tier in tier_order:
    pts = [(b, c, name) for name, b, c, t, *_ in configs if t == tier]
    if not pts:
        continue
    bs  = [p[0] for p in pts]
    cs  = [p[1] for p in pts]
    # non-recommended dots
    reg_idx = [i for i, p in enumerate(pts) if p[2] not in recommended]
    rec_idx = [i for i, p in enumerate(pts) if p[2] in recommended]
    if reg_idx:
        ax.scatter([bs[i] for i in reg_idx], [cs[i] for i in reg_idx],
                   s=120, c=tier_colors[tier], edgecolors="black",
                   linewidths=0.8, marker="o", zorder=3, label=tier)
    if rec_idx:
        ax.scatter([bs[i] for i in rec_idx], [cs[i] for i in rec_idx],
                   s=320, c=tier_colors[tier], edgecolors="black",
                   linewidths=2.0, marker="*", zorder=4,
                   label=None if reg_idx else tier)

# Annotate every point with controlled offsets
for name, b, c, tier, dx, dy in configs:
    is_rec = name in recommended
    ax.annotate(
        name, (b, c),
        textcoords="offset points", xytext=(dx, dy),
        fontsize=9, fontweight="bold" if is_rec else "normal",
        color="black",
    )

ax.set_xlabel("outer blocks (b)", fontsize=13)
ax.set_ylabel("trunk channels (c)", fontsize=13)
ax.set_title("KataGo nested-bottleneck (nbt) configs   "
             "★ = official recommended for tier", fontsize=13)
ax.grid(True, alpha=0.3, zorder=0)
ax.set_xlim(0, 50)
ax.set_ylim(0, 900)
ax.legend(title="tier", loc="lower right", fontsize=10, framealpha=0.9)

out_path = "/home/sky/RL/SkyZero/SkyZero_V7.1/network_arch/katago_nbt_b_vs_c.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
