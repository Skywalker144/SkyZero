"""Central hyperparameters for the 2048 Stochastic Gumbel AlphaZero pipeline.

Shared by net / mcts / selfplay / train / eval so the value scale and discount
stay consistent across the whole loop.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- network (simplified KataGo NestedBottleneck) ---
    num_planes: int = 16
    channels: int = 96          # c_main (trunk width)
    blocks: int = 6             # NestedBottleneck blocks  => "b6c96"
    # c_mid / c_gpool auto-derive from channels (like V7.1's model_config) when
    # left None; set explicitly only to override.
    c_mid: Optional[int] = None      # -> max(16, channels // 2)
    c_gpool: Optional[int] = None    # -> max(16, channels // 8)
    internal_length: int = 2    # InnerRes blocks per bottleneck
    value_hidden: int = 64

    # Value targets are discounted future score (raw 2048 points, reaching tens
    # of thousands). The value head regresses target / VALUE_SCALE for numeric
    # stability; mcts/selfplay multiply back to raw points so Q = reward + g*V
    # stays in one unit.
    value_scale: float = 4000.0

    # --- search ---
    gamma: float = 0.999
    num_simulations: int = 64
    c_puct: float = 1.25
    # Gumbel
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_noise: bool = True

    # --- selfplay ---
    games_per_iter: int = 200
    selfplay_batch: int = 64          # games stepped in parallel (batched NN eval)

    # --- training ---
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    train_steps_per_iter: int = 400
    replay_window: int = 400_000      # max rows kept in the replay buffer
    value_loss_weight: float = 1.0
    grad_clip: float = 4.0

    # --- loop ---
    num_iters: int = 40
    eval_games: int = 50
    device: str = "cuda"

    def __post_init__(self) -> None:
        # KataGo-style auto-derivation (model_config.py): fill width params from
        # the trunk channel count unless the user pinned them explicitly.
        if self.c_mid is None:
            self.c_mid = max(16, self.channels // 2)
        if self.c_gpool is None:
            self.c_gpool = max(16, self.channels // 8)


CONFIG = Config()


# --- V7.1-style network-name + env construction --------------------------
# scripts/run.sh sources run.cfg and exports every hyperparameter to the env;
# the per-network train/export/init scripts then build a Config from the
# network name ("b<blocks>c<channels>") plus those env vars. This mirrors
# SkyZero_V7.1/python/model_config.py's net_config_from_name / _from_env.

_NET_NAME_RE = re.compile(r"^\s*b(\d+)c(\d+)\s*$")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(float(v)) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def config_from_env(**overrides) -> Config:
    """Build a Config from environment variables (set by scripts/run.sh from
    run.cfg). C_MID / C_GPOOL left unset (0/absent) auto-derive from channels.
    `overrides` win and are applied at construction so __post_init__ derives
    widths from the FINAL channel count."""
    kw = dict(
        channels=_env_int("NUM_CHANNELS", Config.channels),
        blocks=_env_int("NUM_BLOCKS", Config.blocks),
        c_mid=(_env_int("C_MID", 0) or None),
        c_gpool=(_env_int("C_GPOOL", 0) or None),
        internal_length=_env_int("INTERNAL_LENGTH", Config.internal_length),
        value_hidden=_env_int("VALUE_HIDDEN", Config.value_hidden),
        value_scale=_env_float("VALUE_SCALE", Config.value_scale),
        gamma=_env_float("GAMMA", Config.gamma),
        num_simulations=_env_int("SIMS", Config.num_simulations),
        lr=_env_float("LR", Config.lr),
        batch_size=_env_int("BATCH_SIZE", Config.batch_size),
        weight_decay=_env_float("WEIGHT_DECAY", Config.weight_decay),
        grad_clip=_env_float("GRAD_CLIP", Config.grad_clip),
        eval_games=_env_int("EVAL_GAMES", Config.eval_games),
        device=os.environ.get("DEVICE", Config.device),
    )
    kw.update(overrides)
    return Config(**kw)


def config_from_name(name: str, **overrides) -> Config:
    """Parse a V7.1-style network name 'b<blocks>c<channels>' (e.g. 'b6c96')
    into a Config, pulling all other hyperparameters from the environment."""
    m = _NET_NAME_RE.match(name)
    if not m:
        raise ValueError(f"bad network name {name!r}; expected like 'b6c96'")
    return config_from_env(blocks=int(m.group(1)), channels=int(m.group(2)), **overrides)
