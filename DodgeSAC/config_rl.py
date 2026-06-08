"""Config-file driven hyperparameters for the off-policy agents (Rainbow / QR-DQN /
SAC), same ``KEY=value`` ``.cfg`` style and precedence as the PPO / DQN folders::

    dataclass defaults  <  .cfg file  <  environment variables  <  --set CLI

    python train_value.py configs/rainbow.cfg
    python train_value.py configs/qrdqn.cfg --set lr=1e-4
    python train_sac.py   configs/sac.cfg   --set num_envs=32

Two dataclasses (:class:`ValueConfig` for Rainbow/QR-DQN, :class:`SACConfig`) share
one generic loader.  NOTE: no ``from __future__ import annotations`` — the cast
relies on ``dataclasses.fields()[i].type`` being the real type object.
"""

import os
from dataclasses import dataclass, fields, asdict


# --------------------------------------------------------------------------- #
# value-based: Rainbow + QR-DQN (algo switch)
# --------------------------------------------------------------------------- #
@dataclass
class ValueConfig:
    # --- which algorithm ---
    algo: str = "rainbow"            # rainbow (C51) | qrdqn (quantile)

    # --- environment (vector obs / discrete actions) ---
    obs_mode: str = "vector"         # vector | grid
    max_steps: int = 18000           # 10-min episode cap (match the PPO recipe)
    reward_scale: float = 1.0
    stationary_bonus: float = 0.005
    reverse_penalty: float = 0.0
    accel_penalty: float = 0.0       # anti-jitter: penalize ||mv_t - mv_{t-1}||
    jerk_penalty: float = 0.0        # penalize 2nd diff of the move command

    # --- network ---
    hidden: tuple = (256, 256)
    channels: tuple = (32, 64)       # grid obs only
    noisy: bool = True               # NoisyNet heads (rainbow default; qrdqn cfg turns it off)
    noisy_std: float = 0.5
    # distributional head sizes
    n_atoms: int = 51                # rainbow (C51)
    v_min: float = -10.0
    v_max: float = 40.0
    n_quantiles: int = 200           # qrdqn

    # --- off-policy core ---
    total_steps: int = 0             # total env steps; 0 = run until Ctrl-C
    num_envs: int = 64               # parallel collectors
    num_workers: int = 8             # processes to spread the envs over
    gamma: float = 0.99
    n_step: int = 3
    buffer_size: int = 400000
    learning_starts: int = 25000     # env steps of random play before training
    batch_size: int = 256
    train_freq: int = 1              # collector ticks (= num_envs env steps) between gradient bursts
    gradient_steps: int = 2          # gradient updates per train tick
    target_update_interval: int = 2000   # gradient steps between hard target-net copies
    # exploration (qrdqn with noisy=0): epsilon-greedy anneal
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 400000
    # PER
    per_alpha: float = 0.5
    per_beta0: float = 0.4
    per_beta_steps: int = 3000000
    per_eps: float = 1e-6

    # --- optimization ---
    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    max_grad_norm: float = 10.0
    resume: str = ""

    # --- logging / eval / io ---
    log_every: int = 20000           # env steps per log line
    eval_every: int = 200000         # env steps per evaluation
    eval_episodes: int = 10
    plot: bool = True
    runs_dir: str = "runs"
    run_name: str = "rainbow"
    seed: int = 0
    device: str = "auto"

    @property
    def tag(self):
        return self.run_name or self.algo

    @property
    def out_dir(self):
        return os.path.join(self.runs_dir, self.tag)

    def pretty(self):
        return "\n".join(f"  {k:20s} = {v}" for k, v in asdict(self).items())

    def to_cfg(self):
        return _to_cfg(self)


# --------------------------------------------------------------------------- #
# SAC (continuous)
# --------------------------------------------------------------------------- #
@dataclass
class SACConfig:
    # --- environment (vector obs / continuous actions) ---
    obs_mode: str = "vector"
    max_steps: int = 18000
    reward_scale: float = 1.0
    stationary_bonus: float = 0.005
    reverse_penalty: float = 0.01
    accel_penalty: float = 0.0       # anti-jitter: penalize ||mv_t - mv_{t-1}|| (TARGET if annealing)
    jerk_penalty: float = 0.0        # penalize 2nd diff of the move command (TARGET if annealing)
    speed_penalty: float = 0.0       # energy: penalize ||mv|| -> rest when safe, move when threatened
    center_weight: float = 0.0       # reward proximity to arena center (small)
    smooth_anneal_steps: int = 0     # ramp accel/jerk 0->target over this many env steps; 0 = full from start

    # --- network ---
    hidden: tuple = (256, 256)

    # --- SAC core ---
    total_steps: int = 0
    num_envs: int = 64
    num_workers: int = 8
    gamma: float = 0.99
    n_step: int = 1                  # SAC is classically 1-step; >1 also supported
    buffer_size: int = 1000000
    learning_starts: int = 25000
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 2
    tau: float = 0.005               # soft target update
    # entropy temperature (auto-tuned to target_entropy = -act_dim)
    autotune_alpha: bool = True
    alpha: float = 0.2               # used when autotune_alpha=0
    target_entropy_scale: float = 1.0

    # --- optimization ---
    lr: float = 3e-4
    adam_eps: float = 1e-8
    max_grad_norm: float = 0.0       # 0 = no clipping
    resume: str = ""

    # --- logging / eval / io ---
    log_every: int = 20000
    eval_every: int = 200000
    eval_episodes: int = 10
    plot: bool = True
    runs_dir: str = "runs"
    run_name: str = "sac"
    seed: int = 0
    device: str = "auto"

    @property
    def tag(self):
        return self.run_name or "sac"

    @property
    def out_dir(self):
        return os.path.join(self.runs_dir, self.tag)

    def pretty(self):
        return "\n".join(f"  {k:20s} = {v}" for k, v in asdict(self).items())

    def to_cfg(self):
        return _to_cfg(self)


# --------------------------------------------------------------------------- #
# TD3 (continuous, deterministic policy)
# --------------------------------------------------------------------------- #
@dataclass
class TD3Config:
    # --- environment (continuous) ---
    obs_mode: str = "vector"
    max_steps: int = 18000
    reward_scale: float = 1.0
    stationary_bonus: float = 0.005
    reverse_penalty: float = 0.01
    accel_penalty: float = 0.0
    jerk_penalty: float = 0.0
    speed_penalty: float = 0.0       # energy: penalize ||mv|| -> rest when safe

    # --- network ---
    hidden: tuple = (256, 256)

    # --- TD3 core ---
    total_steps: int = 0
    num_envs: int = 64
    num_workers: int = 8
    gamma: float = 0.99
    n_step: int = 1
    buffer_size: int = 1000000
    learning_starts: int = 25000
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 2
    tau: float = 0.005
    policy_delay: int = 2            # actor (and target) updates every N critic updates
    expl_noise: float = 0.1         # Gaussian action noise during collection
    target_noise: float = 0.2       # target-policy smoothing noise
    noise_clip: float = 0.5         # clip for the smoothing noise

    # --- optimization ---
    lr: float = 3e-4
    adam_eps: float = 1e-8
    max_grad_norm: float = 0.0
    resume: str = ""

    # --- logging / eval / io ---
    log_every: int = 20000
    eval_every: int = 200000
    eval_episodes: int = 10
    plot: bool = True
    runs_dir: str = "runs"
    run_name: str = "td3"
    seed: int = 0
    device: str = "auto"

    @property
    def tag(self):
        return self.run_name or "td3"

    @property
    def out_dir(self):
        return os.path.join(self.runs_dir, self.tag)

    def pretty(self):
        return "\n".join(f"  {k:20s} = {v}" for k, v in asdict(self).items())

    def to_cfg(self):
        return _to_cfg(self)


# --------------------------------------------------------------------------- #
# parsing (shared)
# --------------------------------------------------------------------------- #
def _to_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _to_tuple(s):
    out = []
    for p in str(s).replace(",", " ").split():
        f = float(p)
        out.append(int(f) if f.is_integer() else f)
    return tuple(out)


def _cast(field_type, raw):
    if field_type is bool:
        return _to_bool(raw)
    if field_type is int:
        return int(float(raw))
    if field_type is float:
        return float(raw)
    if field_type is tuple:
        return _to_tuple(raw)
    return str(raw)


def _clean_value(val):
    val = val.strip()
    if val[:1] in ("\"", "'"):
        q = val[0]
        end = val.find(q, 1)
        return val[1:end] if end != -1 else val[1:]
    if "#" in val:
        val = val.split("#", 1)[0]
    return val.strip()


def _to_cfg(cfg):
    lines = []
    for k, v in asdict(cfg).items():
        if isinstance(v, tuple):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, bool):
            v = int(v)
        lines.append(f"{k.upper()}={v}")
    return "\n".join(lines) + "\n"


def parse_cfg_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            out[key.strip()] = _clean_value(val)
    return out


def parse_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_config(cls, path=None, overrides=None):
    type_by_name = {f.name: f.type for f in fields(cls)}
    known = set(type_by_name)
    merged = {}

    if path:
        for k, v in parse_cfg_file(path).items():
            name = k.lower()
            if name not in known:
                print(f"[config] WARNING: unknown key {k!r} in {path} (ignored)")
                continue
            merged[name] = v

    for name in known:
        env = os.environ.get(name.upper())
        if env is not None:
            merged[name] = env

    for k, v in (overrides or {}).items():
        name = k.lower()
        if name not in known:
            print(f"[config] WARNING: unknown override {k!r} (ignored)")
            continue
        merged[name] = v

    kwargs = {}
    for name, raw in merged.items():
        kwargs[name] = _cast(type_by_name[name], raw) if isinstance(raw, str) else raw
    return cls(**kwargs)
