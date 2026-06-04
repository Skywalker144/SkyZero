"""Config-file driven hyperparameters for PPO (SkyZero-style ``KEY=value`` .cfg).

Same format / precedence as the DQN folder::

    dataclass defaults  <  .cfg file  <  environment variables  <  --set CLI

    python train_ppo.py configs/dodge.cfg
    python train_ppo.py configs/dodge.cfg --set lr=1e-3 --set num_envs=32
    LR=1e-3 python train_ppo.py configs/dodge.cfg

NOTE: do *not* add ``from __future__ import annotations`` here — casting relies on
``dataclasses.fields(Config)[i].type`` being the real type object.
"""

import os
from dataclasses import dataclass, fields, asdict


@dataclass
class Config:
    # --- environment ---
    obs_mode: str = "vector"        # vector | grid
    action_mode: str = "discrete"   # discrete | continuous
    max_steps: int = 4000           # episode truncation (~133s at 30 Hz); 0 = no limit
    reward_scale: float = 1.0
    stationary_bonus: float = 0.005 # per-step reward for holding still (anti-jitter); 0 = off
    reverse_penalty: float = 0.0    # penalty for reversing move direction (anti-oscillation); 0 = off

    # --- network ---
    hidden: tuple = (256, 256)      # MLP hidden sizes (vector) / head after conv (grid)
    channels: tuple = (32, 64)      # CNN conv channels (grid obs only)

    # --- PPO core ---
    total_steps: int = 0            # total env steps across all envs; 0 = run until Ctrl-C
    num_envs: int = 16              # parallel environments
    num_workers: int = 1            # processes to spread the envs over (1 = single-process SyncVecEnv)
    num_steps: int = 256            # rollout length per env (batch = num_envs * num_steps)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True         # clipped value loss
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True           # advantage normalization per minibatch
    target_kl: float = 0.0          # early-stop epoch if approx_kl exceeds this (0 = off)

    # --- optimization ---
    lr: float = 3e-4
    anneal_lr: bool = True
    resume: str = ""                # path to a checkpoint to load (model + optimizer) before training

    # --- logging / eval / io ---
    log_every: int = 1              # updates per log line
    eval_every: int = 50            # updates per evaluation
    eval_episodes: int = 20
    plot: bool = True
    runs_dir: str = "runs"          # per-run output dirs live under here (V7.1-style isolation)
    run_name: str = "dodge"
    seed: int = 0
    device: str = "auto"            # auto | cuda | cpu

    @property
    def tag(self):
        return self.run_name or "ppo"

    @property
    def out_dir(self):
        """This run's isolated output directory: <runs_dir>/<run_name>/.
        Holds best.pt / final.pt / train.csv / eval.csv / progress.png / run.cfg."""
        import os
        return os.path.join(self.runs_dir, self.tag)

    @property
    def batch_size(self):
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self):
        return int(self.batch_size // self.num_minibatches)

    def pretty(self):
        return "\n".join(f"  {k:18s} = {v}" for k, v in asdict(self).items())

    def to_cfg(self):
        """Serialize the resolved config back to KEY=value text (a reproducible
        snapshot, including any CLI/env overrides). Tuples render comma-joined."""
        lines = []
        for k, v in asdict(self).items():
            if isinstance(v, tuple):
                v = ", ".join(str(x) for x in v)
            elif isinstance(v, bool):
                v = int(v)
            lines.append(f"{k.upper()}={v}")
        return "\n".join(lines) + "\n"


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


def load_config(path=None, overrides=None):
    type_by_name = {f.name: f.type for f in fields(Config)}
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
    return Config(**kwargs)


def parse_overrides(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out
