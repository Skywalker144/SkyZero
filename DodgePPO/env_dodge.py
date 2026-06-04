"""Channel-Dodge environment for PPO — Gym-compatible, headless, vectorizable.

Same faithful port of ``SkyZero/SkyZeroWeb/channel-dodge.html`` as the QR-DQN
version, but built for PPO:

* two **observation** modes
    - ``"vector"`` (default): a fixed-size egocentric feature vector of the K
      nearest threats (relative position / velocity / danger) + the player state
      — finer spatial precision than a coarse grid, and an MLP consumes it, which
      keeps thousands of parallel-env steps cheap.  (The coarse grid was the main
      thing capping the value-based agent's dodging skill.)
    - ``"grid"``: the ``(6, 13, 13)`` egocentric danger-field image (CNN), kept
      for apples-to-apples comparison with the QR-DQN run.
* two **action** modes
    - ``"discrete"`` (default): ``Discrete(9)`` unit move vectors (stay + 8 dirs).
    - ``"continuous"``: ``Box(-1, 1, (2,))`` raw move vector, clamped to the unit
      disk inside the env — the game's native control, for the finest steering.
* a small ``SyncVectorDodgeEnv`` that steps N envs, auto-resets on episode end,
  exposes the terminal observation for truncation bootstrapping, and accumulates
  per-episode score / survival / return for logging.

API (single env) follows Gymnasium::

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
# Gym / Gymnasium compatibility (optional dependency) — self-contained shims.
# --------------------------------------------------------------------------- #
try:
    import gymnasium as gym
    from gymnasium import spaces as _spaces

    _GYM = "gymnasium"
    Discrete = _spaces.Discrete
    Box = _spaces.Box
    _EnvBase = gym.Env
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore
        from gym import spaces as _spaces  # type: ignore

        _GYM = "gym"
        Discrete = _spaces.Discrete
        Box = _spaces.Box
        _EnvBase = gym.Env
    except Exception:
        _GYM = None

        class Discrete:
            def __init__(self, n):
                self.n = int(n)
                self.dtype = np.int64
                self._rng = np.random.default_rng()

            def sample(self):
                return int(self._rng.integers(self.n))

            def __repr__(self):
                return f"Discrete({self.n})"

        class Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype
                self._rng = np.random.default_rng()

            def sample(self):
                return self._rng.uniform(-1.0, 1.0, size=self.shape).astype(self.dtype)

            def __repr__(self):
                return f"Box(shape={self.shape}, dtype={self.dtype})"

        class _EnvBase:
            metadata = {"render_modes": ["ansi"]}

            def close(self):
                pass


# --------------------------------------------------------------------------- #
# Discrete action set — unit move vectors (diagonals normalized, like the game)
# --------------------------------------------------------------------------- #
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
ACTION_VECS = np.array([
    (0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0),
    (-_INV_SQRT2, -_INV_SQRT2), (_INV_SQRT2, -_INV_SQRT2),
    (-_INV_SQRT2, _INV_SQRT2), (_INV_SQRT2, _INV_SQRT2),
], dtype=np.float32)
ACTION_NAMES = ["stay", "up", "down", "left", "right",
                "up-left", "up-right", "down-left", "down-right"]
NUM_ACTIONS = len(ACTION_VECS)


# --------------------------------------------------------------------------- #
# Game constants (ported verbatim from channel-dodge.html)
# --------------------------------------------------------------------------- #
PLAYER_R = 11.0
PLAYER_SPEED = 270.0
MAX_HP = 100.0
REGEN_DELAY = 3.0
REGEN_RATE = 7.0
HIT_INVULN = 0.6

TYPES = {
    "bullet": {"score": 1, "dmg": 8.0,  "r": 6.0},
    "split":  {"score": 1, "dmg": 8.0,  "r": 6.0},
    "aimed":  {"score": 2, "dmg": 12.0, "r": 7.0,  "turn": 1.12},   # softened homing (was 1.6)
    "cannon": {"score": 3, "dmg": 24.0, "r": 16.0},
    "bomb":   {"score": 4, "dmg": 28.0, "r": 11.0},
    "laser":  {"score": 5, "dmg": 30.0, "r": 0.0},
}
HEAL_AMOUNT = 25.0
PICKUP_R = 13.0
PICKUP_LIFE = 9.0
BOMB_FUSE = 1.8
BOMB_FLASH = 0.2
LASER_HALFWIDTH = 13.0
LASER_CHARGE = 1.0
LASER_FIRE = 0.55

# Vector-observation layout / normalization
K_THREATS = 32           # nearest projectiles (bullet/split/aimed/cannon) fed to the policy
K_LASERS = 3             # active beams — dedicated slots (full line geometry, not a point)
K_BOMBS = 3              # live bombs — dedicated slots (blast radius + fuse, not a point)
K_PICKUPS = 2
THREAT_FEATS = 7         # rel_x, rel_y, rel_vx, rel_vy, danger, radius, present
LASER_FEATS = 6          # signed_perp, cos_ang, sin_ang, t_to_fire, t_firing, present
BOMB_FEATS = 5           # rel_x, rel_y, blast_radius, fuse_remaining, present
PICKUP_FEATS = 3         # rel_x, rel_y, present
PLAYER_FEATS = 8         # hp, invuln, posx, posy, wall L/R/U/D
VEC_DIM = (K_THREATS * THREAT_FEATS + K_LASERS * LASER_FEATS + K_BOMBS * BOMB_FEATS
           + K_PICKUPS * PICKUP_FEATS + PLAYER_FEATS)
POS_NORM = 250.0         # px -> ~[-1,1] for relative positions
VEL_NORM = 300.0
DMG_NORM = 30.0
R_NORM = 20.0            # projectile radius norm (max is cannon r=16)
BLAST_NORM = 150.0       # bomb blast radius norm (max blastR)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# --------------------------------------------------------------------------- #
# Single environment
# --------------------------------------------------------------------------- #
class ChannelDodgeEnv(_EnvBase):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(
        self,
        width=450.0,
        height=600.0,
        dt=1.0 / 30.0,
        obs_mode="vector",
        action_mode="discrete",
        grid=13,
        view=156.0,
        max_steps=4000,
        survive_bonus=0.02,
        score_weight=0.20,
        damage_weight=0.04,
        heal_weight=0.20,
        death_penalty=1.0,
        stationary_bonus=0.005,
        reverse_penalty=0.0,
        render_mode=None,
    ):
        super().__init__()
        if obs_mode not in ("vector", "grid"):
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        self.W = float(width)
        self.H = float(height)
        self.dt = float(dt)
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.G = int(grid)
        self.view = float(view)
        self.cell = (2.0 * self.view) / self.G
        self.Cgrid = 6
        self.max_steps = int(max_steps) if max_steps else None

        self.survive_bonus = float(survive_bonus)
        self.score_weight = float(score_weight)
        self.damage_weight = float(damage_weight)
        self.heal_weight = float(heal_weight)
        self.death_penalty = float(death_penalty)
        self.stationary_bonus = float(stationary_bonus)
        self.reverse_penalty = float(reverse_penalty)
        self.render_mode = render_mode

        self.rng = None
        if action_mode == "discrete":
            self.action_space = Discrete(NUM_ACTIONS)
        else:
            self.action_space = Box(-1.0, 1.0, (2,), np.float32)
        if obs_mode == "vector":
            self.observation_space = Box(-np.inf, np.inf, (VEC_DIM,), np.float32)
        else:
            self.observation_space = Box(0.0, np.inf, (self.Cgrid, self.G, self.G), np.float32)

        self.player = None
        self.projectiles = []
        self.lasers = []
        self.bombs = []
        self.pickups = []
        self._reset_state_vars()

    # -- core gym API ------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()
        self.player = {"x": self.W * 0.5, "y": self.H * 0.5,
                       "hp": MAX_HP, "sinceHit": 999.0, "invuln": 0.0}
        self.projectiles = []
        self.lasers = []
        self.bombs = []
        self.pickups = []
        self._reset_state_vars()
        return self._obs(), self._info()

    def _reset_state_vars(self):
        self.score = 0
        self.dodged = 0
        self.elapsed = 0.0
        self.steps = 0
        # initial timers match the game's startGame()
        self.spawnTimer = 0.6
        self.ringTimer = 2.5
        self.laserTimer = 5.0
        self.bombTimer = 7.0
        self.pickupTimer = 8.0
        self.recentKinds = []     # recent primary spawn kinds (anti-streak)
        self.lastRing = None      # last split-burst origin (anti-stack)
        self._dmg_step = 0.0
        self._score_step = 0
        self._heal_step = 0.0
        self._prev_move = (0.0, 0.0)   # last step's move vector (for reverse_penalty)

    def _action_to_move(self, action):
        if self.action_mode == "discrete":
            return ACTION_VECS[int(action)]
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        mag = float(math.hypot(a[0], a[1]))
        if mag > 1.0:
            a = a / mag                       # clamp to the unit disk (max speed)
        return a

    def step(self, action):
        self._dmg_step = 0.0
        self._score_step = 0
        self._heal_step = 0.0

        mv = self._action_to_move(action)
        self._update(mv)

        self.steps += 1
        terminated = self.player["hp"] <= 0.0
        truncated = self.max_steps is not None and self.steps >= self.max_steps

        # movement shaping (small, so dodging stays primary): reward holding still,
        # and penalize reversing direction — both target the high-frequency jitter.
        mvx, mvy = float(mv[0]), float(mv[1])
        stay_r = self.stationary_bonus * max(0.0, 1.0 - math.hypot(mvx, mvy))
        reversal = max(0.0, -(mvx * self._prev_move[0] + mvy * self._prev_move[1]))
        self._prev_move = (mvx, mvy)

        reward = (
            self.survive_bonus
            + self.score_weight * self._score_step
            - self.damage_weight * self._dmg_step
            + self.heal_weight * (self._heal_step / HEAL_AMOUNT)
            + stay_r
            - self.reverse_penalty * reversal
        )
        if terminated:
            reward -= self.death_penalty

        if self.render_mode == "ansi":
            print(self.render())
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info()

    def render(self):
        p = self.player
        return (f"t={self.elapsed:5.1f}s hp={p['hp']:5.1f} score={self.score:4d} "
                f"dodged={self.dodged:4d} threats={len(self.projectiles)}"
                f"+{len(self.lasers)}L+{len(self.bombs)}B steps={self.steps}")

    def close(self):
        pass

    # -- random helpers ----------------------------------------------------- #
    def _rand(self, a, b):
        return a + self.rng.random() * (b - a)

    def _edge_spawn(self):
        m = 24.0
        side = int(self.rng.integers(0, 4))
        if side == 0:
            return self._rand(0, self.W), -m
        if side == 1:
            return self.W + m, self._rand(0, self.H)
        if side == 2:
            return self._rand(0, self.W), self.H + m
        return -m, self._rand(0, self.H)

    # -- difficulty & spawning ---------------------------------------------- #
    def difficulty(self):
        return min(1.0 + self.elapsed / 50.0, 3.8)

    def _spawn_bullet(self, speed_mul):
        sx, sy = self._edge_spawn()
        p = self.player
        ang = math.atan2(p["y"] - sy, p["x"] - sx) + self._rand(-0.5, 0.5)
        spd = self._rand(150, 230) * speed_mul
        self.projectiles.append({"kind": "bullet", "x": sx, "y": sy,
                                 "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                                 "r": TYPES["bullet"]["r"], "life": None, "hitOnce": False})

    def _spawn_aimed(self, speed_mul):
        sx, sy = self._edge_spawn()
        p = self.player
        ang = math.atan2(p["y"] - sy, p["x"] - sx) + self._rand(-0.08, 0.08)
        spd = self._rand(190, 260) * speed_mul
        self.projectiles.append({"kind": "aimed", "x": sx, "y": sy,
                                 "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                                 "r": TYPES["aimed"]["r"], "life": 6.0, "hitOnce": False})

    def _spawn_cannon(self, speed_mul):
        sx, sy = self._edge_spawn()
        p = self.player
        ang = math.atan2(p["y"] - sy, p["x"] - sx) + self._rand(-0.2, 0.2)
        spd = self._rand(95, 140) * speed_mul
        self.projectiles.append({"kind": "cannon", "x": sx, "y": sy,
                                 "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                                 "r": TYPES["cannon"]["r"], "life": None, "hitOnce": False})

    def _spawn_ring(self, speed_mul):
        # Pick a burst origin not right on top of the previous one, so
        # consecutive split bursts don't stack in the same spot.
        cx = self._rand(self.W * 0.2, self.W * 0.8)
        cy = self._rand(self.H * 0.15, self.H * 0.5)
        min_sep = min(self.W, self.H) * 0.28
        for _ in range(6):
            cx = self._rand(self.W * 0.2, self.W * 0.8)
            cy = self._rand(self.H * 0.15, self.H * 0.5)
            if self.lastRing is None or \
                    math.hypot(cx - self.lastRing[0], cy - self.lastRing[1]) >= min_sep:
                break
        self.lastRing = (cx, cy)
        n = int(self._rand(8, 14))
        spd = self._rand(120, 170) * speed_mul
        off = self._rand(0, math.tau)
        for i in range(n):
            ang = off + (i / n) * math.tau
            self.projectiles.append({"kind": "split", "x": cx, "y": cy,
                                     "vx": math.cos(ang) * spd, "vy": math.sin(ang) * spd,
                                     "r": TYPES["split"]["r"], "life": None, "hitOnce": False})

    def _spawn_bomb(self):
        m = 56.0
        self.bombs.append({"x": self._rand(m, self.W - m), "y": self._rand(m, self.H - m),
                           "blastR": _clamp(min(self.W, self.H) * 0.30, 70, 150),
                           "fuse": BOMB_FUSE, "flash": 0.0, "phase": "fuse", "hitOnce": False})

    def _spawn_pickup(self):
        m = 36.0
        self.pickups.append({"x": self._rand(m, self.W - m),
                             "y": self._rand(m, self.H - m), "life": PICKUP_LIFE})

    def _spawn_laser(self):
        sx, sy = self._edge_spawn()
        p = self.player
        ang = math.atan2(p["y"] - sy, p["x"] - sx)
        self.lasers.append({"ox": sx, "oy": sy, "ang": ang,
                            "charge": LASER_CHARGE, "fire": LASER_FIRE,
                            "phase": "charge", "halfWidth": LASER_HALFWIDTH, "hitOnce": False})

    def _pick_spawn_kind(self, d, roll):
        """Primary spawn kind with the original mix, but break up long runs of
        the same kind (no more than 2 in a row) so the screen isn't flooded with
        one bullet type. Ring bursts are handled on their own timer, not here."""
        if d > 1.5 and roll < 0.23:
            kind = "cannon"
        elif roll < 0.56:
            kind = "aimed"
        else:
            kind = "bullet"
        rk = self.recentKinds
        if len(rk) >= 2 and rk[-1] == kind and rk[-2] == kind:
            alts = [k for k in ("bullet", "aimed", "cannon")
                    if k != kind and (k != "cannon" or d > 1.5)]
            kind = alts[int(self.rng.integers(len(alts)))]
        rk.append(kind)
        if len(rk) > 4:
            rk.pop(0)
        return kind

    def _run_director(self, dt):
        d = self.difficulty()
        speed_mul = 0.9 + d * 0.12

        self.spawnTimer -= dt
        if self.spawnTimer <= 0:
            self.spawnTimer = self._rand(0.55, 1.0) / (0.7 + d * 0.4)
            kind = self._pick_spawn_kind(d, self.rng.random())
            if kind == "cannon":
                self._spawn_cannon(speed_mul)
            elif kind == "aimed":
                self._spawn_aimed(speed_mul)
            else:
                self._spawn_bullet(speed_mul)
            if d > 1.8 and self.rng.random() < 0.5:
                self._spawn_bullet(speed_mul)
            if d > 3.0 and self.rng.random() < 0.5:
                self._spawn_bullet(speed_mul)

        # Split bursts fire on their own evenly-spaced cadence (a random per-tick
        # roll used to cluster several bursts back-to-back).
        if d > 2.2:
            self.ringTimer -= dt
            if self.ringTimer <= 0:
                self.ringTimer = self._rand(2.6, 3.6) / (0.55 + d * 0.18)
                self._spawn_ring(speed_mul)

        if d > 1.6:
            self.laserTimer -= dt
            if self.laserTimer <= 0:
                self.laserTimer = self._rand(3.2, 5.5) / (0.6 + d * 0.18)
                self._spawn_laser()

        if d > 1.4:
            self.bombTimer -= dt
            if self.bombTimer <= 0:
                self.bombTimer = self._rand(4.5, 7.5) / (0.7 + d * 0.12)
                self._spawn_bomb()

        self.pickupTimer -= dt
        if self.pickupTimer <= 0:
            self.pickupTimer = self._rand(9, 14)
            if len(self.pickups) < 2:
                self._spawn_pickup()

    # -- damage / scoring --------------------------------------------------- #
    def _hurt(self, dmg):
        p = self.player
        if p["invuln"] > 0:
            return False
        p["hp"] = max(0.0, p["hp"] - dmg)
        p["invuln"] = HIT_INVULN
        p["sinceHit"] = 0.0
        self._dmg_step += dmg
        return True

    def _add_score(self, n):
        self.score += n
        self.dodged += 1
        self._score_step += n

    @staticmethod
    def _dist_to_line(px, py, ox, oy, a):
        dx, dy = px - ox, py - oy
        return abs(dx * math.sin(a) - dy * math.cos(a))

    @staticmethod
    def _steer_toward(b, target_ang, max_turn, dt):
        cur = math.atan2(b["vy"], b["vx"])
        diff = target_ang - cur
        while diff > math.pi:
            diff -= math.tau
        while diff < -math.pi:
            diff += math.tau
        step = max_turn * dt
        diff = _clamp(diff, -step, step)
        nxt = cur + diff
        spd = math.hypot(b["vx"], b["vy"])
        b["vx"] = math.cos(nxt) * spd
        b["vy"] = math.sin(nxt) * spd

    # -- physics tick ------------------------------------------------------- #
    def _update(self, mv):
        dt = self.dt
        p = self.player
        self.elapsed += dt
        p["invuln"] = max(0.0, p["invuln"] - dt)
        p["sinceHit"] += dt

        p["x"] = _clamp(p["x"] + float(mv[0]) * PLAYER_SPEED * dt, PLAYER_R, self.W - PLAYER_R)
        p["y"] = _clamp(p["y"] + float(mv[1]) * PLAYER_SPEED * dt, PLAYER_R, self.H - PLAYER_R)

        if p["sinceHit"] >= REGEN_DELAY and p["hp"] < MAX_HP:
            p["hp"] = min(MAX_HP, p["hp"] + REGEN_RATE * dt)

        self._run_director(dt)

        margin = 40.0
        alive = []
        for b in self.projectiles:
            cfg = TYPES[b["kind"]]
            if b["kind"] == "aimed":
                self._steer_toward(b, math.atan2(p["y"] - b["y"], p["x"] - b["x"]),
                                   cfg["turn"], dt)
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            if b["life"] is not None:
                b["life"] -= dt
                if b["life"] <= 0:
                    if not b["hitOnce"]:
                        self._add_score(cfg["score"])
                    continue
            if not b["hitOnce"]:
                dx, dy = b["x"] - p["x"], b["y"] - p["y"]
                rr = b["r"] + PLAYER_R
                if dx * dx + dy * dy <= rr * rr:
                    if self._hurt(cfg["dmg"]):
                        b["hitOnce"] = True
            if (b["x"] < -margin or b["x"] > self.W + margin or
                    b["y"] < -margin or b["y"] > self.H + margin):
                if not b["hitOnce"]:
                    self._add_score(cfg["score"])
                continue
            alive.append(b)
        self.projectiles = alive

        laser_alive = []
        for L in self.lasers:
            if L["phase"] == "charge":
                L["charge"] -= dt
                if L["charge"] <= 0:
                    L["phase"] = "fire"
            else:
                L["fire"] -= dt
                if self._dist_to_line(p["x"], p["y"], L["ox"], L["oy"], L["ang"]) <= \
                        L["halfWidth"] + PLAYER_R:
                    if self._hurt(TYPES["laser"]["dmg"]):
                        L["hitOnce"] = True
                if L["fire"] <= 0:
                    if not L["hitOnce"]:
                        self._add_score(TYPES["laser"]["score"])
                    continue
            laser_alive.append(L)
        self.lasers = laser_alive

        bomb_alive = []
        for bm in self.bombs:
            if bm["phase"] == "fuse":
                bm["fuse"] -= dt
                if bm["fuse"] <= 0:
                    bm["phase"] = "flash"
                    bm["flash"] = BOMB_FLASH
                    dx, dy = p["x"] - bm["x"], p["y"] - bm["y"]
                    if dx * dx + dy * dy <= bm["blastR"] * bm["blastR"]:
                        if self._hurt(TYPES["bomb"]["dmg"]):
                            bm["hitOnce"] = True
            else:
                bm["flash"] -= dt
                if bm["flash"] <= 0:
                    if not bm["hitOnce"]:
                        self._add_score(TYPES["bomb"]["score"])
                    continue
            bomb_alive.append(bm)
        self.bombs = bomb_alive

        pick_alive = []
        for k in self.pickups:
            k["life"] -= dt
            dx, dy = k["x"] - p["x"], k["y"] - p["y"]
            rr = PICKUP_R + PLAYER_R
            if dx * dx + dy * dy <= rr * rr and p["hp"] < MAX_HP:
                heal = min(MAX_HP, p["hp"] + HEAL_AMOUNT) - p["hp"]
                p["hp"] += heal
                self._heal_step += heal
                continue
            if k["life"] <= 0:
                continue
            pick_alive.append(k)
        self.pickups = pick_alive

    # -- observations ------------------------------------------------------- #
    def _obs(self):
        return self._obs_vector() if self.obs_mode == "vector" else self._obs_grid()

    def _obs_vector(self):
        p = self.player
        px, py = p["x"], p["y"]

        out = np.zeros(VEC_DIM, dtype=np.float32)

        # --- nearest projectiles: rel pos / rel vel / danger / radius ---
        pts = []
        for b in self.projectiles:
            cfg = TYPES[b["kind"]]
            pts.append((b["x"] - px, b["y"] - py, b["vx"], b["vy"],
                        cfg["dmg"] / DMG_NORM, cfg["r"] / R_NORM))
        pts.sort(key=lambda q: q[0] * q[0] + q[1] * q[1])
        i = 0
        for q in pts[:K_THREATS]:
            out[i + 0] = _clamp(q[0] / POS_NORM, -1.0, 1.0)
            out[i + 1] = _clamp(q[1] / POS_NORM, -1.0, 1.0)
            out[i + 2] = _clamp(q[2] / VEL_NORM, -1.0, 1.0)
            out[i + 3] = _clamp(q[3] / VEL_NORM, -1.0, 1.0)
            out[i + 4] = q[4]
            out[i + 5] = q[5]
            out[i + 6] = 1.0
            i += THREAT_FEATS
        i = K_THREATS * THREAT_FEATS

        # --- lasers: full beam line (signed perpendicular distance + orientation
        # + charge/fire timing), nearest beam first. Not collapsed to a point. ---
        lz = []
        for L in self.lasers:
            ca, sa = math.cos(L["ang"]), math.sin(L["ang"])
            perp = (px - L["ox"]) * sa - (py - L["oy"]) * ca   # signed dist to beam line
            t_fire = L["charge"] if L["phase"] == "charge" else 0.0
            t_firing = L["fire"] if L["phase"] == "fire" else 0.0
            lz.append((abs(perp), perp, ca, sa, t_fire, t_firing))
        lz.sort(key=lambda q: q[0])
        for q in lz[:K_LASERS]:
            out[i + 0] = _clamp(q[1] / POS_NORM, -1.0, 1.0)
            out[i + 1] = q[2]
            out[i + 2] = q[3]
            out[i + 3] = _clamp(q[4] / LASER_CHARGE, 0.0, 1.0)
            out[i + 4] = _clamp(q[5] / LASER_FIRE, 0.0, 1.0)
            out[i + 5] = 1.0
            i += LASER_FEATS
        i = K_THREATS * THREAT_FEATS + K_LASERS * LASER_FEATS

        # --- bombs: rel pos + blast radius + fuse remaining, nearest first ---
        bz = []
        for bm in self.bombs:
            dx, dy = bm["x"] - px, bm["y"] - py
            fuse_rem = max(0.0, bm["fuse"]) if bm["phase"] == "fuse" else 0.0
            bz.append((dx * dx + dy * dy, dx, dy, bm["blastR"], fuse_rem))
        bz.sort(key=lambda q: q[0])
        for q in bz[:K_BOMBS]:
            out[i + 0] = _clamp(q[1] / POS_NORM, -1.0, 1.0)
            out[i + 1] = _clamp(q[2] / POS_NORM, -1.0, 1.0)
            out[i + 2] = _clamp(q[3] / BLAST_NORM, 0.0, 1.0)
            out[i + 3] = _clamp(q[4] / BOMB_FUSE, 0.0, 1.0)
            out[i + 4] = 1.0
            i += BOMB_FEATS
        i = K_THREATS * THREAT_FEATS + K_LASERS * LASER_FEATS + K_BOMBS * BOMB_FEATS

        picks = sorted(self.pickups, key=lambda k: (k["x"] - px) ** 2 + (k["y"] - py) ** 2)
        j = i
        for k in picks[:K_PICKUPS]:
            out[j + 0] = _clamp((k["x"] - px) / POS_NORM, -1.0, 1.0)
            out[j + 1] = _clamp((k["y"] - py) / POS_NORM, -1.0, 1.0)
            out[j + 2] = 1.0
            j += PICKUP_FEATS
        j = i + K_PICKUPS * PICKUP_FEATS

        out[j + 0] = p["hp"] / MAX_HP
        out[j + 1] = min(1.0, p["invuln"] / HIT_INVULN)
        out[j + 2] = px / self.W
        out[j + 3] = py / self.H
        out[j + 4] = _clamp(px / POS_NORM, 0.0, 1.0)              # dist to left wall
        out[j + 5] = _clamp((self.W - px) / POS_NORM, 0.0, 1.0)   # right
        out[j + 6] = _clamp(py / POS_NORM, 0.0, 1.0)              # top
        out[j + 7] = _clamp((self.H - py) / POS_NORM, 0.0, 1.0)   # bottom
        return out

    def _world_to_cell(self, wx, wy):
        half = (self.G - 1) / 2.0
        return (wx - self.player["x"]) / self.cell + half, (wy - self.player["y"]) / self.cell + half

    def _stamp(self, ch, wx, wy, wr, val):
        col, row = self._world_to_cell(wx, wy)
        rad = wr / self.cell
        c0 = max(0, int(math.floor(col - rad)))
        c1 = min(self.G - 1, int(math.ceil(col + rad)))
        r0 = max(0, int(math.floor(row - rad)))
        r1 = min(self.G - 1, int(math.ceil(row + rad)))
        thr = rad + 0.5
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if math.hypot(c - col, r - row) <= thr:
                    ch[r, c] += val

    def _obs_grid(self):
        p = self.player
        g = np.zeros((self.Cgrid, self.G, self.G), dtype=np.float32)
        contact, telegraph, closing, walls, pickups, hp_plane = g
        half = (self.G - 1) / 2.0
        for r in range(self.G):
            wy = p["y"] + (r - half) * self.cell
            for c in range(self.G):
                wx = p["x"] + (c - half) * self.cell
                if wx < 0 or wx > self.W or wy < 0 or wy > self.H:
                    walls[r, c] = 1.0

        def closing_speed(bx, by, vx, vy):
            dx, dy = p["x"] - bx, p["y"] - by
            d = math.hypot(dx, dy)
            if d < 1e-6:
                return 1.0
            return _clamp((dx * vx + dy * vy) / d / 300.0, 0.0, 1.0)

        for b in self.projectiles:
            cfg = TYPES[b["kind"]]
            self._stamp(contact, b["x"], b["y"], max(b["r"], self.cell * 0.4), cfg["dmg"] / 30.0)
            self._stamp(closing, b["x"], b["y"], b["r"], closing_speed(b["x"], b["y"], b["vx"], b["vy"]))
        for L in self.lasers:
            ch = contact if L["phase"] == "fire" else telegraph
            ca, sa = math.cos(L["ang"]), math.sin(L["ang"])
            span = self.view * 2.0 + 80.0
            n = int(span / (self.cell * 0.5)) + 1
            base = (self.player["x"] - L["ox"]) * ca + (self.player["y"] - L["oy"]) * sa
            for i in range(n):
                t = -span / 2.0 + i * (self.cell * 0.5)
                self._stamp(ch, L["ox"] + ca * (t + base), L["oy"] + sa * (t + base),
                            L["halfWidth"], TYPES["laser"]["dmg"] / 30.0)
        for bm in self.bombs:
            if bm["phase"] == "fuse":
                w = 0.3 + 0.7 * (1.0 - bm["fuse"] / BOMB_FUSE)
                self._stamp(telegraph, bm["x"], bm["y"], bm["blastR"], w)
            else:
                self._stamp(contact, bm["x"], bm["y"], bm["blastR"], TYPES["bomb"]["dmg"] / 30.0)
        for k in self.pickups:
            self._stamp(pickups, k["x"], k["y"], max(PICKUP_R, self.cell * 0.5), 1.0)
        hp_plane[:, :] = p["hp"] / MAX_HP
        return g

    def _info(self):
        p = self.player
        return {
            "score": int(self.score), "dodged": int(self.dodged),
            "hp": float(p["hp"]), "survived": float(self.elapsed),
            "steps": int(self.steps), "difficulty": float(self.difficulty()),
        }


# --------------------------------------------------------------------------- #
# Synchronous vector env: step N independent games, auto-reset, expose terminal
# observation for truncation bootstrapping + per-episode stats for logging.
# --------------------------------------------------------------------------- #
class SyncVectorDodgeEnv:
    def __init__(self, num_envs, **env_kwargs):
        self.num_envs = int(num_envs)
        self.envs = [ChannelDodgeEnv(**env_kwargs) for _ in range(self.num_envs)]
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.obs_shape = tuple(self.single_observation_space.shape)
        self._ep_ret = np.zeros(self.num_envs, dtype=np.float64)

    def reset(self, seed=None):
        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=(None if seed is None else seed + i))
            obs[i] = o
        self._ep_ret[:] = 0.0
        return obs

    def step(self, actions):
        n = self.num_envs
        obs = np.zeros((n, *self.obs_shape), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        terminations = np.zeros(n, dtype=bool)
        truncations = np.zeros(n, dtype=bool)
        final_obs = [None] * n           # real terminal obs (for truncation bootstrap)
        episodes = []                    # finished-episode stats for logging

        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(actions[i])
            rewards[i] = r
            self._ep_ret[i] += r
            terminations[i] = term
            truncations[i] = trunc
            if term or trunc:
                final_obs[i] = o
                episodes.append({"score": info["score"], "survived": info["survived"],
                                 "steps": info["steps"], "return": float(self._ep_ret[i])})
                self._ep_ret[i] = 0.0
                o, _ = e.reset()
            obs[i] = o

        infos = {"final_obs": final_obs, "terminations": terminations,
                 "truncations": truncations, "episodes": episodes}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Subprocess vector env: split the N envs across W worker processes so the pure-
# Python env stepping (the throughput bottleneck — the net is tiny and the GPU is
# nearly idle) runs on many cores at once.  Each worker owns a contiguous chunk
# of envs and runs them with a SyncVectorDodgeEnv internally, so one IPC round-
# trip carries a whole chunk (not one env) — that amortization is what makes it
# worthwhile.  The interface is identical to SyncVectorDodgeEnv, so train_ppo.py
# is agnostic to which one it uses.
#
# IMPORTANT: construct this BEFORE moving any torch model to CUDA. It forks, and
# the workers only ever touch NumPy (never torch/CUDA), so forking a
# CUDA-uninitialized parent keeps the children clean.
# --------------------------------------------------------------------------- #
import multiprocessing as _mp


def _subproc_worker(remote, parent_remote, chunk_size, env_kwargs, base_seed):
    parent_remote.close()
    venv = SyncVectorDodgeEnv(chunk_size, **env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(venv.step(data))
            elif cmd == "reset":
                remote.send(venv.reset(seed=data))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        venv.close()


class SubprocVectorDodgeEnv:
    """Same API as SyncVectorDodgeEnv, but steps the envs across `num_workers`
    processes. Falls back to behaving like a single chunk if num_workers == 1."""

    def __init__(self, num_envs, num_workers, **env_kwargs):
        self.num_envs = int(num_envs)
        self.num_workers = max(1, min(int(num_workers), self.num_envs))
        # contiguous chunk boundaries (as even as possible)
        bounds = np.linspace(0, self.num_envs, self.num_workers + 1).astype(int)
        self._splits = [(int(bounds[i]), int(bounds[i + 1])) for i in range(self.num_workers)]

        # a throwaway env in the parent just to read the spaces (cheap, no fork hazard)
        probe = ChannelDodgeEnv(**env_kwargs)
        self.single_observation_space = probe.observation_space
        self.single_action_space = probe.action_space
        self.obs_shape = tuple(self.single_observation_space.shape)
        del probe

        ctx = _mp.get_context("fork")
        self.remotes, self.work_remotes, self.procs = [], [], []
        for w, (s, e) in enumerate(self._splits):
            remote, work_remote = ctx.Pipe()
            p = ctx.Process(
                target=_subproc_worker,
                args=(work_remote, remote, e - s, env_kwargs, 1000 * (w + 1)),
                daemon=True,
            )
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.work_remotes.append(work_remote)
            self.procs.append(p)
        self.closed = False

    def reset(self, seed=None):
        for w, (s, e) in enumerate(self._splits):
            self.remotes[w].send(("reset", (None if seed is None else seed + s)))
        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        for w, (s, e) in enumerate(self._splits):
            obs[s:e] = self.remotes[w].recv()
        return obs

    def step(self, actions):
        actions = np.asarray(actions)
        for w, (s, e) in enumerate(self._splits):
            self.remotes[w].send(("step", actions[s:e]))

        obs = np.zeros((self.num_envs, *self.obs_shape), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        final_obs = [None] * self.num_envs
        episodes = []
        for w, (s, e) in enumerate(self._splits):
            o, r, term, trunc, infos = self.remotes[w].recv()
            obs[s:e] = o
            rewards[s:e] = r
            terminations[s:e] = term
            truncations[s:e] = trunc
            final_obs[s:e] = infos["final_obs"]
            episodes.extend(infos["episodes"])
        infos = {"final_obs": final_obs, "terminations": terminations,
                 "truncations": truncations, "episodes": episodes}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self.closed = True


def make_vec_env(num_envs, num_workers=1, **env_kwargs):
    """Pick the synchronous (1 process) or subprocess (many processes) vector env."""
    if int(num_workers) <= 1:
        return SyncVectorDodgeEnv(num_envs, **env_kwargs)
    return SubprocVectorDodgeEnv(num_envs, num_workers, **env_kwargs)


# --------------------------------------------------------------------------- #
# Smoke test: random + flee baselines for both obs modes
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    def run(obs_mode, action_mode, policy, episodes=20, seed0=0):
        env = ChannelDodgeEnv(obs_mode=obs_mode, action_mode=action_mode)
        sc, sv = [], []
        for k in range(episodes):
            obs, info = env.reset(seed=seed0 + k)
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(policy(env))
                done = term or trunc
            sc.append(info["score"]); sv.append(info["survived"])
        return np.mean(sc), np.mean(sv)

    rng = np.random.default_rng(0)
    rand_disc = lambda env: int(rng.integers(NUM_ACTIONS))
    rand_cont = lambda env: rng.uniform(-1, 1, size=2).astype(np.float32)

    def flee_disc(env):
        p = env.player
        nearest, best = None, 1e18
        for b in env.projectiles:
            d = (b["x"] - p["x"]) ** 2 + (b["y"] - p["y"]) ** 2
            if d < best:
                best, nearest = d, b
        if nearest is None:
            return 0
        ax, ay = p["x"] - nearest["x"], p["y"] - nearest["y"]
        return int(max(range(NUM_ACTIONS), key=lambda a: ACTION_VECS[a][0] * ax + ACTION_VECS[a][1] * ay))

    print(f"Gym backend: {_GYM}  VEC_DIM={VEC_DIM}")
    e = ChannelDodgeEnv(obs_mode="vector")
    print(f"vector obs_shape={e.observation_space.shape}  grid obs_shape="
          f"{ChannelDodgeEnv(obs_mode='grid').observation_space.shape}")
    t0 = time.time()
    print("[vector/discrete  random] score={:.1f} surv={:.1f}s".format(*run("vector", "discrete", rand_disc)))
    print("[vector/discrete  flee  ] score={:.1f} surv={:.1f}s".format(*run("vector", "discrete", flee_disc)))
    print("[grid/discrete    flee  ] score={:.1f} surv={:.1f}s".format(*run("grid", "discrete", flee_disc)))
    print("[vector/continuous random] score={:.1f} surv={:.1f}s".format(*run("vector", "continuous", rand_cont)))
    # vector env smoke
    venv = SyncVectorDodgeEnv(4, obs_mode="vector")
    o = venv.reset(seed=0)
    for _ in range(50):
        o, r, te, tr, info = venv.step([venv.single_action_space.sample() for _ in range(4)])
    print(f"vec env ok: obs={o.shape} finished_eps_in_50steps={len(info['episodes'])}")
    print(f"({time.time() - t0:.1f}s)")
