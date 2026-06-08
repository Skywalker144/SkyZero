"""GPU-vectorized Channel-Dodge — all N envs stepped at once as torch tensors.

A faithful batched port of ``env_dodge.ChannelDodgeEnv._update``: same physics
constants, spawn distributions, collision rules, scoring and reward as the CPU env,
but every entity lives in a fixed-size padded buffer ``[N, slots]`` with an
``active`` mask, so thousands of envs step in pure tensor ops with NO Python
per-env loop, NO IPC, and NO CPU<->GPU transfer (obs/actions/state all stay on
device). This is what actually saturates throughput for small-net RL.

Differences vs the CPU env (intentional, documented):
* RNG is torch per-env (not numpy seed-identical) — same *dynamics distribution*,
  validated by matching random-policy score/survival distributions, not seedwise.
* the anti-streak (recentKinds) / ring anti-stack (lastRing) flavor is dropped
  (minor effect on the spawn mix).

API mirrors SyncVectorDodgeEnv but in torch:
    obs = venv.reset()                  # [N, VEC_DIM] float32 cuda
    obs, rew, term, trunc, infos = venv.step(actions)   # actions [N,2] cuda
    infos = {"final_obs", "episodes": [..]}  # auto-reset, final obs for bootstrap
"""

from __future__ import annotations

import math

import torch

from env_dodge import (PLAYER_R, PLAYER_SPEED, MAX_HP, REGEN_DELAY, REGEN_RATE,
                       HIT_INVULN, HEAL_AMOUNT, PICKUP_R, PICKUP_LIFE, BOMB_FUSE,
                       BOMB_FLASH, LASER_HALFWIDTH, LASER_CHARGE, LASER_FIRE,
                       K_THREATS, K_LASERS, K_BOMBS, K_PICKUPS, THREAT_FEATS,
                       LASER_FEATS, BOMB_FEATS, PICKUP_FEATS, PLAYER_FEATS, VEC_DIM,
                       POS_NORM, VEL_NORM, DMG_NORM, R_NORM, BLAST_NORM)

TAU = 2.0 * math.pi
# projectile kinds: 0 bullet, 1 split, 2 aimed, 3 cannon  (bomb/laser are separate buffers)
KIND_R = [6.0, 6.0, 7.0, 16.0]
KIND_DMG = [8.0, 8.0, 12.0, 24.0]
KIND_SCORE = [1.0, 1.0, 2.0, 3.0]
AIMED_TURN = 1.12
BOMB_DMG, BOMB_SCORE = 28.0, 4.0
LASER_DMG, LASER_SCORE = 30.0, 5.0


class VecDodgeGPU:
    def __init__(self, num_envs, device="cuda", *, width=450.0, height=600.0,
                 dt=1.0 / 30.0, max_steps=18000, P=160, L=4, B=4, K=4,
                 survive_bonus=0.02, score_weight=0.20, damage_weight=0.04,
                 heal_weight=0.20, death_penalty=1.0, stationary_bonus=0.005,
                 reverse_penalty=0.0, accel_penalty=0.0, jerk_penalty=0.0,
                 speed_penalty=0.0, center_weight=0.0, reward_scale=1.0):
        self.N = int(num_envs)
        self.dev = torch.device(device)
        self.W, self.H, self.dt = float(width), float(height), float(dt)
        self.max_steps = int(max_steps)
        self.P, self.L, self.B, self.K = P, L, B, K
        # reward shaping
        self.survive_bonus, self.score_weight = survive_bonus, score_weight
        self.damage_weight, self.heal_weight = damage_weight, heal_weight
        self.death_penalty, self.stationary_bonus = death_penalty, stationary_bonus
        self.reverse_penalty, self.accel_penalty = reverse_penalty, accel_penalty
        self.jerk_penalty, self.speed_penalty = jerk_penalty, speed_penalty
        self.center_weight, self.reward_scale = center_weight, reward_scale
        self.obs_dim = VEC_DIM

        self.kind_r = torch.tensor(KIND_R, device=self.dev)
        self.kind_dmg = torch.tensor(KIND_DMG, device=self.dev)
        self.kind_score = torch.tensor(KIND_SCORE, device=self.dev)

        self._alloc_state()

    # ----- allocation ----------------------------------------------------- #
    def _z(self, *shape):
        return torch.zeros(shape, device=self.dev)

    def _alloc_state(self):
        N, P, L, B, K = self.N, self.P, self.L, self.B, self.K
        d = self.dev
        # player
        self.px, self.py = self._z(N), self._z(N)
        self.hp = self._z(N)
        self.sinceHit, self.invuln = self._z(N), self._z(N)
        self.elapsed, self.steps = self._z(N), torch.zeros(N, dtype=torch.long, device=d)
        self.score = self._z(N)
        # timers
        self.spawnT, self.ringT = self._z(N), self._z(N)
        self.laserT, self.bombT, self.pickupT = self._z(N), self._z(N), self._z(N)
        # per-step accumulators
        self.dmg_step, self.score_step, self.heal_step = self._z(N), self._z(N), self._z(N)
        # movement history
        self.prev_mv, self.prev2_mv = self._z(N, 2), self._z(N, 2)
        # episode return
        self.ep_ret = self._z(N)
        # projectiles
        self.p_act = torch.zeros(N, P, dtype=torch.bool, device=d)
        self.p_kind = torch.zeros(N, P, dtype=torch.long, device=d)
        self.p_x, self.p_y = self._z(N, P), self._z(N, P)
        self.p_vx, self.p_vy = self._z(N, P), self._z(N, P)
        self.p_life = torch.full((N, P), float("inf"), device=d)
        self.p_hit = torch.zeros(N, P, dtype=torch.bool, device=d)
        # lasers
        self.l_act = torch.zeros(N, L, dtype=torch.bool, device=d)
        self.l_ox, self.l_oy, self.l_ang = self._z(N, L), self._z(N, L), self._z(N, L)
        self.l_charge, self.l_fire = self._z(N, L), self._z(N, L)
        self.l_phase = torch.zeros(N, L, dtype=torch.long, device=d)   # 0 charge, 1 fire
        self.l_hit = torch.zeros(N, L, dtype=torch.bool, device=d)
        # bombs
        self.b_act = torch.zeros(N, B, dtype=torch.bool, device=d)
        self.b_x, self.b_y = self._z(N, B), self._z(N, B)
        self.b_blast, self.b_fuse, self.b_flash = self._z(N, B), self._z(N, B), self._z(N, B)
        self.b_phase = torch.zeros(N, B, dtype=torch.long, device=d)   # 0 fuse, 1 flash
        self.b_hit = torch.zeros(N, B, dtype=torch.bool, device=d)
        # pickups
        self.k_act = torch.zeros(N, K, dtype=torch.bool, device=d)
        self.k_x, self.k_y, self.k_life = self._z(N, K), self._z(N, K), self._z(N, K)

    # ----- rng helpers ---------------------------------------------------- #
    def _rand(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(shape, device=self.dev) * (hi - lo) + lo

    def _edge_spawn(self):
        """Random edge spawn point per env -> (sx, sy) each [N]."""
        N, m = self.N, 24.0
        side = torch.randint(0, 4, (N,), device=self.dev)
        rx, ry = self._rand(N) * self.W, self._rand(N) * self.H
        sx = torch.where(side == 1, torch.full_like(rx, self.W + m),
             torch.where(side == 3, torch.full_like(rx, -m), rx))
        sy = torch.where(side == 0, torch.full_like(ry, -m),
             torch.where(side == 2, torch.full_like(ry, self.H + m), ry))
        return sx, sy

    # ----- reset ---------------------------------------------------------- #
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self._reset_mask(torch.ones(self.N, dtype=torch.bool, device=self.dev))
        return self._obs()

    def _reset_mask(self, m):
        """Reset the envs where m is True (used for initial + auto-reset)."""
        mf = m
        self.px[m], self.py[m] = self.W * 0.5, self.H * 0.5
        self.hp[m] = MAX_HP
        self.sinceHit[m], self.invuln[m] = 999.0, 0.0
        self.elapsed[m], self.steps[m], self.score[m] = 0.0, 0, 0.0
        self.spawnT[m], self.ringT[m] = 0.6, 2.5
        self.laserT[m], self.bombT[m], self.pickupT[m] = 5.0, 7.0, 8.0
        self.prev_mv[m], self.prev2_mv[m] = 0.0, 0.0
        self.ep_ret[m] = 0.0
        self.p_act[m] = False
        self.l_act[m] = False
        self.b_act[m] = False
        self.k_act[m] = False

    def difficulty(self):
        return torch.clamp(1.0 + self.elapsed / 50.0, max=3.8)

    # ----- spawn into padded slots --------------------------------------- #
    def _proj_slot(self, sel):
        """First free projectile slot per env; returns (do[N] bool, col[N] long)."""
        free = ~self.p_act
        has = free.any(1)
        col = free.float().argmax(1)
        return sel & has, col

    def _spawn_proj(self, sel, kind, sx, sy, vx, vy, life):
        do, col = self._proj_slot(sel)
        rows = do.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        c = col[rows]
        self.p_act[rows, c] = True
        self.p_kind[rows, c] = kind
        self.p_x[rows, c], self.p_y[rows, c] = sx[rows], sy[rows]
        self.p_vx[rows, c], self.p_vy[rows, c] = vx[rows], vy[rows]
        self.p_life[rows, c] = life[rows]
        self.p_hit[rows, c] = False

    def _spawn_edge_kind(self, sel, kind, ang_noise, spd_lo, spd_hi, speed_mul, life_val):
        """Spawn an edge projectile aimed at the player (+noise) for envs in sel."""
        if not bool(sel.any()):
            return
        sx, sy = self._edge_spawn()
        ang = torch.atan2(self.py - sy, self.px - sx) + self._rand(self.N, lo=-ang_noise, hi=ang_noise)
        spd = self._rand(self.N, lo=spd_lo, hi=spd_hi) * speed_mul
        life = torch.full((self.N,), life_val, device=self.dev)
        self._spawn_proj(sel, kind, sx, sy, torch.cos(ang) * spd, torch.sin(ang) * spd, life)

    def _run_director(self):
        dt = self.dt
        d = self.difficulty()
        speed_mul = 0.9 + d * 0.12
        inf = torch.full((self.N,), float("inf"), device=self.dev)

        # ---- primary spawn ----
        self.spawnT -= dt
        fired = self.spawnT <= 0
        roll = self._rand(self.N)
        is_cannon = (d > 1.5) & (roll < 0.23)
        is_aimed = (~is_cannon) & (roll < 0.56)
        is_bullet = (~is_cannon) & (~is_aimed)
        self._spawn_edge_kind(fired & is_bullet, 0, 0.5, 150, 230, speed_mul, float("inf"))
        self._spawn_edge_kind(fired & is_aimed, 2, 0.08, 190, 260, speed_mul, 6.0)
        self._spawn_edge_kind(fired & is_cannon, 3, 0.2, 95, 140, speed_mul, float("inf"))
        # extra bullets at higher difficulty
        extra1 = fired & (d > 1.8) & (self._rand(self.N) < 0.5)
        self._spawn_edge_kind(extra1, 0, 0.5, 150, 230, speed_mul, float("inf"))
        extra2 = fired & (d > 3.0) & (self._rand(self.N) < 0.5)
        self._spawn_edge_kind(extra2, 0, 0.5, 150, 230, speed_mul, float("inf"))
        self.spawnT = torch.where(fired, self._rand(self.N, lo=0.55, hi=1.0) / (0.7 + d * 0.4), self.spawnT)

        # ---- ring bursts ----
        self.ringT = torch.where(d > 2.2, self.ringT - dt, self.ringT)
        rfired = (d > 2.2) & (self.ringT <= 0)
        if bool(rfired.any()):
            cx = self._rand(self.N, lo=self.W * 0.2, hi=self.W * 0.8)
            cy = self._rand(self.N, lo=self.H * 0.15, hi=self.H * 0.5)
            n = torch.randint(8, 14, (self.N,), device=self.dev).float()
            spd = self._rand(self.N, lo=120, hi=170) * speed_mul
            off = self._rand(self.N, lo=0.0, hi=TAU)
            for k in range(13):
                selk = rfired & (k < n)
                ang = off + (k / torch.clamp(n, min=1.0)) * TAU
                self._spawn_proj(selk, 1, cx, cy, torch.cos(ang) * spd, torch.sin(ang) * spd, inf)
            self.ringT = torch.where(rfired, self._rand(self.N, lo=2.6, hi=3.6) / (0.55 + d * 0.18), self.ringT)

        # ---- lasers ----
        self.laserT = torch.where(d > 1.6, self.laserT - dt, self.laserT)
        lfired = (d > 1.6) & (self.laserT <= 0)
        if bool(lfired.any()):
            sx, sy = self._edge_spawn()
            ang = torch.atan2(self.py - sy, self.px - sx)
            free = ~self.l_act
            do = lfired & free.any(1)
            col = free.float().argmax(1)
            rows = do.nonzero(as_tuple=True)[0]
            if rows.numel():
                c = col[rows]
                self.l_act[rows, c] = True
                self.l_ox[rows, c], self.l_oy[rows, c], self.l_ang[rows, c] = sx[rows], sy[rows], ang[rows]
                self.l_charge[rows, c], self.l_fire[rows, c] = LASER_CHARGE, LASER_FIRE
                self.l_phase[rows, c], self.l_hit[rows, c] = 0, False
            self.laserT = torch.where(lfired, self._rand(self.N, lo=3.2, hi=5.5) / (0.6 + d * 0.18), self.laserT)

        # ---- bombs ----
        self.bombT = torch.where(d > 1.4, self.bombT - dt, self.bombT)
        bfired = (d > 1.4) & (self.bombT <= 0)
        if bool(bfired.any()):
            bx = self._rand(self.N, lo=56, hi=self.W - 56)
            by = self._rand(self.N, lo=56, hi=self.H - 56)
            blast = max(min(self.W, self.H) * 0.30, 70.0)
            blast = min(blast, 150.0)
            free = ~self.b_act
            do = bfired & free.any(1)
            col = free.float().argmax(1)
            rows = do.nonzero(as_tuple=True)[0]
            if rows.numel():
                c = col[rows]
                self.b_act[rows, c] = True
                self.b_x[rows, c], self.b_y[rows, c] = bx[rows], by[rows]
                self.b_blast[rows, c], self.b_fuse[rows, c] = blast, BOMB_FUSE
                self.b_phase[rows, c], self.b_hit[rows, c] = 0, False
            self.bombT = torch.where(bfired, self._rand(self.N, lo=4.5, hi=7.5) / (0.7 + d * 0.12), self.bombT)

        # ---- pickups ----
        self.pickupT -= dt
        pfired = (self.pickupT <= 0) & (self.k_act.sum(1) < 2)
        if bool(pfired.any()):
            kx = self._rand(self.N, lo=36, hi=self.W - 36)
            ky = self._rand(self.N, lo=36, hi=self.H - 36)
            free = ~self.k_act
            do = pfired & free.any(1)
            col = free.float().argmax(1)
            rows = do.nonzero(as_tuple=True)[0]
            if rows.numel():
                c = col[rows]
                self.k_act[rows, c] = True
                self.k_x[rows, c], self.k_y[rows, c], self.k_life[rows, c] = kx[rows], ky[rows], PICKUP_LIFE
        self.pickupT = torch.where(self.pickupT <= 0, self._rand(self.N, lo=9, hi=14), self.pickupT)

    # ----- single-hit damage helper (respects invuln, one source per step) - #
    def _apply_hit(self, env_can_hit, dmg_per):
        """env_can_hit[N] bool (already gated by invuln); dmg_per[N] damage to apply."""
        rows = env_can_hit.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return
        self.hp[rows] = torch.clamp(self.hp[rows] - dmg_per[rows], min=0.0)
        self.invuln[rows] = HIT_INVULN
        self.sinceHit[rows] = 0.0
        self.dmg_step[rows] += dmg_per[rows]

    # ----- physics step --------------------------------------------------- #
    def _update(self, mv):
        dt = self.dt
        self.dmg_step.zero_(); self.score_step.zero_(); self.heal_step.zero_()
        self.elapsed += dt
        self.invuln = torch.clamp(self.invuln - dt, min=0.0)
        self.sinceHit += dt
        # player move (mv already clamped to unit disk)
        self.px = torch.clamp(self.px + mv[:, 0] * PLAYER_SPEED * dt, PLAYER_R, self.W - PLAYER_R)
        self.py = torch.clamp(self.py + mv[:, 1] * PLAYER_SPEED * dt, PLAYER_R, self.H - PLAYER_R)
        # regen
        regen = (self.sinceHit >= REGEN_DELAY) & (self.hp < MAX_HP)
        self.hp = torch.where(regen, torch.clamp(self.hp + REGEN_RATE * dt, max=MAX_HP), self.hp)

        self._run_director()

        px, py = self.px[:, None], self.py[:, None]
        # ---- projectiles: aimed steering ----
        aimed = self.p_act & (self.p_kind == 2)
        if bool(aimed.any()):
            spd = torch.hypot(self.p_vx, self.p_vy).clamp(min=1e-6)
            cur = torch.atan2(self.p_vy, self.p_vx)
            tgt = torch.atan2(py - self.p_y, px - self.p_x)
            diff = (tgt - cur + math.pi) % TAU - math.pi
            step = torch.clamp(diff, -AIMED_TURN * dt, AIMED_TURN * dt)
            nxt = cur + step
            self.p_vx = torch.where(aimed, torch.cos(nxt) * spd, self.p_vx)
            self.p_vy = torch.where(aimed, torch.sin(nxt) * spd, self.p_vy)
        # move
        self.p_x = torch.where(self.p_act, self.p_x + self.p_vx * dt, self.p_x)
        self.p_y = torch.where(self.p_act, self.p_y + self.p_vy * dt, self.p_y)
        # life decrement (finite only)
        has_life = self.p_act & torch.isfinite(self.p_life)
        self.p_life = torch.where(has_life, self.p_life - dt, self.p_life)
        life_exp = has_life & (self.p_life <= 0)
        # radius/dmg/score per slot
        r = self.kind_r[self.p_kind]
        dmg = self.kind_dmg[self.p_kind]
        scr = self.kind_score[self.p_kind]
        # collision (only un-hit, active)
        dx, dy = self.p_x - px, self.p_y - py
        rr = r + PLAYER_R
        collide = self.p_act & (~self.p_hit) & ~life_exp & (dx * dx + dy * dy <= rr * rr)
        can = (self.invuln <= 0)[:, None] & collide
        dmg_slot = torch.where(can, dmg, torch.full_like(dmg, -1.0))
        best = dmg_slot.argmax(1)
        env_hit = can.any(1)
        self._apply_hit(env_hit, dmg.gather(1, best[:, None]).squeeze(1))
        hit_rows = env_hit.nonzero(as_tuple=True)[0]
        if hit_rows.numel():
            self.p_hit[hit_rows, best[hit_rows]] = True
        # out of bounds
        m = 40.0
        oob = self.p_act & ((self.p_x < -m) | (self.p_x > self.W + m) |
                            (self.p_y < -m) | (self.p_y > self.H + m))
        # despawn: life expired OR oob -> award score if not hitOnce
        despawn = life_exp | oob
        award = despawn & (~self.p_hit)
        self.score_step += (award.float() * scr).sum(1)
        self.p_act = self.p_act & (~despawn)

        # ---- lasers ----
        lox, loy = self.l_ox, self.l_oy
        ca, sa = torch.cos(self.l_ang), torch.sin(self.l_ang)
        # charge phase
        charging = self.l_act & (self.l_phase == 0)
        self.l_charge = torch.where(charging, self.l_charge - dt, self.l_charge)
        to_fire = charging & (self.l_charge <= 0)
        self.l_phase = torch.where(to_fire, torch.ones_like(self.l_phase), self.l_phase)
        # fire phase
        firing = self.l_act & (self.l_phase == 1)
        self.l_fire = torch.where(firing, self.l_fire - dt, self.l_fire)
        line_d = ((px - lox) * sa - (py - loy) * ca).abs()
        lcollide = firing & (~self.l_hit) & (line_d <= LASER_HALFWIDTH + PLAYER_R)
        lcan = (self.invuln <= 0)[:, None] & lcollide
        lenv = lcan.any(1)
        self._apply_hit(lenv, torch.full((self.N,), LASER_DMG, device=self.dev))
        lrows = lenv.nonzero(as_tuple=True)[0]
        if lrows.numel():
            lbest = lcan.float().argmax(1)
            self.l_hit[lrows, lbest[lrows]] = True
        ldesp = firing & (self.l_fire <= 0)
        self.score_step += (ldesp & (~self.l_hit)).float().sum(1) * LASER_SCORE
        self.l_act = self.l_act & (~ldesp)

        # ---- bombs ----
        fusing = self.b_act & (self.b_phase == 0)
        self.b_fuse = torch.where(fusing, self.b_fuse - dt, self.b_fuse)
        deton = fusing & (self.b_fuse <= 0)
        # blast check at detonation
        bdx, bdy = px - self.b_x, py - self.b_y
        inblast = deton & (bdx * bdx + bdy * bdy <= self.b_blast * self.b_blast)
        bcan = (self.invuln <= 0)[:, None] & inblast & (~self.b_hit)
        benv = bcan.any(1)
        self._apply_hit(benv, torch.full((self.N,), BOMB_DMG, device=self.dev))
        brows = benv.nonzero(as_tuple=True)[0]
        if brows.numel():
            bbest = bcan.float().argmax(1)
            self.b_hit[brows, bbest[brows]] = True
        self.b_phase = torch.where(deton, torch.ones_like(self.b_phase), self.b_phase)
        self.b_flash = torch.where(deton, torch.full_like(self.b_flash, BOMB_FLASH), self.b_flash)
        flashing = self.b_act & (self.b_phase == 1)
        self.b_flash = torch.where(flashing, self.b_flash - dt, self.b_flash)
        bdesp = flashing & (self.b_flash <= 0)
        self.score_step += (bdesp & (~self.b_hit)).float().sum(1) * BOMB_SCORE
        self.b_act = self.b_act & (~bdesp)

        # ---- pickups ----
        self.k_life = torch.where(self.k_act, self.k_life - dt, self.k_life)
        kdx, kdy = self.k_x - px, self.k_y - py
        krr = PICKUP_R + PLAYER_R
        touch = self.k_act & (kdx * kdx + kdy * kdy <= krr * krr) & (self.hp[:, None] < MAX_HP)
        # heal: one pickup per env is enough; apply total possible heal (cap at MAX)
        any_touch = touch.any(1)
        heal = torch.clamp(self.hp + HEAL_AMOUNT, max=MAX_HP) - self.hp
        self.hp = torch.where(any_touch, self.hp + heal, self.hp)
        self.heal_step += torch.where(any_touch, heal, torch.zeros_like(heal))
        # deactivate the (first) touched pickup per env + expired
        if bool(any_touch.any()):
            trows = any_touch.nonzero(as_tuple=True)[0]
            tcol = touch.float().argmax(1)
            self.k_act[trows, tcol[trows]] = False
        self.k_act = self.k_act & (self.k_life > 0)

    # ----- observation (vector) ------------------------------------------ #
    def _obs(self):
        N = self.N
        out = torch.zeros(N, VEC_DIM, device=self.dev)
        px, py = self.px[:, None], self.py[:, None]

        # threats: K nearest active projectiles
        dx, dy = self.p_x - px, self.p_y - py
        d2 = torch.where(self.p_act, dx * dx + dy * dy, torch.full_like(dx, float("inf")))
        kk = min(K_THREATS, self.P)
        _, idx = torch.topk(d2, kk, dim=1, largest=False)
        g = lambda t: t.gather(1, idx)
        present = torch.isfinite(g(d2))
        relx = torch.clamp(g(dx) / POS_NORM, -1, 1)
        rely = torch.clamp(g(dy) / POS_NORM, -1, 1)
        relvx = torch.clamp(g(self.p_vx) / VEL_NORM, -1, 1)
        relvy = torch.clamp(g(self.p_vy) / VEL_NORM, -1, 1)
        gd = (self.kind_dmg[self.p_kind] / DMG_NORM).gather(1, idx)
        gr = (self.kind_r[self.p_kind] / R_NORM).gather(1, idx)
        feat = torch.stack([relx, rely, relvx, relvy, gd, gr, present.float()], dim=2)
        feat = feat * present.float()[:, :, None]
        out[:, :K_THREATS * THREAT_FEATS] = feat.reshape(N, -1)
        off = K_THREATS * THREAT_FEATS

        # lasers: K_LASERS nearest beams by |signed perp|
        ca, sa = torch.cos(self.l_ang), torch.sin(self.l_ang)
        perp = (px - self.l_ox) * sa - (py - self.l_oy) * ca
        ld = torch.where(self.l_act, perp.abs(), torch.full_like(perp, float("inf")))
        kl = min(K_LASERS, self.L)
        _, li = torch.topk(ld, kl, dim=1, largest=False)
        lp = torch.isfinite(ld.gather(1, li))
        t_fire = torch.where(self.l_phase == 0, self.l_charge, torch.zeros_like(self.l_charge))
        t_firing = torch.where(self.l_phase == 1, self.l_fire, torch.zeros_like(self.l_fire))
        lfeat = torch.stack([
            torch.clamp(perp.gather(1, li) / POS_NORM, -1, 1),
            ca.gather(1, li), sa.gather(1, li),
            torch.clamp(t_fire.gather(1, li) / LASER_CHARGE, 0, 1),
            torch.clamp(t_firing.gather(1, li) / LASER_FIRE, 0, 1),
            lp.float()], dim=2) * lp.float()[:, :, None]
        out[:, off:off + K_LASERS * LASER_FEATS] = lfeat.reshape(N, -1)
        off += K_LASERS * LASER_FEATS

        # bombs: K_BOMBS nearest
        bdx, bdy = self.b_x - px, self.b_y - py
        bd2 = torch.where(self.b_act, bdx * bdx + bdy * bdy, torch.full_like(bdx, float("inf")))
        kb = min(K_BOMBS, self.B)
        _, bi = torch.topk(bd2, kb, dim=1, largest=False)
        bp = torch.isfinite(bd2.gather(1, bi))
        fuse_rem = torch.where(self.b_phase == 0, torch.clamp(self.b_fuse, min=0.0), torch.zeros_like(self.b_fuse))
        bfeat = torch.stack([
            torch.clamp(bdx.gather(1, bi) / POS_NORM, -1, 1),
            torch.clamp(bdy.gather(1, bi) / POS_NORM, -1, 1),
            torch.clamp(self.b_blast.gather(1, bi) / BLAST_NORM, 0, 1),
            torch.clamp(fuse_rem.gather(1, bi) / BOMB_FUSE, 0, 1),
            bp.float()], dim=2) * bp.float()[:, :, None]
        out[:, off:off + K_BOMBS * BOMB_FEATS] = bfeat.reshape(N, -1)
        off += K_BOMBS * BOMB_FEATS

        # pickups: K_PICKUPS nearest
        kdx, kdy = self.k_x - px, self.k_y - py
        kd2 = torch.where(self.k_act, kdx * kdx + kdy * kdy, torch.full_like(kdx, float("inf")))
        kp_k = min(K_PICKUPS, self.K)
        _, ki = torch.topk(kd2, kp_k, dim=1, largest=False)
        kp = torch.isfinite(kd2.gather(1, ki))
        kfeat = torch.stack([
            torch.clamp(kdx.gather(1, ki) / POS_NORM, -1, 1),
            torch.clamp(kdy.gather(1, ki) / POS_NORM, -1, 1),
            kp.float()], dim=2) * kp.float()[:, :, None]
        out[:, off:off + K_PICKUPS * PICKUP_FEATS] = kfeat.reshape(N, -1)
        off += K_PICKUPS * PICKUP_FEATS

        # player feats
        out[:, off + 0] = self.hp / MAX_HP
        out[:, off + 1] = torch.clamp(self.invuln / HIT_INVULN, max=1.0)
        out[:, off + 2] = self.px / self.W
        out[:, off + 3] = self.py / self.H
        out[:, off + 4] = torch.clamp(self.px / POS_NORM, 0, 1)
        out[:, off + 5] = torch.clamp((self.W - self.px) / POS_NORM, 0, 1)
        out[:, off + 6] = torch.clamp(self.py / POS_NORM, 0, 1)
        out[:, off + 7] = torch.clamp((self.H - self.py) / POS_NORM, 0, 1)
        return out

    # ----- public step ---------------------------------------------------- #
    def step(self, actions):
        a = actions
        mag = torch.hypot(a[:, 0], a[:, 1]).clamp(min=1e-8)
        scale = torch.clamp(mag, max=1.0) / mag
        mv = a * scale[:, None]                       # clamp to unit disk
        self._update(mv)
        self.steps += 1

        mvx, mvy = mv[:, 0], mv[:, 1]
        speed = torch.hypot(mvx, mvy)
        stay = self.stationary_bonus * torch.clamp(1.0 - speed, min=0.0)
        reversal = torch.clamp(-(mvx * self.prev_mv[:, 0] + mvy * self.prev_mv[:, 1]), min=0.0)
        ax, ay = mvx - self.prev_mv[:, 0], mvy - self.prev_mv[:, 1]
        accel = torch.hypot(ax, ay)
        pax, pay = self.prev_mv[:, 0] - self.prev2_mv[:, 0], self.prev_mv[:, 1] - self.prev2_mv[:, 1]
        jerk = torch.hypot(ax - pax, ay - pay)
        self.prev2_mv = self.prev_mv.clone()
        self.prev_mv = mv.clone()
        cdist = torch.hypot(self.px - self.W * 0.5, self.py - self.H * 0.5)
        center_r = self.center_weight * torch.clamp(1.0 - cdist / math.hypot(self.W * 0.5, self.H * 0.5), min=0.0)

        self.score += self.score_step
        terminated = self.hp <= 0.0
        truncated = self.steps >= self.max_steps

        reward = (self.survive_bonus + self.score_weight * self.score_step
                  - self.damage_weight * self.dmg_step
                  + self.heal_weight * (self.heal_step / HEAL_AMOUNT)
                  + stay - self.reverse_penalty * reversal
                  - self.accel_penalty * accel - self.jerk_penalty * jerk
                  - self.speed_penalty * speed + center_r)
        reward = torch.where(terminated, reward - self.death_penalty, reward) * self.reward_scale
        self.ep_ret += reward

        done = terminated | truncated
        obs_now = self._obs()
        final_obs = obs_now
        episodes = {"score": self.score.clone(), "survived": self.elapsed.clone(),
                    "ret": self.ep_ret.clone(), "done": done.clone()}
        if bool(done.any()):
            final_obs = obs_now.clone()
            self._reset_mask(done)
            obs_now = self._obs()                       # fresh obs for reset envs
        return obs_now, reward, terminated, truncated, {"final_obs": final_obs, "episodes": episodes}


# --------------------------------------------------------------------------- #
# parity test: random-policy score/survival distribution vs the CPU env
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    import numpy as np
    from env_dodge import ChannelDodgeEnv

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, STEPS = 2048, 2000
    torch.manual_seed(0)
    venv = VecDodgeGPU(N, device=dev, max_steps=18000)
    obs = venv.reset(seed=0)
    print(f"[gpu] obs {tuple(obs.shape)} dtype={obs.dtype} dev={obs.device}")

    sc_done, sv_done = [], []
    t0 = time.time()
    for _ in range(STEPS):
        a = (torch.rand(N, 2, device=dev) * 2 - 1)
        obs, r, term, trunc, info = venv.step(a)
        ep = info["episodes"]; dmask = ep["done"]
        if bool(dmask.any()):
            sc_done += ep["score"][dmask].tolist()
            sv_done += ep["survived"][dmask].tolist()
    dt = time.time() - t0
    sps = int(N * STEPS / dt)
    print(f"[gpu] {N} envs x {STEPS} steps in {dt:.1f}s -> {sps:,} sps  ({len(sc_done)} eps done)")
    if sc_done:
        print(f"[gpu] random-policy: score mean={np.mean(sc_done):.1f} surv mean={np.mean(sv_done):.1f}s")

    # CPU random baseline
    rng = np.random.default_rng(0)
    csc, csv = [], []
    for k in range(30):
        e = ChannelDodgeEnv(action_mode="continuous", max_steps=18000)
        o, _ = e.reset(seed=k)
        done = False
        while not done:
            o, rr, te, tr, inf = e.step(rng.uniform(-1, 1, 2).astype(np.float32))
            done = te or tr
        csc.append(inf["score"]); csv.append(inf["survived"])
    print(f"[cpu] random-policy: score mean={np.mean(csc):.1f} surv mean={np.mean(csv):.1f}s  (30 eps)")
