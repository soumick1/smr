"""The full scaffold: three HD rings + M grid modules, HD-gated.

State xi = (yaw, pitch, roll, {phi_m in R^3}_{m=1..M})  -- dimension 3 + 3M.
Placement is by clamp input (anchoring); motion is by conjunctive drive; the
gate rotates egocentric velocity by the *rings' own decoded orientation*
(Eq. gate in the plan), so the integrated trajectory is a property of the
dynamics, never of an external pose.

Position decode is the chained residue decode: coarse-to-fine over modules,
valid within a window of +-lambda_max/2 per axis (scenes are normalised well
inside this).  The CRT-fragility demonstration lives in tests, on a toy
config, as specified in the plan (Prop. range / Rem. fragility).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.geometry import euler_zyx_to_R, R_to_euler_zyx, wrap_pi
from .fields import Ring, Torus2D


@dataclass
class GridModule:
    lam: float                # spatial period (scene units)
    torus: Torus2D            # horizontal (x, y)
    zring: Ring               # height (z)

    def phases(self):
        px, py = self.torus.decode()
        return np.array([px, py, self.zring.decode()])

    def place(self, pos_xyz, offset=np.zeros(3), steps=250):
        ph = (2 * np.pi / self.lam) * np.asarray(pos_xyz) + offset
        self.torus.place((ph[0] % (2 * np.pi), ph[1] % (2 * np.pi)), steps=steps)
        self.zring.place(ph[2] % (2 * np.pi), steps=steps)

    def step(self, v_world, dt_scale=1.0, noise=0.0):
        r = (2 * np.pi / self.lam) * np.asarray(v_world)   # rad/t per axis
        self.torus.step((r[0], r[1]), noise=noise)
        self.zring.step(r[2], noise=noise)


class ScaffoldState:
    """Rings + modules + gate.  All fields share dt."""

    def __init__(self, periods, ring_N=256, torus_N=48, seed=0,
                 omega_max=None, v_max=None):
        self.rings = {k: Ring(N=ring_N, seed=seed + i)
                      for i, k in enumerate(("yaw", "pitch", "roll"))}
        self.modules = [
            GridModule(lam=l,
                       torus=Torus2D(N=torus_N, seed=seed + 10 + m),
                       zring=Ring(N=torus_N, seed=seed + 20 + m))
            for m, l in enumerate(periods)
        ]
        self.dt = self.rings["yaw"].dt
        self.offsets = np.zeros((len(self.modules), 3))   # remapping, per scene
        self.omega_max = omega_max      # measured in T2; may be set later
        self.v_max = v_max

    # ------------------------------------------------------------ calibration
    def calibrate(self, verbose=False):
        s = self.rings["yaw"].calibrate()
        for k in ("pitch", "roll"):
            self.rings[k].v_scale = self.rings["yaw"].v_scale
        st = self.modules[0].torus.calibrate()
        sz = self.modules[0].zring.calibrate()
        for m in self.modules:
            m.torus.v_scale = self.modules[0].torus.v_scale
            m.zring.v_scale = self.modules[0].zring.v_scale
        if self.omega_max is None:
            self.omega_max = self.rings["yaw"].measure_vmax()
        if self.v_max is None:
            # translation speed limit set by the FINEST module's phase-rate
            # ceiling: v * 2*pi/lam_min <= omega_max; 0.6 safety factor
            lam_min = min(m.lam for m in self.modules)
            self.v_max = 0.6 * self.omega_max * lam_min / (2 * np.pi)
        if verbose:
            print(f"  calib: ring slope={s:.3f} torus={st:.3f} z={sz:.3f} "
                  f"omega_max={self.omega_max:.2f}")
        return dict(ring_slope=s, torus_slope=st, z_slope=sz,
                    omega_max=self.omega_max)

    # --------------------------------------------------------------- decoding
    def decode_euler(self):
        return np.array([wrap_pi(self.rings[k].decode())
                         for k in ("yaw", "pitch", "roll")])

    def decode_R(self):
        y, p, r = self.decode_euler()
        return euler_zyx_to_R(y, p, r)

    def phases(self):
        return np.stack([m.phases() for m in self.modules])   # (M, 3)

    def decode_position(self):
        """Chained coarse-to-fine residue decode (window +-lam_max/2)."""
        order = np.argsort([-m.lam for m in self.modules])
        ph = self.phases() - self.offsets
        x = np.zeros(3)
        first = True
        for j in order:
            lam = self.modules[j].lam
            frac = (ph[j] / (2 * np.pi))            # position / lam, mod 1
            if first:
                x = wrap_pi(2 * np.pi * frac) / (2 * np.pi) * lam
                first = False
            else:
                corr = frac - x / lam
                x = x + lam * ((corr + 0.5) % 1.0 - 0.5)
        return x

    def state(self):
        return np.concatenate([self.decode_euler(), self.phases().ravel()])

    # -------------------------------------------------------------- placement
    def place_pose(self, T, steps=250):
        y, p, r = R_to_euler_zyx(T[:3, :3])
        self.rings["yaw"].place(y % (2 * np.pi), steps=steps)
        self.rings["pitch"].place(p % (2 * np.pi), steps=steps)
        self.rings["roll"].place(r % (2 * np.pi), steps=steps)
        for m, mod in enumerate(self.modules):
            mod.place(T[:3, 3], offset=self.offsets[m], steps=steps)

    # ----------------------------------------------------------------- motion
    def step_ego(self, omega_body_rates_euler, v_ego, gate=True, noise=0.0):
        """One step: rings driven by Euler rates; modules by gated velocity."""
        for k, rate in zip(("yaw", "pitch", "roll"), omega_body_rates_euler):
            self.rings[k].step(rate, noise=noise)
        R = self.decode_R() if gate else np.eye(3)
        v_w = R @ np.asarray(v_ego)
        for mod in self.modules:
            mod.step(v_w, noise=noise)

    def drive_to(self, T_target, gate=True, speed_frac=0.8, tol_ang=0.02,
                 tol_pos=0.02, max_steps=6000, record=False,
                 snapshot_fn=None, snapshot_every=5):
        """Propagate bumps to a target pose.  Returns (n_steps, trace|None).
        snapshot_fn(self) is called every `snapshot_every` steps (for
        visualisation) and once on arrival."""
        wmax = speed_frac * (self.omega_max or 1.0)
        vmax = speed_frac * (self.v_max or 1.0)
        y_t, p_t, r_t = R_to_euler_zyx(T_target[:3, :3])
        target_e = np.array([y_t, p_t, r_t])
        trace = [] if record else None
        for n in range(max_steps):
            e = self.decode_euler()
            err_e = wrap_pi(target_e - e)
            x = self.decode_position()
            err_x = T_target[:3, 3] - x
            if np.all(np.abs(err_e) < tol_ang) and np.linalg.norm(err_x) < tol_pos:
                if record:
                    trace.append(self.state())
                if snapshot_fn is not None:
                    snapshot_fn(self)
                return n, (np.array(trace) if record else None)
            rates = np.clip(err_e / self.dt, -wmax, wmax)
            v_w_des = np.clip(err_x / self.dt, -vmax, vmax)
            v_ego = (self.decode_R().T @ v_w_des) if gate else v_w_des
            self.step_ego(rates, v_ego, gate=gate)
            if record and n % 5 == 0:
                trace.append(self.state())
            if snapshot_fn is not None and n % snapshot_every == 0:
                snapshot_fn(self)
        return max_steps, (np.array(trace) if record else None)
