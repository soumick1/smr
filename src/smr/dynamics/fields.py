"""Cosine-kernel attractor fields with conjunctive (double-ring) velocity.

Dynamics (forward Euler, never differentiated):
    tau * du/dt = -u + (w (*) f(u)) + B*(1 + alpha * e_i . v) + h
with (*) circular convolution (FFT) and f = ReLU (soft cap).  The kernel is
w(t) = J0 + J1 cos t (ring) / J0 + J1 (cos x + cos y)/2 (torus), designed by
`kernels.design_from_width` so the ring bump width has the closed form
F1(width/2) = 1/J1 -- test T1 checks simulation against that prediction.

Velocity mechanism (Xie-Hahnloser-Seung / Zhang; Burak-Fiete on the torus):
neuron i carries preferred direction e_i; its OUTGOING kernel is shifted by
`shift` sites along e_i; its drive is modulated by (1 + alpha e_i . v).  At
v = 0 pools balance (stationary bump); v != 0 propagates the bump.  The sign
convention (roll(+shift * e)) is fixed by the directional unit test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import rng
from . import kernels as K


def _relu(u, cap=25.0):
    return np.clip(u, 0.0, cap)


def popvec_angle(f_vals, angles):
    z = np.sum(f_vals * np.exp(1j * angles))
    return float(np.angle(z) % (2 * np.pi))


def _euler_dt_ok(dt, tau, p: K.CosineParams):
    g0, g1 = K.linear_gains(p)
    lam = (1.0 + abs(g0) + g1) / tau          # conservative stiffness bound
    return dt <= 2.0 / lam * 0.6


# ============================================================== 1-D ring field
@dataclass
class Ring:
    N: int = 256
    dt: float = 0.05
    tau: float = 1.0
    B: float = 0.35
    h: float = 0.0
    alpha: float = 0.9
    shift: int | None = None   # auto: max(2, N//32)
    bump_width: float = 2.0          # target full width, radians
    kappa: float = 1.35
    seed: int = 0

    def __post_init__(self):
        self.dx = 2 * np.pi / self.N
        self.angles = 2 * np.pi * np.arange(self.N) / self.N
        self.params, self.A_coef = K.design_from_width(
            self.bump_width, self.B + self.h, self.kappa)
        self.A_pred = self.A_coef * (1.0 - np.cos(self.bump_width / 2.0))
        assert _euler_dt_ok(self.dt, self.tau, self.params), \
            f"dt={self.dt} unstable for gains {K.linear_gains(self.params)}"
        if self.shift is None:
            self.shift = max(2, self.N // 32)
        self._Kf = np.fft.rfft(K.ring_kernel(self.N, self.params))
        self.prefs = np.where(np.arange(self.N) % 2 == 0, 1.0, -1.0)
        self._masks = {c: (self.prefs == c).astype(float) for c in (+1.0, -1.0)}
        self.v_scale = 1.0
        self.ignite(rng(self.seed).uniform(0, 2 * np.pi))

    # ---------------------------------------------------------------- helpers
    def ignite(self, theta: float = 0.0):
        d = np.abs((self.angles - theta + np.pi) % (2 * np.pi) - np.pi)
        self.u = 0.8 * self.A_pred * np.exp(
            -0.5 * (d / (self.bump_width / 4)) ** 2) - 0.05
        return self

    def f(self, u):
        return _relu(u)

    def _recurrent(self, fu):
        out = np.zeros(self.N)
        for c, m in self._masks.items():
            conv = np.fft.irfft(np.fft.rfft(fu * m) * self._Kf, n=self.N)
            out += np.roll(conv, int(c) * self.shift)
        return out * self.dx

    # ------------------------------------------------------------------ steps
    def step(self, rate: float = 0.0, ext=None, noise: float = 0.0,
             g: np.random.Generator | None = None):
        v = self.v_scale * rate
        drive = self.B * (1.0 + self.alpha * self.prefs * v)
        inp = drive if ext is None else drive + ext
        du = (-self.u + self._recurrent(self.f(self.u)) + inp + self.h) / self.tau
        self.u = self.u + self.dt * du
        if noise > 0.0:
            gg = g if g is not None else np.random.default_rng()
            self.u += noise * np.sqrt(self.dt) * gg.standard_normal(self.N)
        return self.u

    def settle(self, steps: int = 300, **kw):
        for _ in range(steps):
            self.step(0.0, **kw)
        return self.u

    def decode(self) -> float:
        return popvec_angle(self.f(self.u), self.angles)

    def width(self) -> float:
        return float(np.count_nonzero(self.u > 0.0)) * self.dx

    def amplitude(self) -> float:
        return float(np.max(self.u))

    # ------------------------------------------------------------- operations
    def clamp_input(self, target: float, A_L: float = None, sigma_L: float = None):
        A_L = A_L if A_L is not None else 0.8 * self.A_pred
        sigma_L = sigma_L or self.bump_width / 4.0
        d = np.abs((self.angles - target + np.pi) % (2 * np.pi) - np.pi)
        return A_L * np.exp(-0.5 * (d / sigma_L) ** 2)

    def place(self, target: float, steps: int = 150):
        """Strong-clamp placement: the A_L -> large limit of landmark input,
        in which a bump re-forms at the target and the old one is
        extinguished by global inhibition.  `place_soft` keeps the
        moderate-input regime for the migrate-vs-reform study (T3)."""
        self.ignite(target)
        self.settle(steps)
        return self.decode()

    def place_soft(self, target: float, steps: int = 400, A_L: float = None):
        ext = self.clamp_input(target, A_L)
        for _ in range(steps):
            self.step(0.0, ext=ext)
        for _ in range(60):
            self.step(0.0)
        return self.decode()

    # ------------------------------------------------------------ calibration
    def _track(self, rate, T):
        prev, unw, t = self.decode(), 0.0, 0.0
        while t < T:
            self.step(rate)
            th = self.decode()
            unw += (th - prev + np.pi) % (2 * np.pi) - np.pi
            prev, t = th, t + self.dt
        return unw / T

    def calibrate(self, v_probe: float = 0.4, T: float = 50.0):
        self.v_scale = 1.0
        self.settle(400)
        slope = self._track(v_probe, T) / v_probe
        self.v_scale = 1.0 / slope
        return slope

    def measure_vmax(self, rates=None, T: float = 20.0, tol: float = 0.12):
        rates = rates if rates is not None else np.array([0.02,0.04,0.06,0.08,0.10,0.13,0.16,0.20,0.25,0.30,0.40,0.55,0.75,1.0])
        ok = 0.0
        for r in rates:
            self.settle(250)
            realized = self._track(r, T)
            if abs(realized - r) <= tol * r and self.amplitude() > 0.2 * self.A_pred:
                ok = r
            else:
                break
        return float(ok)


# ============================================================ 2-D torus field
@dataclass
class Torus2D:
    N: int = 48
    dt: float = 0.05
    tau: float = 1.0
    B: float = 0.35
    h: float = 0.0
    alpha: float = 0.9
    shift: int = 3
    bump_width: float = 2.0          # per-axis target (ring closed form)
    kappa: float = 1.35
    seed: int = 0

    DIRS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)

    def __post_init__(self):
        self.dx = 2 * np.pi / self.N
        ax = 2 * np.pi * np.arange(self.N) / self.N
        self.ang_x, self.ang_y = np.meshgrid(ax, ax, indexing="ij")
        self.params, self.A_coef = K.design_from_width(
            self.bump_width, self.B + self.h, self.kappa)
        self.A_pred = self.A_coef * (1.0 - np.cos(self.bump_width / 2.0))
        assert _euler_dt_ok(self.dt, self.tau, self.params)
        self._Kf = np.fft.rfft2(K.torus_kernel(self.N, self.params))
        idx = (np.add.outer(np.arange(self.N), np.arange(self.N))) % 4
        self._masks = [(idx == k).astype(float) for k in range(4)]
        self.v_scale = 1.0
        g0 = rng(self.seed)
        self.ignite((g0.uniform(0, 2 * np.pi), g0.uniform(0, 2 * np.pi)))

    def ignite(self, target_xy=(0.0, 0.0)):
        dxa = np.abs((self.ang_x - target_xy[0] + np.pi) % (2 * np.pi) - np.pi)
        dya = np.abs((self.ang_y - target_xy[1] + np.pi) % (2 * np.pi) - np.pi)
        self.u = 0.8 * self.A_pred * np.exp(
            -0.5 * (dxa ** 2 + dya ** 2) / (self.bump_width / 4) ** 2) - 0.05
        return self

    def f(self, u):
        return _relu(u)

    def _recurrent(self, fu):
        out = np.zeros_like(fu)
        for k, m in enumerate(self._masks):
            conv = np.fft.irfft2(np.fft.rfft2(fu * m) * self._Kf, s=fu.shape)
            sx, sy = (self.DIRS[k] * self.shift).astype(int)
            out += np.roll(np.roll(conv, sx, axis=0), sy, axis=1)
        return out * self.dx * self.dx

    def step(self, rate_xy=(0.0, 0.0), ext=None, noise: float = 0.0,
             g: np.random.Generator | None = None):
        v = self.v_scale * np.asarray(rate_xy, dtype=float)
        drive = np.zeros_like(self.u)
        for k, m in enumerate(self._masks):
            drive += m * self.B * (1.0 + self.alpha * float(self.DIRS[k] @ v))
        inp = drive if ext is None else drive + ext
        du = (-self.u + self._recurrent(self.f(self.u)) + inp + self.h) / self.tau
        self.u = self.u + self.dt * du
        if noise > 0.0:
            gg = g if g is not None else np.random.default_rng()
            self.u += noise * np.sqrt(self.dt) * gg.standard_normal(self.u.shape)
        return self.u

    def settle(self, steps: int = 300, **kw):
        for _ in range(steps):
            self.step((0.0, 0.0), **kw)
        return self.u

    def decode(self):
        fu = self.f(self.u)
        return (popvec_angle(fu, self.ang_x), popvec_angle(fu, self.ang_y))

    def width(self) -> float:
        area = float(np.count_nonzero(self.u > 0.0)) * self.dx * self.dx
        return 2.0 * np.sqrt(area / np.pi)

    def amplitude(self) -> float:
        return float(np.max(self.u))

    def clamp_input(self, target_xy, A_L: float = None, sigma_L: float = None):
        A_L = A_L if A_L is not None else 0.8 * self.A_pred
        sigma_L = sigma_L or self.bump_width / 4.0
        dxa = np.abs((self.ang_x - target_xy[0] + np.pi) % (2 * np.pi) - np.pi)
        dya = np.abs((self.ang_y - target_xy[1] + np.pi) % (2 * np.pi) - np.pi)
        return A_L * np.exp(-0.5 * (dxa ** 2 + dya ** 2) / sigma_L ** 2)

    def place(self, target_xy, steps: int = 150):
        self.ignite(target_xy)
        self.settle(steps)
        return self.decode()

    def place_soft(self, target_xy, steps: int = 400, A_L: float = None):
        ext = self.clamp_input(target_xy, A_L)
        for _ in range(steps):
            self.step((0.0, 0.0), ext=ext)
        for _ in range(60):
            self.step((0.0, 0.0))
        return self.decode()

    def calibrate(self, v_probe: float = 0.4, T: float = 50.0):
        self.v_scale = 1.0
        self.settle(400)
        prev, unw, t = self.decode()[0], 0.0, 0.0
        while t < T:
            self.step((v_probe, 0.0))
            th = self.decode()[0]
            unw += (th - prev + np.pi) % (2 * np.pi) - np.pi
            prev, t = th, t + self.dt
        slope = unw / T / v_probe
        self.v_scale = 1.0 / slope
        return slope
