"""Recurrent kernels for Amari-type fields.

Operational kernels are cosine-type (Ben-Yishai et al. 1995; Zhang 1996):
    ring : w(t)   = J0 + J1 * cos(t)
    torus: w(x,y) = J0 + J1 * (cos(x) + cos(y)) / 2
with J0 < 0 global inhibition and J1 > 1/pi so the first harmonic is
pattern-forming.  With f = ReLU the ring bump has a CLOSED-FORM width:
writing the bump as u(t) = A (cos t - cos tc) on |t| < tc, self-consistency
of the first harmonic gives
    F1(tc) = tc - sin(tc) cos(tc) = 1 / J1,
independent of the uniform drive -- `ring_width_closed_form` solves it and
T1 checks the simulation against it.  The uniform drive sets the amplitude:
    A = -c / (cos tc + J0 * F0(tc)),   F0 = 2 (sin tc - tc cos tc),
with c = B + h the net uniform input (must satisfy the sign condition,
verified at construction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CosineParams:
    J0: float = -0.10     # uniform (global) inhibition
    J1: float = 0.55      # first-harmonic excitation; pattern iff J1 > 1/pi


def F1(tc):
    return tc - np.sin(tc) * np.cos(tc)


def F0(tc):
    return 2.0 * (np.sin(tc) - tc * np.cos(tc))


def ring_width_closed_form(p: CosineParams) -> float:
    """Full bump width 2*tc from F1(tc) = 1/J1.  Requires J1 > 1/pi."""
    if p.J1 <= 1.0 / np.pi:
        raise ValueError("J1 <= 1/pi: no pattern-forming harmonic.")
    tc = brentq(lambda t: F1(t) - 1.0 / p.J1, 1e-6, np.pi - 1e-6)
    return 2.0 * float(tc)


def ring_amplitude_closed_form(p: CosineParams, c_uniform: float) -> float:
    tc = ring_width_closed_form(p) / 2.0
    denom = np.cos(tc) + p.J0 * F0(tc)
    A = -c_uniform / denom
    if A <= 0:
        raise ValueError("Amplitude sign condition violated; adjust J0/B/h.")
    return float(A)


def ring_kernel(N: int, p: CosineParams):
    t = 2 * np.pi * np.arange(N) / N
    return p.J0 + p.J1 * np.cos(t)


def torus_kernel(N: int, p: CosineParams):
    t = 2 * np.pi * np.arange(N) / N
    cx, cy = np.meshgrid(np.cos(t), np.cos(t), indexing="ij")
    return p.J0 + p.J1 * 0.5 * (cx + cy)


# --------------------------------------------------------- legacy DoG (ref.)
@dataclass(frozen=True)
class DoGParams:
    A_e: float = 1.0
    A_i: float = 0.60
    sigma_e: float = 0.35
    sigma_i: float = 0.90


def dog(d, p: DoGParams):
    d = np.asarray(d, dtype=float)
    return (p.A_e * np.exp(-0.5 * (d / p.sigma_e) ** 2)
            - p.A_i * np.exp(-0.5 * (d / p.sigma_i) ** 2))


def design_from_width(width: float, c_uniform: float, kappa: float = 1.35):
    """CosineParams whose closed-form bump width equals `width` (radians).

    J1 from F1(tc) = 1/J1 at tc = width/2;  J0 = -kappa * cos(tc)/F0(tc)
    (kappa > 1 gives a negative amplitude denominator with margin).
    Returns (params, predicted_amplitude)."""
    tc = width / 2.0
    if not (0.0 < tc < np.pi):
        raise ValueError("width must be in (0, 2*pi)")
    J1 = 1.0 / F1(tc)
    J0 = -kappa * np.cos(tc) / F0(tc)
    p = CosineParams(J0=float(J0), J1=float(J1))
    A = ring_amplitude_closed_form(p, c_uniform)
    return p, A


def linear_gains(p: CosineParams):
    """(g0, g1) continuum Fourier gains: g0 = 2*pi*J0, g1 = pi*J1."""
    return 2 * np.pi * p.J0, np.pi * p.J1
