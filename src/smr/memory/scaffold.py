"""Field-phase block scaffold: the Vector-HaSH loop over analytic bump codes.

Joint states are encoded by *template* bumps at given module phases (forming
associations needs codes, not dynamics), formed over a contiguous 3-D grid,
and settled through h.  Rings are excluded by design: orientation
perturbations are tangent to the attractor set (Cor. base-blind) and are
corrected only by landmark clamps, never by the settle.
"""
from __future__ import annotations

import numpy as np

from ..utils import rng


class BlockScaffold:
    def __init__(self, periods, torus_N=32, N_h=1024, k=64, seed=0):
        self.periods = np.asarray(periods, dtype=float)
        self.M, self.Nt = len(periods), torus_N
        self.ax = 2 * np.pi * np.arange(torus_N) / torus_N
        self.dim_mod = torus_N * torus_N + torus_N        # torus + z-ring
        self.N_g = self.M * self.dim_mod
        self.N_h, self.k = N_h, k
        self.W_gh = rng(seed).standard_normal((N_h, self.N_g)) / np.sqrt(self.N_g)
        self.W_hg = np.zeros((self.N_g, N_h))
        self._formed_h = []
        self.sigma = 2.0 / 4.0        # template bump width/4, matches fields

    # --------------------------------------------------------------- encoding
    def _bump1d(self, phase):
        d = np.abs((self.ax - phase + np.pi) % (2 * np.pi) - np.pi)
        b = np.exp(-0.5 * (d / self.sigma) ** 2)
        return b / (np.linalg.norm(b) + 1e-12)

    def encode_phases(self, phases):                      # (M, 3) -> g
        parts = []
        for m in range(self.M):
            bx, by, bz = (self._bump1d(phases[m, i]) for i in range(3))
            parts.append(np.outer(bx, by).ravel())
            parts.append(bz)
        return np.concatenate(parts)

    def phases_of_pos(self, x, offsets=None):
        ph = (2 * np.pi / self.periods)[:, None] * np.asarray(x)[None, :]
        if offsets is not None:
            ph = ph + offsets
        return ph % (2 * np.pi)

    def encode_pos(self, x, offsets=None):
        return self.encode_phases(self.phases_of_pos(x, offsets))

    def decode_phases(self, g):
        ph = np.zeros((self.M, 3))
        for m in range(self.M):
            off = m * self.dim_mod
            tor = g[off:off + self.Nt ** 2].reshape(self.Nt, self.Nt)
            z = g[off + self.Nt ** 2: off + self.dim_mod]
            fx, fy = tor.sum(1), tor.sum(0)
            ph[m, 0] = np.angle(np.sum(fx * np.exp(1j * self.ax))) % (2 * np.pi)
            ph[m, 1] = np.angle(np.sum(fy * np.exp(1j * self.ax))) % (2 * np.pi)
            ph[m, 2] = np.angle(np.sum(z * np.exp(1j * self.ax))) % (2 * np.pi)
        return ph

    def decode_pos(self, g, offsets=None):
        """Chained coarse-to-fine residue decode (window +-max(period)/2)."""
        ph = self.decode_phases(g)
        if offsets is not None:
            ph = (ph - offsets) % (2 * np.pi)
        order = np.argsort(-self.periods)
        x = np.zeros(3)
        for j_i, j in enumerate(order):
            lam, frac = self.periods[j], ph[j] / (2 * np.pi)
            if j_i == 0:
                x = ((frac + 0.5) % 1.0 - 0.5) * lam
            else:
                x = x + lam * (((frac - x / lam) + 0.5) % 1.0 - 0.5)
        return x

    # ------------------------------------------------------------ hippocampus
    def h_of(self, g):
        a = self.W_gh @ g
        h = np.zeros_like(a)
        idx = np.argpartition(a, -self.k)[-self.k:]
        h[idx] = a[idx]
        return h / (np.linalg.norm(h) + 1e-12)

    def form_grid(self, extent, spacing, offsets=None, rule="ridge",
                  lam=1e-3):
        """Formation over a contiguous grid in [-extent, extent]^3.

        rule="ridge" (default): W_hg = G (H^T H + lam I)^{-1} H^T -- the
        pseudoinverse-form return, exact at formed states.  For the sparse
        binary h of the original Vector-HaSH the plain Hebbian outer-product
        sum approximates this (near-orthogonal codes); our h is continuous
        top-k, where neighbour overlaps make the Hebbian mean blur basins,
        so the ridge form is the faithful implementation of the same
        associative return.  rule="hebb" keeps the outer-product variant
        for comparison."""
        xs = np.arange(-extent, extent + 1e-9, spacing)
        G, H = [], []
        for x in xs:
            for y in xs:
                for z in xs:
                    g = self.encode_pos(np.array([x, y, z]), offsets)
                    G.append(g)
                    H.append(self.h_of(g))
        G = np.stack(G, axis=1)                    # (N_g, P)
        H = np.stack(H, axis=1)                    # (N_h, P)
        if rule == "ridge":
            P = G.shape[1]
            gram = H.T @ H + lam * np.eye(P)
            self.W_hg = G @ np.linalg.solve(gram, H.T)
        else:
            self.W_hg = (G @ H.T) / G.shape[1]
        self._formed_h = H.T.copy()
        self._formed_g = G
        return xs

    def settle(self, g, iters=15, damp=0.5):
        """Damped loop g <- renorm((1-damp) g + damp * W_hg h(g)).

        Role (measured, not assumed): with continuous template codes the
        ridge return acts as a smooth projector onto the affine span of
        formed codes, so the loop is a *detector* (its reconstruction
        residual separates on/off-manifold by ~3 orders of magnitude; see
        Novelty and T11/T13), not a large-amplitude corrector.  Correction
        proper is landmark re-anchoring (place), which the architecture
        makes mandatory for base components anyway (Cor. base-blind)."""
        for _ in range(iters):
            g_hat = self._renorm(self.W_hg @ self.h_of(g))
            g = self._renorm((1.0 - damp) * g + damp * g_hat)
        return g

    def _renorm(self, g):
        out = np.zeros_like(g)
        for m in range(self.M):
            off = m * self.dim_mod
            for sl in (slice(off, off + self.Nt ** 2),
                       slice(off + self.Nt ** 2, off + self.dim_mod)):
                out[sl] = g[sl] / (np.linalg.norm(g[sl]) + 1e-12)
        return out

    # --------------------------------------------- coherent/incoherent split
    def coherent_basis(self):
        """Orthonormal basis of the coherent subspace C in phase space."""
        Bc = np.zeros((3 * self.M, 3))
        for kk in range(3):
            for m in range(self.M):
                Bc[3 * m + kk, kk] = 2 * np.pi / self.periods[m]
        Q, _ = np.linalg.qr(Bc)
        return Q

    def split_phase_err(self, dphi):
        Q = self.coherent_basis()
        v = dphi.reshape(-1)
        c = Q @ (Q.T @ v)
        return c, v - c
