"""Faithful discrete Vector-HaSH (Chandra et al. 2025) for T9/T10/T14.

Grid modules are one-hot rings of coprime sizes p_m; g is their
concatenation (N_g = sum p_m); hippocampus h = theta(W_gh g) with fixed
random W_gh; the return W_hg is Hebbian over the joint grid states (or a
contiguous subset, for the strong-generalisation test); content attaches by
pseudoinverse.  The paper's Fig. 2-4 phenomenology is reproduced here before
the field version is trusted.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..utils import rng


def all_states(periods):
    for combo in itertools.product(*[range(p) for p in periods]):
        yield combo


def encode(combo, periods):
    g = np.zeros(sum(periods))
    off = 0
    for c, p in zip(combo, periods):
        g[off + c] = 1.0
        off += p
    return g


class ToyVectorHash:
    def __init__(self, periods, N_h, seed=0, theta=0.5):
        self.periods, self.N_h = list(periods), N_h
        self.N_g = sum(periods)
        self.W_gh = rng(seed).standard_normal((N_h, self.N_g)) / np.sqrt(self.N_g)
        self.theta = theta
        self.W_hg = np.zeros((self.N_g, N_h))

    def h_of(self, g):
        return (self.W_gh @ g > self.theta).astype(float)

    def form(self, combos=None):
        combos = list(combos) if combos is not None else \
            list(all_states(self.periods))
        for c in combos:
            g = encode(c, self.periods)
            h = self.h_of(g)
            n = float(h @ h)
            if n > 0:
                self.W_hg += np.outer(g - self.W_hg @ h / n, h) / n
        return self

    def g_round(self, g_hat):
        """Project onto the nearest one-hot-per-module code."""
        out, off = np.zeros_like(g_hat), 0
        for p in self.periods:
            out[off + np.argmax(g_hat[off:off + p])] = 1.0
            off += p
        return out

    def settle(self, g, iters=5):
        for _ in range(iters):
            g = self.g_round(self.W_hg @ self.h_of(g))
        return g

    def is_fixed(self, combo):
        g = encode(combo, self.periods)
        return np.array_equal(self.settle(g, iters=2), g)

    def frac_fixed(self, sample=400, seed=1):
        gg = rng(seed)
        combos = [tuple(gg.integers(0, p) for p in self.periods)
                  for _ in range(sample)]
        return float(np.mean([self.is_fixed(c) for c in combos]))


def nh_star(periods, seed=0, lo=4, hi=600, frac=0.99):
    """Smallest N_h with >= frac of joint states fixed (bisection)."""
    def ok(nh):
        return ToyVectorHash(periods, nh, seed=seed).form().frac_fixed() >= frac
    if not ok(hi):
        return hi
    while hi - lo > 4:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


# ------------------------------- heteroassociation & Hopfield control (T10)
def pseudoinverse_recall_error(N_h, P_list, dim_s=64, seed=0):
    g0 = rng(seed)
    errs = []
    for P in P_list:
        H = (g0.standard_normal((N_h, P)) > 0.8).astype(float)
        S = g0.standard_normal((dim_s, P))
        W = S @ np.linalg.pinv(H)
        errs.append(float(np.linalg.norm(W @ H - S) / np.linalg.norm(S)))
    return np.array(errs)


def hopfield_recall_error(N, P_list, seed=0, flips=0.05, iters=30):
    g0 = rng(seed)
    errs = []
    for P in P_list:
        X = np.sign(g0.standard_normal((N, P)))
        W = (X @ X.T) / N
        np.fill_diagonal(W, 0.0)
        ers = []
        for j in range(min(P, 20)):
            x = X[:, j].copy()
            x[g0.random(N) < flips] *= -1
            for _ in range(iters):
                x = np.sign(W @ x + 1e-9)
            ers.append(np.mean(x != X[:, j]))
        errs.append(float(np.mean(ers)))
    return np.array(errs)
