"""One-shot heteroassociative binding by recursive least squares (primal).

Exact (equals the batch ridge solution), no learning rate, no epochs:
    k = C h / (1 + h^T C h);  W += (s - W h) k^T;  C -= k (h^T C).
Both directions are maintained (h -> s readout; s -> h cue path).
Scene boundaries call reset().
"""
from __future__ import annotations

import numpy as np


class RLSMemory:
    def __init__(self, N_h, N_s, lam=1e2):
        self.N_h, self.N_s, self.lam = N_h, N_s, lam
        self.reset()

    def reset(self):
        self.W_hs = np.zeros((self.N_s, self.N_h))
        self.W_sh = np.zeros((self.N_h, self.N_s))
        self.C = np.eye(self.N_h) * self.lam
        self.Cs = np.eye(self.N_s) * self.lam
        self.n = 0

    def write(self, h, s):
        k = self.C @ h / (1.0 + h @ self.C @ h)
        self.W_hs += np.outer(s - self.W_hs @ h, k)
        self.C -= np.outer(k, h @ self.C)
        ks = self.Cs @ s / (1.0 + s @ self.Cs @ s)
        self.W_sh += np.outer(h - self.W_sh @ s, ks)
        self.Cs -= np.outer(ks, s @ self.Cs)
        self.n += 1

    def read_s(self, h):
        return self.W_hs @ h

    def cue_h(self, s_tilde):
        return self.W_sh @ s_tilde
