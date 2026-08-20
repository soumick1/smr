"""Remapping offsets, novelty detector, neocortical store."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def remap_offsets(context_id, M):
    """Deterministic per-module phase offsets in [0, 2pi)^{M x 3}."""
    hsh = hashlib.sha256(str(context_id).encode()).digest()
    seed = int.from_bytes(hsh[:8], "little")
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, size=(M, 3))


class Novelty:
    """Loop-reconstruction familiarity: score(g) = 1 - cos(g, W_hg h(g)).

    Formed states reconstruct themselves (score ~ 0); anything off the
    formed manifold reconstructs poorly (score ~ 1).  This is the
    hippocampal mismatch signal: it fires exactly for error components the
    scaffold can represent as "not a state", i.e. the detectable ones
    (Thm. selective) -- coherent shifts land on other formed states and
    are, by the same token, invisible to it."""

    def fit(self, scaffold):
        self.sc = scaffold
        return self

    def score(self, g):
        sc = self.sc
        gg = sc._renorm(g)
        ghat = sc._renorm(sc.W_hg @ sc.h_of(gg))
        c = float(gg @ ghat) / (np.linalg.norm(gg) * np.linalg.norm(ghat) + 1e-12)
        return 1.0 - c


@dataclass
class Store:
    """Indexed neocortical store + state table (xi_i, idx_i)."""
    states: list = field(default_factory=list)
    content: dict = field(default_factory=dict)

    def add(self, xi, payload):
        idx = len(self.states)
        self.states.append(np.asarray(xi, dtype=float))
        self.content[idx] = payload
        return idx

    def nearest(self, xi, k=4, w_pos=1.0):
        if not self.states:
            return []
        S = np.stack(self.states)
        d_a = np.linalg.norm(((S[:, :3] - xi[:3] + np.pi) % (2 * np.pi)) - np.pi, axis=1)
        d_p = np.linalg.norm(((S[:, 3:] - xi[3:] + np.pi) % (2 * np.pi)) - np.pi, axis=1)
        return list(np.argsort(d_a + w_pos * d_p)[:k])
