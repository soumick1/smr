"""Backbone abstraction: any frozen geometry model that maps K images to
poses / intrinsics / depth / pointmaps / features plugs in here.

The scaffold and memory consume ONLY this dataclass, which is what makes
the mechanism backbone-agnostic (plan, Sec. Portability).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_REGISTRY: dict = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_backbone(name, **kw):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backbone '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


@dataclass
class BackboneOutput:
    poses: np.ndarray                   # (K, 4, 4) world-from-camera
    intrinsics: np.ndarray              # (3, 3)
    depth: np.ndarray                   # (K, H, W)
    rgb: np.ndarray                     # (K, H, W, 3)
    mask: np.ndarray                    # (K, H, W) bool
    features: Optional[np.ndarray] = None    # (K, P, d_f) or None
    extras: dict = field(default_factory=dict)

    def descriptor(self, i, dim=448):
        """Pooled per-view descriptor, deterministic.  Placeholder for the
        plan's PCA-pooled backbone features (adapter v2); built from cues
        that survive corruption and view symmetry: block-MEAN thumbnails
        (colour 8x8, grey 12x12, per-descriptor mean removed) plus
        per-channel 16-bin histograms (which walls are visible)."""
        img = self.rgb[i]

        def blockmean(a, gy, gx):
            ye = np.linspace(0, a.shape[0], gy + 1).astype(int)
            xe = np.linspace(0, a.shape[1], gx + 1).astype(int)
            out = np.empty((gy, gx) + a.shape[2:])
            for r in range(gy):
                for c in range(gx):
                    out[r, c] = a[ye[r]:ye[r + 1], xe[c]:xe[c + 1]].mean((0, 1))
            return out

        # v2 (validated): natural part magnitudes act as implicit weights;
        # per-part unit-normalisation and a depth channel were tried and
        # REJECTED by measurement -- depth/grey layouts are common across
        # ring views and equal weighting injects a shared component that
        # collapses inter-view separation (97.5% -> 64.5% direct reloc).
        col = blockmean(img, 8, 8).ravel()
        col = col - col.mean()
        grey = blockmean(img.mean(-1, keepdims=True), 12, 12).ravel()
        grey = grey - grey.mean()
        hist = np.concatenate([np.histogram(img[..., c], bins=16,
                                            range=(0, 1), density=True)[0]
                               for c in range(3)])
        s = np.concatenate([col, grey, hist / (np.linalg.norm(hist) + 1e-9)])
        out = np.zeros(dim)
        out[:min(dim, s.size)] = s[:dim]
        return out / (np.linalg.norm(out) + 1e-9)


class Backbone:
    """Interface: infer(images | scene handle) -> BackboneOutput."""

    name = "abstract"

    def infer(self, *a, **kw) -> BackboneOutput:      # pragma: no cover
        raise NotImplementedError
