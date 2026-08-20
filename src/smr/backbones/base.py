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
        """Pooled per-view descriptor (fixed pooling; PCA refit per backbone
        on the server -- here: downsampled RGB, deterministic)."""
        img = self.rgb[i]
        s = img[::img.shape[0] // 12 or 1, ::img.shape[1] // 12 or 1].ravel()
        out = np.zeros(dim)
        out[:min(dim, s.size)] = s[:dim]
        n = np.linalg.norm(out)
        return out / (n + 1e-9)


class Backbone:
    """Interface: infer(images | scene handle) -> BackboneOutput."""

    name = "abstract"

    def infer(self, *a, **kw) -> BackboneOutput:      # pragma: no cover
        raise NotImplementedError
