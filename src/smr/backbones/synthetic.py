"""Ground-truth synthetic backbone: renders the analytic room and returns
exact poses/depth, standing in for a frozen geometry model so every
integration test runs on CPU with known ground truth."""
from __future__ import annotations

import numpy as np

from ..scene import Camera, build_room, camera_ring, render
from .base import Backbone, BackboneOutput, register


@register("synthetic")
class SyntheticBackbone(Backbone):
    name = "synthetic"

    def __init__(self, seed=0, H=192, W=192, f=160.0):
        self.seed = seed
        self.cam = Camera(H=H, W=W, f=f)
        self.points, self.colors = build_room(seed=seed)

    def infer(self, n_views=8, radius=1.05, jitter=0.0) -> BackboneOutput:
        Ts = camera_ring(n_views, radius=radius, seed=self.seed, jitter=jitter)
        rgbs, deps, msks = [], [], []
        for T in Ts:
            r, d, m = render(self.points, self.colors, T, self.cam)
            rgbs.append(r); deps.append(d); msks.append(m)
        return BackboneOutput(poses=np.stack(Ts), intrinsics=self.cam.K,
                              depth=np.stack(deps), rgb=np.stack(rgbs),
                              mask=np.stack(msks))
