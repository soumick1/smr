"""A GPU-free world with REVISITS for validating the stitchers.

The analytic room is rendered along a two-lap ring trajectory: lap two
retraces lap one with small radial/angular jitter, so every frame of lap
two is a revisit of a lap-one viewpoint (the ground truth of a loop
closure).  Descriptors are computed from the RENDERED images with the
same function used on real captures, so anchor proposal is exercised
through appearance, not through an oracle.

Poses per pass come from SimulatedRunner (regauged GT + per-pass
similarity error), so chaining drifts and loop closure has something to
remove.  Nothing here is a result.
"""
from __future__ import annotations

import numpy as np

from ..scene import Camera, build_room, render
from ..utils.geometry import make_T
from .passes import SimulatedRunner, array_descriptors


def two_lap_trajectory(n_per_lap=32, laps=2, radius=1.05, height=1.0,
                       jitter=0.02, seed=0):
    """c2w poses on a ring looking at the room centre, `laps` times round."""
    g = np.random.default_rng(seed)
    Ts = []
    for lap in range(laps):
        for i in range(n_per_lap):
            a = 2 * np.pi * i / n_per_lap + jitter * g.standard_normal()
            r = radius * (1 + jitter * g.standard_normal())
            pos = np.array([r * np.cos(a), height + jitter * g.standard_normal(),
                            r * np.sin(a)])
            fwd = np.array([0.0, height, 0.0]) - pos
            fwd /= np.linalg.norm(fwd)
            up = np.array([0.0, -1.0, 0.0])
            right = np.cross(up, fwd); right /= np.linalg.norm(right)
            upo = np.cross(fwd, right)
            Ts.append(make_T(np.stack([right, upo, fwd], axis=1), pos))
    return np.stack(Ts)


class LoopWorld:
    def __init__(self, n_per_lap=32, laps=2, H=96, W=96, f=80.0, seed=0,
                 jitter=0.02):
        self.cam = Camera(H=H, W=W, f=f)
        self.points, self.colors = build_room(seed=seed)
        self.gt = two_lap_trajectory(n_per_lap, laps, jitter=jitter, seed=seed)
        self.n_per_lap, self.laps = n_per_lap, laps
        rgbs = [render(self.points, self.colors, T, self.cam)[0] for T in self.gt]
        self.rgb = np.stack(rgbs)
        self.descriptors = array_descriptors(self.rgb)

    def runner(self, noise=0.02, distortion=0.0, seed=0):
        return SimulatedRunner(self.gt, noise=noise, distortion=distortion,
                               seed=seed)

    def revisit_pairs(self):
        """(i, i + n_per_lap): the same ring position one lap later."""
        n = self.n_per_lap
        return [(i, i + n) for i in range(len(self.gt) - n)]
