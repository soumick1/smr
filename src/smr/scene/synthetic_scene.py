"""Synthetic surfel rooms with a numpy pinhole z-buffer renderer.

Provides ground-truth geometry for Tier-3 integration tests without any GPU
or learned backbone: a box room whose walls/floor carry textured surfels
plus a few colored blobs, cameras on a ring looking inward, and rendering
by point projection with a z-buffer (2x2 splat for hole reduction).
Camera convention: x right, y down, z forward; world-from-camera pose T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import rng
from ..utils.geometry import make_T, Rz, Ry


@dataclass
class Camera:
    H: int = 192
    W: int = 192
    f: float = 160.0

    @property
    def K(self):
        return np.array([[self.f, 0, self.W / 2],
                         [0, self.f, self.H / 2],
                         [0, 0, 1.0]])


def build_room(seed=0, n_wall=26000, n_blobs=6, half=1.5, height=2.0):
    """Returns (points (N,3), colors (N,3) in [0,1])."""
    g = rng(seed)
    pts, cols = [], []

    def patch(n, origin, ex, ey, base):
        uv = g.random((n, 2))
        p = origin[None] + uv[:, :1] * ex[None] + uv[:, 1:] * ey[None]
        checker = ((np.floor(uv[:, 0] * 6) + np.floor(uv[:, 1] * 6)) % 2)
        c = base[None] * (0.55 + 0.45 * checker[:, None])
        pts.append(p); cols.append(c)

    s = half
    patch(n_wall // 5, np.array([-s, 0, -s]), np.array([2 * s, 0, 0]),
          np.array([0, 0, 2 * s]), np.array([0.55, 0.52, 0.50]))       # floor
    patch(n_wall // 5, np.array([-s, height, -s]), np.array([2 * s, 0, 0]),
          np.array([0, 0, 2 * s]), np.array([0.75, 0.75, 0.78]))       # ceil
    patch(n_wall // 5, np.array([-s, 0, s]), np.array([2 * s, 0, 0]),
          np.array([0, height, 0]), np.array([0.35, 0.55, 0.70]))      # +z wall
    patch(n_wall // 10, np.array([-s, 0, -s]), np.array([0, height, 0]),
          np.array([0, 0, 2 * s]), np.array([0.70, 0.45, 0.35]))       # -x wall
    patch(n_wall // 10, np.array([s, 0, -s]), np.array([0, height, 0]),
          np.array([0, 0, 2 * s]), np.array([0.45, 0.70, 0.40]))       # +x wall
    for b in range(n_blobs):
        c0 = g.uniform([-.8 * s, .2, -.8 * s], [.8 * s, .8 * height, .8 * s])
        col = g.uniform(0.25, 0.95, 3)
        n = 1400
        p = c0[None] + 0.12 * g.standard_normal((n, 3))
        pts.append(p); cols.append(np.tile(col, (n, 1)))
    return np.concatenate(pts), np.clip(np.concatenate(cols), 0, 1)


def camera_ring(n_views, radius=1.05, height=1.0, look_h=1.0, seed=0,
                jitter=0.0):
    """World-from-camera poses on a ring, +z (optical axis) toward centre."""
    g = rng(seed)
    Ts = []
    for i in range(n_views):
        a = 2 * np.pi * i / n_views + (jitter * g.standard_normal() if jitter else 0)
        pos = np.array([radius * np.cos(a), height, radius * np.sin(a)])
        fwd = np.array([0.0, look_h, 0.0]) - pos
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, -1.0, 0.0])           # y-down convention
        right = np.cross(up, fwd); right /= np.linalg.norm(right)
        upo = np.cross(fwd, right)
        R = np.stack([right, upo, fwd], axis=1)
        Ts.append(make_T(R, pos))
    return Ts


def render(points, colors, T_wc, cam: Camera):
    """Z-buffer point render.  Returns (rgb HxWx3, depth HxW, mask HxW)."""
    Rcw, tcw = T_wc[:3, :3].T, -T_wc[:3, :3].T @ T_wc[:3, 3]
    pc = points @ Rcw.T + tcw
    z = pc[:, 2]
    vis = z > 0.05
    pc, cc, z = pc[vis], colors[vis], z[vis]
    u = cam.f * pc[:, 0] / z + cam.W / 2
    v = cam.f * pc[:, 1] / z + cam.H / 2
    depth = np.full((cam.H, cam.W), np.inf)
    rgb = np.zeros((cam.H, cam.W, 3))
    order = np.argsort(-z)                       # far first, near overwrites
    ui, vi = u[order].astype(int), v[order].astype(int)
    zi, ci = z[order], cc[order]
    for du in (0, 1):                            # 2x2 splat
        for dv in (0, 1):
            uu, vv = ui + du, vi + dv
            ok = (uu >= 0) & (uu < cam.W) & (vv >= 0) & (vv < cam.H)
            depth[vv[ok], uu[ok]] = zi[ok]
            rgb[vv[ok], uu[ok]] = ci[ok]
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return rgb, depth, mask


def unproject(depth, mask, cam: Camera):
    """Per-pixel camera-frame points (H, W, 3) where mask."""
    vs, us = np.mgrid[0:cam.H, 0:cam.W]
    x = (us - cam.W / 2) / cam.f * depth
    y = (vs - cam.H / 2) / cam.f * depth
    return np.stack([x, y, depth], axis=-1), mask
