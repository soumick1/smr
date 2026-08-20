"""Zero-retrain content: per-view surfels, rigid transform, splat readout."""
from __future__ import annotations

import numpy as np

from .synthetic_scene import Camera, render


def surfels_from_view(rgb, depth, mask, cam: Camera, stride=2):
    """Camera-frame surfels (points, colors) subsampled by `stride`."""
    vs, us = np.mgrid[0:cam.H:stride, 0:cam.W:stride]
    d = depth[::stride, ::stride]
    m = mask[::stride, ::stride] & (d > 0)
    x = (us - cam.W / 2) / cam.f * d
    y = (vs - cam.H / 2) / cam.f * d
    pts = np.stack([x[m], y[m], d[m]], axis=-1)
    cols = rgb[::stride, ::stride][m]
    return pts, cols


def transform(points_cam, T_src_wc, T_query_wc):
    """Map view-frame surfels of the source into the QUERY camera's world:
    identical world points, expressed once; splat then uses T_query."""
    Pw = points_cam @ T_src_wc[:3, :3].T + T_src_wc[:3, 3]
    return Pw


def splat(points_world, colors, T_query_wc, cam: Camera):
    return render(points_world, colors, T_query_wc, cam)
