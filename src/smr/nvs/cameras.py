"""Camera protocol for the NVS benchmark (pure numpy; tested).

Conventions (documented once, used by the renderer, the loaders and the evaluator):
  * Object normalisation: bounding-box centre at the origin, then scaled so the bounding
    SPHERE has radius 0.5 (every vertex within 0.5 of the origin).  At radius 2.0 and
    FOV 40 deg the ball subtends 29 deg: the whole object is always in frame.
  * World: y up (Blender's z-up is handled inside the renderer).  Cameras look at the
    origin from a sphere of radius `radius` (default 2.0), vertical FOV 40 deg, square
    images of `res` pixels (default 256).
  * c2w is OpenCV-style: camera x right, y down, z forward (into the scene); the
    returned intrinsics use the corner convention (principal point at res/2).
  * Training objects: `n` random views, azimuth ~ U(0, 360), elevation ~ U(-20, 60) deg.
  * GSO test protocol (LVSM/Instant3D style): 4 inputs at elevation 20 deg and azimuths
    0/90/180/270, plus 10 targets drawn as training views with a per-object seed.
  * Pluecker rays (for target-view conditioning, VGGT-NVS/LVSM): per pixel (d, o x d) with
    d the unit ray direction in world coordinates and o the camera centre.
"""
from __future__ import annotations

import numpy as np

RADIUS, FOV_DEG, RES = 2.0, 40.0, 256


def intrinsics(res=RES, fov_deg=FOV_DEG):
    f = 0.5 * res / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])


def look_at(cam, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """OpenCV c2w (4x4): z toward target, y down, x right."""
    cam, target, up = (np.asarray(v, float) for v in (cam, target, up))
    z = target - cam; z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:                     # looking straight along `up`
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                              # points down for y-up worlds seen from above
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, cam
    return M


def spherical(az_deg, el_deg, radius=RADIUS):
    """Camera centre on the sphere; azimuth about +y, elevation from the x-z plane."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return radius * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def train_views(n, seed, radius=RADIUS, el_range=(-20.0, 60.0)):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n); el = rng.uniform(*el_range, n)
    return [dict(az=float(a), el=float(e), c2w=look_at(spherical(a, e, radius))) for a, e in zip(az, el)]


def gso_views(seed, radius=RADIUS, n_targets=10):
    inputs = [dict(az=float(a), el=20.0, c2w=look_at(spherical(a, 20.0, radius)), role="input") for a in (0, 90, 180, 270)]
    targets = [dict(**v, role="target") for v in train_views(n_targets, seed, radius)]
    return inputs + targets


def project(K, c2w, X):
    """World points (N,3) -> pixel (N,2) and depth (N,) through an OpenCV c2w."""
    w2c = np.linalg.inv(c2w)
    Xc = (w2c[:3, :3] @ np.asarray(X, float).T).T + w2c[:3, 3]
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def pluecker_rays(K, c2w, res=RES):
    """(res, res, 6) Pluecker coordinates (d, o x d) of every pixel centre, world frame."""
    ys, xs = np.mgrid[0:res, 0:res]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1).reshape(-1, 3)
    d_cam = (np.linalg.inv(K) @ pix.T).T
    d = (c2w[:3, :3] @ d_cam.T).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = c2w[:3, 3]
    return np.concatenate([d, np.cross(np.broadcast_to(o, d.shape), d)], -1).reshape(res, res, 6)


def cams_to_json(views, K, res=RES):
    return dict(res=res, K=K.tolist(), fov_deg=FOV_DEG, radius=RADIUS, convention="opencv_c2w_y_up",
                views=[dict(index=i, az=v["az"], el=v["el"], role=v.get("role", "train"), c2w=np.asarray(v["c2w"]).tolist())
                       for i, v in enumerate(views)])
