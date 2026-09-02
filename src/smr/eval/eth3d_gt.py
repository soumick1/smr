"""ETH3D ground truth on the backbone's pixel grid (pure numpy).

VGGT Sec. 4.3 scores *point maps*: per-pixel 3D points of 10 sampled frames,
Umeyama-aligned to the ground truth, invalid pixels removed with "the
official masks", then Chamfer Acc/Comp.  That needs a per-pixel ground-truth
point map for each frame.  Two sources are implemented:

official   ETH3D's rendered depth maps (``<scene>_dslr_depth.7z`` ->
           ``ground_truth_depth/dslr_images/<name>``; raw little-endian float32,
           row-major, one value per pixel of the ORIGINAL DISTORTED image, inf =
           no ground truth; documentation, "Training data").  They are defined
           on the THIN_PRISM_FISHEYE camera (``dslr_calibration_jpg``), while our
           pipeline runs on the pre-undistorted PINHOLE images
           (``dslr_calibration_undistorted``).  Both share the pose; depth is the
           camera-z, identical in both parametrisations.  We therefore map every
           undistorted pixel forward through the fisheye model (closed form,
           ETH3D BenchmarkCamera::Distort; same formulas as robustmvd's
           compute_eth3d_undistorted.py) to read the depth of the distorted pixel
           it lands on, then pool the full-resolution values into the backbone
           grid (per-cell median of finite depths; ``pool="min"`` = nearest surface).
           The inf pixels are the official mask.

scan       fallback with no extra download: z-buffer the *eval* laser scan
           (scan_alignment.mlp + plys, the cloud the official evaluator uses)
           into each camera at the backbone grid (nearest depth per cell).
           No occlusion mesh, so thin gaps in the scan can leak a farther
           surface; reported as a fallback, never as the official protocol.

Grid convention: the backbone works on the undistorted image resized to
(Wd, Hd) anisotropically (VGGT: width 518, height rounded to a multiple of
14), then optionally strided.  A depth-grid pixel (x, y) covers undistorted
pixels u in [x*Wu/Wd, (x+1)*Wu/Wd), v likewise (COLMAP corner convention:
pixel centres at +0.5).
"""
from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

SCENES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office",
          "pipes", "playground", "relief", "relief_2", "terrace", "terrains"]


def grid_map(wh_img, depth_hw, target=518, patch=14):
    """Backbone depth grid -> source image coordinates.

    VGGT's load_and_preprocess_images (mode "crop"): resize to width `target`,
    height round(H*target/W/patch)*patch (anisotropic), then centre-crop the
    height to `target` if it exceeds it (portrait frames).  Returns (sx, sy, oy)
    such that grid pixel (x, y) covers source u in [x*sx, (x+1)*sx),
    v in [(y+oy)*sy, (y+oy+1)*sy).  Falls back to plain anisotropic scaling
    when the grid does not match VGGT's rule (other backbones).
    """
    W, H = wh_img; Hd, Wd = depth_hw
    sx = W / Wd
    new_h = int(round(H * Wd / W / patch) * patch)
    if new_h == Hd:
        return sx, H / Hd, 0
    if new_h > Hd and Wd == target:               # VGGT centre crop
        return sx, H / new_h, (new_h - Hd) // 2
    return sx, H / Hd, 0


# ------------------------------------------------------------- COLMAP text ---
def read_cameras_txt(path):
    cams = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = dict(model=t[1], width=int(t[2]), height=int(t[3]),
                               params=np.array([float(x) for x in t[4:]]))
    return cams


def read_images_txt(path):
    """{basename: (camera_id, qvec, tvec)} for a COLMAP images.txt."""
    out = {}
    lines = [l for l in pathlib.Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    for l in lines[::2]:
        t = l.split()
        out[pathlib.Path(t[9]).name] = (int(t[8]), np.array(t[1:5], float), np.array(t[5:8], float))
    return out


# ---------------------------------------------------------- fisheye model ----
def thin_prism_distort(u, v, dist):
    k1, k2, p1, p2, k3, k4, sx1, sy1 = dist
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    du = 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2
    dv = 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2
    return u * radial + du, v * radial + dv


def fisheye_project(x, y, dist):
    """Undistorted normalised (x, y) -> distorted normalised (ETH3D THIN_PRISM_FISHEYE)."""
    r = np.sqrt(x * x + y * y)
    scale = np.where(r > 1e-6, np.arctan(r) / np.where(r > 1e-6, r, 1.0), 1.0)
    return thin_prism_distort(x * scale, y * scale, dist)


# ---------------------------------------------------------------- pooling ----
def _pool_bins(bins, vals, n_bins, how="median"):
    """Per-bin median (or min) of vals; NaN for empty bins."""
    out = np.full(n_bins, np.nan)
    if len(vals) == 0:
        return out
    if how == "min":
        mn = np.full(n_bins, np.inf)
        np.minimum.at(mn, bins, vals)
        mn[~np.isfinite(mn)] = np.nan
        return mn
    order = np.lexsort((vals, bins))
    b, v = bins[order], vals[order]
    start = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    count = np.diff(np.r_[start, len(b)])
    med = v[start + count // 2]
    out[b[start]] = med
    return out


def gt_depth_official(scene_dir, name, K_und, wh_und, depth_hw, pool="median", chunk=256):
    """Low-resolution GT depth (Hd, Wd) for undistorted image `name`.

    Args:
        scene_dir: <root>/<scene> containing dslr_calibration_jpg/ and ground_truth_depth/.
        name: image basename (e.g. DSC_0286.JPG) -- identical for distorted/undistorted.
        K_und: (3,3) PINHOLE intrinsics of the undistorted image; wh_und: (Wu, Hu).
        depth_hw: (Hd, Wd) backbone depth-map grid.
    Returns:
        depth (Hd, Wd) float64 with NaN where no ground truth (the official mask),
        stats dict.
    """
    scene_dir = pathlib.Path(scene_dir)
    cams = read_cameras_txt(scene_dir / "dslr_calibration_jpg" / "cameras.txt")
    imgs = read_images_txt(scene_dir / "dslr_calibration_jpg" / "images.txt")
    if name not in imgs:
        raise FileNotFoundError(f"{name} not in dslr_calibration_jpg/images.txt")
    cam = cams[imgs[name][0]]
    assert cam["model"] == "THIN_PRISM_FISHEYE", cam["model"]
    fxd, fyd, cxd, cyd = cam["params"][:4]; dist = cam["params"][4:]
    Wd_img, Hd_img = cam["width"], cam["height"]
    dfile = scene_dir / "ground_truth_depth" / "dslr_images" / name
    if not dfile.exists():                       # some archives store lowercase/other ext
        cands = list((scene_dir / "ground_truth_depth" / "dslr_images").glob(pathlib.Path(name).stem + "*"))
        if not cands:
            raise FileNotFoundError(f"GT depth for {name} not under {dfile.parent}")
        dfile = cands[0]
    D = np.fromfile(dfile, dtype=np.float32)
    assert D.size == Wd_img * Hd_img, f"{dfile}: {D.size} floats != {Wd_img}x{Hd_img}"
    D = D.reshape(Hd_img, Wd_img)
    Wu, Hu = wh_und; Hd, Wd = depth_hw
    sx, sy, oy = grid_map((Wu, Hu), (Hd, Wd))
    fx, fy, cx, cy = K_und[0, 0], K_und[1, 1], K_und[0, 2], K_und[1, 2]
    bins_all, vals_all = [], []
    xs = (np.arange(Wu) + 0.5 - cx) / fx
    for v0 in range(0, Hu, chunk):
        v1 = min(v0 + chunk, Hu)
        ys = (np.arange(v0, v1) + 0.5 - cy) / fy
        gx, gy = np.meshgrid(xs, ys)
        dx, dy = fisheye_project(gx, gy, dist)
        px = np.floor(dx * fxd + cxd).astype(np.int64)       # corner convention -> pixel index
        py = np.floor(dy * fyd + cyd).astype(np.int64)
        ok = (px >= 0) & (px < Wd_img) & (py >= 0) & (py < Hd_img)
        val = np.full(gx.shape, np.inf)
        val[ok] = D[py[ok], px[ok]]
        fin = np.isfinite(val) & (val > 0)
        if not fin.any():
            continue
        uu, vv = np.meshgrid(np.arange(Wu), np.arange(v0, v1))
        bx = np.floor((uu[fin] + 0.5) / sx).astype(np.int64)
        by = np.floor((vv[fin] + 0.5) / sy).astype(np.int64) - oy
        inb = (bx >= 0) & (bx < Wd) & (by >= 0) & (by < Hd)       # outside the crop -> dropped
        bins_all.append((by * Wd + bx)[inb]); vals_all.append(val[fin][inb].astype(np.float64))
    bins = np.concatenate(bins_all) if bins_all else np.zeros(0, np.int64)
    vals = np.concatenate(vals_all) if vals_all else np.zeros(0)
    depth = _pool_bins(bins, vals, Hd * Wd, pool).reshape(Hd, Wd)
    stats = dict(gt_pixels_fullres=int(len(vals)), grid_valid=float(np.isfinite(depth).mean()))
    return depth, stats


# ------------------------------------------------------------ scan fallback --
def read_ply_xyz(path):
    try:
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    except ImportError:
        raw = pathlib.Path(path).read_bytes()
        head, body = raw.split(b"end_header\n", 1)
        n = int(re.search(rb"element vertex (\d+)", head).group(1))
        assert b"format ascii" in head, "pip install plyfile for binary PLY"
        return np.loadtxt(body.splitlines()[:n], usecols=(0, 1, 2))


def load_scan(mlp_path):
    """Merged laser scan (world frame) from a MeshLab .mlp (ETH3D dslr_scan_eval)."""
    mlp = pathlib.Path(mlp_path)
    pts = []
    root = ET.parse(mlp).getroot()
    for mesh in root.iter("MLMesh"):
        fn = (mlp.parent / mesh.get("filename")).resolve()
        M = np.eye(4)
        mm = mesh.find("MLMatrix44")
        if mm is not None and mm.text and mm.text.split():
            M = np.array(mm.text.split(), float).reshape(4, 4)
        P = read_ply_xyz(fn)
        pts.append((M[:3, :3] @ P.T).T + M[:3, 3])
    return np.concatenate(pts)


def gt_depth_from_scan(scan_xyz, K_und, c2w, wh_und, depth_hw):
    """Z-buffer the laser scan into camera (K_und, c2w) at the backbone grid."""
    Wu, Hu = wh_und; Hd, Wd = depth_hw
    w2c = np.linalg.inv(c2w)
    Xc = (w2c[:3, :3] @ scan_xyz.T).T + w2c[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-6
    u = K_und[0, 0] * Xc[ok, 0] / z[ok] + K_und[0, 2]          # corner convention
    v = K_und[1, 1] * Xc[ok, 1] / z[ok] + K_und[1, 2]
    sx, sy, oy = grid_map((Wu, Hu), (Hd, Wd))
    bx = np.floor(u / sx).astype(np.int64); by = np.floor(v / sy).astype(np.int64) - oy
    inside = (bx >= 0) & (bx < Wd) & (by >= 0) & (by < Hd)
    out = np.full(Hd * Wd, np.inf)
    np.minimum.at(out, (by * Wd + bx)[inside], z[ok][inside])
    out[~np.isfinite(out)] = np.nan
    return out.reshape(Hd, Wd)


# ------------------------------------------------------------ unprojection --
def unproject_grid(depth, K_und, c2w, wh_und, stride=1):
    """Depth on the (Hd, Wd) grid -> world points (gh, gw, 3) on the strided grid.

    Uses corner-convention pixel centres ((x+0.5) in the depth grid), i.e. the
    exact geometry of the undistorted PINHOLE camera; NaN depth -> NaN point.
    """
    Hd, Wd = depth.shape; Wu, Hu = wh_und
    sx, sy, oy = grid_map((Wu, Hu), (Hd, Wd))
    ys, xs = np.mgrid[0:Hd:stride, 0:Wd:stride]
    d = depth[ys, xs]
    u = (xs + 0.5) * sx; v = (ys + oy + 0.5) * sy
    x = (u - K_und[0, 2]) / K_und[0, 0]; y = (v - K_und[1, 2]) / K_und[1, 1]
    Xc = np.stack([x * d, y * d, d], -1)
    Xw = (c2w[:3, :3] @ Xc.reshape(-1, 3).T).T + c2w[:3, 3]
    return Xw.reshape(Xc.shape)
