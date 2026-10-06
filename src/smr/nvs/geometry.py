"""Per-view geometry for the NVS head (numpy; tested with a stub backbone).

Frozen backbone -> world-frame point maps on the render grid:
  * `load_views` reads an object folder (RGBA PNGs + cams.json): colour composited on white
    (the LVSM/GS-LRM convention), alpha kept as the object mask, GT c2w and K.
  * `place_views` puts a backbone output into the render's world frame with the KNOWN input
    cameras (the "Known Input Cam" setting of GS-LRM/LVSM): each view's predicted depth (or
    point head, expressed in that view's camera) is unprojected through the GT camera; the
    metric scale comes from the Sim(3) fitting the backbone's predicted cameras to the GT ones
    (four cameras, well posed).  Maps are resampled from the backbone grid (518^2) to the
    render grid (256^2); the square image is never cropped, so pixel grids align.
  * `predict_geometry` runs `reads` orderings of the same views and applies the memory's
    in-window read: symmetric re-measure of the per-pass scale about the GT cameras
    (content_align, ref="mean") and per-pixel consensus (median of the witnesses).  reads=1
    is the raw row.  The read touches geometry only; the Gaussian head is shared by all rows.
"""
from __future__ import annotations

import json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))
from smr.eval.trajectory import sim3_from_poses  # noqa: E402


def _imread_rgba(path):
    """RGB in [0,1] and an object mask: the alpha channel of RGBA renders (v142+); for RGB-only
    renders (pre-v142 dry runs) the mask is 'not white' (all channels > 0.98 -> background)."""
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        a = np.asarray(im).astype(np.float32) / 255.0
        return a[..., :3], a[..., 3]
    rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    return rgb, (rgb < 0.98).any(-1).astype(np.float32)


def load_views(obj_dir, ids=None):
    """-> dict(rgb (V,H,W,3) on white, alpha (V,H,W), c2w (V,4,4), K (3,3), paths, res)."""
    obj_dir = pathlib.Path(obj_dir)
    cams = json.load(open(obj_dir / "cams.json"))
    views = cams["views"] if ids is None else [cams["views"][i] for i in ids]
    rgb, alpha, c2w, paths = [], [], [], []
    for v in views:
        p = obj_dir / f"{v['index']:03d}.png"
        c, a = _imread_rgba(p)
        rgb.append(c * a[..., None] + (1.0 - a[..., None]))         # composite on white
        alpha.append(a); c2w.append(np.asarray(v["c2w"], float)); paths.append(str(p))
    return dict(rgb=np.stack(rgb), alpha=np.stack(alpha), c2w=np.stack(c2w), K=np.asarray(cams["K"], float),
                paths=paths, res=int(cams["res"]), roles=[v.get("role", "train") for v in views])


def resize_map(x, res):
    """Nearest-neighbour resample of an (h,w[,c]) map to (res,res[,c]) by index mapping."""
    h, w = x.shape[:2]
    ys = np.minimum((np.arange(res) + 0.5) * h / res, h - 1).astype(int)
    xs = np.minimum((np.arange(res) + 0.5) * w / res, w - 1).astype(int)
    return x[ys[:, None], xs[None, :]]


def unproject_grid(depth, K, c2w):
    """(res,res) depth through K and an OpenCV c2w -> (res,res,3) world points (NaN where invalid)."""
    res = depth.shape[0]
    ys, xs = np.mgrid[0:res, 0:res]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1).reshape(-1, 3).astype(float)
    rays = (np.linalg.inv(K) @ pix.T).T
    z = depth.reshape(-1, 1).astype(float)
    pts = (c2w[:3, :3] @ (rays * z).T).T + c2w[:3, 3]
    pts[~(np.isfinite(z[:, 0]) & (z[:, 0] > 0))] = np.nan
    return pts.reshape(res, res, 3)


def place_views(o, gt_c2w, K, res, source="auto"):
    """Backbone output `o` (poses (V,4,4) c2w in its own frame, depth (V,h,w), extras) -> world
    points (V,res,res,3) through the GT cameras, confidence (V,res,res), and the metric scale.
    source: 'depth' (depth x GT camera), 'pointhead' (o.extras['world_points'] moved into each
    view's own predicted camera, then placed by the GT camera), 'auto' (pointhead if exposed)."""
    V = len(gt_c2w)
    s, _, _ = sim3_from_poses(np.asarray(o.poses)[:V], np.asarray(gt_c2w))
    use_ph = source == "pointhead" or (source == "auto" and "world_points" in getattr(o, "extras", {}))
    pts, conf = [], []
    for j in range(V):
        if use_ph:
            Xw = np.asarray(o.extras["world_points"][j], float)               # backbone frame
            Tinv = np.linalg.inv(np.asarray(o.poses[j], float))
            Xc = (Tinv[:3, :3] @ Xw.reshape(-1, 3).T).T + Tinv[:3, 3]       # that view's camera frame
            Xc = resize_map(Xc.reshape(Xw.shape), res).reshape(-1, 3) * s
            Xg = (gt_c2w[j][:3, :3] @ Xc.T).T + gt_c2w[j][:3, 3]
            bad = ~np.isfinite(Xc).all(1) | (Xc[:, 2] <= 0)
            Xg[bad] = np.nan
            pts.append(Xg.reshape(res, res, 3))
        else:
            d = resize_map(np.asarray(o.depth[j], float), res) * s
            pts.append(unproject_grid(d, K, gt_c2w[j]))
        c = o.extras.get("conf", None) if hasattr(o, "extras") else None
        conf.append(resize_map(np.asarray(c[j], float), res) if c is not None else np.ones((res, res)))
    return np.stack(pts), np.stack(conf), float(s)


def predict_geometry(bb, paths, gt_c2w, K, res, reads=1, source="auto", tau_abs=None, fallback=None, seed=0, abstain_rel=None, align=None):
    """Geometry of the input views from ONE backbone pass, placed into the GT camera frame (v198: the repeated-read
    consensus over input orderings was removed from the method; `reads` > 1 is rejected so old call sites fail loudly).
    Returns dict(pts (V,res,res,3), conf (V,res,res), scales [s], orderings [identity])."""
    if int(reads) != 1:
        raise ValueError("repeated reads (reads > 1) were removed in v198; the memory supplies geometry through stored "
                         "windows and their revised placements, not by re-reading the same views in other orders")
    V = len(paths)
    perm = tuple(range(V))
    o = bb.infer([paths[i] for i in perm])
    p, c, s = place_views(o, np.asarray(gt_c2w)[list(perm)], K, res, source)
    return dict(pts=p, conf=c, scales=[s], orderings=[perm])


def head_input(rgb, pts, conf, alpha, depth_from_cams=None):
    """(V,res,res,8) float32 features for the Gaussian head: rgb(3), xyz(3), depth(1), conf(1); NaN->0.
    Background (alpha<0.5) and invalid points are reported in the returned mask."""
    V, res = rgb.shape[0], rgb.shape[1]
    valid = np.isfinite(pts).all(-1) & (alpha > 0.5)
    xyz = np.where(valid[..., None], pts, 0.0)
    depth = depth_from_cams if depth_from_cams is not None else np.linalg.norm(xyz, axis=-1)
    c = np.where(np.isfinite(conf), conf, 0.0)
    c = c / (np.percentile(c[valid], 95) + 1e-6) if valid.any() else c
    x = np.concatenate([rgb, xyz, depth[..., None], np.clip(c, 0, 1)[..., None]], -1).astype(np.float32)
    return x, valid
