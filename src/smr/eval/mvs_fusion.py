"""Multi-view depth fusion utilities (pure numpy / scipy, no GPU).

Two protocol pieces the DTU/ETH3D reproduction chase needs, each traced to
the paper/code it copies so no step is a silent guess:

1. ``geo_consistency``: the geometric-consistency filter of PatchmatchNet's
   ``eval.py`` (Wang et al., CVPR'21) that MASt3R Sec. 4.5 cites for removing
   spurious points before the official DTU evaluation ("We remove spurious
   3D points via geometric consistency post-processing [99]"), and hence the
   post-processing VGGT Table 2 inherits by "following MASt3R".  A reference
   pixel with depth d is kept when, for at least ``min_views`` source views,
   projecting it into the source, reading the source depth there, and
   projecting back lands within ``pix_thr`` pixels (reference default 1.0)
   and within a relative depth difference of ``rel_thr`` (reference default
   0.01).  The fused depth is the mean of the reference depth and the
   consistent reprojected depths (reference ``depth_est_averaged``).
   Reference defaults: pix 1.0, rel 0.01, ``geo_mask_thres`` 5.

2. ``umeyama_trimmed``: closed-form Sim(3) Umeyama on point correspondences
   (VGGT Sec. 4.3: "aligned to the ground truth using the Umeyama algorithm")
   with an optional iterative trim of the largest residuals for robustness;
   ``trim=0`` is the plain algorithm.

Conventions: poses are (V, 4, 4) camera-to-world; intrinsics (V, 3, 3) are
at the depth-map resolution with integer pixel indices (u, v) = (col, row),
the convention of BackboneOutput / points_suite.unproject.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .trajectory import umeyama_sim3


# ----------------------------------------------------------------- helpers ---
def select_sources(c2ws, n_src, pair_lists=None):
    """Source views per reference view.

    With ``pair_lists`` (e.g. MVSNet pair.txt) those are used verbatim;
    otherwise the ``n_src`` nearest cameras by centre distance (the DTU
    pair.txt is itself a nearest-camera ranking, so this is the same idea
    without the file).  Returns a list of index lists.
    """
    V = len(c2ws)
    if pair_lists is not None:
        return [[j for j in p if j != i][:n_src] for i, p in enumerate(pair_lists)]
    C = np.asarray(c2ws)[:, :3, 3]
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    out = []
    for i in range(V):
        order = np.argsort(D[i])
        out.append([int(j) for j in order if j != i][:n_src])
    return out


def _reproject(d_ref, K_ref, c2w_ref, d_src, K_src, c2w_src):
    """PatchmatchNet reproject_with_depth, vectorised, numpy.

    Returns (depth_reprojected, x_reprojected, y_reprojected, sampled_src_depth)
    each (H, W).  Invalid (non-finite / <=0) depths propagate as 0 so they
    fail the consistency test.
    """
    H, W = d_ref.shape
    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], 0).astype(np.float64)
    dr = np.where(np.isfinite(d_ref), d_ref, 0.0).ravel().astype(np.float64)
    X_ref = (np.linalg.inv(K_ref) @ pix) * dr                      # (3, N) cam
    w2c_src = np.linalg.inv(c2w_src)
    M = w2c_src @ c2w_ref                                          # ref cam -> src cam
    X_src = M[:3, :3] @ X_ref + M[:3, 3:4]
    z = X_src[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (K_src @ X_src)[:2] / z
    in_front = z > 1e-9
    # bilinear source-depth lookup (cv2.remap INTER_LINEAR equivalent)
    d_src0 = np.where(np.isfinite(d_src) & (d_src > 0), d_src, 0.0).astype(np.float64)
    coords = np.stack([np.where(in_front, uv[1], -10.0), np.where(in_front, uv[0], -10.0)])
    ds = map_coordinates(d_src0, coords, order=1, mode="constant", cval=0.0)
    inside = in_front & (uv[0] >= 0) & (uv[0] <= W - 1) & (uv[1] >= 0) & (uv[1] <= H - 1)
    ds = np.where(inside, ds, 0.0)
    uv1 = np.stack([np.where(inside, uv[0], 0.0), np.where(inside, uv[1], 0.0), np.ones(H * W)])
    X_src2 = (np.linalg.inv(K_src) @ uv1) * ds                      # src cam, via src depth
    Minv = np.linalg.inv(c2w_ref) @ c2w_src                         # src cam -> ref cam
    X_ref2 = Minv[:3, :3] @ X_src2 + Minv[:3, 3:4]
    z2 = X_ref2[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy2 = (K_ref @ X_ref2)[:2] / z2
    ok = inside & (z2 > 1e-9) & (ds > 0)
    d_rep = np.where(ok, z2, 0.0).reshape(H, W)
    x2 = np.where(ok, xy2[0], 1e9).reshape(H, W)
    y2 = np.where(ok, xy2[1], 1e9).reshape(H, W)
    return d_rep, x2, y2, ds.reshape(H, W)


def geo_consistency(depths, Ks, c2ws, sources, pix_thr=1.0, rel_thr=0.01,
                    min_views=5, average=True):
    """Geometric-consistency filter over a set of views.

    Args:
        depths: (V, H, W) depth maps (same grid), NaN/<=0 = invalid.
        Ks: (V, 3, 3) intrinsics at that grid; c2ws: (V, 4, 4) camera-to-world.
        sources: list (len V) of source-view index lists (see select_sources).
        pix_thr, rel_thr, min_views: PatchmatchNet thresholds (1.0, 0.01, 5).
        average: replace kept depths by the mean over {ref} U consistent sources.
    Returns:
        keep: (V, H, W) bool mask; depth_out: (V, H, W) fused depth (NaN where dropped);
        counts: (V, H, W) int number of consistent source views.
    """
    depths = np.asarray(depths, np.float64)
    V, H, W = depths.shape
    ys, xs = np.mgrid[0:H, 0:W]
    keep = np.zeros((V, H, W), bool)
    counts = np.zeros((V, H, W), np.int32)
    depth_out = np.full((V, H, W), np.nan)
    for i in range(V):
        d_ref = depths[i]
        valid = np.isfinite(d_ref) & (d_ref > 0)
        acc = np.where(valid, d_ref, 0.0).copy()
        n = np.zeros((H, W), np.int32)
        for j in sources[i]:
            d_rep, x2, y2, _ = _reproject(d_ref, Ks[i], c2ws[i], depths[j], Ks[j], c2ws[j])
            dist = np.hypot(x2 - xs, y2 - ys)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(d_rep - d_ref) / d_ref
            m = valid & (dist < pix_thr) & (rel < rel_thr)
            n += m
            acc += np.where(m, d_rep, 0.0)
        k = valid & (n >= min_views)
        keep[i] = k; counts[i] = n
        depth_out[i] = np.where(k, (acc / (n + 1)) if average else d_ref, np.nan)
    return keep, depth_out, counts


# ---------------------------------------------------------------- umeyama ---
def umeyama_trimmed(X, Y, trim=0.0, iters=3, with_scale=True):
    """Sim(3) Umeyama X -> Y on correspondences, optionally trimming the
    ``trim`` fraction of largest residuals for ``iters`` rounds.

    Returns (s, R, t, inlier_mask).  trim=0 reproduces the plain algorithm
    (the wording of VGGT Sec. 4.3); trim>0 is our robust diagnostic and is
    labelled as such wherever it is reported.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    idx = np.where(ok)[0]
    s, R, t = umeyama_sim3(X[idx], Y[idx], with_scale)
    if trim > 0:
        for _ in range(iters):
            res = np.linalg.norm((s * (R @ X[idx].T)).T + t - Y[idx], axis=1)
            cut = np.quantile(res, 1.0 - trim)
            idx = idx[res <= cut]
            if len(idx) < 10:
                break
            s, R, t = umeyama_sim3(X[idx], Y[idx], with_scale)
    inl = np.zeros(len(X), bool); inl[idx] = True
    return s, R, t, inl


def apply_sim3(P, s, R, t):
    """Points (..., 3) -> s R p + t (NaNs pass through)."""
    P = np.asarray(P, float)
    return (s * (R @ P.reshape(-1, 3).T)).T.reshape(P.shape) + t


def chamfer(pred, gt, max_dist=None):
    """Mean nearest-neighbour distances pred->gt (Acc) and gt->pred (Comp).

    ``max_dist`` optionally discards larger distances (DTU convention);
    ETH3D/VGGT Table 3 report the plain means (max_dist=None).
    """
    from scipy.spatial import cKDTree
    pred = pred[np.isfinite(pred).all(1)]; gt = gt[np.isfinite(gt).all(1)]
    dp = cKDTree(gt).query(pred, workers=-1)[0]
    dg = cKDTree(pred).query(gt, workers=-1)[0]
    if max_dist is not None:
        dp, dg = dp[dp < max_dist], dg[dg < max_dist]
    acc, comp = float(dp.mean()), float(dg.mean())
    return acc, comp, 0.5 * (acc + comp), dp, dg
