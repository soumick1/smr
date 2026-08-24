"""Similarity-transform helpers for chunk stitching.

A Sim(3) is the triple (s, R, t): x -> s R x + t.  Camera-to-world poses go
through it as apply_sim3_to_poses does (orientation R R_c, centre s R c + t).
Everything here is pure numpy so it can be tested against hand-derivable
cases.
"""
from __future__ import annotations

import numpy as np

from ..eval.trajectory import apply_sim3_to_poses, rotation_angle_deg

Sim3 = tuple  # (s, R, t)


def identity():
    return 1.0, np.eye(3), np.zeros(3)


def compose(S2, S1):
    """S2 o S1 : apply S1 first, then S2."""
    s2, R2, t2 = S2
    s1, R1, t1 = S1
    return s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2


def inverse(S):
    s, R, t = S
    return 1.0 / s, R.T, -(R.T @ t) / s


def apply(S, poses):
    return apply_sim3_to_poses(poses, *S)


def rotvec(R):
    """Rotation matrix -> axis-angle vector (numpy only)."""
    ang = np.radians(rotation_angle_deg(R))
    if ang < 1e-12:
        return np.zeros(3)
    if abs(ang - np.pi) < 1e-6:            # near pi: axis from R + I
        A = R + np.eye(3)
        k = int(np.argmax(np.linalg.norm(A, axis=0)))
        ax = A[:, k] / np.linalg.norm(A[:, k])
        return ang * ax
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * w / (2.0 * np.sin(ang))


def rotmat(v):
    """Axis-angle vector -> rotation matrix (Rodrigues)."""
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def to_vec(S):
    """(s, R, t) -> 7-vector [log s, rotvec, t]."""
    s, R, t = S
    return np.concatenate([[np.log(s)], rotvec(R), t])


def from_vec(v):
    v = np.asarray(v, float)
    return float(np.exp(v[0])), rotmat(v[1:4]), v[4:7].copy()


def interpolate(S, alpha):
    """Component-wise fraction `alpha` of the similarity S: identity at 0,
    S at 1.  Rotation via the axis-angle vector, scale geometrically,
    translation linearly.  Not the Lie-group geodesic (that mixes the
    parts), but continuous, monotone in alpha, and exact at both ends,
    which is all distributing a loop-closure correction needs."""
    s, R, t = S
    return float(s ** alpha), rotmat(alpha * rotvec(R)), alpha * np.asarray(t, float)


def fit_poses(A, B):
    """Similarity taking c2w poses A onto poses B, orientation-aware.

    Rotation: orthogonal Procrustes on the frame orientations (well posed
    from a single pose).  Scale: ratio of centre spreads (needs >= 2
    poses; with one pose scale is left at 1 and reported as unresolved).
    Translation: centroids.  Returns (S, info) where info carries per-pose
    residuals so callers can reject inconsistent correspondences.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    assert A.shape == B.shape and A.ndim == 3 and A.shape[1:] == (4, 4)
    n = len(A)
    M = np.zeros((3, 3))
    for i in range(n):
        M += B[i, :3, :3] @ A[i, :3, :3].T
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    ca, cb = A[:, :3, 3], B[:, :3, 3]
    ma, mb = ca.mean(0), cb.mean(0)
    da, db = ca - ma, cb - mb
    na = float(np.sqrt((da ** 2).sum()))
    scale_ok = n >= 2 and na > 1e-9
    s = float(np.sqrt((db ** 2).sum()) / na) if scale_ok else 1.0
    t = mb - s * (R @ ma)
    S = (s, R, t)
    moved = apply(S, A)
    rot_res = np.array([rotation_angle_deg(moved[i, :3, :3].T @ B[i, :3, :3])
                        for i in range(n)])
    pos_res = np.linalg.norm(moved[:, :3, 3] - cb, axis=1)
    spread = float(np.median(np.linalg.norm(db, axis=1))) if n >= 2 else 1.0
    return S, dict(rot_res_deg=rot_res, pos_res=pos_res, spread=spread,
                   scale_ok=scale_ok, n=n)


def fit_poses_robust(A, B, rot_thresh_deg=10.0, pos_thresh_rel=0.5,
                     min_inliers=2, max_iter=5):
    """fit_poses with iterative rejection of inconsistent correspondences.

    After each fit, any correspondence whose rotation residual exceeds
    rot_thresh_deg or whose centre residual exceeds pos_thresh_rel times
    the median centre spread of B is an outlier; the worst is dropped and
    the fit repeated, never below min_inliers.  Returns (S, inlier_mask,
    info).  A single wrong anchor (a false loop closure) must not be able
    to move a chunk, and this is the guard that ensures it.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = len(A)
    keep = np.ones(n, bool)
    S, info = fit_poses(A, B)
    for _ in range(max_iter):
        S, info = fit_poses(A[keep], B[keep])
        rr, pr = info["rot_res_deg"], info["pos_res"]
        thr_pos = pos_thresh_rel * max(info["spread"], 1e-9)
        bad = (rr > rot_thresh_deg) | (pr > thr_pos)
        if not bad.any() or keep.sum() <= min_inliers:
            break
        score = rr / rot_thresh_deg + pr / thr_pos
        score[~bad] = -1.0
        worst_local = int(np.argmax(score))
        worst_global = np.nonzero(keep)[0][worst_local]
        keep[worst_global] = False
    S, info = fit_poses(A[keep], B[keep])
    return S, keep, info
