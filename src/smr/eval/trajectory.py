"""Trajectory metrics for the long-sequence experiment.

Pure numpy, no torch, no GPU -- so the definitions can be unit-tested
against analytically known cases (see tests/test_trajectory.py) rather than
trusted.  Every metric here follows the convention used by the papers we
compare against; where a choice exists it is stated in the docstring.

Poses are ALWAYS (K, 4, 4) camera-to-world, matching BackboneOutput.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------- alignment --
def umeyama_sim3(X, Y, with_scale=True):
    """Least-squares similarity aligning X onto Y (Umeyama 1991).

    Args:
        X: (N, 3) source points, Y: (N, 3) target points.
    Returns:
        (s, R, t) minimising sum ||s R x_i + t - y_i||^2.

    Scale matters here: each backbone chunk carries its own scale gauge
    (median depth := 1), so stitching chunks is a Sim(3) problem, not SE(3).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 3
    n = X.shape[0]
    assert n >= 3, f"need >=3 correspondences for Sim(3), got {n}"

    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(S)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:      # reflection guard
        W[2, 2] = -1.0
    R = U @ W @ Vt
    if with_scale:
        var_x = (Xc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ W) / var_x) if var_x > 1e-12 else 1.0
    else:
        s = 1.0
    t = my - s * R @ mx
    return s, R, t


def collinearity(points):
    """0 = collinear, 1 = well spread.  Ratio of the 2nd to the 1st singular
    value of the centred point cloud."""
    X = np.asarray(points, float)
    X = X - X.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    return float(sv[1] / sv[0]) if sv[0] > 1e-12 else 0.0


def sim3_from_poses(A, B):
    """Similarity taking camera-to-world poses A onto poses B.

    Point-based Umeyama uses camera CENTRES only, and a handful of
    consecutive frames from a smooth trajectory are nearly COLLINEAR -- the
    rotation about that line is then unconstrained and the fit is garbage.
    Chunk overlaps are exactly that situation, which is why stitching with
    Umeyama alone produced an unusable trajectory.

    Using orientations fixes the conditioning:
      * rotation from orthogonal Procrustes on the frame orientations
        (well-posed from a SINGLE pose, and averaged over all of them),
      * scale from the spread of the centres,
      * translation from the centroids.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    assert A.shape == B.shape and A.ndim == 3 and A.shape[1:] == (4, 4)
    M = np.zeros((3, 3))
    for i in range(len(A)):
        M += B[i, :3, :3] @ A[i, :3, :3].T
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt

    ca, cb = A[:, :3, 3], B[:, :3, 3]
    ma, mb = ca.mean(0), cb.mean(0)
    da, db = ca - ma, cb - mb
    na = float(np.sqrt((da ** 2).sum()))
    s = float(np.sqrt((db ** 2).sum()) / na) if na > 1e-9 else 1.0
    t = mb - s * (R @ ma)
    return s, R, t


def apply_sim3_to_poses(poses, s, R, t):
    """Push camera-to-world poses through the similarity (s, R, t).

    A c2w pose is [R_c | c]; under x -> sRx + t the centre maps to
    sR c + t and the orientation to R R_c.  Scale does not rotate.
    """
    poses = np.asarray(poses, float)
    out = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    out[:, :3, :3] = R @ poses[:, :3, :3]
    out[:, :3, 3] = (s * (R @ poses[:, :3, 3].T).T) + t
    return out


def align_to_gt(est, gt, with_scale=True):
    """Sim(3)-align an estimated trajectory to ground truth (ATE convention)."""
    s, R, t = umeyama_sim3(np.asarray(est)[:, :3, 3], np.asarray(gt)[:, :3, 3],
                           with_scale=with_scale)
    return apply_sim3_to_poses(est, s, R, t), (s, R, t)


# --------------------------------------------------------------- metrics --
def rotation_angle_deg(R):
    """Geodesic angle of a rotation matrix, in degrees."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def ate_rmse(est, gt, with_scale=True):
    """Absolute Trajectory Error: RMSE of camera centres after Sim(3)."""
    aligned, _ = align_to_gt(est, gt, with_scale=with_scale)
    d = aligned[:, :3, 3] - np.asarray(gt, float)[:, :3, 3]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def rpe(est, gt, delta=1, with_scale=True):
    """Relative Pose Error over a fixed frame gap.

    Returns (trans_rmse, rot_rmse_deg).  The estimate is Sim(3)-aligned
    first so that translations are comparable in scale; the relative errors
    themselves are then gauge-free.
    """
    aligned, _ = align_to_gt(est, gt, with_scale=with_scale)
    gt = np.asarray(gt, float)
    tr, ro = [], []
    for i in range(len(gt) - delta):
        j = i + delta
        d_gt = np.linalg.inv(gt[i]) @ gt[j]
        d_es = np.linalg.inv(aligned[i]) @ aligned[j]
        E = np.linalg.inv(d_gt) @ d_es
        tr.append(float(np.linalg.norm(E[:3, 3])))
        ro.append(rotation_angle_deg(E[:3, :3]))
    if not tr:
        return float("nan"), float("nan")
    return (float(np.sqrt(np.mean(np.square(tr)))),
            float(np.sqrt(np.mean(np.square(ro)))))


def pairwise_pose_errors(est, gt):
    """Relative rotation / translation-direction errors over ALL pairs.

    This is the quantity behind AUC@30 in VGGT, VGGSfM and PoseDiffusion:
    for every image pair, the angular error of the relative rotation (RRA)
    and of the relative translation DIRECTION (RTA).  Both are gauge- and
    scale-free, which is why no alignment is needed.
    """
    est = np.asarray(est, float)
    gt = np.asarray(gt, float)
    n = len(gt)
    rra, rta = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dg = np.linalg.inv(gt[i]) @ gt[j]
            de = np.linalg.inv(est[i]) @ est[j]
            rra.append(rotation_angle_deg(de[:3, :3].T @ dg[:3, :3]))
            a, b = de[:3, 3], dg[:3, 3]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                rta.append(0.0)
            else:
                rta.append(float(np.degrees(np.arccos(
                    np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))))
    return np.array(rra), np.array(rta)


def auc_at(rra, rta, max_threshold=30):
    """AUC@max_threshold over the per-pair max(RRA, RTA) error.

    Definition (the one used by PoseDiffusion and inherited by VGGT):
    accuracy at threshold tau is the fraction of pairs whose max(RRA, RTA)
    is below tau; the AUC is the mean of that accuracy over integer
    thresholds 1..max_threshold, i.e. the normalised area under the
    accuracy-threshold curve.  Reported in percent.
    """
    if len(rra) == 0:
        return float("nan")
    err = np.maximum(np.asarray(rra, float), np.asarray(rta, float))
    taus = np.arange(1, int(max_threshold) + 1)
    acc = [(err < tau).mean() for tau in taus]
    return float(100.0 * np.mean(acc))


def drift_at_revisit(est, gt, revisit_pairs, with_scale=True):
    """Mean centre error over frame pairs that revisit the same place.

    The quantity chunk-stitching cannot control and a persistent memory
    can: if frames (i, j) image the same viewpoint, their estimated centres
    should coincide as closely as the ground-truth ones do.
    """
    aligned, _ = align_to_gt(est, gt, with_scale=with_scale)
    gt = np.asarray(gt, float)
    errs = []
    for i, j in revisit_pairs:
        d_est = np.linalg.norm(aligned[i, :3, 3] - aligned[j, :3, 3])
        d_gt = np.linalg.norm(gt[i, :3, 3] - gt[j, :3, 3])
        errs.append(abs(d_est - d_gt))
    return float(np.mean(errs)) if errs else float("nan")


def summarise(est, gt, delta=1):
    """Every Pilot-A column for one trajectory, in one call."""
    rra, rta = pairwise_pose_errors(est, gt)
    rpe_t, rpe_r = rpe(est, gt, delta=delta)
    return dict(ate_rmse=ate_rmse(est, gt),
                rpe_trans=rpe_t,
                rpe_rot_deg=rpe_r,
                auc30=auc_at(rra, rta),
                rra_median_deg=float(np.median(rra)) if len(rra) else float("nan"),
                rta_median_deg=float(np.median(rta)) if len(rta) else float("nan"),
                n_frames=int(len(gt)))
