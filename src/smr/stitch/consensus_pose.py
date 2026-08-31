"""Memory as test-time context consensus for the camera-pose protocol.

The single-window benchmarks (CO3Dv2 / RealEstate10K, 10 frames, AUC@30)
give the backbone one context and take its answer.  A backbone with a
first-frame reference convention is order-sensitive, so the answer varies
under permutation of the SAME frames -- variance the protocol never sees.
The memory turns that variance into accuracy: run k permuted (optionally
overlapping-subset) passes, bind every pass, express each pass in the frame
of a shared reference by the pose-head Sim(3) fit (the one channel the
robust backend can trust), and read each pairwise pose out by robust
consensus across contexts.  Training-free, protocol-compliant: only the
given frames are ever seen.
"""
from __future__ import annotations

import numpy as np

from . import sim3


def relative_rotation_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def translation_angle_deg(ta, tb):
    na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(ta @ tb / (na * nb), -1, 1))))


def pairwise_errors(pred, gt):
    """RRA/RTA errors for all ordered pairs of c2w poses."""
    n = len(gt)
    rot, trn = [], []
    for i in range(n):
        for j in range(i + 1, n):
            Rg = gt[j][:3, :3].T @ gt[i][:3, :3]
            tg = gt[j][:3, :3].T @ (gt[i][:3, 3] - gt[j][:3, 3])
            Rp = pred[j][:3, :3].T @ pred[i][:3, :3]
            tp = pred[j][:3, :3].T @ (pred[i][:3, 3] - pred[j][:3, 3])
            rot.append(relative_rotation_deg(Rg, Rp))
            trn.append(translation_angle_deg(tg, tp))
    return np.array(rot), np.array(trn)


def auc_at(rot, trn, tau=30.0, step=1.0):
    """Area under min(RRA,RTA) accuracy curve up to tau degrees (the
    RelPose++/VGGT convention), in [0, 1]."""
    err = np.maximum(rot, trn)
    ts = np.arange(step, tau + step, step)
    acc = [(err <= t).mean() for t in ts]
    return float(np.mean(acc))


def consensus_poses(passes, n_frames, min_ctx=2):
    """Fuse k passes into one pose set per frame.

    passes: list of (frame_ids, poses) -- each pass covers a subset (or a
    permutation) of range(n_frames), poses c2w in the pass's own gauge.
    Every pass is expressed in the gauge of a shared reference (the pass
    with the most frames) via a robust Sim(3) fit on the pose head, then
    each frame's pose is the geodesic/robust mean over contexts.  Frames
    seen by fewer than min_ctx contexts keep the reference pass's answer.
    Returns (n_frames, 4, 4) and the per-frame context counts."""
    ref = int(np.argmax([len(f) for f, _ in passes]))
    ref_ids, ref_P = passes[ref]
    ref_pose = {g: ref_P[i] for i, g in enumerate(ref_ids)}
    votes = {g: [ref_pose[g]] for g in ref_ids}
    for pi, (ids, P) in enumerate(passes):
        if pi == ref:
            continue
        shared = [g for g in ids if g in ref_pose]
        if len(shared) < 3:
            continue
        A = np.stack([P[list(ids).index(g)] for g in shared])
        B = np.stack([ref_pose[g] for g in shared])
        S, keep, info = sim3.fit_poses_robust(A, B, min_inliers=3)
        if not info.get("scale_ok", True):
            continue
        M = sim3.apply(S, np.stack([P[i] for i in range(len(ids))]))
        for i, g in enumerate(ids):
            votes.setdefault(g, []).append(M[i])
    out = np.stack([_robust_pose_mean(votes[g]) if len(votes.get(g, [])) >= min_ctx
                    else ref_pose.get(g, np.eye(4)) for g in range(n_frames)])
    return out, {g: len(votes.get(g, [])) for g in range(n_frames)}


def _robust_pose_mean(Ts):
    """Median translation; chordal-mean rotation with one reweighting round."""
    Ts = np.stack(Ts)
    t = np.median(Ts[:, :3, 3], axis=0)
    Rs = Ts[:, :3, :3]
    M = Rs.mean(0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    ang = np.array([relative_rotation_deg(R, Ri) for Ri in Rs])
    w = 1.0 / np.maximum(ang, 1.0)
    M = (Rs * w[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out = np.eye(4); out[:3, :3] = R; out[:3, 3] = t
    return out


def pairwise_median_errors(passes, gt):
    """Per-pair robust consensus: each pass containing frames (i, j) casts one
    vote for their relative pose; the chordal-median rotation and the median
    translation direction are scored against GT.  No global alignment, so a
    bad-reference pass cannot poison pairs it estimates well, and pairs it
    estimates badly are outvoted.  Returns (rot_errs, trn_errs) over pairs."""
    n = len(gt)
    where = []
    for ids, P in passes:
        where.append({g: k for k, g in enumerate(ids)})
    rot, trn = [], []
    for i in range(n):
        for j in range(i + 1, n):
            Rs, ts = [], []
            for (ids, P), pos in zip(passes, where):
                if i not in pos or j not in pos:
                    continue
                Rs.append(P[pos[j]][:3, :3].T @ P[pos[i]][:3, :3])
                tv = P[pos[j]][:3, :3].T @ (P[pos[i]][:3, 3] - P[pos[j]][:3, 3])
                nv = np.linalg.norm(tv)
                if nv > 1e-9:
                    ts.append(tv / nv)
            if not Rs:
                continue
            R = _chordal_median(Rs)
            tm = _direction_median(ts) if ts else np.zeros(3)
            Rg = gt[j][:3, :3].T @ gt[i][:3, :3]
            tg = gt[j][:3, :3].T @ (gt[i][:3, 3] - gt[j][:3, 3])
            rot.append(relative_rotation_deg(Rg, R))
            trn.append(translation_angle_deg(tg, tm))
    return np.array(rot), np.array(trn)


def _chordal_median(Rs, iters=3):
    Rs = np.stack(Rs)
    w = np.ones(len(Rs))
    R = None
    for _ in range(iters):
        M = (Rs * w[:, None, None]).sum(0)
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        ang = np.array([relative_rotation_deg(R, Ri) for Ri in Rs])
        w = 1.0 / np.maximum(ang, 2.0)
    return R


def _direction_median(ts, iters=3):
    ts = np.stack(ts)
    w = np.ones(len(ts))
    v = ts.mean(0)
    for _ in range(iters):
        v = (ts * w[:, None]).sum(0)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return ts[0]
        v = v / nv
        ang = np.degrees(np.arccos(np.clip(ts @ v, -1, 1)))
        w = 1.0 / np.maximum(ang, 2.0)
    return v
