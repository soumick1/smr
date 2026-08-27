"""Relocalisation into memory -- Pilot A's mechanism applied to ONE image.

A memory built from mapping sessions is asked where a new image was taken.
The backbone cannot answer this on its own: it has no map.  The rows of the
relocalisation table are the same question asked with less and less memory:

  smr     memory proposes anchor sites for the query (address cue -> stored
          views), ONE small backbone pass on [query + site frames] places
          the query against the sites' stored poses (consensus between two
          sites when available).
  plain   identical, but sites come from plain descriptor nearest neighbour
          (no scaffold address) -- what a database gives.
  lastk   no memory at all: the query is passed with the LAST K map
          keyframes (recency is all a backbone has) and placed against
          their poses.
  oracle  sites chosen by ground-truth proximity: separates retrieval
          failure from placement failure (diagnostic, never a row a system
          could run).

Every row places the query with the same pass machinery; only the anchors
differ.  Errors are metric because the MAP is aligned to ground truth by one
Sim(3) (the SLAM-protocol alignment); the query's pose is never aligned.
"""
from __future__ import annotations

import numpy as np

from ..eval.trajectory import align_to_gt, rotation_angle_deg
from . import sim3
from .memory_index import DescriptorIndex


# ------------------------------------------------------------------ map ----
class MemoryMap:
    """A stitched multi-session map aligned to ground truth, plus its index."""

    def __init__(self, result, gt_map, index, descriptors, owner_paths=None):
        frames = sorted(result["local"])
        est = result["est"]
        aligned, (s, R, t) = align_to_gt(est, gt_map)
        self.align = (float(s), R, t)
        self.frames = frames                       # keyframe ids (map index space)
        self.pose = {g: aligned[i] for i, g in enumerate(frames)}
        self.owner = {g: int(o) for g, o in zip(frames, result["owner"])}
        self.index = index
        for g in frames:                           # index now speaks metric
            index.update_pose(g, self.pose[g])
        self.descriptors = descriptors
        self.map_ate = float(np.sqrt(np.mean(np.sum((aligned[:, :3, 3] - gt_map[:, :3, 3]) ** 2, 1))))
        self.owner_paths = owner_paths

    def sites_for(self, desc, mode, n_sites=2, top=5, desc_thresh=0.5,
                  partner_gap=3, gt_pose=None, K=15):
        """Anchor frames for one query, per mode.  Returns list of (j, partner)."""
        stored = self.pose
        if mode == "lastk":
            last = self.frames[-K:]
            return [(g, None) for g in last]
        if mode == "oracle":
            assert gt_pose is not None
            c = np.stack([stored[g][:3, 3] for g in self.frames])
            d = np.linalg.norm(c - gt_pose[:3, 3], axis=1)
            order = [self.frames[i] for i in np.argsort(d)]
            cands = [(g, 0.0, 1.0) for g in order[:top * 3]]
        else:
            cands = self.index.propose(desc, exclude=(), top=top * 3,
                                       desc_thresh=desc_thresh)
        sites, used = [], set()
        for (j, score, cos) in cands:
            oc = self.owner[j]
            partner = None
            for gap in range(partner_gap, 0, -1):
                for p in (j + gap, j - gap):
                    if p in stored and self.owner.get(p) == oc:
                        partner = p
                        break
                if partner is not None:
                    break
            if partner is None or oc in used:
                continue
            sites.append((j, partner))
            used.add(oc)
            if len(sites) >= n_sites:
                break
        return sites


# ------------------------------------------------------------- localise ----
def _dir_err_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1, 1))))


def localise(mmap, q_desc, run_pass, mode="smr", n_sites=2, gt_pose=None,
             site_rot_deg=10.0, site_dir_deg=25.0, agree_rot_deg=10.0,
             agree_pos=0.3, K=15, **kw):
    """Place one query.  run_pass(anchor_frames) -> poses of [query] + anchors
    in one pass (the query first).  Returns a dict with T (4x4 or None)."""
    sites = mmap.sites_for(q_desc, mode, n_sites=n_sites, gt_pose=gt_pose, K=K, **kw)
    if not sites:
        return dict(T=None, reason="no_proposal", n_sites=0)
    anchors = []
    for j, p in sites:
        for g in (j, p):
            if g is not None and g not in anchors:
                anchors.append(g)
    P = run_pass(anchors)                       # (1 + len(anchors), 4, 4)
    pos = {g: i + 1 for i, g in enumerate(anchors)}
    stored = mmap.pose

    if mode == "lastk":
        A = P[[pos[g] for g in anchors]]
        B = np.stack([stored[g] for g in anchors])
        S, keep, info = sim3.fit_poses_robust(A, B, min_inliers=3)
        return dict(T=sim3.apply(S, P[:1])[0], reason="ok", n_sites=len(anchors),
                    inliers=int(keep.sum()))

    # verify each site inside the pass, then fit per site
    good, fits = [], []
    for j, p in sites:
        rel_pass = np.linalg.inv(P[pos[j]]) @ P[pos[p]]
        rel_st = np.linalg.inv(stored[j]) @ stored[p]
        rot = rotation_angle_deg(rel_pass[:3, :3].T @ rel_st[:3, :3])
        dr = _dir_err_deg(rel_pass[:3, 3], rel_st[:3, 3])
        if rot < site_rot_deg and dr < site_dir_deg:
            S, info = sim3.fit_poses(P[[pos[j], pos[p]]], np.stack([stored[j], stored[p]]))
            if info["scale_ok"]:
                good.append((j, p)); fits.append(S)
    if not good:
        return dict(T=None, reason="no_verified_site", n_sites=0)
    if len(good) >= 2:
        qs = [sim3.apply(S, P[:1])[0] for S in fits]
        rot_d = rotation_angle_deg(qs[0][:3, :3].T @ qs[1][:3, :3])
        pos_d = float(np.linalg.norm(qs[0][:3, 3] - qs[1][:3, 3]))
        if rot_d <= agree_rot_deg and pos_d <= agree_pos:
            frames = [g for s in good for g in s]
            S, keep, info = sim3.fit_poses_robust(P[[pos[g] for g in frames]],
                                                  np.stack([stored[g] for g in frames]), min_inliers=3)
            return dict(T=sim3.apply(S, P[:1])[0], reason="consensus", n_sites=2)
        # disagreement: a look-alike among the two.  Keep the site whose
        # internal geometry matched memory more tightly; mark ambiguous.
        return dict(T=sim3.apply(fits[0], P[:1])[0], reason="ambiguous", n_sites=1,
                    disagreement=dict(rot_deg=rot_d, pos=pos_d))
    return dict(T=sim3.apply(fits[0], P[:1])[0], reason="single_site", n_sites=1)


# ------------------------------------------------------------ corruptions ---
def corrupt(img, spec, rng):
    """ImageNet-C-style degradations on an (H, W, 3) image in [0, 1].
    spec: 'gauss:0.08' | 'occlude:0.3' | 'blur:4' | 'dark:0.5' | 'none'."""
    img = np.asarray(img, float)
    if not spec or spec == "none":
        return img
    kind, _, val = spec.partition(":")
    v = float(val) if val else 0.0
    if kind == "gauss":
        return np.clip(img + rng.normal(scale=v, size=img.shape), 0, 1)
    if kind == "occlude":                       # one grey rectangle of area fraction v
        H, W = img.shape[:2]
        h = int(np.sqrt(v) * H); w = int(np.sqrt(v) * W)
        y = rng.integers(0, max(1, H - h)); x = rng.integers(0, max(1, W - w))
        out = img.copy(); out[y:y + h, x:x + w] = 0.5
        return out
    if kind == "blur":                          # separable Gaussian, sigma v px
        sig = max(v, 1e-3); r = int(3 * sig)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
        out = img.copy()
        for c in range(out.shape[2]):
            ch = np.pad(out[..., c], ((r, r), (r, r)), mode="edge")
            ch = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, ch)
            ch = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, ch)
            out[..., c] = ch
        return out
    if kind == "dark":
        return np.clip(img * (1.0 - v), 0, 1)
    raise KeyError(spec)


# ---------------------------------------------------------------- metrics --
def pose_error(T_est, T_gt):
    """(translation metres, rotation degrees) of a query pose; both metric
    because the map is aligned to GT and the query is placed in the map."""
    if T_est is None:
        return float("inf"), float("inf")
    return (float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])),
            rotation_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3]))


def summarise(errs, thresholds=((0.05, 5.0), (0.10, 10.0), (0.25, 25.0))):
    """errs: list of (t, r).  Median over ALL queries (misses count as inf,
    as in the 7-Scenes relocalisation tables) and recall at thresholds."""
    t = np.array([e[0] for e in errs]); r = np.array([e[1] for e in errs])
    out = dict(n=len(errs), median_t_cm=float(np.median(t) * 100) if len(t) else float("nan"),
               median_r_deg=float(np.median(r)) if len(r) else float("nan"),
               fail_rate=float(np.mean(~np.isfinite(t))) if len(t) else float("nan"))
    for tt, rr in thresholds:
        out[f"recall_{int(tt * 100)}cm_{int(rr)}deg"] = float(np.mean((t <= tt) & (r <= rr))) if len(t) else float("nan")
    return out
