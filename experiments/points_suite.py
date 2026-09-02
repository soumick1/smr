#!/usr/bin/env python3
"""Multi-view depth / point-map suite: single pass vs +SMR cross-context
consensus fusion (VGGT Tables 2 & 3 regime).

Design rule (from six recorded refutations): depth is only ever the
ESTIMAND; every transform comes from the pose head (robust Sim(3) on
shared-view poses). Fusion = per-point cross-context consensus: keep a
point if >= m contexts agree within tau_fuse (relative to depth), fuse
by coordinate-wise median of the agreeing set.

Modes
-----
--synthetic            no GPU: constructed GT surface, simulated contexts
                       with correlated bias + iid noise + outliers; asserts
                       fused Overall < single-context Overall. Run this
                       before any real data.
--gt scan.npz          real: image_paths + K + c2w poses (prep scripts);
                       contexts = --k-ctx view subsets of size --w built by
                       spread sampling with shared views; backbone runs one
                       pass per context; alignment context->ref via
                       sim3_from_poses on shared views (pose head ONLY);
                       writes fused + single point clouds as .ply for the
                       dataset evaluator (dtu_eval.py / eth3d_eval.sh).

    python experiments/points_suite.py --synthetic
    python experiments/points_suite.py --gt data/gt/dtu/scan24.npz \
        --backbone vggt_omega --w 16 --k-ctx 6 --m 2 --tau-fuse 0.01 \
        --out-dir outputs/points/dtu_scan24_omega
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, "src")
from smr.eval.trajectory import sim3_from_poses            # pose-head alignment

# ----------------------------------------------------------------- fusion ---
def consensus_fuse(ctx_points, m=2, tau=0.01, ref_depth=None):
    """ctx_points: list over contexts of dicts view->(P,3) arrays for the SAME
    pixel grid (NaN where invalid). Returns fused view->(P,3) with NaN where
    consensus fails, plus per-view keep masks."""
    fused = {}
    for v in ctx_points[0]:
        stacks = np.stack([c[v] for c in ctx_points if v in c])   # (C,P,3)
        med = np.nanmedian(stacks, axis=0)                        # (P,3)
        scale = np.nanmedian(np.abs(med[:, 2])) if ref_depth is None else ref_depth
        d = np.linalg.norm(stacks - med[None], axis=-1)           # (C,P)
        agree = (d < tau * max(scale, 1e-6)) & np.isfinite(d)
        n_ok = agree.sum(0)
        sel = np.where(agree[..., None], stacks, np.nan)
        out = np.nanmedian(sel, axis=0)
        out[n_ok < m] = np.nan
        fused[v] = out
    return fused

def acc_comp(pred, gt, max_dist=20.0):
    """Symmetric chamfer-style Acc/Comp with MaxDist discard (official DTU
    convention; distances in the clouds' units)."""
    from scipy.spatial import cKDTree
    dp = cKDTree(gt).query(pred, workers=-1)[0]
    dg = cKDTree(pred).query(gt, workers=-1)[0]
    acc = float(dp[dp < max_dist].mean()) if (dp < max_dist).any() else float("inf")
    comp = float(dg[dg < max_dist].mean()) if (dg < max_dist).any() else float("inf")
    return acc, comp, 0.5 * (acc + comp)

# -------------------------------------------------------------- synthetic ---
def synthetic(seed=0, n_ctx=5, n_pts=4000, bias=0.02, noise=0.01, outlier=0.08):
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(-1, 1, (2, n_pts))
    gt = np.stack([u, v, 0.2 * np.sin(3 * u) * np.cos(2 * v) + 2.0], -1)
    ctxs = []
    for c in range(n_ctx):
        b = rng.normal(0, bias, 3)                       # correlated per-context bias
        p = gt + b + rng.normal(0, noise, gt.shape)      # iid noise
        out = rng.random(n_pts) < outlier                # gross outliers
        p[out] += rng.normal(0, 0.5, (out.sum(), 3))
        ctxs.append({0: p})
    single = ctxs[0][0]
    fused = consensus_fuse(ctxs, m=2, tau=0.02, ref_depth=2.0)[0]
    ok = np.isfinite(fused).all(-1)
    a1, c1, o1 = acc_comp(single, gt, max_dist=1.0)
    a2, c2, o2 = acc_comp(fused[ok], gt, max_dist=1.0)
    print(f"synthetic  single: Acc {a1:.4f}  Comp {c1:.4f}  Overall {o1:.4f}")
    print(f"synthetic  fused : Acc {a2:.4f}  Comp {c2:.4f}  Overall {o2:.4f}  "
          f"(kept {ok.mean()*100:.1f}% of pixels)")
    assert o2 < o1, "consensus fusion must beat a single context on synthetic GT"
    print("SYNTHETIC PROOF PASSED: fused Overall < single Overall")

# ------------------------------------------------------------------- real ---
def unproject(depth, K, c2w, stride=2):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    pix = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3).astype(float)
    rays = (np.linalg.inv(K) @ pix.T).T
    pts_c = rays * z.reshape(-1, 1)
    pts_w = (c2w[:3, :3] @ pts_c.T).T + c2w[:3, 3]
    pts_w[~ok.reshape(-1)] = np.nan
    return pts_w

def real(a):
    from smr.backbones import get_backbone
    d = np.load(a.gt, allow_pickle=True)
    paths = [str(p) for p in d["image_paths"]]
    V = len(paths)
    bb = get_backbone(a.backbone)
    # contexts: k spread windows with guaranteed pairwise shared views
    idx = np.arange(V)
    ctx_ids = [sorted(set(np.linspace(o, V - 1, a.w).astype(int))) for o in
               np.linspace(0, max(1, V - a.w), a.k_ctx).astype(int)]
    outs, ctx_pts = [], []
    for ci, ids in enumerate(ctx_ids):
        o = bb.infer([paths[i] for i in ids])
        outs.append((ids, o))
        print(f"ctx {ci}: {len(ids)} views  scale {o.extras['scene_scale']:.3f}")
    ref_ids, ref = outs[0]
    pose_of = {v: ref.poses[j] for j, v in enumerate(ref_ids)}
    for ids, o in outs:
        shared = [v for v in ids if v in pose_of]
        A = np.stack([o.poses[ids.index(v)] for v in shared])
        B = np.stack([pose_of[v] for v in shared])
        s, R, t = sim3_from_poses(A, B)                    # ctx -> ref (pose head only)
        pts = {}
        for j, v in enumerate(ids):
            K = o.extras["intrinsics_all"][j]
            P = unproject(o.depth[j], K, o.poses[j], a.stride)
            pts[v] = (s * (R @ P.T)).T + t
        ctx_pts.append(pts)
    single = np.concatenate([p for p in ctx_pts[0].values()])
    fused_v = consensus_fuse(ctx_pts, m=a.m, tau=a.tau_fuse)
    fused = np.concatenate([p for p in fused_v.values()])
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, P in (("single", single), ("fused", fused)):
        P = P[np.isfinite(P).all(-1)]
        with open(out / f"{name}.ply", "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex %d\n"
                    "property float x\nproperty float y\nproperty float z\nend_header\n" % len(P))
            np.savetxt(f, P, fmt="%.5f")
        print(f"{name}: {len(P):,} pts -> {out/f'{name}.ply'}")
    json.dump(dict(gt=a.gt, backbone=a.backbone, k_ctx=a.k_ctx, w=a.w,
                   m=a.m, tau_fuse=a.tau_fuse), open(out / "run.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--gt"); ap.add_argument("--backbone", default="vggt_omega")
    ap.add_argument("--w", type=int, default=16); ap.add_argument("--k-ctx", type=int, default=6)
    ap.add_argument("--m", type=int, default=2); ap.add_argument("--tau-fuse", type=float, default=0.01)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--out-dir", default="outputs/points/run")
    a = ap.parse_args()
    synthetic() if a.synthetic else real(a)
