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
def consensus_fuse(ctx_points, m=2, tau=0.01, ref_depth=None, tau_abs=None,
                   tau_rel=None, zdist=None, zdist_all=None, keep_singles=True):
    """ctx_points: list over contexts of dicts view->(P,3) arrays for the SAME
    pixel grid (NaN where invalid). Returns fused view->(P,3) with NaN where
    consensus fails, plus per-view keep masks."""
    fused = {}
    all_views = sorted({v for c in ctx_points for v in c})
    for v in all_views:
        stacks = np.stack([c[v] for c in ctx_points if v in c])   # (C,P,3)
        med = np.nanmedian(stacks, axis=0)                        # (P,3)
        d = np.linalg.norm(stacks - med[None], axis=-1)           # (C,P)
        if zdist is not None and v in zdist:                       # physics gate:
            z = np.nanmedian(np.stack([zc[v] for zc in zdist_all if v in zc]), 0)
            thr = np.maximum(tau_abs if tau_abs else 0.01, tau_rel * z)  # (P,)
        elif tau_abs is not None:
            thr = tau_abs
        else:
            scale = np.nanmedian(np.abs(med[:, 2])) if ref_depth is None else ref_depth
            thr = tau * max(scale, 1e-6)
        agree = (d < thr) & np.isfinite(d)
        n_ok = agree.sum(0)
        if stacks.shape[0] == 1 and keep_singles:     # sole, uncontradicted witness
            fused[v] = stacks[0]; continue
        sel = np.where(agree[..., None], stacks, np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
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
    # contexts: OVERLAPPING WINDOWS over the (ordered) view list, exactly
    # like the pose pipeline: consecutive contexts share `--overlap` views,
    # so every Sim(3) junction is anchored on many poses, and alignment is
    # chained ctx_k -> ctx_{k-1} -> ... -> ctx_0 with robust fits.
    step = max(1, a.w - a.overlap)
    starts = list(range(0, max(1, V - a.w + 1), step))
    if starts[-1] + a.w < V: starts.append(V - a.w)
    ctx_ids = [list(range(s, min(V, s + a.w))) for s in starts[:a.k_ctx] if s >= 0]               if a.k_ctx > 0 else [list(range(s, min(V, s + a.w))) for s in starts]
    outs, ctx_pts = [], []
    for ci, ids in enumerate(ctx_ids):
        o = bb.infer([paths[i] for i in ids])
        outs.append((ids, o))
        print(f"ctx {ci}: views {ids[0]}..{ids[-1]}  scale {o.extras['scene_scale']:.3f}")
    ref_ids, ref = outs[0]
    T = [np.eye(4)]; scales = [1.0]                 # ctx -> ref, chained
    for k in range(1, len(outs)):
        ids_a, oa = outs[k - 1]; ids_b, ob = outs[k]
        shared = sorted(set(ids_a) & set(ids_b))
        A = np.stack([ob.poses[ids_b.index(v)] for v in shared])
        B = np.stack([oa.poses[ids_a.index(v)] for v in shared])
        s, R, tt = sim3_from_poses(A, B)            # ctx_k -> ctx_{k-1}
        M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = tt
        Tk = T[k - 1] @ M
        T.append(Tk); scales.append(scales[k - 1] * s)
        res = np.linalg.norm((s * (R @ np.stack([p[:3, 3] for p in A]).T)).T + tt
                             - np.stack([p[:3, 3] for p in B]), axis=1)
        print(f"junction {k-1}->{k}: {len(shared)} shared views, "
              f"residual med {np.median(res):.4f} (ctx units)")
        if np.median(res) > 0.02:
            print(f"  WARNING: junction {k-1}->{k} weak; its context will "
                  f"contribute mainly where the relative gate tolerates it")
    for k, (ids, o) in enumerate(outs):
        pts = {}
        for j, v in enumerate(ids):
            K = o.extras["intrinsics_all"][j]
            P = unproject(o.depth[j], K, o.poses[j], a.stride)
            Ph = (T[k][:3, :3] @ P.T).T + T[k][:3, 3]
            pts[v] = Ph
        ctx_pts.append(pts)
    pose_of = {v: ref.poses[j] for j, v in enumerate(ref_ids)}
    # align the reference (backbone) frame to the dataset GT frame so the
    # official evaluators score in the scan's metric coordinates. Pose-head
    # only: Sim(3) from ctx0's backbone poses onto the npz GT poses.
    gt_poses = d["poses"]
    sG, RG, tG = sim3_from_poses(np.stack([pose_of[v] for v in ref_ids]),
                                 np.stack([gt_poses[v] for v in ref_ids]))
    print(f"ref->GT: scale {sG:.4f}")
    def to_gt(P): return (sG * (RG @ P.T)).T + tG
    if a.gt_cams:
        # DTU-style protocol: cameras are PROVIDED. Depth stays the estimand;
        # rays come from GT K/pose; only monocular scale rides the pose fit.
        Ks_gt = d["K"]
        from PIL import Image
        W0, H0 = Image.open(paths[0]).size          # calibration resolution
        ctx_pts = []
        for k, (ids, o) in enumerate(outs):
            sc = scales[k] * sG
            pts = {}
            for j, v in enumerate(ids):
                Hd, Wd = o.depth[j].shape
                Kv = Ks_gt[v].copy()
                Kv[0] *= Wd / W0; Kv[1] *= Hd / H0   # K at depth-map resolution
                pts[v] = unproject(o.depth[j] * sc, Kv, gt_poses[v], a.stride)
            ctx_pts.append(pts)
        print(f"gt-cams mode: metric depth scale per ctx = chain x {sG:.2f}")
    else:
        ctx_pts = [{v: to_gt(P) for v, P in c.items()} for c in ctx_pts]   # metric FIRST
    cams_gt = {v: gt_poses[v][:3, 3] for v in range(V)}
    zs = [{v: np.linalg.norm(P - cams_gt[v][None], axis=1) for v, P in c.items()}
          for c in ctx_pts]                                        # per-point depth (m)
    single = np.concatenate([p for p in ctx_pts[0].values()])
    fused_v = consensus_fuse(ctx_pts, m=a.m, tau=None, tau_abs=a.tau_m,
                             tau_rel=a.tau_rel, zdist=zs[0], zdist_all=zs,
                             keep_singles=not a.strict)
    fused = np.concatenate([p for p in fused_v.values()])
    print(f"consensus gate: max({a.tau_m} m, {a.tau_rel} x depth)")
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
    ap.add_argument("--tau-m", type=float, default=0.01, help="consensus gate floor (GT units)")
    ap.add_argument("--tau-rel", type=float, default=0.03, help="depth-proportional gate (fraction of z)")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--gt-cams", action="store_true",
                    help="unproject with provided cameras (DTU protocol); depth remains the estimand")
    ap.add_argument("--strict", action="store_true", help="drop sole-witness views (old behavior)")
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--out-dir", default="outputs/points/run")
    a = ap.parse_args()
    synthetic() if a.synthetic else real(a)
