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

v122 (protocol facts, see UPDATE_NOTES_v122.md):
  --geo-views N   PatchmatchNet geometric-consistency filter (MASt3R Sec. 4.5
                  post-processing; VGGT Table 2 "follows MASt3R"): keep a pixel
                  if >= N source views agree within --geo-pix px and --geo-rel
                  relative depth; depth averaged over the agreeing views
                  (reference: 1.0 px, 0.01, N=5, 10 nearest sources).
  --geo-cams      pred: the backbone's own cameras (self-consistency; DUSt3R/
                  VGGT x-block); gt: provided cameras (requires --gt-cams).
  --save-maps     write maps.npz with per-view point maps on the pixel grid
                  (single context + fused) so ETH3D can be scored with the
                  correspondence-Umeyama protocol offline (eth3d_pointmap_eval).
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, "src")
from smr.eval.trajectory import sim3_from_poses            # pose-head alignment
from smr.eval.mvs_fusion import geo_consistency, select_sources   # v122: MASt3R/PatchmatchNet filter
from smr.eval.eth3d_gt import grid_map                              # v122: resize/crop of the backbone grid


def K_at_grid(K0, wh_img, depth_hw):
    """Calibration K (image pixels) -> K on the backbone depth grid, honouring
    VGGT's anisotropic resize and centre crop (grid_map)."""
    sx, sy, oy = grid_map(wh_img, depth_hw)
    K = K0.copy(); K[0] /= sx; K[1] /= sy; K[1, 2] -= oy
    return K

# ----------------------------------------------------------------- fusion ---
def consensus_fuse(ctx_points, m=2, tau=0.01, ref_depth=None, tau_abs=None,
                   tau_rel=None, zdist=None, zdist_all=None, keep_singles=True,
                   fallback="median", abstain_rel=0.10, abstain_factor=5.0):
    """ctx_points: list over contexts of dicts view->(P,3) arrays for the SAME
    pixel grid (NaN where invalid). Returns fused view->(P,3).

    Where fewer than m contexts agree within the gate, `fallback` decides:
      "median" (v126 default): the median of ALL witnesses -- the robust
                estimate, symmetric, privileges no pass; never emptier than a pass;
      "first":  the earliest witness (v125; privileges context 0 -- on ETH3D
                relief that read was the failing one and this undid the gain);
      "none":   NaN (abstain; v124 behaviour, most accurate, loses completeness).
    v127: with "median", pixels whose witnesses disagree GROSSLY -- median
    witness-to-consensus distance above abstain_rel x depth (or abstain_factor
    x the gate when no depth is known) -- are abstained (NaN) instead: reads
    that contradict each other by metres carry no usable estimate, and
    keeping a compromise point drags the downstream alignment (ETH3D relief)."""
    fused = {}
    n_fallback = 0; n_abstain = 0
    if fallback is True: fallback = "first"
    if fallback is False or fallback is None: fallback = "none"
    all_views = sorted({v for c in ctx_points for v in c})
    import warnings
    for v in all_views:
        stacks = np.stack([c[v] for c in ctx_points if v in c])   # (C,P,3)
        with warnings.catch_warnings():                            # all-NaN pixels are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stacks, axis=0)                    # (P,3)
            d = np.linalg.norm(stacks - med[None], axis=-1)       # (C,P)
            if zdist is not None and v in zdist:                   # physics gate:
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
        fail = n_ok < m
        out[fail] = np.nan
        if fallback == "median" and fail.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spread = np.nanmedian(d, axis=0)         # typical witness deviation from the consensus
            if zdist is not None and v in zdist:
                limit = abstain_rel * z
            else:
                limit = abstain_factor * (thr if np.ndim(thr) else np.full(len(fail), thr))
            gross = (spread > limit) if abstain_rel > 0 else np.zeros(len(fail), bool)
            use = fail & ~gross & np.isfinite(med).all(1)  # modest disagreement: median of all witnesses
            out[use] = med[use]; n_fallback += int(use.sum())
            n_abstain += int((fail & gross).sum())
        elif fallback == "first" and fail.any():
            first = np.full(out.shape, np.nan)
            for c in range(stacks.shape[0]):           # earliest finite witness per pixel
                take = np.isnan(first).any(1) & np.isfinite(stacks[c]).all(1)
                first[take] = stacks[c][take]
            use = fail & np.isfinite(first).all(1)
            out[use] = first[use]; n_fallback += int(use.sum())
        fused[v] = out
    if fallback != "none":
        print(f"consensus: {n_fallback:,} pixels below agreement m={m} -> fallback '{fallback}'"
              + (f"; {n_abstain:,} abstained (witnesses disagree > {abstain_rel*100:.0f}% of depth)" if fallback == "median" else ""))
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
_CONF_GUARD_NOTE = set()


def conf_mask(o, j, pct, abs_thr=None, key="conf", guard=0.05, guard_pct=32.0):
    """Per-pixel keep mask from the backbone's confidence map `key`:
    percentile cut (--conf-pct) and/or absolute cut (--conf-abs; VGGT's expp1
    confidences are >1, and C>2 <=> positive loss weight -- the threshold the
    independent DTU study of Langendoerfer et al. 2026 recommends).
    v133 guard: if the absolute cut would keep fewer than `guard` of the pixels
    (e.g. a sigmoid confidence in (0,1) against C>2), fall back to keeping the
    top (100-guard_pct)% by percentile and say so once."""
    c = o.extras.get(key) if hasattr(o, "extras") and o.extras else None
    if (pct is None and abs_thr is None) or c is None:
        if (pct is not None or abs_thr is not None) and c is None:
            print(f"  [warn] confidence filter set but backbone exposes no '{key}'; keeping all")
        return None
    cj = np.asarray(c[j], dtype=np.float32)
    fin = np.isfinite(cj)
    m = fin.copy()
    if abs_thr is not None:
        m_abs = fin & (cj > abs_thr)
        if m_abs.sum() < guard * fin.sum():
            m &= cj >= np.percentile(cj[fin], guard_pct)
            if key not in _CONF_GUARD_NOTE:
                _CONF_GUARD_NOTE.add(key)
                print(f"  [conf] absolute cut > {abs_thr} keeps {m_abs.mean()*100:.1f}% of '{key}' "
                      f"(range {np.nanmin(cj):.2f}-{np.nanmax(cj):.2f}); using the top {100-guard_pct:.0f}% by percentile instead")
        else:
            m &= m_abs
    if pct:
        m &= cj >= np.percentile(cj[fin], pct)
    return m


def pointmap_points(pm, stride=2, mask=None):
    """Point-map head (H, W, 3) -> (P, 3) on the strided grid, NaN where masked."""
    H, W = pm.shape[:2]
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    P = np.asarray(pm, np.float64)[ys, xs].reshape(-1, 3).copy()
    if mask is not None:
        P[~mask[ys, xs].ravel()] = np.nan
    return P

def unproject(depth, K, c2w, stride=2, mask=None):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0)
    if mask is not None:
        ok &= mask[ys, xs]
    pix = np.stack([xs, ys, np.ones_like(xs)], -1).reshape(-1, 3).astype(float)
    rays = (np.linalg.inv(K) @ pix.T).T
    pts_c = rays * z.reshape(-1, 1)
    pts_w = (c2w[:3, :3] @ pts_c.T).T + c2w[:3, 3]
    pts_w[~ok.reshape(-1)] = np.nan
    return pts_w

def content_align(ctx_pts, ctx_cams, model="scale", trim=0.2, iters=2, max_corr=300000, seed=0,
                  ref="mean"):
    """Re-measure each later context against the memory's cards on shared views.

    ctx_pts: list of {view: (P,3)} in ONE frame; shared views of context k and
    any earlier context give per-pixel correspondences (same grid).
    model="scale": the per-pass point-vs-camera scale bias is a scaling about
    each view's camera centre (ctx_cams[k][view], the pass's own placed cameras
    or the GT cameras); solved in closed form with trimming.  model="sim3": a
    trimmed Umeyama Sim(3) on the correspondences (general misplacement).
    ref="first": context 0 is the reference (its scale is kept).  ref="mean"
    (v125 default, model="scale" only): after the pairwise factors are known
    the whole set is renormalised so the geometric mean of the applied depth
    scales is 1 -- the spread between passes is removed while the consensus
    scale is preserved, and no pass is privileged.  Returns the corrected list.
    """
    cams_all = ctx_cams; ref_mode = ref
    from smr.eval.mvs_fusion import umeyama_trimmed, apply_sim3
    rng = np.random.default_rng(seed)
    out = [dict(c) for c in ctx_pts]
    applied = [1.0] * len(out)                          # cumulative depth scale per context
    for k in range(1, len(out)):
        for it in range(iters):
            X, Y, C = [], [], []
            for v, P in out[k].items():
                ref = [j for j in range(k) if v in out[j]]
                if not ref:
                    continue
                Q = out[ref[-1]][v]                     # latest card for this view
                ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
                if ok.sum() < 50:
                    continue
                X.append(P[ok]); Y.append(Q[ok])
                if model == "scale":
                    C.append(np.repeat(cams_all[k][v][None], ok.sum(), 0))
            if not X:
                print(f"content-align ctx {k}: no shared views, left as placed")
                break
            X = np.concatenate(X); Y = np.concatenate(Y)
            C = np.concatenate(C) if C else None
            if len(X) > max_corr:
                sel = rng.choice(len(X), max_corr, replace=False)
                X, Y = X[sel], Y[sel]; C = C[sel] if C is not None else None
            res0 = np.linalg.norm(X - Y, axis=1)
            if model == "sim3":
                s, R, t, inl = umeyama_trimmed(X, Y, trim=trim, iters=3)
                out[k] = {v: apply_sim3(P, s, R, t) for v, P in out[k].items()}
                res1 = np.linalg.norm(apply_sim3(X, s, R, t) - Y, axis=1)
                print(f"content-align ctx {k} it{it}: {len(X):,} corr, scale {s:.4f} |t| {np.linalg.norm(t):.4f}, "
                      f"median residual {np.median(res0):.4f} -> {np.median(res1):.4f}")
            else:
                dX, dY = X - C, Y - C
                keep = np.ones(len(X), bool)
                for _ in range(3):
                    s = float((dX[keep] * dY[keep]).sum() / max((dX[keep] ** 2).sum(), 1e-12))
                    r = np.linalg.norm(C + s * dX - Y, axis=1)
                    keep = r <= np.quantile(r, 1.0 - trim)
                cam_of = cams_all[k]
                out[k] = {v: cam_of[v] + s * (P - cam_of[v]) for v, P in out[k].items()}
                applied[k] *= s
                res1 = np.linalg.norm(C + s * dX - Y, axis=1)
                print(f"content-align ctx {k} it{it}: {len(X):,} corr, depth scale x{s:.4f} about the pass cameras, "
                      f"median residual {np.median(res0):.4f} -> {np.median(res1):.4f}")
    same_views = len(out) > 1 and all(set(c) == set(out[0]) for c in out[1:])
    if ref_mode == "mean" and model == "scale" and same_views:
        # v130: the symmetric reference is a REPEATED-READS operation (all contexts see
        # the same views).  For chained windows a global depth scale is absorbed by the
        # evaluator's gauge, and scaling each window about its OWN placed cameras pulls
        # shared views apart by (1/g-1)*(camera-placement difference) -- on ETH3D windows
        # that cost 45% of the pixels.  Reads only, and about ONE common centre per view
        # (the earliest context's camera), so coincident points stay coincident.
        g = float(np.exp(np.mean(np.log(applied))))     # geometric mean of the applied factors
        common = {}
        for k in range(len(out)):
            for v in out[k]:
                common.setdefault(v, cams_all[k][v])
        for k in range(len(out)):
            out[k] = {v: common[v] + (1.0 / g) * (P - common[v]) for v, P in out[k].items()}
        print(f"content-align: symmetric reference -- factors {[round(a, 4) for a in applied]} "
              f"renormalised by 1/{g:.4f} about one centre per view (geometric mean = 1; no pass privileged)")
    elif ref_mode == "mean" and model == "scale" and len(out) > 1:
        print("content-align: contexts are windows (different views) -> no renormalisation; the global scale is the gauge's")
    return out


def real(a):
    from smr.backbones import get_backbone
    d = np.load(a.gt, allow_pickle=True)
    sel_ids = list(range(len(d["image_paths"])))          # indices into the GT npz
    if a.n_views and a.n_views < len(d["image_paths"]):
        rng = np.random.RandomState(a.view_seed)
        sel = np.sort(rng.choice(len(d["image_paths"]), a.n_views, replace=False))
        d = {k: (np.asarray(d[k])[sel] if k in ("image_paths", "poses", "K") else d[k])
             for k in d.files}
        sel_ids = [int(x) for x in sel]
        print(f"protocol subsample: {a.n_views} views (seed {a.view_seed}): {sel_ids}")
    pilot = None
    if a.pilot:
        # v123: place the SAME backbone passes under a pose-pipeline row's
        # junctions (chained / smr / smr_pgo / ...): frames = the pilot's
        # keyframes, contexts = its chunks, placement = Sim(3) from each pass
        # onto the row's trajectory.  Depth is identical across rows by
        # construction; the rows differ only in how the passes are placed.
        Pz = np.load(a.pilot, allow_pickle=True)
        rows = [str(r) for r in Pz["rows"]]
        if a.row not in rows:
            raise SystemExit(f"--row {a.row} not in {a.pilot}: rows={rows}")
        key = [int(i) for i in Pz["key"]]
        if a.n_views and a.n_views < len(np.load(a.gt, allow_pickle=True)["image_paths"]):
            raise SystemExit("--pilot is incompatible with --n-views (frames come from the pilot npz)")
        d = {k: (np.asarray(d[k])[key] if k in ("image_paths", "poses", "K") else d[k]) for k in d.files}
        sel_ids = key
        pilot = dict(est=np.asarray(Pz[f"est_{a.row}"], float),
                     chunks=[[int(i) for i in c] for c in Pz["chunks"]], row=a.row)
        assert len(pilot["est"]) == len(key), (len(pilot["est"]), len(key))
        print(f"pilot row '{a.row}': {len(key)} keyframes, {len(pilot['chunks'])} chunks from {a.pilot}")
    paths = [str(p) for p in d["image_paths"]]
    V = len(paths)
    def _auto(v):
        for cast in (int, float):
            try: return cast(v)
            except ValueError: pass
        return {"true": True, "false": False}.get(v.lower(), v)
    bb_kw = dict((k, _auto(v)) for k, v in (x.split("=", 1) for x in (a.backbone_kw or [])))
    if bb_kw:
        print(f"backbone kwargs: {bb_kw}")
    bb = get_backbone(a.backbone, **bb_kw)
    # contexts: OVERLAPPING WINDOWS over the (ordered) view list, exactly
    # like the pose pipeline: consecutive contexts share `--overlap` views,
    # so every Sim(3) junction is anchored on many poses, and alignment is
    # chained ctx_k -> ctx_{k-1} -> ... -> ctx_0 with robust fits.
    step = max(1, a.w - a.overlap)
    starts = list(range(0, max(1, V - a.w + 1), step))
    if starts[-1] + a.w < V: starts.append(V - a.w)
    ctx_ids = [list(range(s, min(V, s + a.w))) for s in starts[:a.k_ctx] if s >= 0]               if a.k_ctx > 0 else [list(range(s, min(V, s + a.w))) for s in starts]
    if pilot is not None:
        ctx_ids = pilot["chunks"]
    if a.reads > 1:
        # v124: K reads of the SAME view set with rotated orderings (the first
        # frame fixes the backbone's frame and reference).  Each read is one
        # context; the consensus read across them is the memory's
        # error-correcting read exercised inside a single window.
        if pilot is not None:
            raise SystemExit("--reads is a native-window operation; not combinable with --pilot")
        base = list(range(V))
        ctx_ids = [base[(k * V) // a.reads:] + base[:(k * V) // a.reads] for k in range(a.reads)]
        print(f"reads: {a.reads} orderings of all {V} views (starts {[c[0] for c in ctx_ids]})")
    outs, ctx_pts = [], []
    for ci, ids in enumerate(ctx_ids):
        o = bb.infer([paths[i] for i in ids])
        outs.append((ids, o))
        print(f"ctx {ci}: views {ids[0]}..{ids[-1]}  scale {o.extras['scene_scale']:.3f}")
    ref_ids, ref = outs[0]
    gt_poses = d["poses"]
    T = [np.eye(4)]; scales = [1.0]                 # ctx -> ref, chained
    if pilot is not None:
        # every pass (incl. ctx0) -> the row's global frame, by a Sim(3) fit of
        # the pass poses onto the row's trajectory for that chunk's frames
        T, scales = [], []
        for k, (ids, o) in enumerate(outs):
            s, R, tt = sim3_from_poses(np.stack(o.poses), pilot["est"][ids])
            M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = tt
            T.append(M); scales.append(s)
            res = np.linalg.norm((s * (R @ np.stack([p[:3, 3] for p in o.poses]).T)).T + tt
                                 - pilot["est"][ids][:, :3, 3], axis=1)
            print(f"pilot placement ctx {k} ({pilot['row']}): {len(ids)} frames, residual med {np.median(res):.4f} (row units)")
    for k in range(1, len(outs) if pilot is None else 0):
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
    # ---- v122: geometric-consistency filter on each context's depth maps ----
    # depth_used[k][j] / geo_mask[k][j]: fused depth + keep-mask for view j of
    # context k (None = untouched).  In --geo-cams pred the check runs in the
    # backbone's own frame (its cameras, its normalised scale: the test is
    # scale-free); in gt it runs with the provided cameras (needs --gt-cams).
    depth_used = [[None] * len(ids) for ids, _ in outs]
    geo_mask = [[None] * len(ids) for ids, _ in outs]
    if a.geo_views > 0:
        if a.geo_cams == "gt" and not a.gt_cams:
            raise SystemExit("--geo-cams gt requires --gt-cams")
        Ks_gt = d["K"]
        for k, (ids, o) in enumerate(outs):
            Hd, Wd = o.depth[0].shape
            if a.geo_cams == "gt":
                from PIL import Image
                W0, H0 = Image.open(paths[0]).size
                Ks = np.stack([K_at_grid(Ks_gt[v], (W0, H0), (Hd, Wd)) for v in ids])
                cams = np.stack([gt_poses[v] for v in ids])
                # metric scale so that depth and provided cameras agree
                sc_k = scales[k] * sim3_from_poses(np.stack([outs[0][1].poses[j] for j in range(len(outs[0][0]))]),
                                                    np.stack([gt_poses[v] for v in outs[0][0]]))[0]
                D = np.stack([o.depth[j] * sc_k for j in range(len(ids))])
            else:
                Ks = np.stack([o.extras["intrinsics_all"][j] for j in range(len(ids))])
                cams = np.stack([o.poses[j] for j in range(len(ids))])
                D = np.stack([o.depth[j] for j in range(len(ids))])
            src = select_sources(cams, a.geo_src)
            gkeep, D_out, cnt = geo_consistency(D, Ks, cams, src, pix_thr=a.geo_pix,
                                                rel_thr=a.geo_rel, min_views=a.geo_views,
                                                average=not a.no_geo_avg)
            for j in range(len(ids)):
                geo_mask[k][j] = gkeep[j]
                # store fused depth back in the backbone's units for the unprojection paths below
                depth_used[k][j] = (D_out[j] / sc_k) if a.geo_cams == "gt" else D_out[j]
            print(f"ctx {k}: geo-consistency (>= {a.geo_views} of {a.geo_src} src, "
                  f"{a.geo_pix} px, {a.geo_rel*100:.1f}% depth): kept {gkeep.mean()*100:.1f}% of pixels, "
                  f"median support {np.median(cnt[gkeep]) if gkeep.any() else 0:.0f} views")

    def view_depth(k, j):
        o = outs[k][1]
        return o.depth[j] if depth_used[k][j] is None else depth_used[k][j]

    if a.points_from == "auto":                    # v133: the backbone's native point map if it has one
        has_ph = outs[0][1].extras.get("world_points") is not None
        a.points_from = "pointhead" if has_ph else "depth"
        print(f"points-from auto -> {a.points_from} ({'native point map exposed' if has_ph else 'depth x camera'})")
    conf_key = "world_points_conf" if a.points_from == "pointhead" else "conf"

    def view_mask(k, j):
        o = outs[k][1]
        m = conf_mask(o, j, a.conf_pct, a.conf_abs, key=conf_key)
        g = geo_mask[k][j]
        if g is None: return m
        return g if m is None else (m & g)

    if a.points_from == "pointhead":
        if outs[0][1].extras.get("world_points") is None:
            raise SystemExit("--points-from pointhead: backbone exposes no point-map head")
        if a.gt_cams:
            print("  [note] --points-from pointhead ignores --gt-cams (point head is not ray-based); "
                  "GT frame via the camera Sim(3)")
    for k, (ids, o) in enumerate(outs):
        pts = {}
        for j, v in enumerate(ids):
            if a.points_from == "pointhead":
                # VGGT "Ours (Point)": the point-map head, already expressed in the
                # context's first-camera frame at the same scale as depth/poses
                P = pointmap_points(o.extras["world_points"][j], a.stride, mask=view_mask(k, j))
            else:
                K = o.extras["intrinsics_all"][j]
                P = unproject(view_depth(k, j), K, o.poses[j], a.stride, mask=view_mask(k, j))
            Ph = (T[k][:3, :3] @ P.T).T + T[k][:3, 3]
            pts[v] = Ph
        ctx_pts.append(pts)
    pose_of = {v: ref.poses[j] for j, v in enumerate(ref_ids)}
    # align the reference (backbone) frame to the dataset GT frame so the
    # official evaluators score in the scan's metric coordinates. Pose-head
    # only: Sim(3) from ctx0's backbone poses onto the npz GT poses.
    gt_poses = d["poses"]
    if pilot is not None:                       # row frame -> GT over ALL keyframes (ATE-style)
        sG, RG, tG = sim3_from_poses(pilot["est"], gt_poses)
    else:
        # v128: the global frame is fitted over EVERY view's chained camera (each view
        # taken from the first context that holds it, mapped through T[k]), not only
        # context 0's cameras -- a 16-camera arc gave windowed clouds a 30 mm initial
        # misalignment that the evaluator's region-restricted ICP could not recover.
        pose_all = {}
        for k, (ids, o) in enumerate(outs):
            sk = float(np.cbrt(abs(np.linalg.det(T[k][:3, :3]))))
            for j, v in enumerate(ids):
                if v not in pose_all:
                    M = T[k] @ o.poses[j]
                    M[:3, :3] /= sk                 # keep a proper rotation block
                    pose_all[v] = M
        vs = sorted(pose_all)
        sG, RG, tG = sim3_from_poses(np.stack([pose_all[v] for v in vs]), np.stack([gt_poses[v] for v in vs]))
    print(f"ref->GT: scale {sG:.4f}")
    def to_gt(P): return (sG * (RG @ P.T)).T + tG
    if a.gt_cams:
        # DTU-style protocol: cameras are PROVIDED. Depth (or the point head,
        # taken to each view's own camera frame) stays the estimand; rays/poses
        # come from GT; only the metric scale rides the pose fit.
        Ks_gt = d["K"]
        from PIL import Image
        W0, H0 = Image.open(paths[0]).size          # calibration resolution
        ctx_pts = []
        for k, (ids, o) in enumerate(outs):
            sc = scales[k] * sG
            pts = {}
            for j, v in enumerate(ids):
                if a.points_from == "pointhead":
                    # v124: point head -> this view's (predicted) camera frame -> metric -> GT camera
                    P = pointmap_points(o.extras["world_points"][j], a.stride, mask=view_mask(k, j))
                    w2c = np.linalg.inv(o.poses[j])
                    Pc = ((w2c[:3, :3] @ P.T).T + w2c[:3, 3]) * sc
                    pts[v] = (gt_poses[v][:3, :3] @ Pc.T).T + gt_poses[v][:3, 3]
                else:
                    Hd, Wd = o.depth[j].shape
                    Kv = K_at_grid(Ks_gt[v], (W0, H0), (Hd, Wd))   # K at depth-map grid (resize + crop)
                    pts[v] = unproject(view_depth(k, j) * sc, Kv, gt_poses[v], a.stride,
                                       mask=view_mask(k, j))
            ctx_pts.append(pts)
        print(f"gt-cams mode: metric scale per ctx = chain x {sG:.2f}")
        ctx_cams = [{v: gt_poses[v][:3, 3] for v in ids} for ids, _ in outs]
    else:
        ctx_pts = [{v: to_gt(P) for v, P in c.items()} for c in ctx_pts]   # metric FIRST
        ctx_cams = [{v: to_gt((T[k][:3, :3] @ o.poses[j][:3, 3] + T[k][:3, 3])[None])[0]
                     for j, v in enumerate(ids)} for k, (ids, o) in enumerate(outs)]
    for k, c in enumerate(ctx_pts):
        fin = np.concatenate([np.isfinite(P).all(1) for P in c.values()])
        print(f"ctx {k}: {fin.mean()*100:.1f}% of pixels carry a point after masks")
    if a.content_align and len(ctx_pts) > 1:
        # v124: the RE-MEASURE step on content.  Each later pass is compared
        # with what the memory already holds for the views it shares (the
        # cards: per-pixel point maps), and a robust Sim(3) on those
        # correspondences corrects the pass -- the per-pass point-vs-camera
        # scale bias that camera placement cannot see.  With --gt-cams the
        # cameras are exact, so only the metric scale is refined, about each
        # view's camera centre.
        ctx_pts = content_align(ctx_pts, ctx_cams, model=a.content_model,
                                trim=a.content_trim, iters=a.content_iters, ref=a.content_ref)
    cams_gt = {v: gt_poses[v][:3, 3] for v in range(V)}
    zs = [{v: np.linalg.norm(P - cams_gt[v][None], axis=1) for v, P in c.items()}
          for c in ctx_pts]                                        # per-point depth (m)
    single = np.concatenate([p for p in ctx_pts[0].values()])
    fused_v = consensus_fuse(ctx_pts, m=a.m, tau=None, tau_abs=a.tau_m,
                             tau_rel=a.tau_rel, zdist=zs[0], zdist_all=zs,
                             keep_singles=not a.strict, fallback=a.fallback, abstain_rel=a.abstain_rel)
    fused = np.concatenate([p for p in fused_v.values()])
    print(f"consensus gate: max({a.tau_m} m, {a.tau_rel} x depth)")
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    # v153: per-point colours for figures. Points of view v are the depth-grid pixels in mgrid[0:H:stride, 0:W:stride]
    # order (see unproject); the image is put on the same grid (resize width, centre-crop height -- the backbones'
    # preprocessing) and sampled with the same stride. Colours are per view, identical for every context.
    def view_colours(v):
        from PIL import Image
        Hd, Wd = outs[0][1].depth[list(outs[0][0]).index(v)].shape if v in outs[0][0] else outs[0][1].depth[0].shape
        im = Image.open(paths[v]).convert("RGB"); W0, H0 = im.size
        im = im.resize((Wd, max(Hd, int(round(H0 * Wd / W0)))), Image.BILINEAR)
        top = (im.size[1] - Hd) // 2
        arr = np.asarray(im)[top:top + Hd][::a.stride, ::a.stride].reshape(-1, 3)
        return arr
    try:
        cols_v = {v: view_colours(v) for v in ctx_pts[0]}
        single_c = np.concatenate([cols_v[v] for v in ctx_pts[0]]); fused_c = np.concatenate([cols_v[v] for v in fused_v])
        assert len(single_c) == len(single) and len(fused_c) == len(fused)
    except Exception as ex:                                   # colours are a courtesy for figures, never a failure
        print(f"[ply] colours unavailable ({type(ex).__name__}: {ex}); writing xyz only"); single_c = fused_c = None
    if a.per_ctx:
        for k, c in enumerate(ctx_pts):
            Pk = np.concatenate([p for p in c.values()]); Pk = Pk[np.isfinite(Pk).all(-1)].astype(np.float32)
            with open(out / f"ctx_{k:02d}.ply", "wb") as f:
                f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(Pk)).encode())
                Pk.astype("<f4").tofile(f)
        print(f"per-context clouds: {len(ctx_pts)} files ctx_XX.ply -> {out}")
    for name, P, Cc in (("single", single, single_c), ("fused", fused, fused_c)):
        ok = np.isfinite(P).all(-1); P = P[ok].astype(np.float32); Cc = Cc[ok] if Cc is not None else None
        # v134: binary little-endian PLY (float32 xyz [+ uchar rgb]); dtu_eval.py reads both (with or without plyfile).
        with open(out / f"{name}.ply", "wb") as f:
            props = "property float x\nproperty float y\nproperty float z\n" + ("property uchar red\nproperty uchar green\nproperty uchar blue\n" if Cc is not None else "")
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (len(P), props)).encode())
            if Cc is None:
                P.astype("<f4").tofile(f)
            else:
                rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
                rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]; rec["r"], rec["g"], rec["b"] = Cc[:, 0], Cc[:, 1], Cc[:, 2]
                rec.tofile(f)
        print(f"{name}: {len(P):,} pts{' + rgb' if Cc is not None else ''} -> {out/f'{name}.ply'}")
    json.dump({**vars(a), "n_views_used": V, "view_ids_in_gt": sel_ids},
              open(out / "run.json", "w"), indent=1)
    if a.save_maps:
        # per-view point maps on the (strided) pixel grid, in the GT metric
        # frame, for the correspondence-Umeyama protocol (VGGT Sec. 4.3).
        Hd, Wd = ref.depth[0].shape
        grid_h, grid_w = len(range(0, Hd, a.stride)), len(range(0, Wd, a.stride))
        single_maps = np.stack([ctx_pts[0][v].reshape(grid_h, grid_w, 3) for v in ref_ids]).astype(np.float32)
        fused_views = sorted(fused_v)
        fused_maps = np.stack([fused_v[v].reshape(grid_h, grid_w, 3) for v in fused_views]).astype(np.float32)
        conf0 = ref.extras.get("conf")
        conf_maps = (np.stack([np.asarray(conf0[j])[::a.stride, ::a.stride] for j in range(len(ref_ids))]).astype(np.float32)
                     if conf0 is not None else np.zeros((0,)))
        extra = {}
        wp = ref.extras.get("world_points")
        if wp is not None:                     # point-head row (VGGT "Ours (Point)")
            Pm = np.asarray(wp)[:, ::a.stride, ::a.stride, :].reshape(len(ref_ids), -1, 3)
            Pm = np.stack([((T[0][:3, :3] @ Pm[j].T).T + T[0][:3, 3]) for j in range(len(ref_ids))])
            if not a.gt_cams:
                Pm = np.stack([to_gt(Pm[j]) for j in range(len(ref_ids))])
            extra["pointhead_maps"] = Pm.reshape(len(ref_ids), grid_h, grid_w, 3).astype(np.float32)
        np.savez_compressed(out / "maps.npz", gt=a.gt, stride=a.stride, depth_hw=np.array([Hd, Wd]),
                            image_paths=np.array(paths), single_views=np.array(ref_ids),
                            single_maps=single_maps, single_conf=conf_maps,
                            fused_views=np.array(fused_views), fused_maps=fused_maps,
                            gt_K=d["K"], gt_poses=gt_poses, gt_cams=bool(a.gt_cams),
                            view_ids_in_gt=np.array(sel_ids), **extra)
        print(f"maps: single {single_maps.shape} fused {fused_maps.shape} -> {out/'maps.npz'}")

def build_parser():
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
    ap.add_argument("--n-views", type=int, default=0, help="protocol subsample (e.g. 10 for VGGT ETH3D)")
    ap.add_argument("--view-seed", type=int, default=0)
    ap.add_argument("--conf-pct", type=float, default=None,
                    help="keep pixels with confidence >= this percentile (e.g. 50)")
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--out-dir", default="outputs/points/run")
    # v122: MASt3R/PatchmatchNet geometric-consistency post-processing
    ap.add_argument("--geo-views", type=int, default=0,
                    help="keep pixels consistent in >= this many source views (PatchmatchNet default 5; 0=off)")
    ap.add_argument("--geo-pix", type=float, default=1.0, help="reprojection threshold in pixels (ref 1.0)")
    ap.add_argument("--geo-rel", type=float, default=0.01, help="relative depth threshold (ref 0.01)")
    ap.add_argument("--geo-src", type=int, default=10, help="nearest source views per reference view (pair.txt uses 10)")
    ap.add_argument("--geo-cams", choices=["pred", "gt"], default="pred",
                    help="cameras used for the consistency check: backbone's own (pred) or provided (gt, needs --gt-cams)")
    ap.add_argument("--no-geo-avg", action="store_true", help="keep the reference depth instead of the consistent-view average")
    ap.add_argument("--save-maps", action="store_true", help="write maps.npz (per-view point maps) for eth3d_pointmap_eval.py")
    # v123: point-map head route and absolute confidence threshold (Langendoerfer et al. 2026: VGGT-p, C>2.0)
    ap.add_argument("--backbone-kw", action="append", default=[],
                    help="key=value passed to the backbone constructor, repeatable (e.g. scene_graph=swin-5 for dust3r)")
    ap.add_argument("--points-from", choices=["depth", "pointhead", "auto"], default="depth",
                    help="depth: unproject depth with cameras (VGGT 'Depth+Cam'); pointhead: point-map head ('Point'); auto: pointhead if exposed")
    ap.add_argument("--conf-abs", type=float, default=None,
                    help="keep pixels with confidence > this absolute value (VGGT conf is expp1 > 1; 2.0 = positive weight)")
    # v123: downstream matrix -- place the passes under a pose-pipeline row (pilot_a.py --save-est)
    ap.add_argument("--pilot", default="", help="npz from experiments/pilot_a.py --save-est")
    ap.add_argument("--row", default="smr", help="row of the pilot npz to place the passes under (chained|smr|smr_pgo|...)")
    ap.add_argument("--per-ctx", action="store_true", help="also write each context's (window's) points as ctx_XX.ply (figures)")
    # v124: in-window memory operations
    ap.add_argument("--reads", type=int, default=1, help="K reads of the same view set (rotated orderings), fused by consensus")
    ap.add_argument("--content-align", action="store_true",
                    help="re-measure each later context against the stored cards of shared views (Sim(3) on content; scale-only with --gt-cams)")
    ap.add_argument("--content-model", choices=["scale", "sim3"], default="scale",
                    help="scale: per-pass depth scale about the pass cameras (the bias windows show); sim3: free Umeyama")
    ap.add_argument("--content-ref", choices=["mean", "first"], default="mean",
                    help="mean: renormalise so no pass is privileged (default); first: keep context 0's scale")
    ap.add_argument("--content-trim", type=float, default=0.2)
    ap.add_argument("--fallback", choices=["median", "first", "none"], default="median",
                    help="where fewer than --m witnesses agree: median of all witnesses (default), earliest witness, or abstain")
    ap.add_argument("--abstain-rel", type=float, default=0.10,
                    help="with --fallback median: abstain instead when the witnesses' spread exceeds this fraction of depth (0 = never)")
    ap.add_argument("--content-iters", type=int, default=2)
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    synthetic() if a.synthetic else real(a)
