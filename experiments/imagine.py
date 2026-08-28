#!/usr/bin/env python3
"""T4 -- path-integrated view change (imagination) on 7-Scenes.

    python experiments/imagine.py --gt data/gt/7scenes_office_all.npz --backbone vggt_omega \
        --map-seqs 1,3,4,5,8,10 --target-seqs 2,6,7,9 --n-paths 25 --steps 1,2,4,8,16

A memory is built from the mapping sessions exactly as in relocalisation
(same caches).  Each path: localise ONE target image against memory, then
integrate the ground-truth relative motion commands for k steps (no images),
and predict the view at the integrated pose by reading the memory out.
Rows, all scored against the real frame + dataset depth at the target:

  nearest     the stored keyframe nearest the integrated pose, as-is
              (the nearest-view baseline of the NVS literature)
  reproject   surfels of that ONE stored view splatted at the pose
              (what a single stored window can answer)
  smr         the full memory bank splatted at the pose (readout; zero
              backbone cost per query)
  rerun       a fresh backbone pass on the 4 recalled keyframes, fresh
              surfels, splatted at the pose (upper bound; one pass/query)
  external    frames from --score-external DIR (DIR/p{path}_s{step}.png),
              e.g. a generative NVS model given the same recalled views

Metrics: PSNR / SSIM / LPIPS (NVS convention), depth abs-rel vs the
dataset's depth maps (Eigen convention), coverage, and the start
localisation error.  Misses and holes are visible in coverage, not hidden.
"""
import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smr.backbones import get_backbone                                      # noqa: E402
from smr.stitch import AnchoredStitcher, BackboneRunner, PassCache, ScaffoldIndex  # noqa: E402
from smr.stitch.chunks import make_session_chunks                           # noqa: E402
from smr.stitch import sim3, posegraph                                      # noqa: E402
from smr.stitch.relocalise import MemoryMap, localise, pose_error           # noqa: E402
from smr.stitch.imagine import lpips_fn, score_view, splat_points           # noqa: E402
from smr.eval.trajectory import ate_rmse                                    # noqa: E402

K7 = np.array([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1.0]])          # 7-Scenes intrinsics


def depth_path(color_path):
    return str(color_path).replace(".color.png", ".depth.png")


def load_depth(p, hw):
    from PIL import Image
    d = np.asarray(Image.open(p), np.float64)
    d[d == 65535] = 0
    d = d / 1000.0
    if d.shape != hw:
        import numpy as _np
        yi = (_np.linspace(0, d.shape[0] - 1, hw[0])).astype(int)
        xi = (_np.linspace(0, d.shape[1] - 1, hw[1])).astype(int)
        d = d[yi][:, xi]
    return d


def load_rgb(p, hw):
    from PIL import Image
    im = Image.open(p).convert("RGB").resize((hw[1], hw[0]))
    return np.asarray(im, np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--backbone", default="vggt_omega")
    ap.add_argument("--map-seqs", required=True)
    ap.add_argument("--target-seqs", required=True)
    ap.add_argument("--keyframe-stride", type=int, default=10)
    ap.add_argument("--target-stride", type=int, default=10)
    ap.add_argument("--n-paths", type=int, default=25)
    ap.add_argument("--steps", default="1,2,4,8,16", help="path lengths in target keyframes")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--pixel-stride", type=int, default=5, help="surfel subsampling at bind")
    ap.add_argument("--render-hw", default="240,320")
    ap.add_argument("--rows", default="nearest,reproject,smr,rerun")
    ap.add_argument("--rerun-k", type=int, default=4)
    ap.add_argument("--score-external", default="")
    ap.add_argument("--desc-thresh", type=float, default=0.5)
    ap.add_argument("--N-h", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    hw = tuple(int(x) for x in a.render_hw.split(","))
    Kr = K7.copy(); Kr[0] *= hw[1] / 640.0; Kr[1] *= hw[0] / 480.0
    npz = np.load(a.gt, allow_pickle=True)
    paths = [str(s) for s in npz["image_paths"]]
    gt = np.asarray(npz["poses"], float)
    sess = np.asarray(npz["frame_ids"]) // 100000
    scene = str(npz["scene"]) if "scene" in npz else pathlib.Path(a.gt).stem
    map_seqs = [int(x) for x in a.map_seqs.split(",")]
    t_seqs = [int(x) for x in a.target_seqs.split(",")]
    map_idx = [i for i in range(len(paths)) if sess[i] in map_seqs][::a.keyframe_stride]
    t_idx = [i for i in range(len(paths)) if sess[i] in t_seqs][::a.target_stride]
    map_paths = [paths[i] for i in map_idx]
    steps = [int(s) for s in a.steps.split(",")]
    print(f"scene {scene} | map {len(map_idx)} kf | targets {len(t_idx)} | paths {a.n_paths} x steps {steps}")

    cache_dir = ROOT / "outputs" / "cache"
    tag = f"reloc_{scene}_{a.backbone}_m{''.join(map(str, map_seqs))}_s{a.keyframe_stride}"
    from experiments.relocalise import descriptors_for                       # same caches as T2
    map_desc = descriptors_for(map_paths, "dino", a.device, cache_dir / f"{tag}.desc_dino.npy")
    runner = BackboneRunner(a.backbone, map_paths, device=a.device)
    cache = PassCache(cache_dir / f"{tag}.npy")
    chunks = make_session_chunks([int(sess[i]) for i in map_idx], a.chunk, a.overlap)
    index = ScaffoldIndex(N_h=a.N_h, seed=a.seed, desc_dim=map_desc.shape[1])
    res = AnchoredStitcher(index, n_sites=a.sites, desc_thresh=a.desc_thresh).run(
        chunks, cache, runner, map_desc)
    pg, _ = posegraph.solve(res, chunks)
    res = dict(res); res["est"] = pg
    mmap = MemoryMap(res, gt[map_idx], index, map_desc)
    print(f"  map ATE {mmap.map_ate:.3f} m, {res['n_loops']} closures")

    # ---------------- surfel bank: lift each chunk once, place metrically
    bb = get_backbone(a.backbone, device=a.device)
    pts_all, col_all, own_all = [], [], []
    t0 = time.time()
    for ci, ch in enumerate(chunks):
        rv = bb.infer([map_paths[i] for i in ch])
        A = rv.poses
        B = np.stack([mmap.pose[map_idx[i]] for i in ch])
        S, _, _ = sim3.fit_poses_robust(A, B, min_inliers=3)
        H, W = rv.depth.shape[1:3]
        Kp = rv.intrinsics
        ys, xs = np.mgrid[0:H:a.pixel_stride, 0:W:a.pixel_stride]
        for li, gi in enumerate(ch):
            if mmap.owner[map_idx[gi]] != ci:      # bind each keyframe once, by its owner chunk
                continue
            d = rv.depth[li][ys, xs]
            ok = d > 1e-6
            if hasattr(rv, "conf") and rv.conf is not None:
                c = rv.conf[li][ys, xs]; ok &= c >= np.quantile(c[ok], 0.3) if ok.any() else ok
            K_ = Kp[li] if Kp.ndim == 3 else Kp
            X = (xs[ok] - K_[0, 2]) / K_[0, 0] * d[ok]
            Y = (ys[ok] - K_[1, 2]) / K_[1, 1] * d[ok]
            Pcam = np.stack([X, Y, d[ok]], 1)
            Pw_pass = Pcam @ rv.poses[li][:3, :3].T + rv.poses[li][:3, 3]
            Pw = S[0] * (Pw_pass @ S[1].T) + S[2]
            pts_all.append(Pw); own_all.append(np.full(len(Pw), map_idx[gi]))
            col_all.append(rv.rgb[li][ys, xs][ok] / (255.0 if rv.rgb.max() > 2 else 1.0))
    pts = np.concatenate(pts_all); cols = np.concatenate(col_all); owners = np.concatenate(own_all)
    print(f"  bank: {len(pts):,} surfels from {len(chunks)} passes ({time.time() - t0:.0f}s)")

    # ---------------- query machinery (start localisation, T2-style)
    rng = np.random.default_rng(a.seed)
    max_step = max(steps)
    starts = sorted(rng.choice(len(t_idx) - max_step, size=min(a.n_paths, len(t_idx) - max_step), replace=False))
    q_paths = [paths[i] for i in t_idx]
    q_desc = descriptors_for(q_paths, "dino", a.device,
                             cache_dir / f"{tag}_imgq{''.join(map(str, t_seqs))}_qs{a.target_stride}.desc_dino.npy")
    all_paths = map_paths + q_paths
    qrunner = BackboneRunner(a.backbone, all_paths, device=a.device)
    qcache = PassCache(cache_dir / f"{tag}_imgq{''.join(map(str, t_seqs))}_qs{a.target_stride}.npy")
    kf_centres = np.stack([mmap.pose[g][:3, 3] for g in mmap.frames])
    lp = lpips_fn()
    ext = pathlib.Path(a.score_external) if a.score_external else None
    rows = a.rows.split(",") + (["external"] if ext else [])
    results = {r: {s: [] for s in steps} for r in rows}
    loc_errs = []

    for pi, s0 in enumerate(starts):
        def run_pass(anchors, q=len(map_paths) + s0):
            return qcache.get([q] + [int(g) for g in anchors], qrunner)["poses"]
        out = localise(mmap, q_desc[s0], run_pass, mode="plain", n_sites=a.sites,
                       desc_thresh=a.desc_thresh)
        if out["T"] is None:
            loc_errs.append((float("inf"), float("inf"))); continue
        T0 = out["T"]
        loc_errs.append(pose_error(T0, gt[t_idx[s0]]))
        for k in steps:
            tgt = t_idx[s0 + k]
            Tk = T0 @ np.linalg.inv(gt[t_idx[s0]]) @ gt[tgt]     # integrate GT commands
            rgb_gt = load_rgb(paths[tgt], hw)
            dep_gt = load_depth(depth_path(paths[tgt]), hw)
            near = np.argsort(np.linalg.norm(kf_centres - Tk[:3, 3], axis=1))
            g_near = [mmap.frames[j] for j in near[: a.rerun_k]]
            for row in rows:
                if row == "nearest":
                    est = load_rgb(map_paths[map_idx.index(g_near[0])], hw)
                    de = load_depth(depth_path(map_paths[map_idx.index(g_near[0])]), hw)
                    sc = score_view(est, de, np.ones(hw, bool), rgb_gt, dep_gt, lp)
                elif row == "reproject":
                    m = owners == g_near[0]
                    e, d, msk = splat_points(pts[m], cols[m], Tk, Kr, hw)
                    sc = score_view(e, d, msk, rgb_gt, dep_gt, lp)
                elif row == "smr":
                    e, d, msk = splat_points(pts, cols, Tk, Kr, hw)
                    sc = score_view(e, d, msk, rgb_gt, dep_gt, lp)
                elif row == "rerun":
                    rv = bb.infer([map_paths[map_idx.index(g)] for g in g_near])
                    S, _, _ = sim3.fit_poses_robust(rv.poses, np.stack([mmap.pose[g] for g in g_near]), min_inliers=3)
                    H2, W2 = rv.depth.shape[1:3]
                    ys, xs = np.mgrid[0:H2:a.pixel_stride, 0:W2:a.pixel_stride]
                    P, C = [], []
                    for li in range(len(g_near)):
                        d2 = rv.depth[li][ys, xs]; ok = d2 > 1e-6
                        K_ = rv.intrinsics[li] if rv.intrinsics.ndim == 3 else rv.intrinsics
                        X = (xs[ok] - K_[0, 2]) / K_[0, 0] * d2[ok]
                        Y = (ys[ok] - K_[1, 2]) / K_[1, 1] * d2[ok]
                        Pc = np.stack([X, Y, d2[ok]], 1)
                        Pw = Pc @ rv.poses[li][:3, :3].T + rv.poses[li][:3, 3]
                        P.append(S[0] * (Pw @ S[1].T) + S[2])
                        C.append(rv.rgb[li][ys, xs][ok] / (255.0 if rv.rgb.max() > 2 else 1.0))
                    e, d, msk = splat_points(np.concatenate(P), np.concatenate(C), Tk, Kr, hw)
                    sc = score_view(e, d, msk, rgb_gt, dep_gt, lp)
                elif row == "external":
                    f = ext / f"p{pi}_s{k}.png"
                    if not f.exists():
                        continue
                    e = load_rgb(f, hw)
                    sc = score_view(e, np.zeros(hw), np.ones(hw, bool), rgb_gt, dep_gt, lp)
                results[row][k].append(sc)
        if pi == 0:
            print(f"  [path 0 localised at {loc_errs[0][0]*100:.1f} cm; rows OK]", flush=True)

    fin = np.array([e[0] for e in loc_errs if np.isfinite(e[0])])
    print(f"\n  start localisation: median {np.median(fin)*100:.1f} cm over {len(fin)}/{len(loc_errs)} paths")
    hdr = f"  {'row':<10}{'k':>4}{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'absrel':>8}{'coverage':>9}{'n':>4}"
    print(hdr)
    summary = {}
    for row in rows:
        for k in steps:
            L = results[row][k]
            if not L: continue
            m = {q: float(np.nanmean([x[q] for x in L])) for q in L[0]}
            summary[f"{row}_k{k}"] = m
            print(f"  {row:<10}{k:>4}{m['psnr']:>7.2f}{m['ssim']:>7.3f}"
                  f"{m.get('lpips', float('nan')):>7.3f}{m['absrel']:>8.3f}{m['coverage']:>9.2f}{len(L):>4}")
    out = pathlib.Path(a.json or (ROOT / "outputs" / "reports" / f"imagine_{scene}_{a.backbone}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(scene=scene, backbone=a.backbone, map_ate=mmap.map_ate,
                                   n_paths=len(starts), steps=steps,
                                   loc_err=[[float(x) for x in e] for e in loc_errs],
                                   summary=summary,
                                   raw={r: {str(k): results[r][k] for k in steps} for r in rows}),
                             indent=1, default=float))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
