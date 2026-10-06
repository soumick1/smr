#!/usr/bin/env python3
"""Novel view synthesis from the memory map of a long sequence (v199, stage 1: frozen decoders).

    python experiments/nvs_sequence.py --gt data/gt/7scenes_pumpkin_seq01.npz --backbone vggt_omega --keyframe-stride 5 \
        --head outputs/nvs/vggt_omega/ckpt_best.pt --K 585,585,320,240 --out outputs/nvs_seq/pumpkin --save-images 6
    python experiments/nvs_sequence.py --gt data/gt/co3d_full/apple_110_13051_23361.npz --backbone vggt --keyframe-stride 1 \
        --max-frames 200 --head outputs/nvs/vggt/ckpt_best.pt --out outputs/nvs_seq/apple_110

Protocol. Every --holdout-th keyframe is a TARGET and is never shown to the backbone; the other keyframes are processed in
32/16 windows exactly as in the pose experiments (paper preset of pilot_a v198: --legacy switches to the old rules).
Each input frame's points come from the backbone pass the stitcher used for its window (re-run here, the pass index lists
are recorded in the run) and stay in that pass's coordinates; the map is the union of all frames' points placed by the
per-frame placement T_g = est[g] local[g]^-1 of a method:
    raw        chained windows            smr        online revision            smr_pgo    final pose graph
    oracle     local reconstructions placed by their ground-truth cameras (per-window Sim(3) fit): the ceiling
Target cameras are mapped into each method's map by the Sim(3) between its estimated input cameras and the GT ones. The
map is normalised to a unit sphere around the input cameras; points of every frame are subsampled by --stride pixels and
turned into Gaussians by the frozen decoder (same weights for every method). Rendering at --res x --res (square squash of
the frames), metrics against the GT target frames: PSNR, SSIM, LPIPS, plus the target's window distance from the start.
Output: <out>/<method>.jsonl (one record per target), <out>/summary.json, optional renders under <out>/images.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from smr.stitch import sim3  # noqa: E402
from smr.stitch.anchored import AnchoredStitcher, stitch_chained  # noqa: E402
from smr.stitch.chunks import keyframe_indices, make_chunks  # noqa: E402
from smr.stitch.memory_index import ScaffoldIndex  # noqa: E402
from smr.stitch.passes import BackboneRunner, PassCache, dino_descriptors  # noqa: E402
from smr.stitch import posegraph  # noqa: E402
from smr.nvs.geometry import resize_map, unproject_grid  # noqa: E402

PAPER = dict(N_h=1024, torus_N=48, k=64, ring_N=256, correction="distribute", site_agree=(3.0, 0.15), seq_scale_gate=1.5,
             fit_mode="irls", pair_strict=True, require_two_sites=True, reject_on_disagree=True,
             extent_factor=1e9, site_rot_deg=180.0, site_dir_deg=180.0, budget_rot=(1e9, 0, 1e9), budget_pos=(1e9, 0),
             tight_rot=(1e9, 0, 1e9), tight_pos=(1e9, 0))
LEGACY = dict(N_h=2048, torus_N=32, k=None, ring_N=0, correction="relax", site_agree=(1e9, 1e9), seq_scale_gate=None,
              fit_mode="robust", pair_strict=False, require_two_sites=False, reject_on_disagree=False,
              extent_factor=3.0, site_rot_deg=10.0, site_dir_deg=25.0, budget_rot=(10.0, 3.0, 45.0), budget_pos=(1.0, 0.5),
              tight_rot=(3.0, 1.0, 15.0), tight_pos=(0.5, 0.15))


# ------------------------------------------------------------------ helpers
def scaled_K(K, w, h, res):
    K = np.asarray(K, float).copy()
    K[0] *= res / w; K[1] *= res / h
    return K


def square_image(path, res):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def placements(est, local, frames):
    """Per-frame Sim(3) placement T_g = est[g] local[g]^-1 as (s, R, t) with the scale from the stored pose pair."""
    out = {}
    for g in frames:
        E, L = np.asarray(est[g], float), np.asarray(local[g], float)
        R = E[:3, :3] @ L[:3, :3].T
        out[g] = (1.0, R, E[:3, 3] - R @ L[:3, 3])
    return out


def window_sim3(local, gt, frames):
    """Oracle: one Sim(3) per window from the frames' local poses onto their GT cameras."""
    A = np.stack([local[g] for g in frames]); B = np.stack([gt[g] for g in frames])
    S, _ = sim3.fit_poses(A, B)
    return S


def apply_pts(S, X):
    s, R, t = S
    return s * (R @ X.reshape(-1, 3).T).T + t


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--backbone", required=True); ap.add_argument("--head", required=True, help="decoder checkpoint (ckpt_best.pt of this backbone)")
    ap.add_argument("--out", required=True); ap.add_argument("--keyframe-stride", type=int, default=5); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=16); ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=8, help="every n-th keyframe is a target")
    ap.add_argument("--K", default=None, help="GT intrinsics fx,fy,cx,cy at the frames' native size (7-Scenes: 585,585,320,240); default: from the npz or the backbone")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--stride", type=int, default=2, help="pixel subsampling of each frame's points")
    ap.add_argument("--footprint", default="pixel", choices=["pixel", "head"], help="Gaussian scale: geometric pixel footprint (depth x pixel / f) or the decoder's")
    ap.add_argument("--methods", nargs="+", default=["raw", "smr", "smr_pgo", "oracle"])
    ap.add_argument("--legacy", action="store_true"); ap.add_argument("--desc-thresh", type=float, default=0.5)
    ap.add_argument("--save-images", type=int, default=0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=None, help="pass cache path (default outputs/cache/nvsseq_<stem>_<bb>_s<stride>_h<holdout>.npy)")
    a = ap.parse_args()
    import torch
    from smr.nvs.gaussian_head import GaussianHead, Perceptual, render, ssim as ssim_fn
    from smr.nvs.geometry import head_input
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = LEGACY if a.legacy else PAPER
    t_start = time.time()

    # ---- data
    z = np.load(a.gt, allow_pickle=True)
    gt_all = np.asarray(z["poses"], float); paths_all = [str(p) for p in z["image_paths"]]
    key = keyframe_indices(len(gt_all), a.keyframe_stride)
    if a.max_frames:
        key = key[: a.max_frames]
    kf_paths = [paths_all[i] for i in key]; kf_gt = gt_all[key]
    tgt_local = list(range(0, len(key), a.holdout))[1:]                       # keyframe positions used as targets (never the first)
    inp_local = [i for i in range(len(key)) if i not in set(tgt_local)]
    in_paths = [kf_paths[i] for i in inp_local]; in_gt = kf_gt[inp_local]
    print(f"{len(key)} keyframes: {len(inp_local)} inputs, {len(tgt_local)} targets (every {a.holdout}th)")
    from PIL import Image
    W0, H0 = Image.open(kf_paths[0]).size
    if a.K:
        fx, fy, cx, cy = [float(v) for v in a.K.split(",")]; K_native = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif "K" in z or "intrinsics" in z:
        Kz = np.asarray(z["K"] if "K" in z else z["intrinsics"], float); K_native = Kz if Kz.ndim == 2 else Kz[0]
    else:
        K_native = None
    chunks = make_chunks(len(in_paths), a.chunk, a.overlap)

    # ---- pipeline on the inputs
    device = a.device if torch.cuda.is_available() else "cpu"
    runner = BackboneRunner(a.backbone, in_paths, device=device)
    cache_path = a.cache or f"outputs/cache/nvsseq_{pathlib.Path(a.gt).stem}_{a.backbone}_s{a.keyframe_stride}_h{a.holdout}.npy"
    cache = PassCache(cache_path)
    desc = dino_descriptors(in_paths, device=device)
    pos_scale = 1.0
    if not a.legacy:
        o0 = runner.bb.infer([in_paths[i] for i in chunks[0]])
        d0 = np.asarray(o0.depth, float) if getattr(o0, "depth", None) is not None else None
        if d0 is not None and np.isfinite(d0).any():
            pos_scale = 1.0 / float(np.median(d0[np.isfinite(d0) & (d0 > 0)]))
    index = ScaffoldIndex(N_h=cfg["N_h"], torus_N=cfg["torus_N"], k=cfg["k"], ring_N=cfg["ring_N"], desc_dim=desc.shape[1], pos_scale=pos_scale)
    AnchoredStitcher.fit_mode = cfg["fit_mode"]
    st = AnchoredStitcher(index, n_sites=a.sites, desc_thresh=a.desc_thresh, correction=cfg["correction"],
                          site_agree_rot=cfg["site_agree"][0], site_agree_pos=cfg["site_agree"][1], seq_scale_gate=cfg["seq_scale_gate"],
                          pair_strict=cfg["pair_strict"], require_two_sites=cfg["require_two_sites"], reject_on_disagree=cfg["reject_on_disagree"],
                          extent_factor=cfg["extent_factor"], site_rot_deg=cfg["site_rot_deg"], site_dir_deg=cfg["site_dir_deg"],
                          budget_rot=cfg["budget_rot"], budget_pos=cfg["budget_pos"], tight_rot=cfg["tight_rot"], tight_pos=cfg["tight_pos"])
    est_raw = stitch_chained(chunks, cache, runner)
    r = st.run(chunks, cache, runner, desc)
    est_smr, local_smr = r["est"], r["local"]
    pg, _ = posegraph.solve(r, chunks)
    print(f"stitching: {r['n_loops']} accepted closures, {len(chunks)} windows, {time.time() - t_start:.0f} s")

    # ---- per-frame points from the passes the stitcher used (and ordinary passes for the raw chain)
    owner = {}
    for k, ch in enumerate(chunks):
        for g in ch:
            owner.setdefault(int(g), k)
    frames_of_window = {k: [g for g in ch if owner[int(g)] == k] for k, ch in enumerate(chunks)}
    res = a.res
    pts_smr, pts_raw, rgb_of, conf_of, Kpred = {}, {}, {}, {}, {}

    def unpack(o, pidx, dest):
        d = np.asarray(o.depth, float) if getattr(o, "depth", None) is not None else None
        Ki = np.asarray(o.intrinsics, float) if getattr(o, "intrinsics", None) is not None else None
        for j, g in enumerate(pidx):
            if g in dest:
                continue
            Kj = (Ki if Ki.ndim == 2 else Ki[j]) if Ki is not None else None
            if d is None or Kj is None:
                raise SystemExit("backbone returned no depth/intrinsics; nvs_sequence needs both (use a depth-predicting backbone)")
            h, w = d[j].shape
            dj = resize_map(d[j], res); Kr = scaled_K(Kj, w, h, res)
            dest[g] = unproject_grid(dj, Kr, np.asarray(o.poses[j], float))          # pass coordinates
            if g not in rgb_of:
                rgb_of[g] = resize_map(np.asarray(o.rgb[j], float), res)
                c = np.asarray(o.conf[j], float) if getattr(o, "conf", None) is not None else np.ones_like(d[j])
                conf_of[g] = resize_map(c, res); Kpred[g] = Kr
    for k, pidx in enumerate(r["passes"]):
        o = runner.bb.infer([in_paths[i] for i in pidx]); unpack(o, [int(i) for i in pidx], pts_smr)
        if list(pidx) != list(chunks[k]):
            o2 = runner.bb.infer([in_paths[i] for i in chunks[k]]); unpack(o2, [int(i) for i in chunks[k]], pts_raw)
        else:
            for g in pidx:
                pts_raw.setdefault(int(g), pts_smr[int(g)])
        print(f"  pass {k}: {len(pidx)} frames", flush=True)
    # local poses of the raw chain = ordinary pass poses (consistent with pts_raw); est_raw places them
    local_raw = {}
    for k, ch in enumerate(chunks):
        o = cache.get(list(ch), runner)["poses"]
        for j, g in enumerate(ch):
            local_raw.setdefault(int(g), o[j])

    # ---- methods -> per-frame placements
    all_in = sorted(owner)
    method_T = {}
    if "raw" in a.methods:
        method_T["raw"] = (placements(est_raw, local_raw, all_in), pts_raw, est_raw)
    if "smr" in a.methods:
        method_T["smr"] = (placements(est_smr, local_smr, all_in), pts_smr, est_smr)
    if "smr_pgo" in a.methods:
        method_T["smr_pgo"] = (placements(pg, local_smr, all_in), pts_smr, pg)
    if "oracle" in a.methods:
        T_or = {}
        for k, fr in frames_of_window.items():
            S = window_sim3(local_smr, in_gt, fr)
            for g in fr:
                T_or[g] = S
        est_or = np.stack([sim3.apply_one(T_or[g], local_smr[g]) for g in all_in]) if hasattr(sim3, "apply_one") else in_gt
        method_T["oracle"] = (T_or, pts_smr, est_or)

    # ---- decoder
    head = GaussianHead().to(device); ck = torch.load(a.head, map_location=device, weights_only=False); head.load_state_dict(ck["head"]); head.eval()
    perc = Perceptual().to(device)
    K_tgt = scaled_K(K_native if K_native is not None else Kpred[all_in[0]], W0, H0, res)
    tgt_imgs = np.stack([square_image(kf_paths[i], res) for i in tgt_local])
    tgt_gt = kf_gt[tgt_local]
    nearest_input = [int(np.argmin([abs(i - j) for j in inp_local])) for i in tgt_local]
    results = {}
    for name, (T_g, pts, est) in method_T.items():
        # map normalisation around the input cameras
        cams = np.stack([est[g][:3, 3] for g in all_in]); centre = cams.mean(0); rad = np.linalg.norm(cams - centre, axis=1).max() + 1e-6
        norm = lambda X: (X - centre) / rad
        xs, rgbs, deps, cfs, foot = [], [], [], [], []
        for g in all_in:
            P = apply_pts(T_g[g], pts[g]).reshape(res, res, 3)
            cam = est[g][:3, 3]
            Xn = norm(P)
            dep = np.linalg.norm(P - cam, axis=-1) / rad
            xs.append(Xn); rgbs.append(rgb_of[g]); deps.append(dep); cfs.append(conf_of[g])
            foot.append(dep * a.stride / float(Kpred[g][0, 0]))                            # pixel footprint in normalised units
        xs, rgbs, deps, cfs, foot = (np.stack(v) for v in (xs, rgbs, deps, cfs, foot))
        sub = (slice(None), slice(0, None, a.stride), slice(0, None, a.stride))
        xs, rgbs, deps, cfs, foot = xs[sub], rgbs[sub], deps[sub], cfs[sub], foot[sub]
        x_in, valid = head_input(rgbs, xs, cfs, np.ones(xs.shape[:3]), depth_from_cams=deps)
        with torch.no_grad():
            xt = torch.from_numpy(x_in).to(device).permute(0, 3, 1, 2)
            g = head(xt, xt[:, 3:6], xt[:, 0:3], torch.from_numpy(valid).to(device))
            if a.footprint == "pixel":
                f = torch.from_numpy(foot.reshape(-1)[valid.reshape(-1)]).to(device).float().clamp(1e-4, 0.1)
                g["scales"] = f[:, None].expand(-1, 3).contiguous()
            # target cameras into this map: Sim(3) from estimated input cameras to GT, inverted, then normalised
            S_e2g = sim3.fit_poses(np.stack([est[g_] for g_ in all_in]), in_gt)[0]
            S_g2e = sim3.inverse(S_e2g)
            c2w = []
            for Tg in tgt_gt:
                Te = sim3.apply_one(S_g2e, Tg) if hasattr(sim3, "apply_one") else Tg
                Te = np.asarray(Te, float).copy(); Te[:3, 3] = (Te[:3, 3] - centre) / rad
                c2w.append(Te)
            c2w = torch.from_numpy(np.stack(c2w)).to(device).float()
            Kt = torch.from_numpy(K_tgt).to(device).float()
            pred, _ = render(g, c2w, Kt, res)
        pred_np = pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        recs = []
        for ti, (pi, gi) in enumerate(zip(pred_np, tgt_imgs)):
            pt = torch.from_numpy(pi).permute(2, 0, 1)[None].to(device); gtt = torch.from_numpy(gi).permute(2, 0, 1)[None].to(device)
            recs.append(dict(target_kf=int(tgt_local[ti]), window=int(owner[int(nearest_input[ti])]), psnr=psnr(pi, gi),
                             ssim=float(ssim_fn(pt, gtt).item()), lpips=float(perc(pt, gtt).item())))
        results[name] = recs
        with open(out / f"{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        if a.save_images:
            (out / "images").mkdir(exist_ok=True)
            for ti in np.linspace(0, len(recs) - 1, min(a.save_images, len(recs))).astype(int):
                Image.fromarray((pred_np[ti] * 255).astype("uint8")).save(out / "images" / f"{name}_t{tgt_local[ti]:03d}.png")
                Image.fromarray((tgt_imgs[ti] * 255).astype("uint8")).save(out / "images" / f"gt_t{tgt_local[ti]:03d}.png")
        m = np.array([[rc["psnr"], rc["ssim"], rc["lpips"]] for rc in recs]).mean(0)
        print(f"{name:<8} PSNR {m[0]:.2f}  SSIM {m[1]:.3f}  LPIPS {m[2]:.3f}   ({g['means'].shape[0]} Gaussians)", flush=True)
    summary = dict(gt=a.gt, backbone=a.backbone, n_inputs=len(inp_local), n_targets=len(tgt_local), closures=int(r["n_loops"]), windows=len(chunks),
                   config="legacy" if a.legacy else "paper", pos_scale=pos_scale,
                   means={k: dict(psnr=float(np.mean([x["psnr"] for x in v])), ssim=float(np.mean([x["ssim"] for x in v])), lpips=float(np.mean([x["lpips"] for x in v]))) for k, v in results.items()},
                   by_window={k: {str(w): float(np.mean([x["psnr"] for x in v if x["window"] == w])) for w in sorted({x["window"] for x in v})} for k, v in results.items()},
                   elapsed_s=time.time() - t_start)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("summary ->", out / "summary.json")


if __name__ == "__main__":
    main()
