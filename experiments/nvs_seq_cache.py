#!/usr/bin/env python3
"""Build the per-sequence cache for sequence NVS training / evaluation (v202).

    python experiments/nvs_seq_cache.py --gt data/gt/7scenes_pumpkin_seq01.npz --backbone vggt_omega --out cache/nvsseq/vggt_omega/pumpkin_seq01 --holdout 8
    python experiments/nvs_seq_cache.py --gt data/gt/7scenes_chess_seq03.npz   --backbone vggt_omega --out cache/nvsseq/vggt_omega/chess_seq03  --holdout 0

--holdout 8: every 8th keyframe is a held-out target never shown to the backbone (evaluation sequences).
--holdout 0: every keyframe is an input; decoder training uses input frames as targets with their neighbours excluded
from the sources (training sequences).
Pipeline: the paper configuration (address 1024/64 with orientation rings, median-depth units, pairs at +-3, two pairs,
agreement 10 deg / 1.0 E, joint pass gated at 5 deg, Sim(3) interpolation, IRLS); --legacy for the old rules.
Stored (one .npy per array, memory-mappable, plus meta.json):
    in_kf, in_gt (n,4,4), in_paths.json, owner (n,), K_src (n,3,3) at H x W, rgb_src (n,H,W,3) uint8, conf (n,H,W) f16,
    pts_raw / pts_smr (n,H,W,3) f16 in the coordinates of the pass the method used,
    S_<m> (n,7) per-frame Sim(3) [s, rotvec(3), t(3)] placing those points for m in raw, smr, smr_pgo, gt,
    est_<m> (n,4,4) placed cameras, E2G_<m> (7,) Sim(3) from the method's map to the GT frame (fit on all inputs),
    tgt_kf, tgt_gt (m,4,4), tgt_paths.json, K_gt (3,3) at H x W.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smr.stitch import sim3  # noqa: E402
from smr.stitch import posegraph  # noqa: E402
from smr.stitch.anchored import AnchoredStitcher, stitch_chained  # noqa: E402
from smr.stitch.chunks import keyframe_indices, make_chunks  # noqa: E402
from smr.stitch.memory_index import ScaffoldIndex  # noqa: E402
from smr.stitch.passes import BackboneRunner, PassCache, dino_descriptors  # noqa: E402

PAPER = dict(N_h=1024, torus_N=48, k=64, ring_N=256, correction="distribute", site_agree=(10.0, 1.0), seq_scale_gate=1.5, fit_mode="irls",
             pair_strict=True, require_two_sites=True, reject_on_disagree=True, local_from="gated", gate=5.0,
             extent_factor=1e9, site_rot_deg=180.0, site_dir_deg=180.0, budget_rot=(1e9, 0, 1e9), budget_pos=(1e9, 0), tight_rot=(1e9, 0, 1e9), tight_pos=(1e9, 0))
LEGACY = dict(N_h=2048, torus_N=32, k=None, ring_N=0, correction="relax", site_agree=(1e9, 1e9), seq_scale_gate=None, fit_mode="robust",
              pair_strict=False, require_two_sites=False, reject_on_disagree=False, local_from="anchored", gate=5.0,
              extent_factor=3.0, site_rot_deg=10.0, site_dir_deg=25.0, budget_rot=(10.0, 3.0, 45.0), budget_pos=(1.0, 0.5), tight_rot=(3.0, 1.0, 15.0), tight_pos=(0.5, 0.15))


def resize_hw(x, H, W):
    """Nearest-neighbour resample of an (h,w[,c]) map to (H,W[,c])."""
    h, w = x.shape[:2]
    ys = np.minimum((np.arange(H) + 0.5) * h / H, h - 1).astype(int)
    xs = np.minimum((np.arange(W) + 0.5) * w / W, w - 1).astype(int)
    return x[ys][:, xs]


def unproject_hw(depth, K, c2w):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1).reshape(-1, 3).astype(float)
    rays = (np.linalg.inv(K) @ pix.T).T
    z = depth.reshape(-1, 1).astype(float)
    pts = (c2w[:3, :3] @ (rays * z).T).T + c2w[:3, 3]
    pts[~(np.isfinite(z[:, 0]) & (z[:, 0] > 0))] = np.nan
    return pts.reshape(H, W, 3)


def scaled_K(K, w, h, W, H):
    K = np.asarray(K, float).copy(); K[0] *= W / w; K[1] *= H / h
    return K


def to7(S):
    s, R, t = S
    from scipy.spatial.transform import Rotation
    return np.concatenate([[s], Rotation.from_matrix(R).as_rotvec(), t])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True); ap.add_argument("--backbone", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keyframe-stride", type=int, default=5); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=16); ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=8); ap.add_argument("--H", type=int, default=288); ap.add_argument("--W", type=int, default=384)
    ap.add_argument("--K", default=None, help="GT intrinsics fx,fy,cx,cy at native size (7-Scenes 585,585,320,240)")
    ap.add_argument("--legacy", action="store_true"); ap.add_argument("--desc-thresh", type=float, default=0.5); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from PIL import Image
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        print("cache exists:", out); return
    cfg = LEGACY if a.legacy else PAPER
    t0 = time.time()
    z = np.load(a.gt, allow_pickle=True)
    gt_all = np.asarray(z["poses"], float); paths_all = [str(p) for p in z["image_paths"]]
    key = keyframe_indices(len(gt_all), a.keyframe_stride)
    if a.max_frames:
        key = key[: a.max_frames]
    kf_paths = [paths_all[i] for i in key]; kf_gt = gt_all[key]
    tgt = list(range(0, len(key), a.holdout))[1:] if a.holdout > 0 else []
    inp = [i for i in range(len(key)) if i not in set(tgt)]
    in_paths = [kf_paths[i] for i in inp]; in_gt = kf_gt[inp]
    W0, H0 = Image.open(kf_paths[0]).size
    H, W = (a.H, a.W) if W0 >= H0 else (a.W, a.H)        # portrait sequences (some CO3D orbits) keep their orientation
    if a.K:
        fx, fy, cx, cy = [float(v) for v in a.K.split(",")]; Kn = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    elif "K" in z:
        Kn = np.asarray(z["K"], float); Kn = Kn if Kn.ndim == 2 else Kn[0]
    else:
        Kn = None
    chunks = make_chunks(len(in_paths), a.chunk, a.overlap)
    device = a.device if torch.cuda.is_available() else "cpu"
    runner = BackboneRunner(a.backbone, in_paths, device=device)
    cache = PassCache(f"outputs/cache/nvsseqc_{pathlib.Path(a.gt).stem}_{a.backbone}_s{a.keyframe_stride}_h{a.holdout}{'_legacy' if a.legacy else ''}.npy")
    desc = dino_descriptors(in_paths, device=device)
    pos_scale = 1.0
    if not a.legacy:
        o0 = runner.bb.infer([in_paths[i] for i in chunks[0]])
        d0 = np.asarray(o0.depth, float) if getattr(o0, "depth", None) is not None else np.asarray(o0.pts_local, float)[..., 2]
        pos_scale = 1.0 / float(np.median(d0[np.isfinite(d0) & (d0 > 0)]))
    index = ScaffoldIndex(N_h=cfg["N_h"], torus_N=cfg["torus_N"], k=cfg["k"], ring_N=cfg["ring_N"], desc_dim=desc.shape[1], pos_scale=pos_scale)
    AnchoredStitcher.fit_mode = cfg["fit_mode"]; AnchoredStitcher.local_from = cfg["local_from"]; AnchoredStitcher.distortion_gate = cfg["gate"]
    st = AnchoredStitcher(index, n_sites=a.sites, desc_thresh=a.desc_thresh, correction=cfg["correction"],
                          site_agree_rot=cfg["site_agree"][0], site_agree_pos=cfg["site_agree"][1], seq_scale_gate=cfg["seq_scale_gate"],
                          pair_strict=cfg["pair_strict"], require_two_sites=cfg["require_two_sites"], reject_on_disagree=cfg["reject_on_disagree"],
                          extent_factor=cfg["extent_factor"], site_rot_deg=cfg["site_rot_deg"], site_dir_deg=cfg["site_dir_deg"],
                          budget_rot=cfg["budget_rot"], budget_pos=cfg["budget_pos"], tight_rot=cfg["tight_rot"], tight_pos=cfg["tight_pos"])
    est_raw = stitch_chained(chunks, cache, runner)
    r = st.run(chunks, cache, runner, desc)
    est_smr, local_smr = r["est"], r["local"]
    est_pgo, _ = posegraph.solve(r, chunks)
    print(f"{pathlib.Path(a.gt).stem}: {len(inp)} inputs, {len(tgt)} targets, {r['n_loops']} closures, {len(chunks)} windows ({time.time() - t0:.0f} s)", flush=True)

    n = len(in_paths)
    owner = np.full(n, -1, int)
    for k, ch in enumerate(chunks):
        for g in ch:
            if owner[g] < 0:
                owner[g] = k
    used = {int(e.get("chunk", -1)): (e.get("anchored_distortion") or {}).get("used") for e in r["events"]}
    pts_raw = np.full((n, H, W, 3), np.nan, np.float16); pts_smr = np.full((n, H, W, 3), np.nan, np.float16)
    rgb_src = np.zeros((n, H, W, 3), np.uint8); conf = np.zeros((n, H, W), np.float16); K_src = np.zeros((n, 3, 3))
    local_raw = np.zeros((n, 4, 4)); have_raw = np.zeros(n, bool); have_smr = np.zeros(n, bool)

    def unpack(o, pidx, dest, have, local=None, first=False):
        """Per frame of a pass: points at H x W in the pass's coordinates (from depth + intrinsics, or from the camera-frame
        point map when the backbone reports no depth), antialiased source rgb, confidence, intrinsics at H x W."""
        d = np.asarray(o.depth, float) if getattr(o, "depth", None) is not None else None
        pl = np.asarray(o.pts_local, float) if getattr(o, "pts_local", None) is not None else None
        Ki = np.asarray(o.intrinsics, float) if getattr(o, "intrinsics", None) is not None else None
        if d is None and pl is None:
            raise SystemExit("backbone returned neither depth nor pts_local")
        for j, g in enumerate(pidx):
            if have[g]:
                continue
            h, w = (d[j].shape if d is not None else pl[j].shape[:2])
            c2w = np.asarray(o.poses[j], float)
            if Ki is not None:
                Kr = scaled_K(Ki if Ki.ndim == 2 else Ki[j], w, h, W, H)
            elif pl is not None:                                   # focal from the point map (median of z / x-slope), principal point at the centre
                z = pl[j][..., 2]; xs_ = (np.arange(w) + 0.5 - w / 2)[None, :]
                fx = float(np.nanmedian(np.abs(z * xs_ / np.where(np.abs(pl[j][..., 0]) > 1e-6, pl[j][..., 0], np.nan))))
                Kr = scaled_K(np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]]), w, h, W, H)
            else:
                Kr = scaled_K(Kn, W0, H0, W, H)
            if d is not None:
                dest[g] = unproject_hw(resize_hw(d[j], H, W), Kr, c2w).astype(np.float16)
            else:
                X = resize_hw(pl[j], H, W).reshape(-1, 3)
                dest[g] = ((c2w[:3, :3] @ X.T).T + c2w[:3, 3]).reshape(H, W, 3).astype(np.float16)
            have[g] = True
            if local is not None:
                local[g] = c2w
            if first:
                im = Image.fromarray((np.clip(np.asarray(o.rgb[j], float), 0, 1) * 255).astype(np.uint8))
                rgb_src[g] = np.asarray(im.resize((W, H), Image.BOX), np.uint8)
                c = np.asarray(o.conf[j], float) if getattr(o, "conf", None) is not None else np.ones((h, w))
                conf[g] = resize_hw(c, H, W).astype(np.float16); K_src[g] = Kr

    for k, pidx in enumerate(r["passes"]):
        pidx = list(chunks[k]) if used.get(k) == "plain" else [int(i) for i in pidx]
        o = runner.bb.infer([in_paths[i] for i in pidx]); unpack(o, pidx, pts_smr, have_smr, first=True)
        o2 = o if pidx == list(chunks[k]) else runner.bb.infer([in_paths[i] for i in chunks[k]])
        unpack(o2, [int(i) for i in chunks[k]], pts_raw, have_raw, local=local_raw)
        print(f"  pass {k} done", flush=True)

    frames_of = {k: [int(g) for g in range(n) if owner[g] == k] for k in range(len(chunks))}

    def per_window(est, local):
        S = np.zeros((n, 7))
        for k, fr in frames_of.items():
            Sk, _ = sim3.fit_poses(np.stack([local[g] for g in fr]), np.stack([est[g] for g in fr]))
            for g in fr:
                S[g] = to7(Sk)
        return S

    local_s = np.stack([local_smr[g] for g in range(n)])
    S = dict(raw=per_window(est_raw, local_raw), smr=per_window(est_smr, local_s), smr_pgo=per_window(est_pgo, local_s), gt=per_window(in_gt, local_s))
    from scipy.spatial.transform import Rotation
    est = dict(raw=np.asarray(est_raw, float), smr=np.asarray(est_smr, float), smr_pgo=np.asarray(est_pgo, float))
    est["gt"] = np.stack([np.eye(4) for _ in range(n)])
    for g in range(n):                                    # GT placement: local poses placed by the per-window GT fit
        s, rv, t = S["gt"][g][0], S["gt"][g][1:4], S["gt"][g][4:7]
        R = Rotation.from_rotvec(rv).as_matrix(); L = local_s[g]
        est["gt"][g][:3, :3] = R @ L[:3, :3]; est["gt"][g][:3, 3] = s * (R @ L[:3, 3]) + t
    for m in ("raw", "smr", "smr_pgo", "gt"):
        E2G, _ = sim3.fit_poses(est[m], in_gt)
        np.save(out / f"S_{m}.npy", S[m]); np.save(out / f"est_{m}.npy", est[m]); np.save(out / f"E2G_{m}.npy", to7(E2G))
    np.save(out / "pts_raw.npy", pts_raw); np.save(out / "pts_smr.npy", pts_smr); np.save(out / "rgb_src.npy", rgb_src)
    np.save(out / "conf.npy", conf); np.save(out / "K_src.npy", K_src); np.save(out / "owner.npy", owner)
    np.save(out / "in_kf.npy", np.array(inp)); np.save(out / "in_gt.npy", in_gt); np.save(out / "tgt_kf.npy", np.array(tgt, int))
    np.save(out / "tgt_gt.npy", kf_gt[tgt] if tgt else np.zeros((0, 4, 4)))
    np.save(out / "K_gt.npy", scaled_K(Kn if Kn is not None else K_src[0], W0, H0, W, H) if Kn is not None else K_src[0])
    json.dump(in_paths, open(out / "in_paths.json", "w")); json.dump([kf_paths[i] for i in tgt], open(out / "tgt_paths.json", "w"))
    json.dump(dict(gt=a.gt, backbone=a.backbone, H=H, W=W, n_inputs=n, n_targets=len(tgt), closures=int(r["n_loops"]), windows=len(chunks),
                   config="legacy" if a.legacy else "paper", holdout=a.holdout, gt_K=Kn is not None, elapsed_s=time.time() - t0), open(out / "meta.json", "w"), indent=1)
    print(f"cache -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
