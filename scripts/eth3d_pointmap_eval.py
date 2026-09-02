#!/usr/bin/env python3
r"""ETH3D point-map evaluation, VGGT Table 3 protocol, offline (no GPU).

Protocol (VGGT Sec. 4.3, verbatim facts): per scene 10 random frames; the
predicted point cloud is aligned to the ground truth with the Umeyama
algorithm; invalid points are filtered with the official masks; Accuracy,
Completeness, Overall (Chamfer) are reported.  "Ours (Depth+Cam)" unprojects
the predicted depth with the predicted cameras; "Ours (Point)" uses the point
head.  Distances are in metres, plain means (no MaxDist cutoff).

What this script does with a maps.npz from `points_suite.py --save-maps`:
  1. builds a ground-truth point map on the same pixel grid for every frame
     (official rendered depth by default, laser-scan z-buffer as fallback);
  2. correspondences = pixels with finite GT and finite prediction; Umeyama
     Sim(3) on them (plain = the paper's wording; a trimmed variant is
     printed as a labelled diagnostic);
  3. Chamfer Acc (pred->GT) / Comp (GT->pred) / Overall over the union of the
     frames' points; predictions at pixels without GT are dropped (mask
     applied to both sides; --pred-all keeps them).
It also prints the pre-alignment numbers (camera-Sim(3) from the suite) so
the effect of the correspondence alignment is visible.

    python scripts/eth3d_pointmap_eval.py --maps outputs/points/ethrep_courtyard/maps.npz \
        --scene-dir ~/data/eth3d/courtyard --gt official        # needs *_dslr_depth.7z + *_dslr_jpg.7z (calib)
    python scripts/eth3d_pointmap_eval.py --maps ... --scene-dir ... --gt scan   # no download, fallback
    python scripts/eth3d_pointmap_eval.py --maps ... --scene-dir ... --source fused   # windowed+SMR maps
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from smr.eval.eth3d_gt import (gt_depth_official, gt_depth_from_scan, load_scan,  # noqa: E402
                               unproject_grid)
from smr.eval.mvs_fusion import umeyama_trimmed, apply_sim3, chamfer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--maps", required=True, help="maps.npz written by points_suite.py --save-maps")
ap.add_argument("--scene-dir", required=True, help="<root>/<scene> with dslr_calibration_* and GT")
ap.add_argument("--gt", choices=["official", "scan"], default="official")
ap.add_argument("--source", choices=["single", "fused", "pointhead"], default="single",
                help="single = ctx0 Depth+Cam (VGGT protocol); fused = +SMR maps; pointhead = point head")
ap.add_argument("--pool", choices=["median", "min"], default="median", help="full-res -> grid pooling (official)")
ap.add_argument("--trim", type=float, default=0.2, help="trim fraction for the robust diagnostic")
ap.add_argument("--pred-all", action="store_true", help="keep predictions at pixels without GT")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--tolerances", default="0.02,0.05", help="also print F1@tol (our instrument)")
ap.add_argument("--json", default=None, help="append a result line to this json-lines file")
a = ap.parse_args()

M = np.load(a.maps, allow_pickle=True)
stride = int(M["stride"]); Hd, Wd = [int(x) for x in M["depth_hw"]]
paths = [str(p) for p in M["image_paths"]]
gt_K = M["gt_K"]; gt_poses = M["gt_poses"]
if a.source == "single":
    views = [int(v) for v in M["single_views"]]; P = M["single_maps"].astype(np.float64)
elif a.source == "fused":
    views = [int(v) for v in M["fused_views"]]; P = M["fused_maps"].astype(np.float64)
else:
    if "pointhead_maps" not in M.files:
        raise SystemExit("maps.npz has no pointhead_maps (backbone exposes no point head)")
    views = [int(v) for v in M["single_views"]]; P = M["pointhead_maps"].astype(np.float64)
gh, gw = P.shape[1:3]
from PIL import Image
Wu, Hu = Image.open(paths[0]).size
scene_dir = pathlib.Path(a.scene_dir).expanduser()
print(f"pred maps: {a.source} {P.shape} (stride {stride}, grid {Hd}x{Wd}, image {Wu}x{Hu}); views {views}")

# ------------------------------------------------------------ GT point maps --
cache = pathlib.Path(a.maps).parent / f"gt_maps_{a.gt}_{a.pool}_{a.source}.npz"
if cache.exists() and not a.no_cache:
    G = np.load(cache)["G"]; print(f"GT maps from cache {cache.name}")
else:
    G = np.full(P.shape, np.nan)
    scan = load_scan(scene_dir / "dslr_scan_eval" / "scan_alignment.mlp") if a.gt == "scan" else None
    for i, v in enumerate(views):
        name = pathlib.Path(paths[v]).name
        if a.gt == "official":
            depth, st = gt_depth_official(scene_dir, name, gt_K[v], (Wu, Hu), (Hd, Wd), pool=a.pool)
            print(f"  view {v} {name}: GT depth {st['gt_pixels_fullres']:,} full-res px -> "
                  f"{st['grid_valid']*100:.1f}% of grid")
        else:
            depth = gt_depth_from_scan(scan, gt_K[v], gt_poses[v], (Wu, Hu), (Hd, Wd))
            print(f"  view {v} {name}: scan z-buffer -> {np.isfinite(depth).mean()*100:.1f}% of grid")
        G[i] = unproject_grid(depth, gt_K[v], gt_poses[v], (Wu, Hu), stride)
    np.savez_compressed(cache, G=G.astype(np.float32))

# --------------------------------------------------------------- alignment --
ok_gt = np.isfinite(G).all(-1); ok_pr = np.isfinite(P).all(-1)
corr = ok_gt & ok_pr
X = P[corr]; Y = G[corr]
print(f"correspondences: {corr.sum():,} pixels ({corr.mean()*100:.1f}% of grid); "
      f"GT valid {ok_gt.mean()*100:.1f}%, pred valid {ok_pr.mean()*100:.1f}%")
res0 = np.linalg.norm(X - Y, axis=1)
print(f"pre-align   per-pixel |pred-gt|: median {np.median(res0):.3f} m  mean {res0.mean():.3f} m")

gt_pts = G[ok_gt]
pred_mask = ok_pr if a.pred_all else (ok_pr & ok_gt)
tols = [float(t) for t in a.tolerances.split(",")]

def report(tag, Pal):
    acc, comp, ov, dp, dg = chamfer(Pal[pred_mask], gt_pts)
    line = f"{tag:<14} Acc {acc:.3f} m   Comp {comp:.3f} m   Overall {ov:.3f} m"
    f1s = []
    for t in tols:
        pa, pc = (dp < t).mean(), (dg < t).mean()
        f1s.append(0 if pa + pc == 0 else 2 * pa * pc / (pa + pc))
    line += "   | " + "  ".join(f"F1@{t:g}={f*100:.1f}%" for t, f in zip(tols, f1s))
    print(line)
    return dict(acc=acc, comp=comp, overall=ov, f1={str(t): f for t, f in zip(tols, f1s)})

out = dict(maps=a.maps, source=a.source, gt=a.gt, views=views, n_corr=int(corr.sum()))
out["pre_align"] = report("pre-align", P)
s, R, t, inl = umeyama_trimmed(X, Y, trim=0.0)
Pu = apply_sim3(P, s, R, t)
res1 = np.linalg.norm(Pu[corr] - Y, axis=1)
print(f"umeyama     scale {s:.4f}  per-pixel median {np.median(res1):.3f} m")
out["umeyama"] = report("umeyama", Pu); out["umeyama_scale"] = s
if a.trim > 0:
    s2, R2, t2, inl2 = umeyama_trimmed(X, Y, trim=a.trim, iters=3)
    Pt = apply_sim3(P, s2, R2, t2)
    out[f"umeyama_trim{a.trim:g}"] = report(f"umeyama-trim{a.trim:g}", Pt)
if a.json:
    with open(a.json, "a") as f:
        f.write(json.dumps(out) + "\n")
