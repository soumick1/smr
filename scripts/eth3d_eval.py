#!/usr/bin/env python3
r"""ETH3D evaluation, pure Python (no PCL/VTK/cmake).

Parses the scene's scan_alignment.mlp (MeshLab XML: per-scan .ply +
4x4 transform), loads the laser GT, and scores a predicted cloud with
the standard ETH3D quantities at given tolerances:
  accuracy(t)     = fraction of prediction points within t of GT
  completeness(t) = fraction of GT points within t of prediction
  F1(t)           = harmonic mean
This is the community proxy of the official evaluator (identical
definitions; the official adds voxel bookkeeping for speed). Use one
instrument consistently when comparing clouds.

    python scripts/eth3d_eval.py --pred outputs/points/smoke_courtyard/fused.ply \
        --mlp ~/data/eth3d/courtyard/dslr_scan_eval/scan_alignment.mlp \
        --tolerances 0.02,0.05 --gt-voxel 0.01
"""
import argparse, pathlib, re, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree

def read_ply(path):
    try:
        from plyfile import PlyData
        v = PlyData.read(str(path))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    except ImportError:
        raw = pathlib.Path(path).read_bytes()
        head, body = raw.split(b"end_header\n", 1)
        n = int(re.search(rb"element vertex (\d+)", head).group(1))
        assert b"format ascii" in head, "pip install plyfile for binary PLY"
        return np.loadtxt(body.splitlines()[:n], usecols=(0, 1, 2))

def load_gt(mlp_path, voxel):
    mlp = pathlib.Path(mlp_path)
    pts = []
    try:
        root = ET.parse(mlp).getroot()
        for mesh in root.iter("MLMesh"):
            fn = (mlp.parent / mesh.get("filename")).resolve()
            M = np.eye(4)
            mm = mesh.find("MLMatrix44")
            if mm is not None and mm.text and mm.text.split():
                M = np.array(mm.text.split(), float).reshape(4, 4)
            P = read_ply(fn)
            pts.append((M[:3, :3] @ P.T).T + M[:3, 3])
            print(f"  GT part: {fn.name}  {len(P):,} pts")
    except ET.ParseError:
        pass
    if not pts:                                   # fallback: plys next to the mlp
        for fn in sorted(mlp.parent.glob("*.ply")):
            P = read_ply(fn); pts.append(P); print(f"  GT part: {fn.name}  {len(P):,} pts")
    G = np.concatenate(pts)
    if voxel > 0:
        key = np.floor(G / voxel).astype(np.int64)
        _, keep = np.unique(key, axis=0, return_index=True)
        G = G[keep]
        print(f"  GT after {voxel} m voxel thin: {len(G):,} pts")
    return G

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--mlp", required=True)
ap.add_argument("--tolerances", default="0.02,0.05")
ap.add_argument("--gt-voxel", type=float, default=0.01)
a = ap.parse_args()
pred = read_ply(a.pred); pred = pred[np.isfinite(pred).all(-1)]
print(f"pred: {a.pred}  {len(pred):,} pts")
gt = load_gt(a.mlp, a.gt_voxel)
dp = cKDTree(gt).query(pred, workers=-1)[0]       # pred -> GT  (accuracy)
dg = cKDTree(pred).query(gt, workers=-1)[0]       # GT -> pred (completeness)
print(f"mean-dist  Acc {dp.mean():.3f} m   Comp {dg.mean():.3f} m   "
      f"Overall {(dp.mean()+dg.mean())/2:.3f} m")
print(f"{'tol':>6} {'accuracy':>9} {'complete':>9} {'F1':>7}")
for tol in [float(x) for x in a.tolerances.split(",")]:
    acc = (dp < tol).mean(); comp = (dg < tol).mean()
    f1 = 0.0 if acc + comp == 0 else 2 * acc * comp / (acc + comp)
    print(f"{tol:>6} {acc*100:>8.2f}% {comp*100:>8.2f}% {f1*100:>6.2f}%")
