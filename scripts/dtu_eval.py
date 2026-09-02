#!/usr/bin/env python3
r"""DTU point-cloud evaluation: Accuracy / Completeness / Overall (mm),
official convention (ObsMask + BB filter, MaxDist=20 discard).

GT: SampleSet 'Points/stl/stlXXX_total.ply'; masks 'ObsMask/ObsMaskX_10.mat'
(keys ObsMask, BB, Res) and 'ObsMask/PlaneX.mat' (ground-plane clip for
completeness). CALIBRATE FIRST: run on one scan against a published
method's cloud and match its printed numbers to ~0.01 mm before trusting.

    python scripts/dtu_eval.py --pred outputs/points/dtu_scan24/fused.ply \
        --scan 24 --sampleset ~/data/dtu/SampleSet/MVS\ Data
"""
import argparse, pathlib, re
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

def read_ply(p):
    txt = pathlib.Path(p).read_bytes()
    if txt[:3] != b"ply": raise ValueError("ascii/binary ply expected")
    try:
        import plyfile
        v = plyfile.PlyData.read(str(p))["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], -1).astype(float)
    except ImportError:
        head = txt.split(b"end_header\n", 1)
        n = int(re.search(rb"element vertex (\d+)", head[0]).group(1))
        return np.loadtxt(head[1].splitlines()[:n], usecols=(0, 1, 2))

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--scan", type=int, required=True)
ap.add_argument("--sampleset", required=True); ap.add_argument("--max-dist", type=float, default=20.0)
ap.add_argument("--points-dir", default=None, help="dir with stlXXX_total.ply (Points.zip)")
ap.add_argument("--down", type=float, default=0.2, help="pred downsample voxel (mm); 0=off")
ap.add_argument("--icp", type=int, default=0, help="Sim(3) ICP refinement iterations (0=off)")
a = ap.parse_args()
ss = pathlib.Path(a.sampleset)
stl = (pathlib.Path(a.points_dir) if a.points_dir else ss / "Points" / "stl")
gt = read_ply(stl / f"stl{a.scan:03d}_total.ply")
M = loadmat(ss / "ObsMask" / f"ObsMask{a.scan}_10.mat")
mask = np.asarray(M["ObsMask"])
BB = np.asarray(M["BB"], float).reshape(2, 3)
Res = float(np.asarray(M["Res"]).squeeze())
pl = np.asarray(loadmat(ss / "ObsMask" / f"Plane{a.scan}.mat")["P"], float).reshape(4)
pred = read_ply(a.pred)
if a.down > 0:                                     # voxel thin (official uses 0.2mm reduce)
    key = np.floor(pred / a.down).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True); pred = pred[keep]
if a.icp > 0:
    # Sim(3) point-to-point ICP refinement of pred onto GT (DUSt3R-lineage
    # evals refine the gauge cloud-to-cloud; ours arrives from a camera fit).
    rng = np.random.RandomState(0)
    Gs = gt[rng.choice(len(gt), min(len(gt), 200_000), replace=False)]
    tree = cKDTree(Gs)
    P = pred[rng.choice(len(pred), min(len(pred), 60_000), replace=False)].copy()
    T_s, T_R, T_t = 1.0, np.eye(3), np.zeros(3)
    for it in range(a.icp):
        d_nn, idx = tree.query(P, workers=-1)
        keep_nn = d_nn < np.percentile(d_nn, 80)
        A, B = P[keep_nn], Gs[idx[keep_nn]]
        muA, muB = A.mean(0), B.mean(0)
        A0, B0 = A - muA, B - muB
        U, S, Vt = np.linalg.svd(A0.T @ B0)
        D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ D @ U.T
        s = (S * np.diag(D)).sum() / (A0 ** 2).sum()
        tvec = muB - s * (R @ muA)
        P = (s * (R @ P.T)).T + tvec
        T_R = R @ T_R; T_s = s * T_s; T_t = s * (R @ T_t) + tvec
    pred = (T_s * (T_R @ pred.T)).T + T_t
    print(f"icp({a.icp}): scale {T_s:.5f}  |t| {np.linalg.norm(T_t):.3f} mm")
# Accuracy: pred filtered by BB + ObsMask grid, distances to GT, discard >MaxDist
g = np.floor((pred - BB[0:1]) / Res).astype(int)
inbb = ((pred >= BB[0:1]) & (pred < BB[1:2])).all(1)
ok = inbb.copy()
gi = np.clip(g, 0, np.array(mask.shape) - 1)
ok &= mask[gi[:, 0], gi[:, 1], gi[:, 2]].astype(bool)
dp = cKDTree(gt).query(pred[ok], workers=-1)[0]
acc = dp[dp < a.max_dist].mean()
# Completeness: GT above plane, distances to pred, discard >MaxDist
above = gt @ pl[:3] + pl[3] > 0
dg = cKDTree(pred).query(gt[above], workers=-1)[0]
comp = dg[dg < a.max_dist].mean()
print(f"scan{a.scan}: Acc {acc:.3f}  Comp {comp:.3f}  Overall {(acc+comp)/2:.3f}  "
      f"(pred kept {ok.mean()*100:.1f}%, MaxDist {a.max_dist})")
