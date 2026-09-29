#!/usr/bin/env python3
r"""DTU prep directly from the official SampleSet (no MVSNet download).

Parses Calibration/Results/pos_XXX.txt (3x4 projection matrices P=K[R|t])
via RQ decomposition and pairs them with Rectified/scanX/rect_XXX_L_r5000.png
(one lighting condition L, default 3 = 'max'). Writes data/gt/dtu/scanX.npz
(image_paths, K, poses c2w) compatible with points_suite.py.

    python scripts/dtu_prep_sampleset.py \
        --sampleset ~/data/dtu/SampleSet/MVS\ Data --out-dir data/gt/dtu
"""
import argparse, pathlib, re
import numpy as np
from scipy.linalg import rq

def decompose_P(P):
    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))          # enforce positive diagonal
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0: K, R = -K, -R
    t = np.linalg.inv(K) @ P[:, 3]
    K /= K[2, 2]
    c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
    return K, c2w

ap = argparse.ArgumentParser()
ap.add_argument("--sampleset", required=True)
ap.add_argument("--out-dir", default="data/gt/dtu")
ap.add_argument("--lighting", default="3", help="rect_XXX_<L>_r5000.png; 3='max' typical")
a = ap.parse_args()
ss = pathlib.Path(a.sampleset)
cal_dirs = [p for p in (ss / "Calibration").glob("*") if p.is_dir()]
assert cal_dirs, f"no Calibration/* under {ss}"
cal = cal_dirs[0]
pos = {}
for f in sorted(cal.glob("pos_*.txt")):
    i = int(re.search(r"pos_(\d+)", f.name).group(1))
    pos[i] = np.loadtxt(f).reshape(3, 4)
print(f"calibration: {len(pos)} cameras from {cal.name}")
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for scan in sorted((ss / "Rectified").glob("scan*")):
    paths, Ks, poses = [], [], []
    for i in sorted(pos):
        for pat in (f"rect_{i:03d}_{a.lighting}_r5000.png", f"rect_{i:03d}_max.png"):
            img = scan / pat
            if img.exists(): break
        else:
            continue
        K, c2w = decompose_P(pos[i])
        paths.append(str(img)); Ks.append(K); poses.append(c2w)
    if not paths:
        print(f"{scan.name}: no images matched lighting {a.lighting}; ls one dir to check pattern"); continue
    np.savez_compressed(out / f"{scan.name}.npz", image_paths=np.array(paths),
                        poses=np.stack(poses), K=np.stack(Ks), scene=scan.name)
    print(f"{scan.name}: {len(paths)} views -> {out}/{scan.name}.npz")
