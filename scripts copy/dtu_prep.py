#!/usr/bin/env python3
"""DTU test-set prep: MVSNet layout -> one npz per scan.

Expects the standard MVSNet/CasMVSNet 'dtu test' layout:
    <root>/scanX/images/00000000.jpg ...   (49 views)
    <root>/scanX/cams/00000000_cam.txt     (extrinsic 4x4 w2c, intrinsic 3x3)
Writes data/gt/dtu/scanX.npz with image_paths, K (V,3,3), poses c2w (V,4,4).

    python scripts/dtu_prep.py --root ~/data/dtu/mvsnet_test --out-dir data/gt/dtu
"""
import argparse, pathlib, re
import numpy as np

def read_cam(p):
    tok = p.read_text().split()
    i = tok.index("extrinsic"); E = np.array(tok[i+1:i+17], float).reshape(4, 4)
    j = tok.index("intrinsic"); K = np.array(tok[j+1:j+10], float).reshape(3, 3)
    return np.linalg.inv(E), K            # c2w, K

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out-dir", default="data/gt/dtu")
a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for scan in sorted(pathlib.Path(a.root).glob("scan*")):
    imgs = sorted((scan / "images").glob("*.jpg")) or sorted((scan / "images").glob("*.png"))
    cam_by_idx = {int(re.search(r"(\d+)", c.stem).group(1)): c
                  for c in (scan / "cams").glob("*_cam.txt")}
    imgs = [i for i in imgs if int(re.search(r"(\d+)", i.stem).group(1)) in cam_by_idx]
    assert imgs, f"{scan.name}: no image/cam index matches"
    poses, Ks = zip(*[read_cam(cam_by_idx[int(re.search(r"(\d+)", i.stem).group(1))])
                      for i in imgs])
    np.savez_compressed(out / f"{scan.name}.npz",
                        image_paths=np.array([str(p) for p in imgs]),
                        poses=np.stack(poses), K=np.stack(Ks), scene=scan.name)
    print(scan.name, len(imgs), "views")
