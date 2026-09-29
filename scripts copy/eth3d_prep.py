#!/usr/bin/env python3
"""ETH3D DSLR prep: COLMAP text model -> one npz per scene.

Expects multi_view_training_dslr_undistorted layout:
    <root>/<scene>/images/dslr_images_undistorted/*.JPG
    <root>/<scene>/dslr_calibration_undistorted/{cameras.txt,images.txt}

    python scripts/eth3d_prep.py --root ~/data/eth3d --out-dir data/gt/eth3d
"""
import argparse, pathlib
import numpy as np

def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out-dir", default="data/gt/eth3d")
a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for scene in sorted(pathlib.Path(a.root).iterdir()):
    calib = scene / "dslr_calibration_undistorted"
    if not calib.exists(): continue
    cams = {}
    for l in (calib / "cameras.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip(): continue
        t = l.split(); fx, fy, cx, cy = map(float, t[4:8])
        cams[int(t[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    paths, poses, Ks = [], [], []
    lines = [l for l in (calib / "images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for l in lines[::2]:
        t = l.split(); q = np.array(t[1:5], float); tv = np.array(t[5:8], float)
        R = qvec2rot(q); c2w = np.eye(4)
        c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ tv
        img = scene / "images" / t[9]
        if not img.exists(): img = scene / "images" / "dslr_images_undistorted" / pathlib.Path(t[9]).name
        paths.append(str(img)); poses.append(c2w); Ks.append(cams[int(t[8])])
    np.savez_compressed(out / f"{scene.name}.npz", image_paths=np.array(paths),
                        poses=np.stack(poses), K=np.stack(Ks), scene=scene.name)
    print(scene.name, len(paths), "views")
