#!/usr/bin/env python3
"""KITTI odometry -> SMR npz.

    python scripts/kitti_gt_poses.py --root ~/kitti_odometry --seq 00 \
        --out data/gt/kitti_00.npz

Expects <root>/poses/<seq>.txt (12 numbers/line, cam0-to-world) and
<root>/sequences/<seq>/image_2/*.png.  GT is for cam0; we evaluate the
image_2 trajectory against it -- the fixed cam0->cam2 offset (~6 cm) is
absorbed by the one Sim(3) alignment and is negligible at metre-level ATE.
"""
import argparse, pathlib, sys

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()
    pose_file = root / "poses" / f"{a.seq}.txt"
    img_dir = root / "sequences" / a.seq / "image_2"
    P = np.loadtxt(pose_file).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(P), 1, 1)); poses[:, :3, :] = P
    imgs = sorted(img_dir.glob("*.png"))
    if len(imgs) != len(P):
        print(f"WARNING: {len(imgs)} images vs {len(P)} poses; truncating to min")
        n = min(len(imgs), len(P)); imgs, poses = imgs[:n], poses[:n]
    c = poses[:, :3, 3]
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    d = np.linalg.norm(c[None] - c[:, None], axis=-1)
    iu = np.triu_indices(len(c), k=120)
    rev = int(((d[iu] < 15.0)).sum())
    np.savez_compressed(a.out, image_paths=np.array([str(p) for p in imgs]),
                        poses=poses, frame_ids=np.arange(len(imgs)),
                        scene=f"kitti_{a.seq}")
    print(f"kitti {a.seq}: {len(imgs)} frames, path {seg.sum():.0f} m, "
          f"extent {np.linalg.norm(c.max(0)-c.min(0)):.0f} m, "
          f"revisit pairs (<15 m, >=120 frames apart): {rev}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
