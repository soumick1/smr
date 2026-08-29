#!/usr/bin/env python3
"""Virtual KITTI 1.3.1 -> SMR npz (one clip = one world x one variation).

    python scripts/vkitti_gt_poses.py --root ~/vkitti --world 0001 --var clone \
        --out data/gt/vkitti_0001_clone.npz

Expects <root>/vkitti_1.3.1_rgb/<world>/<var>/*.png and
<root>/vkitti_1.3.1_extrinsicsgt/<world>_<var>.txt (header line, then per
frame: index + 16 row-major world-to-camera entries).
"""
import argparse, pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--var", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()
    ext = root / "vkitti_1.3.1_extrinsicsgt" / f"{a.world}_{a.var}.txt"
    img_dir = root / "vkitti_1.3.1_rgb" / a.world / a.var
    rows = []
    for line in ext.read_text().splitlines()[1:]:
        v = line.replace(",", " ").split()
        if len(v) >= 17:
            rows.append([float(x) for x in v[-16:]])
    W2C = np.array(rows).reshape(-1, 4, 4)
    poses = np.linalg.inv(W2C)                     # c2w
    imgs = sorted(img_dir.glob("*.png"))
    n = min(len(imgs), len(poses))
    if len(imgs) != len(poses):
        print(f"WARNING: {len(imgs)} images vs {len(poses)} poses; truncating to {n}")
    imgs, poses = imgs[:n], poses[:n]
    c = poses[:, :3, 3]
    seg = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    np.savez_compressed(a.out, image_paths=np.array([str(p) for p in imgs]),
                        poses=poses, frame_ids=np.arange(n),
                        scene=f"vkitti_{a.world}_{a.var}")
    print(f"vkitti {a.world}/{a.var}: {n} frames, path {seg:.0f} m -> {a.out}")


if __name__ == "__main__":
    main()
