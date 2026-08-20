#!/usr/bin/env python3
"""Render the synthetic room to PNG frames + GT poses, for feeding real
backbones (the week-3 backbone-swap sanity check)."""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio                      # noqa: E402
from smr.scene import Camera, build_room, camera_ring, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/synth_frames")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--size", type=int, default=518)     # VGGT-native width
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cam = Camera(H=a.size, W=a.size, f=a.size * 0.833)
    pts, cols = build_room(seed=0, n_wall=90000)
    Ts = camera_ring(a.views, radius=1.05, seed=0)
    paths, deps = [], []
    for i, T in enumerate(Ts):
        rgb, dep, msk = render(pts, cols, T, cam)
        p = out / f"view_{i:02d}.png"
        imageio.imwrite(p, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        paths.append(str(p)); deps.append(dep)
    np.savez(out / "gt.npz", poses=np.stack(Ts), K=cam.K,
             depth=np.stack(deps), paths=np.array(paths))
    print(f"wrote {a.views} frames + gt.npz -> {out}")


if __name__ == "__main__":
    main()
