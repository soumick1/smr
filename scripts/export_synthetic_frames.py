#!/usr/bin/env python3
"""Render the synthetic room to PNG frames + GT poses, for feeding real
backbones (the week-3 backbone-swap sanity check)."""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio                      # noqa: E402
from smr.scene import Camera, build_room, camera_ring, render, fill_holes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/synth_frames")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--size", type=int, default=518)     # VGGT-native width
    ap.add_argument("--density", type=int, default=400000)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cam = Camera(H=a.size, W=a.size, f=a.size * 0.833)
    # dense + finely textured: correspondence-friendly for learned backbones
    pts, cols = build_room(seed=0, n_wall=a.density, checker=10,
                           tex_jitter=0.25, closed=True, distinct=True,
                           blob_n=max(1400, a.density // 60))
    Ts = camera_ring(a.views, radius=1.05, seed=0)
    paths, deps = [], []
    for i, T in enumerate(Ts):
        rgb, dep, msk = render(pts, cols, T, cam, splat=3)
        rgb, dep, msk = fill_holes(rgb, dep, msk, max_dist=6)
        p = out / f"view_{i:02d}.png"
        imageio.imwrite(p, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        paths.append(str(p)); deps.append(dep)
    np.savez(out / "gt.npz", poses=np.stack(Ts), K=cam.K,
             depth=np.stack(deps), paths=np.array(paths))
    print(f"wrote {a.views} frames + gt.npz -> {out}")


if __name__ == "__main__":
    main()
