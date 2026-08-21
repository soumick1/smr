#!/usr/bin/env python3
"""Walk a dataset root of scene directories and harvest completion pairs
from each -- the established-corpus driver (CO3D / ScanNet++ / RE10K /
anything arranged as root/<scene>/<images>/*.jpg).

    python scripts/harvest_root.py --root data/co3d_frames \
           --images-sub "" --backbone vggt --out data/completion_co3d \
           --holdouts every:5 --context 1,4,all --max-views 32

Per scene: subsamples to --max-views evenly spaced frames (backbones and
binds scale as K^2), then calls harvest_completion_pairs once.  Skips
scenes already fully harvested (resume-safe via the inner script)."""
import argparse, pathlib, subprocess, sys, tempfile, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--images-sub", default="",
                    help="subdir inside each scene holding frames "
                         "('images_8' for LLFF-style; '' = scene dir itself)")
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdouts", default="every:5")
    ap.add_argument("--context", default="1,4,all")
    ap.add_argument("--max-views", type=int, default=32)
    ap.add_argument("--limit-scenes", type=int, default=0)
    a = ap.parse_args()

    scenes = sorted(p for p in pathlib.Path(a.root).iterdir() if p.is_dir())
    if a.limit_scenes:
        scenes = scenes[:a.limit_scenes]
    print(f"{len(scenes)} scenes under {a.root}")
    for sc in scenes:
        src = sc / a.images_sub if a.images_sub else sc
        imgs = sorted(sum([list(src.glob(p))
                           for p in ("*.png", "*.jpg", "*.jpeg", "*.JPG")],
                          []))
        if len(imgs) < 3:
            print(f"  [skip] {sc.name}: {len(imgs)} images"); continue
        if len(imgs) > a.max_views:
            step = len(imgs) / a.max_views
            pick = [imgs[int(i * step)] for i in range(a.max_views)]
        else:
            pick = imgs
        with tempfile.TemporaryDirectory() as td:
            # scene-named symlink dir so output files carry the scene name
            named = pathlib.Path(td) / sc.name
            named.mkdir()
            for q, p in enumerate(pick):
                (named / f"view_{q:03d}{p.suffix}").symlink_to(p.resolve())
            gt = src / "gt.npz"
            if gt.exists():        # synthetic-backbone smoke path; real
                (named / "gt.npz").symlink_to(gt.resolve())  # datasets are
                # GT-free and ignore this (note: synthetic loader reads the
                # full gt view list, so --max-views is a no-op there)
            r = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "harvest_completion_pairs.py"),
                 "--frames", str(named), "--backbone", a.backbone,
                 "--out", a.out, "--holdouts", a.holdouts,
                 "--context", a.context])
            if r.returncode != 0:
                print(f"  [warn] {sc.name}: harvester exit {r.returncode}")


if __name__ == "__main__":
    main()
