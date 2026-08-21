#!/usr/bin/env python3
"""Orbital video -> frames directory for the SMR harness.

    python scripts/video_to_frames.py --video orbit.mp4 --out orbit_frames --n 32

Samples ~3x the requested frame count via ffmpeg (system binary, or the
imageio-ffmpeg static binary as fallback), scores each candidate's
sharpness (Laplacian variance), keeps the sharpest frame per group, and
writes view_%03d.jpg upright (ffmpeg applies rotation metadata).  Frames
are resized so max(H, W) <= --maxdim; backbones resize to <=518 anyway."""
import argparse, pathlib, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image


def ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: apt install ffmpeg, or pip install "
                 "imageio-ffmpeg")


def sharpness(path):
    a = np.asarray(Image.open(path).convert("L"), float)
    lap = (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]
           - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--maxdim", type=int, default=1024)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dur = None
    probe = shutil.which("ffprobe")
    if probe:
        r = subprocess.run([probe, "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", a.video],
                           capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            dur = None
    with tempfile.TemporaryDirectory() as td:
        cand = pathlib.Path(td)
        n_cand = 3 * a.n
        scale = (f"scale='if(gt(iw,ih),{a.maxdim},-2)':"
                 f"'if(gt(iw,ih),-2,{a.maxdim})'")
        if dur:
            vf = f"{scale},fps={n_cand / dur:.4f}"
            subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", a.video,
                            "-vf", vf, "-q:v", "2",
                            str(cand / "c_%04d.jpg")], check=False)
        else:                          # no ffprobe: decode-and-decimate
            subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", a.video,
                            "-vf", scale, "-frames:v", str(n_cand * 20),
                            "-q:v", "2", str(cand / "c_%04d.jpg")],
                           check=False)
        got = sorted(cand.glob("c_*.jpg"))
        assert got, "ffmpeg produced no frames"
        idx = np.linspace(0, len(got) - 1, 3 * a.n).round().astype(int)
        idx = np.unique(idx)
        groups = np.array_split(idx, a.n)
        kept = []
        for k, grp in enumerate(groups):
            best = max(grp, key=lambda i: sharpness(got[i]))
            dst = out / f"view_{k:03d}.jpg"
            shutil.copy(got[best], dst)
            kept.append(dst)
    print(f"wrote {len(kept)} frames -> {out}  "
          f"(sharpest-in-group of {len(idx)} candidates)")


if __name__ == "__main__":
    main()
