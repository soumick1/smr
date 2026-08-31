#!/usr/bin/env python3
"""RealEstate10K test clips -> per-clip npz (frames + c2w) for the pose protocol.

    pip install yt-dlp
    python scripts/re10k_fetch.py --camera-dir ~/re10k/test --out-dir data/gt/re10k \
        --frames-dir ~/re10k/frames --n-frames 10 --max-clips 300

--camera-dir holds the official per-clip txt files (line 1: video URL; then
per frame: timestamp_us fx fy cx cy k1 k2 + 12 numbers of the 3x4
world-from-camera... RE10K stores CAMERA-FROM-WORLD (w2c) row-major; we
invert to c2w).  Dead/blocked videos are skipped and listed; reruns resume.
"""
import argparse, pathlib, subprocess, sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--max-clips", type=int, default=300)
    ap.add_argument("--min-rows", type=int, default=0,
                    help="skip clips with fewer camera rows than this (use 200 "
                         "with --n-frames 200 for the input-length table)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cam_dir = pathlib.Path(a.camera_dir).expanduser()
    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fr_root = pathlib.Path(a.frames_dir).expanduser(); fr_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    txts = sorted(cam_dir.glob("*.txt"))[: a.max_clips]
    ok = dead = 0
    for txt in txts:
        clip = txt.stem
        out_npz = out_dir / f"{clip}.npz"
        if out_npz.exists():
            ok += 1; continue
        lines = txt.read_text().strip().splitlines()
        url = lines[0].strip()
        rows = [l.split() for l in lines[1:] if l.strip()]
        if len(rows) < max(a.n_frames, a.min_rows):
            continue
        sel = sorted(rng.choice(len(rows), a.n_frames, replace=False))
        ts = [int(rows[i][0]) for i in sel]
        poses = []
        for i in sel:
            M = np.array([float(x) for x in rows[i][7:19]]).reshape(3, 4)
            w2c = np.eye(4); w2c[:3, :] = M
            poses.append(np.linalg.inv(w2c))
        vid_dir = fr_root / clip; vid_dir.mkdir(exist_ok=True)
        mp4 = vid_dir / "video.mp4"
        if not mp4.exists():
            r = subprocess.run(["yt-dlp", "-q", "-f", "bv*[height<=720][ext=mp4]/bv*[ext=mp4]/best",
                                "-o", str(mp4), url], capture_output=True, text=True)
            if r.returncode != 0 or not mp4.exists():
                print(f"  DEAD {clip} ({url.split('=')[-1]})"); dead += 1; continue
        paths = []
        good = True
        for k, t_us in enumerate(ts):
            jpg = vid_dir / f"f{k:02d}_{t_us}.jpg"
            if not jpg.exists():
                r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t_us/1e6:.6f}",
                                    "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(jpg)],
                                   capture_output=True, text=True)
                if not jpg.exists():
                    good = False; break
            paths.append(str(jpg))
        if not good:
            print(f"  extract-fail {clip}"); continue
        mp4.unlink(missing_ok=True)                       # keep frames, drop the video
        np.savez_compressed(out_npz, image_paths=np.array(paths),
                            poses=np.stack(poses), frame_ids=np.arange(a.n_frames),
                            scene=f"re10k_{clip}")
        ok += 1
        if ok % 20 == 0:
            print(f"  [{ok} clips ready, {dead} dead]", flush=True)
    print(f"done: {ok} clips ready, {dead} dead -> {out_dir}")


if __name__ == "__main__":
    main()
