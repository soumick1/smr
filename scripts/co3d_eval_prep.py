#!/usr/bin/env python3
"""CO3Dv2 -> per-sequence npz for the Table-1 pose protocol.

    python scripts/co3d_eval_prep.py --root ~/co3d --categories apple,bench \
        --n-frames 10 --max-seq 20 --out-dir data/gt/co3d

Reads <root>/<category>/frame_annotations.jgz (CO3D v2), groups frames by
sequence, samples n-frames per sequence with a fixed seed, converts the
PyTorch3D world-to-camera (row-vector) convention to c2w, and writes one
npz per sequence.  --max-seq caps sequences per category so a scoped run
is a stated subset, not a silent one.
"""
import argparse, gzip, json, pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--max-seq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/gt/co3d")
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()
    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    n_written = 0
    for cat in a.categories.split(","):
        fa = root / cat / "frame_annotations.jgz"
        if not fa.exists():
            print(f"SKIP {cat}: {fa} missing"); continue
        frames = json.loads(gzip.open(fa, "rt").read())
        by_seq = {}
        for fr in frames:
            by_seq.setdefault(fr["sequence_name"], []).append(fr)
        seqs = sorted(by_seq)[: a.max_seq]
        for sq in seqs:
            fr = sorted(by_seq[sq], key=lambda x: x["frame_number"])
            if len(fr) < a.n_frames:
                continue
            sel = sorted(rng.choice(len(fr), a.n_frames, replace=False))
            paths, poses = [], []
            for i in sel:
                f = fr[i]
                R = np.array(f["viewpoint"]["R"], float)      # x_cam = x_world @ R + T
                T = np.array(f["viewpoint"]["T"], float)
                w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = T
                poses.append(np.linalg.inv(w2c))
                paths.append(str(root / f["image"]["path"]))
            np.savez_compressed(out_dir / f"{cat}_{sq}.npz",
                                image_paths=np.array(paths),
                                poses=np.stack(poses),
                                frame_ids=np.arange(a.n_frames),
                                scene=f"co3d_{cat}_{sq}")
            n_written += 1
        print(f"{cat}: {min(len(seqs), len(by_seq))} sequences prepared")
    print(f"wrote {n_written} sequence npz files -> {out_dir}")


if __name__ == "__main__":
    main()
