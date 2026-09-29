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
    ap.add_argument("--bake-convention", default="right-flip",
                    choices=["none", "right-flip"],
                    help="bake the PyTorch3D->OpenCV camera-axis flip into the "
                         "written poses (needed by consumers without their own "
                         "convention handling, e.g. pilot_a)")
    ap.add_argument("--window", type=int, default=60,
                    help="temporal sampling window (frames); 0 disables -- use with "
                         "--n-frames 200 for the full-sequence Table-1b protocol")
    ap.add_argument("--max-adj-rot", type=float, default=30.0,
                    help="model-free GT validity check: drop a sequence if any "
                         "adjacent-frame GT rotation exceeds this (a smooth CO3D "
                         "orbit moves ~2 deg/frame; broken COLMAP jumps 30-170)")
    ap.add_argument("--min-quality", type=float, default=0.5,
                    help="drop sequences whose dataset-provided COLMAP viewpoint "
                         "quality score is below this (the RelPose++/PoseDiffusion "
                         "protocol filter; CO3D GT is auto-annotated and some "
                         "sequences' poses are broken)")
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
        # sequence-level GT quality filter (standard protocol)
        quality = {}
        sa = root / cat / "sequence_annotations.jgz"
        if sa.exists():
            for s_ in json.loads(gzip.open(sa, "rt").read()):
                q = s_.get("viewpoint_quality_score")
                if q is not None:
                    quality[s_["sequence_name"]] = float(q)
        else:
            print(f"  {cat}: sequence_annotations.jgz missing -- NO quality filter applied")
        by_seq = {}
        for fr in frames:
            if not (root / fr["image"]["path"]).exists():
                continue                       # annotation without an image on disk
            by_seq.setdefault(fr["sequence_name"], []).append(fr)
        # prefer the official test split when set_lists are present
        test_seqs = None
        for sl in sorted((root / cat).glob("set_lists*")):
            try:
                import json as _j
                data = _j.load(open(sl)) if sl.suffix == ".json" else None
                if isinstance(data, dict):
                    for k in data:
                        if "test" in k:
                            test_seqs = {row[0] for row in data[k]}
                            break
            except Exception:
                pass
            if test_seqs:
                break
        import numpy as _np
        def _gt_ok(fr_list):
            fr_s = sorted(fr_list, key=lambda x: x["frame_number"])
            Rp = None
            worst = 0.0
            for f in fr_s:
                R = _np.array(f["viewpoint"]["R"], float)
                if Rp is not None:
                    c = (_np.trace(Rp.T @ R) - 1) / 2
                    worst = max(worst, float(_np.degrees(_np.arccos(_np.clip(c, -1, 1)))))
                Rp = R
            return worst <= a.max_adj_rot, worst
        gt_bad = {s: _gt_ok(by_seq[s]) for s in by_seq}
        n_gtbad = sum(1 for s in gt_bad if not gt_bad[s][0])
        n_lowq = sum(1 for s in by_seq if quality.get(s, 1.0) < a.min_quality)
        pool = [s for s in sorted(by_seq)
                if (test_seqs is None or s in test_seqs) and len(by_seq[s]) >= a.n_frames
                and quality.get(s, 1.0) >= a.min_quality and gt_bad[s][0]]
        print(f"  {cat}: {len(by_seq)} sequences with images on disk, "
              f"{n_lowq} low-quality, {n_gtbad} broken-GT (adj-rot>{a.max_adj_rot:.0f}deg), "
              f"{len(pool)} usable"
              + (" (official test split)" if test_seqs else ""))
        seqs = pool[: a.max_seq]
        for sq in seqs:
            fr = sorted(by_seq[sq], key=lambda x: x["frame_number"])
            W = a.window                       # temporal window; 0 = whole sequence
            if W and len(fr) > W:
                s0 = int(rng.integers(0, len(fr) - W))
                sel = sorted(rng.choice(range(s0, s0 + W), a.n_frames, replace=False))
            else:
                sel = sorted(rng.choice(len(fr), min(a.n_frames, len(fr)), replace=False))
            paths, poses = [], []
            for i in sel:
                f = fr[i]
                R = np.array(f["viewpoint"]["R"], float)      # x_cam = x_world @ R + T
                T = np.array(f["viewpoint"]["T"], float)
                w2c = np.eye(4); w2c[:3, :3] = R.T; w2c[:3, 3] = T
                c2w = np.linalg.inv(w2c)
                if a.bake_convention == "right-flip":
                    c2w = c2w @ np.diag([-1.0, -1.0, 1.0, 1.0])
                poses.append(c2w)
                paths.append(str(root / f["image"]["path"]))
            np.savez_compressed(out_dir / f"{cat}_{sq}.npz",
                                image_paths=np.array(paths),
                                poses=np.stack(poses),
                                frame_ids=np.arange(a.n_frames),
                                scene=f"co3d_{cat}_{sq}")
            n_written += 1
        print(f"  {cat}: {len(seqs)} sequences prepared")
    print(f"wrote {n_written} sequence npz files -> {out_dir}")


if __name__ == "__main__":
    main()
