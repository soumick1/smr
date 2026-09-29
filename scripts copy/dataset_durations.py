#!/usr/bin/env python3
"""Physical duration and sampling rate of the pose-length sequences (CO3Dv2, RE10K).

    python scripts/dataset_durations.py --co3d-gt data/gt/co3d_full --co3d-root /path/to/CO3Dv2 \
        --re10k-root /path/to/RealEstate10K --re10k-list data/re10k/length_study_clips.txt --re10k-list10 data/re10k/test1800_clips.txt

CO3Dv2: for every GT npz in --co3d-gt (the 37 validated sequences; image paths give <category>/<sequence>/images/frameNNNNNN.jpg),
  read the category's frame_annotations.jgz under --co3d-root, take frame_timestamp (seconds) of the frames in the npz,
  and report n_frames, duration = t_last - t_first, effective fps, and for N in {10, 50, 100, 200}: N / n_frames and the
  mean temporal spacing duration / (N - 1) of a uniform sample across the trajectory (the paper's protocol).
RE10K: every clip listed (one clip id per line; default: all *.txt under --re10k-root/test) is a camera file whose lines
  after the URL are "timestamp_us fx fy cx cy 0 0 r11 ... t3"; n_frames = number of frame lines, duration from the
  timestamps, fps = n_frames / duration. The N = 10 evaluation uses all usable test-1800 clips (--re10k-list10), the length
  study the 100 clips with >= 200 recoverable frames (--re10k-list); both summaries are printed when the lists are given.
Prints per-dataset mean / median of duration, n_frames, fps, sampling fraction at each N, and temporal spacing.
"""
import argparse, glob, gzip, json, os, pathlib, re
import numpy as np

NS = (10, 50, 100, 200)


def summarize(name, rows):
    if not rows:
        print(f"{name}: no sequences"); return
    dur = np.array([r["duration"] for r in rows if r["duration"] is not None], float)
    n = np.array([r["n_frames"] for r in rows], float)
    print(f"\n== {name}: {len(rows)} sequences ==")
    print(f"frames per sequence: mean {n.mean():.0f}, median {np.median(n):.0f}, min {n.min():.0f}, max {n.max():.0f}")
    if len(dur):
        fps = n[: len(dur)] / np.maximum(dur, 1e-9) if len(dur) == len(n) else None
        print(f"duration (s): mean {dur.mean():.1f}, median {np.median(dur):.1f}, min {dur.min():.1f}, max {dur.max():.1f}"
              + (f"; effective frame rate: mean {fps.mean():.1f} fps, median {np.median(fps):.1f} fps" if fps is not None else ""))
    else:
        print("duration: no timestamps found (frame counts only)")
    for N in NS:
        frac = N / n
        line = f"N = {N:>3}: sampled fraction of frames mean {100 * frac.mean():.1f} %, median {100 * np.median(frac):.1f} %"
        if len(dur) == len(n):
            sp = dur / (N - 1)
            line += f"; mean temporal spacing {sp.mean():.2f} s (median {np.median(sp):.2f} s)"
        ok = (n >= N).sum()
        line += f"; sequences with >= N frames: {ok}/{len(n)}"
        print(line)


def co3d(gt_dir, root):
    rows, ann_cache = [], {}
    for p in sorted(glob.glob(str(pathlib.Path(gt_dir) / "*.npz"))):
        z = np.load(p, allow_pickle=True)
        paths = [str(x) for x in z["image_paths"]] if "image_paths" in z else []
        n = int(len(z["poses"]))
        duration = None
        if paths:
            m = re.search(r"([^/]+)/([^/]+)/images/frame(\d+)", paths[0])
            if m:
                cat, seq = m.group(1), m.group(2)
                if root is None:                       # infer: the directory that contains <category>/
                    idx = paths[0].find("/" + cat + "/" + seq + "/")
                    root = paths[0][:idx] if idx > 0 else None
                    if root:
                        print(f"  CO3D root inferred from image paths: {root}")
                if cat not in ann_cache:
                    f = pathlib.Path(root) / cat / "frame_annotations.jgz" if root else None
                    ann_cache[cat] = json.load(gzip.open(f)) if (f and f.exists()) else []
                    if not ann_cache[cat]:
                        print(f"  no frame_annotations.jgz for category {cat} under {root}; durations unavailable for it")
                frames = {int(re.search(r"frame(\d+)", q).group(1)) for q in paths if re.search(r"frame(\d+)", q)}
                seq_ann = [a for a in ann_cache[cat] if a["sequence_name"] == seq]
                ts = [a["frame_timestamp"] for a in seq_ann if a["frame_number"] in frames]
                if ts:
                    duration = float(max(ts) - min(ts))
                n_ann = len(seq_ann) or None
        rows.append(dict(id=pathlib.Path(p).stem, n_frames=n, duration=duration, n_annotated=locals().get("n_ann")))
    for r in rows:
        print(f"  {r['id']:<40} frames in GT {r['n_frames']:>4}  annotated in CO3D {r.get('n_annotated') or 'n/a':>4}  "
              f"duration {'%.1f s' % r['duration'] if r['duration'] is not None else 'n/a'}")
    na = [r["n_annotated"] for r in rows if r.get("n_annotated")]
    if na:
        na = np.array(na, float)
        print(f"  annotated frames per sequence in CO3D: mean {na.mean():.0f}, median {np.median(na):.0f}, min {na.min():.0f}, max {na.max():.0f}; "
              f"GT keeps {100 * np.mean([r['n_frames'] / r['n_annotated'] for r in rows if r.get('n_annotated')]):.0f} % of them")
    return rows


def _find_txt(root, clip):
    for sub in ("test", "train", ""):
        f = pathlib.Path(root) / sub / (clip + ".txt")
        if f.exists():
            return f
    hits = glob.glob(str(pathlib.Path(root) / "**" / (clip + ".txt")), recursive=True)
    return pathlib.Path(hits[0]) if hits else None


def re10k(root, clip_list=None, npz_glob=None):
    """Clips from a list file, or from GT npz files (stem = clip id, and the camera txt is searched under root or
    next to the frames named in image_paths), or every test/*.txt under root."""
    files = []
    if clip_list:
        files = [_find_txt(root, c.strip()) for c in open(clip_list) if c.strip()]
    elif npz_glob:
        for p in sorted(glob.glob(npz_glob)):
            clip = pathlib.Path(p).stem.replace("re10k_", "")
            f = _find_txt(root, clip) if root else None
            if f is None:
                z = np.load(p, allow_pickle=True)
                if "image_paths" in z and len(z["image_paths"]):
                    d = pathlib.Path(str(z["image_paths"][0])).parent
                    for up in [d] + list(d.parents)[:4]:
                        hits = glob.glob(str(up / "**" / (clip + ".txt")), recursive=True)
                        if hits:
                            f = pathlib.Path(hits[0]); break
            files.append(f)
    else:
        files = [pathlib.Path(f) for f in sorted(glob.glob(str(pathlib.Path(root) / "test" / "*.txt")))]
    missing = sum(1 for f in files if f is None or not f.exists())
    if missing:
        print(f"  {missing} of {len(files)} clips have no camera txt found")
    rows = []
    for f in files:
        if f is None or not f.exists():
            continue
        lines = [l for l in open(f) if l.strip()][1:]
        ts = np.array([float(l.split()[0]) for l in lines]) / 1e6
        if len(ts) < 2:
            continue
        rows.append(dict(id=f.stem, n_frames=len(ts), duration=float(ts.max() - ts.min())))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co3d-gt", default="data/gt/co3d_full"); ap.add_argument("--co3d-root", default=None)
    ap.add_argument("--re10k-root", default=None); ap.add_argument("--re10k-list", default=None, help="100 length-study clips")
    ap.add_argument("--re10k-list10", default=None, help="1,653 N=10 clips")
    ap.add_argument("--re10k-npz", default=None, help="glob of RE10K GT npz files (clip id = file stem), e.g. 'data/gt/re10k*/*.npz'")
    a = ap.parse_args()
    if os.path.isdir(a.co3d_gt):
        summarize("CO3Dv2 (37 validated sequences)", co3d(a.co3d_gt, a.co3d_root))
    else:
        print(f"no CO3D GT dir at {a.co3d_gt}")
    if a.re10k_npz is None and not a.re10k_root:
        cands = [g for g in glob.glob("data/gt/**/*.npz", recursive=True) if "re10k" in g.lower() or "realestate" in g.lower()]
        if cands:
            a.re10k_npz = "data/gt/**/*re10k*.npz" if any("re10k" in c.lower() for c in cands) else "data/gt/**/*realestate*.npz"
            print(f"RE10K GT npz files found: {len(cands)} (glob {a.re10k_npz})")
        else:
            print("no RE10K GT npz found under data/gt; pass --re10k-root and/or --re10k-npz")
    if a.re10k_list and a.re10k_root:
        summarize("RE10K length study (100 clips)", re10k(a.re10k_root, a.re10k_list))
    if a.re10k_list10 and a.re10k_root:
        summarize("RE10K N=10 evaluation (test-1800 clips)", re10k(a.re10k_root, a.re10k_list10))
    if a.re10k_npz:
        summarize("RE10K clips with GT npz", re10k(a.re10k_root, None, a.re10k_npz))
    elif a.re10k_root and not (a.re10k_list or a.re10k_list10):
        summarize("RE10K all test clips", re10k(a.re10k_root, None))