#!/usr/bin/env python3
"""Extract ground-truth camera poses for a CO3D sequence.

Pilot A scores estimated trajectories against truth, so this is the file
that makes a real result possible at all.

CONVENTION, taken from CO3D's own schema (co3d/dataset/data_types.py:65,
"In right-multiply (PyTorch3D) format. X_cam = X_world @ R + T"):

    column convention:  X_cam = R^T X_world + T
    so                  R_w2c = R^T ,  t_w2c = T
    and                 R_c2w = R    ,  t_c2w = -R T

PyTorch3D cameras also look down +Z with +X LEFT and +Y UP, whereas we (and
OpenCV, and every backbone here) use +X right, +Y down.  That is a rotation
by pi about Z, D = diag(-1, -1, 1), applied on the RIGHT of the c2w
rotation:

    R_c2w_opencv = R @ D ,   t_c2w unchanged (D D = I)

Getting this wrong leaves camera CENTRES correct -- so ATE would look fine
-- while every orientation is off by 180 degrees, quietly wrecking RRA,
RPE-rot and AUC.  Hence --validate: it runs a real backbone on a handful of
frames and reports AUC@30 against the extracted poses.  A correct
extraction gives a high AUC; a convention error gives a near-zero one.
There is no reason to take my derivation on trust when it can be measured.

    python scripts/co3d_gt_poses.py --root ~/co3d_data --list
    python scripts/co3d_gt_poses.py --root ~/co3d_data --category apple \
        --out data/gt/apple.npz
    python scripts/co3d_gt_poses.py --root ~/co3d_data --category apple \
        --out data/gt/apple.npz --validate --backbone vggt
"""
import argparse, gzip, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLIP = np.diag([-1.0, -1.0, 1.0])          # PyTorch3D camera -> OpenCV


def pose_variants(R, T):
    """Every plausible reading of CO3D's (R, T), as camera-to-world.

    The schema says "right-multiply (PyTorch3D) format, X_cam = X_world @ R
    + T", which gives `row_w2c`.  But the axis convention (PyTorch3D has +X
    left / +Y up; we and OpenCV use +X right / +Y down) and the exact
    transpose are easy to get wrong, and a wrong choice leaves camera
    CENTRES plausible while ruining every orientation.  So rather than
    argue, we score all of them against a real backbone and take the one
    that wins by a mile.
    """
    out = {}
    for tag, R_w2c, t_w2c in (
            ("row_w2c", R.T, T),        # X_cam = R^T X_world + T
            ("col_w2c", R, T),          # X_cam = R   X_world + T
    ):
        Rc = R_w2c.T
        tc = -Rc @ t_w2c
        for flip, D in (("", np.eye(3)), ("_flip", FLIP)):
            P = np.eye(4)
            P[:3, :3] = Rc @ D
            P[:3, 3] = tc
            out[tag + flip] = P
    return out


def load_frame_annotations(category_dir):
    p = pathlib.Path(category_dir) / "frame_annotations.jgz"
    if not p.exists():
        raise SystemExit(
            f"{p} not found.\nThis file carries the GT poses.  NOTE: an "
            f"early version of scripts/co3d_cycle.sh deleted *.jgz during "
            f"pruning, so cycled categories have none -- use the download "
            f"subset (~/co3d_data), or re-download this category.")
    with gzip.open(p, "rt") as f:
        return json.load(f)


def sequences_in(category_dir):
    seqs = {}
    for fa in load_frame_annotations(category_dir):
        seqs.setdefault(fa["sequence_name"], 0)
        seqs[fa["sequence_name"]] += 1
    return seqs


def extract(category_dir, sequence=None, max_frames=0,
            convention="row_w2c_flip"):
    """-> (poses c2w (K,4,4), image paths, sequence name)"""
    cat = pathlib.Path(category_dir)
    anns = load_frame_annotations(cat)
    if sequence is None:
        counts = {}
        for fa in anns:
            counts[fa["sequence_name"]] = counts.get(fa["sequence_name"], 0) + 1
        sequence = max(counts, key=counts.get)      # the longest one
    rows = [fa for fa in anns if fa["sequence_name"] == sequence]
    rows.sort(key=lambda fa: fa["frame_number"])
    if max_frames:
        rows = rows[:max_frames]
    if len(rows) < 8:
        raise SystemExit(f"sequence {sequence} has only {len(rows)} frames")

    poses, paths = [], []
    root = cat.parent
    for fa in rows:
        R = np.asarray(fa["viewpoint"]["R"], float)     # right-multiply
        T = np.asarray(fa["viewpoint"]["T"], float)
        poses.append(pose_variants(R, T)[convention])
        ip = root / fa["image"]["path"]
        if not ip.exists():                 # paths are stored category-first
            ip = cat / pathlib.Path(fa["image"]["path"]).relative_to(cat.name)
        paths.append(str(ip))
    return np.stack(poses), paths, sequence


def diagnose(category_dir, sequence, backbone, device, n=10):
    """Score every candidate convention against ONE backbone run."""
    from smr.backbones import get_backbone
    from smr.eval.trajectory import ate_rmse, auc_at, pairwise_pose_errors
    anns = load_frame_annotations(pathlib.Path(category_dir))
    rows = sorted([fa for fa in anns if fa["sequence_name"] == sequence],
                  key=lambda fa: fa["frame_number"])
    step = max(1, len(rows) // n)
    sel = rows[::step][:n]
    root = pathlib.Path(category_dir).parent
    paths = [str(root / fa["image"]["path"]) for fa in sel]
    est = np.asarray(get_backbone(backbone, device=device).infer(paths).poses,
                     float)
    names = list(pose_variants(np.eye(3), np.zeros(3)))
    scores = {}
    for name in names:
        gt = np.stack([pose_variants(np.asarray(fa["viewpoint"]["R"], float),
                                     np.asarray(fa["viewpoint"]["T"], float))
                       [name] for fa in sel])
        rra, rta = pairwise_pose_errors(est, gt)
        scores[name] = dict(auc30=auc_at(rra, rta),
                            rra_med=float(np.median(rra)),
                            rta_med=float(np.median(rta)),
                            ate=ate_rmse(est, gt))
    return scores


def validate(poses, paths, backbone, device, n=10, contiguous=True):
    """Run a real backbone on n frames and score it against the extraction.

    Sample a CONTIGUOUS window by default.  Spreading n frames across a
    202-frame orbit puts ~45 degrees between neighbours, which several of
    these backbones simply fail at -- that failure then looks like a bad
    extraction.  The pilot feeds contiguous chunks, so validate the way the
    pilot runs.
    """
    from smr.backbones import get_backbone
    from smr.eval.trajectory import ate_rmse, auc_at, pairwise_pose_errors
    if contiguous:
        start = max(0, len(paths) // 2 - n // 2)
        idx = list(range(start, min(start + n, len(paths))))
    else:
        step = max(1, len(paths) // n)
        idx = list(range(0, len(paths), step))[:n]
    bb = get_backbone(backbone, device=device)
    out = bb.infer([paths[i] for i in idx])
    gt = poses[idx]
    rra, rta = pairwise_pose_errors(np.asarray(out.poses, float), gt)
    return dict(n=len(idx), auc30=auc_at(rra, rta),
                rra_median_deg=float(np.median(rra)),
                rta_median_deg=float(np.median(rta)),
                ate=ate_rmse(np.asarray(out.poses, float), gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. ~/co3d_data")
    ap.add_argument("--category", default="")
    ap.add_argument("--sequence", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--diagnose-convention", action="store_true",
                    help="score every candidate reading of (R,T) against one "
                         "backbone run and print the ranking")
    ap.add_argument("--convention", default="row_w2c_flip")
    ap.add_argument("--wide-baseline", action="store_true",
                    help="validate on frames spread across the sequence "
                         "instead of a contiguous window (harder; several "
                         "backbones fail there)")
    ap.add_argument("--backbone", default="vggt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    root = pathlib.Path(a.root).expanduser()
    if a.list:
        cats = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p / "frame_annotations.jgz").exists())
        print(f"{len(cats)} categories with GT annotations under {root}\n")
        for c in cats[:60]:
            try:
                seqs = sequences_in(c)
            except SystemExit:
                continue
            best = max(seqs, key=seqs.get)
            print(f"  {c.name:<16} {len(seqs)} seq, longest {best} "
                  f"({seqs[best]} frames)")
        return

    cat = root / a.category if a.category else None
    if cat is None:
        raise SystemExit("pass --category (or --list to see what is there)")
    if a.diagnose_convention:
        anns = load_frame_annotations(cat)
        counts = {}
        for fa in anns:
            counts[fa["sequence_name"]] = counts.get(fa["sequence_name"], 0) + 1
        seq = a.sequence or max(counts, key=counts.get)
        print(f"category {a.category} | sequence {seq} | scoring conventions "
              f"with {a.backbone} (one run, {10} frames)\n")
        sc = diagnose(cat, seq, a.backbone, a.device)
        print(f"  {'convention':<16}{'AUC@30':>9}{'RRA med':>10}"
              f"{'RTA med':>10}{'ATE':>10}")
        print("  " + "-" * 55)
        for k in sorted(sc, key=lambda k: -sc[k]["auc30"]):
            v = sc[k]
            print(f"  {k:<16}{v['auc30']:>9.1f}{v['rra_med']:>10.2f}"
                  f"{v['rta_med']:>10.2f}{v['ate']:>10.4f}")
        best = max(sc, key=lambda k: sc[k]["auc30"])
        print(f"\n  winner: {best} (AUC {sc[best]['auc30']:.1f})")
        print(f"  re-run with:  --convention {best}")
        return

    poses, paths, seq = extract(cat, a.sequence or None, a.max_frames,
                                a.convention)
    print(f"category {a.category} | sequence {seq} | {len(poses)} frames")
    missing = [p for p in paths[:5] if not pathlib.Path(p).exists()]
    if missing:
        print(f"  WARNING: image paths do not resolve, e.g. {missing[0]}")

    if a.validate:
        print(f"  validating the pose convention with {a.backbone} ...")
        v = validate(poses, paths, a.backbone, a.device,
                     contiguous=not a.wide_baseline)
        print(f"  AUC@30 {v['auc30']:.1f} | median RRA {v['rra_median_deg']:.2f} "
              f"deg | median RTA {v['rta_median_deg']:.2f} deg | ATE {v['ate']:.4f}")
        if v["auc30"] > 60:
            print("  -> convention CONFIRMED (a wrong flip would collapse this)")
        else:
            print("  -> SUSPECT. A near-zero AUC with sane ATE means the "
                  "camera-axis flip is wrong; check FLIP in this file.")

    if a.out:
        o = ROOT / a.out if not pathlib.Path(a.out).is_absolute() else pathlib.Path(a.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        np.savez(o, poses=poses, image_paths=np.array(paths),
                 sequence=seq, category=a.category, convention=a.convention)
        print(f"  wrote {o}  (poses {poses.shape}, image_paths included)")


if __name__ == "__main__":
    main()
